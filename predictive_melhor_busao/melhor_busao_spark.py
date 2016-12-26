#!/usr/bin/env python
# -*- coding: latin-1 -*-

"""
Build trips location with social data .

the file can be local or if you setup cluster.

It can be hdfs file path

"""

## Imports

# pyspark
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, Row
from pyspark.sql.functions import regexp_replace, col, round, unix_timestamp, udf, row_number, first, count, when, last
from pyspark.sql.types import *

# geospatial
import shapefile
from pygeocoder import Geocoder
from matplotlib.path import Path

# time
import time
from operator import add
from datetime import datetime

# tools
from itertools import chain

# logging
import logging
logging.basicConfig(level=logging.INFO)

# sys
import sys  
reload(sys) 
#sys.setdefaultencoding('utf-8') 
# External file
# import trajectory_miner

## Constants
APP_NAME = "Build trips location"

##OTHER FUNCTIONS/CLASSES
def get_sum_hour_minute(x):
	formatted_date = '%Y-%m-%d %H:%M:%S'
	date_ = datetime.strptime(x, formatted_date)
	return (60*date_.hour+date_.minute)

def replacerUDF(value):
	return udf(lambda x: get_sum_hour_minute(x))

''' 
	Function to execute Step 2 - Remove columns, Replace Values, create bins (of time),
	Select values, Groupby gps and join ticketing and gps.
'''
def build_user_trips_origin_dataset(ticketing_data_df, gps_data_df, group_size):
	print "=== (Step 2 - User trips origin)"

	# Remove coluna de ID
	print " === Remove columns of ID"
	gps_data_df = gps_data_df.drop("ID_POSONIBUS")
	ticketing_data_df = ticketing_data_df.drop("ID_BILHETAGEM")

	# replace , in a column dataframe
	print " === Replace ',' on the columns "
	gps_data_df = gps_data_df.withColumn("LON", regexp_replace(col("LON"), ",", ".").alias('LONGITUDE'))
	gps_data_df = gps_data_df.withColumn("LAT", regexp_replace(col("LAT"), ",", ".").alias('LATITUDE'))

	# user defined function
	# Bins of 5 minutes GPS data
	print " === Create bins every 5 minutes - gps "
	gps_data_df = gps_data_df.withColumn("group5min", round(replacerUDF(group_size)(col("DTHR"))/group_size))
	# gps_data_df = gps_data_df.withColumn("group5min", replacerUDF(group_size)(col("DTHR")))
	
	# Bins of 5 minutes Ticketing data
	print " === Create bins every 5 minutes - ticketing "
	ticketing_data_df = ticketing_data_df.withColumn("group5min", round(replacerUDF(group_size)(col("DATAUTILIZACAO"))/group_size))

	print " === Selecting columns"
	ticketing_data_df = ticketing_data_df.select('CODLINHA','CODVEICULO','group5min','NUMEROCARTAO','DATAUTILIZACAO'). \
		orderBy(['CODLINHA','CODVEICULO','group5min','DATAUTILIZACAO']). \
		cache()


	print " === Selecting the first element of groupBy "
	grouped_data_df = gps_data_df.groupBy(['COD_LINHA','VEIC','group5min']).agg(first('LAT').alias('LAT'), first('LON').alias('LON')). \
		orderBy(['COD_LINHA','VEIC','group5min']). \
		filter(row_number() == 1).cache()

	## Trips users
	print " === Inner join - gps+ticketing "
	users_trips = ticketing_data_df.join(grouped_data_df, (ticketing_data_df.CODLINHA == grouped_data_df.COD_LINHA) & \
		(ticketing_data_df.CODVEICULO == grouped_data_df.VEIC ) & (ticketing_data_df.group5min == grouped_data_df.group5min), 'inner'). \
		select(ticketing_data_df.CODLINHA,ticketing_data_df.CODVEICULO,ticketing_data_df.group5min,ticketing_data_df.NUMEROCARTAO, \
			ticketing_data_df.DATAUTILIZACAO,grouped_data_df.LAT, grouped_data_df.LON). \
		orderBy(['NUMEROCARTAO','DATAUTILIZACAO']). \
		cache()

	print " === Freq user by card number "
	count_users_trips_df = users_trips.groupBy('NUMEROCARTAO').\
		agg(count('NUMEROCARTAO').alias('COUNT')). \
		select('NUMEROCARTAO','COUNT').\
		filter('COUNT >= 2'). \
		cache()

	### Fazer inner join novamente
	'''
	SQL:
		SELECT 	td.CODLINHA, td.CODVEICULO, td.group5min,td.NUMEROCARTAO, td.DATAUTILIZACAO,gd.LAT, gd.LON
		FROM grouped_data_df gd INNER JOIN ticketing_data_df td
		ON 	(td.CODLINHA == gd.COD_LINHA) & (td.CODVEICULO == gd.VEIC ) & (td.group5min == gd.group5min)
		ORDER BY NUMEROCARTAO, DATAUTILIZACAO
	'''
	users_trips = users_trips.join(count_users_trips_df, (users_trips.NUMEROCARTAO == count_users_trips_df.NUMEROCARTAO), 'inner'). \
		select(users_trips.CODLINHA,users_trips.CODVEICULO,users_trips.group5min,users_trips.NUMEROCARTAO,users_trips.DATAUTILIZACAO, \
			users_trips.LAT, users_trips.LON).\
		orderBy(['NUMEROCARTAO','DATAUTILIZACAO'])

	print " === Drop user that are not duplicated "
	### Drop Duplicated
	users_trips = users_trips.dropDuplicates().cache()
	return users_trips


''' 
	Function to execute Step 3 - Find sector of each user and enrich the data with sector, neighboorhood name, and neighborhood code.
'''
def match_locations_to_city_sectors(city_zone_shp_str, users_trip_location):
	print "=== (Step 3 - Find locations zones)"
	# Opens the Shapefile
	shape = shapefile.Reader(city_zone_shp_str)
	sectors = shape.shapeRecords()


	# Filter Curitiba's sectors
	curitiba_sectors = []
	for current_sector in sectors:
		if current_sector.record[10] == 'CURITIBA':
			sector = {}
			# Get sector border points (reversed order)
			sector['points'] = []
			for current_point in current_sector.shape.points:
				point = []
				point.append(current_point[1])
				point.append(current_point[0])
				sector['points'].append(point)
			sector['paths'] = Path(sector['points'])
			# Get id
			# sector['id'] = current_sector.record[0]
			sector['id'] = current_sector.record[3]
			# Get sector setor
			sector['setor'] = current_sector.record[1]
			sector['bairro'] = current_sector.record[4]
			curitiba_sectors.append(sector)

	### Mostra se cada ponto pertence ou nao em um setor
	# @data: points (lat,lon) and curitiba_sectors
	###

	# Adding broadcast variable
	broadcastVar = sc.broadcast(curitiba_sectors)
	# print broadcastVar.value[0]['setor']

	# rdd_result = users_trip_location.select("LAT","LON").rdd.map( \
	# 		lambda row: (map(lambda n: n['paths'].contains_point([float(row[0]),float(row[1])]), broadcastVar.value),float(row[0]),float(row[1])) \
	# 	). \
	# 		map( \
	# 			lambda rows: list(chain.from_iterable(((broadcastVar.value[id]['setor']),(broadcastVar.value[id]['setor']), \
	# 				(broadcastVar.value[id]['bairro']).decode("latin-1"),rows[1], rows[2]) \
	# 				for id in xrange(0,len(rows[0])) if rows[0][id] == True)) \
	# 		). \
	# 			filter(lambda x: len(x) > 0). \
	# 			filter(lambda y: len(y) <=5).cache()


	rdd_result = users_trip_location.select("LAT","LON").rdd.map(lambda row: \
							(  \
								map(lambda n: n['paths'].contains_point([float(row[0]),float(row[1])]), curitiba_sectors),  float(row[0]), float(row[1]) \
							) \
					).map(lambda rows: list(chain.from_iterable(\
																next(\
																	( (curitiba_sectors[id]['id']),(curitiba_sectors[id]['setor']),\
																		(curitiba_sectors[id]['bairro']).decode("latin-1"),rows[1], rows[2] \
																	))for id in xrange(0,len(rows[0])) if rows[0][id] == True )\
				 						)\
				 )#filter(lambda x: len(x) > 0)

					

	# Without broadcast variable
	# rdd_result = users_trip_location.select("LAT","LON").rdd.map( \
	# 		lambda row: (map(lambda n: n['paths'].contains_point([float(row[0]),float(row[1])]), curitiba_sectors),float(row[0]),float(row[1])) \
	# 	). \
	# 		map( \
	# 			lambda rows: list(chain.from_iterable(((curitiba_sectors[id]['id']),(curitiba_sectors[id]['setor']), \
	# 				(curitiba_sectors[id]['bairro']).decode("latin-1"),rows[1], rows[2]) \
	# 				for id in xrange(0,len(rows[0])) if rows[0][id] == True)) \
	# 		). \
	# 			filter(lambda x: len(x) > 0). \
	# 			filter(lambda y: len(y) <=5)
	logging.info ("dataframe to rdd.")
	
	# Convert rdd to Dataframe
	proj_locations_zones = rdd_result.toDF(['id', 'setor','bairro','latitude','longitude']).cache()
	logging.info("rdd to df")

	'''
	SQL:
		SELECT 	ut.CODLINHA, ut.CODVEICUlO, ut.group5min, ut.NUMEROCARTAO,ut.DATAUTILIZACAO, ut.LAT, ut.LON, x.id, x.setor, x.bairro
		FROM users_trip_location ut INNER JOIN x
		ON 	((ut.LAT == x.latitude) & (ut.LON == x.longitude))
		ORDER BY NUMEROCARTAO, DATAUTILIZACAO
	'''
	matched_locations_zones = users_trip_location.join(proj_locations_zones, (users_trip_location.LAT == proj_locations_zones.latitude) & \
			(users_trip_location.LON == proj_locations_zones.longitude),'inner'). \
		select(users_trip_location.CODLINHA,users_trip_location.CODVEICULO, users_trip_location.group5min, users_trip_location.NUMEROCARTAO, \
			users_trip_location.DATAUTILIZACAO, users_trip_location.LAT, users_trip_location.LON, proj_locations_zones.id, \
			proj_locations_zones.setor, proj_locations_zones.bairro). \
		cache()

	return matched_locations_zones

''' 
	Function to execute Step 4 - Enrich the sector trip data with socioeconomic data from each sector/zones.
'''
def build_user_trips_analysis_dataset(locations_sectors_match,social_data):
	print "=== (Step 4 - Fill data with social data)"

	'''
	SQL:
		SELECT codlinha, codveiculo,group_5_min, timestamp, numerocartao, latitude, longitude, CODSETOR, CODBAIRR, NOMEBAIR
		FROM locations_sectors_match
	'''
	locations_sectors_match_df = locations_sectors_match. \
		select('CODLINHA','CODVEICULO','group5min','NUMEROCARTAO','DATAUTILIZACAO', 'LAT', 'LON', 'id','setor', 'bairro')

	print " === Freq user by card number "
	# users_trips = users_trips.groupBy('NUMEROCARTAO').agg(count('NUMEROCARTAO').alias('COUNT'))
	'''
	SQL:
		SELECT 	numerocartao, count(numerocartao) as COUNT
		FROM locations_sectors_match_df
		GROUP BY numerocartao
		HAVING COUNT >= 2
	'''
	count_location_users_trips_df = locations_sectors_match_df.groupBy('NUMEROCARTAO').\
		agg(count('NUMEROCARTAO').alias('COUNT')). \
		select('NUMEROCARTAO','COUNT').\
		filter('COUNT >= 2').cache()


	print " === users and trips > 2 "

	'''
	SQL:
		SELECT 	ls.codlinha, ls.codveiculo, ls.group_5_min, ls.timestamp,ls.latitude, ls.longitude, ls.CODSETOR, ls.CODBAIRR, ls.NOMEBAIR, ls.numerocartao
		FROM count_location_users_trips_df cl INNER JOIN locations_sectors_match_df ls
		ON (ls.numerocartao == cl.numerocartao)
		ORDER BY numerocartao, timestamp
	'''
	# Inner join to get only users > 2		

	locations_sectors_match_df = locations_sectors_match_df.join(count_location_users_trips_df, \
			(locations_sectors_match_df.NUMEROCARTAO == count_location_users_trips_df.NUMEROCARTAO), 'inner').\
		select(locations_sectors_match_df.CODLINHA, locations_sectors_match_df.CODVEICULO,locations_sectors_match_df.group5min, \
			locations_sectors_match_df.DATAUTILIZACAO, locations_sectors_match_df.LAT, locations_sectors_match_df.LON, \
			locations_sectors_match_df.setor, locations_sectors_match_df.id, locations_sectors_match_df.bairro, \
			locations_sectors_match_df.NUMEROCARTAO). \
		orderBy(['NUMEROCARTAO','DATAUTILIZACAO']). \
		cache()
	# count_location_users_trips_df.unpersist()

	print " === Filtering social data fields "
	social_data_df = social_data.select('CODSETOR', 'CODSETTX','BA_002', 'BA_005','P1_001'). \
		filter(social_data.CODSETOR.isNotNull() & social_data.CODSETTX.isNotNull() & \
			social_data.BA_001.isNotNull()& social_data.BA_002.isNotNull() & \
			social_data.BA_003.isNotNull() & social_data.BA_005.isNotNull() & \
			social_data.BA_007.isNotNull() & social_data.BA_009.isNotNull() & \
			social_data.BA_011.isNotNull() & social_data.P1_001.isNotNull()). \
		na.drop('any', subset=['CODSETTX']). \
		cache() #,'CODSETTX')
	print " === Adding Social data to Trip zones "
	'''
	SQL:
		SELECT 	ls.numerocartao, ls.codlinha, ls.codveiculo, ls.timestamp, ls.CODSETOR, ls.CODBAIRR, ls.NOMEBAIR, ls.latitude, ls.longitude, sd.BA_002, sd.BA_005, sd.P1_001
		FROM locations_sectors_match_df ls INNER JOIN social_data_df sd
		ON (ls.CODSETOR == sd.CODSETOR)
		ORDER BY numerocartao, timestamp
	'''
	# cod.bairro, latitude, longitude, populacao, renda, num.alfabetizados)
	
	trip_zone_social_data = locations_sectors_match_df.join(social_data_df, (locations_sectors_match_df.setor == social_data_df.CODSETOR) , 'inner'). \
		select(locations_sectors_match_df.NUMEROCARTAO,locations_sectors_match_df.CODLINHA,locations_sectors_match_df.CODVEICULO, \
			locations_sectors_match_df.DATAUTILIZACAO, locations_sectors_match_df.setor, locations_sectors_match_df.id, \
			locations_sectors_match_df.bairro, locations_sectors_match_df.LAT, locations_sectors_match_df.LON, social_data_df.BA_002, \
			social_data_df.BA_005, social_data_df.P1_001 ). \
		orderBy(['NUMEROCARTAO','DATAUTILIZACAO'])

	print " === Defining orig/dest vairables "

	'''
	SQL:
		SELECT 	numerocartao, first('CODBAIRR')  AS o_neight_code,
							  first('NOMEBAIR')  AS o_neight_name,
							  first('latitude')  AS o_lat,
							  first('longitude') AS o_long,
							  first('timestamp') AS o_timestamp,
							  first('BA_002') AS o_num_pop,
							  first('BA_005') AS o_renda,
							  first('P1_001') AS o_num_alfa,
							  last('CODBAIRR') AS d_neight_code, 
							  last('NOMEBAIR') AS d_neight_name, 
							  last('latitude') AS d_lat, 
							  last('longitude') AS d_long, 
							  last('timestamp') ASd_timestamp,
							  last('BA_002') AS	d_num_pop,
							  last('BA_005') AS	d_renda,
							  last('P1_001') AS	d_num_alfa
		FROM trip_zone_social_data
		GROUP BY numerocartao
	'''

	print " === Select first and last trip and ordering by card "
	trip_zone_social_data_final = trip_zone_social_data.groupBy('NUMEROCARTAO'). \
		agg( first('id').alias('o_neight_code'), \
			first('bairro').alias('o_neight_name'), \
			first('LAT').alias('o_lat'), \
			first('LON').alias('o_long'), \
			first('DATAUTILIZACAO').alias('o_timestamp'), \
			first('BA_002').alias('o_num_pop'), \
			first('BA_005').alias('o_renda'), \
			first('P1_001').alias('o_num_alfa'), 
			last('bairro').alias('d_neight_code'), \
			last('bairro').alias('d_neight_name'), \
			last('LAT').alias('d_lat'), \
			last('LON').alias('d_long'), \
			last('DATAUTILIZACAO').alias('d_timestamp'), \
			last('BA_002').alias('d_num_pop'), \
			last('BA_005').alias('d_renda'), \
			last('P1_001').alias('d_num_alfa')). \
		select('NUMEROCARTAO','o_neight_code','o_neight_name','o_lat','o_long','o_timestamp', 'o_num_pop','o_renda','o_num_alfa', \
				'd_neight_code', 'd_neight_name', 'd_lat', 'd_long', 'd_timestamp','d_num_pop','d_renda','d_num_alfa'). \
		orderBy('o_timestamp'). \
		cache()

	return trip_zone_social_data_final


def main(sqlContext, ticketing_data, gps_data, city_zone_shape, city_zone_name_layer, trip_social_data, trip_location_output):
	
	### = Step 1
	# @datas: ticketing, gps_bus, social_data and city_zone_shp
	###
	ticketing_data_df = sqlContext.read.csv(ticketing_data, sep=";", header="True")
	gps_data_df = sqlContext.read.csv(gps_data, sep=";", header="True")
	social_data_df = sqlContext.read.csv(trip_social_data, sep=";", header="True")
	city_zone_shp_str = city_zone_shape

	### = Step 2
	# @data: ticketing, gps_bus and minutos to create bins of time
	###
	minuteGroup = 5

	users_trip_location = build_user_trips_origin_dataset(ticketing_data_df, gps_data_df, minuteGroup)

	logging.info("=== users trip location calculated")

	### Save STEP 2
	start_time = time.time()
	users_trip_location.write.csv("/gavelar/user_trip_location", mode="overwrite",sep=";", header="True")
	print("=== time to write step 2 - %s seconds" % (time.time() - start_time))
	
	### = Step 3
	location_sectors_match = match_locations_to_city_sectors(city_zone_shp_str, users_trip_location)
	# location_sectors_match = match_locations_to_city_sectors(city_zone_shp_str, users_trip_location_1)

	### SAVE STEP 3 - Slow to.
	start_time = time.time()
	location_sectors_match.write.csv("/gavelar/user_trip_location_sectors", mode="overwrite",sep=";", header="True")
	print("=== time to write step 3 - %s seconds" % (time.time() - start_time))

	### = Step 4
	# @datas: location_sector_match and social_data
	###
	trip_zone_social_data = build_user_trips_analysis_dataset(location_sectors_match,social_data_df)

	### = Step 5
	# data: Write file with the origin and destiny from each user
	###
	# replace for output_directory_file
	start_time = time.time()
	trip_zone_social_data.write.csv("/gavelar/trip_zone_social_data", mode="overwrite",sep=";", header="True")
	print("=== time to write step 5 - %s seconds" % (time.time() - start_time))

	# print "=== Writing results"
	logging.info('Finished')
	# return 0
	# liberar variables cached



##Main function
if __name__ == "__main__":
	### Configure Spark
	conf = SparkConf().setAppName(APP_NAME)
	### Run locally
	# conf = conf.setMaster("local[4]")
	conf = conf.setMaster("spark://spark01:7077").set("spark.executor.memory", "8g")
	
	### Set spark Context and sqlContext
	sc = SparkContext(conf=conf)
	
	### Add a external file (to test)
	sc.addPyFile("/mnt/gavelar/bigsea_transport/trajectory_miner.py")
	sqlContext = SQLContext(sc)

	# Ticketing data from bus users
	ticketing_data = sys.argv[1]
	# Gps data from bus
	gps_data = sys.argv[2]
	# City zone shape
	city_zone_shape = sys.argv[3]
	# City zone shape layer name
	city_zone_name_layer = sys.argv[4]
	# Trip zone/region data
	trip_social_data = sys.argv[5]
	# Outputfile
	trip_location_output = sys.argv[6]
	# Execute Main functionality
	main(sqlContext, ticketing_data, gps_data, city_zone_shape, city_zone_name_layer, trip_social_data, trip_location_output)
  
