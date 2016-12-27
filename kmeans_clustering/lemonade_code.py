# -*- coding: utf-8 -*-
#!/usr/bin/env python

#
# Auto-generated Spark code from Lemonade Workflow
# (c) Speed Labs - Departamento de Ciência da Computação
#     Universidade Federal de Minas Gerais
# More information about Lemonade to be provided
#
import os
import json
import string
import sys
import time
import unicodedata
from pyspark.ml import Pipeline
from pyspark.ml.classification import *
from pyspark.ml.clustering import *
from pyspark.ml.evaluation import *
from pyspark.ml.feature import *
from pyspark.ml.tuning import *
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import *
from pyspark.sql.types import *
from timeit import default_timer as timer

reload(sys)
sys.setdefaultencoding('utf8')
# Global utilities functions definitions
strip_accents = udf(
    lambda text: ''.join(c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'), StringType())
strip_punctuation = udf(lambda text:
                        text.translate(
                            dict((ord(char), None)
                                for char in string.punctuation)),
                        StringType())

def juicer_debug(name, variable, df):
    """ Debug code """
    print '#' * 20
    print '|| {} ||'.format(name)
    print '== {} =='.format(variable)
    df.show()
    schema = df.schema
    for attr in schema:
        print attr.name, attr.dataType, attr.nullable, attr.metadata

def k_means_clustering_gen_df1(spark_session):
    start = timer()
    df1 = KMeans()
    df1.setMaxIter(10000)
    df1.setK(10)
    df1.setInitMode("k-means||")

    time_elapsed = timer() - start
    return df1, time_elapsed


def data_reader_gen_df3(spark_session):
    start = timer()
    schema_df3 = None
    url_df3 = 'hdfs://spark01.ctweb.inweb.org.br:9000/lemonade/samples/diabetic_data.csv'
    df3 = spark_session.read\
        .option('nullValue', '')\
        .option('treatEmptyValuesAsNulls',
                'true')\
        .csv(url_df3, schema=schema_df3,
             header=True, sep=',',
             inferSchema=True, mode='DROPMALFORMED')
    df3.cache()

    time_elapsed = timer() - start
    return df3, time_elapsed


def feature_indexer_gen_df4_models(spark_session, df3):
    start = timer()

    col_alias = dict(
        [
            [
                "gender", "gender_indexed"], [
                "race", "race_indexed"], [
                    "age", "age_indexed"], [
                        "weight", "weight_indexed"], [
                            "readmitted", "readmitted_indexed"], [
                                "diag1", "diag1_indexed"], [
                                    "diag2", "diag2_indexed"], [
                                        "diag3", "diag3_indexed"]])
    indexers = [StringIndexer(inputCol=col, outputCol=alias,
                              handleInvalid='skip')
                for col, alias in col_alias.iteritems()]

    # Use Pipeline to process all attributes once
    pipeline = Pipeline(stages=indexers)
    models = dict([(col[0], indexers[i].fit(df3)) for i, col in
                   enumerate(col_alias)])
    labels = [model.labels for model in models.itervalues()]

    # Spark ML 2.0.1 do not deal with null in indexer.
    # See SPARK-11569
    df3_without_null = df3.na.fill('NA', subset=col_alias.keys())

    df4 = pipeline.fit(df3_without_null).transform(df3_without_null)

    time_elapsed = timer() - start
    return df4, models, time_elapsed


def feature_assembler_gen_df0(spark_session, df4):
    start = timer()

    assembler = VectorAssembler(
        inputCols=[
            "admission_type_id",
            "discharge_disposition_id",
            "admission_source_id",
            "time_in_hospital",
            "num_lab_procedures",
            "num_procedures",
            "num_medications",
            "number_outpatient",
            "number_emergency",
            "number_inpatient",
            "number_diagnoses",
            "gender_indexed",
            "race_indexed",
            "age_indexed",
            "weight_indexed",
            "readmitted_indexed",
            "diag1_indexed",
            "diag2_indexed",
            "diag3_indexed"],
        outputCol="features")
    df4_without_null = df4.na.drop(
        subset=[
            "admission_type_id",
            "discharge_disposition_id",
            "admission_source_id",
            "time_in_hospital",
            "num_lab_procedures",
            "num_procedures",
            "num_medications",
            "number_outpatient",
            "number_emergency",
            "number_inpatient",
            "number_diagnoses",
            "gender_indexed",
            "race_indexed",
            "age_indexed",
            "weight_indexed",
            "readmitted_indexed",
            "diag1_indexed",
            "diag2_indexed",
            "diag3_indexed"])
    df0 = assembler.transform(df4_without_null)

    time_elapsed = timer() - start
    return df0, time_elapsed


def clustering_model_gen_df2_df2_model(spark_session, df0, df1):
    start = timer()

    df1.setFeaturesCol('features')
    df2_model = df1.fit(df0)
    # There is no way to pass which attribute was used in clustering, so
    # this information will be stored in uid (hack).
    df2_model.uid += '|features'
    df2 = df2_model.transform(df0)

    time_elapsed = timer() - start
    return df2, df2_model, time_elapsed


def projection_gen_df2_tmp_5(spark_session, df2):
    start = timer()
    df2_tmp_5 = df2.select("patient_nbr", "prediction")
    juicer_debug('juicer.spark.etl_operation.Select', 'df2_tmp_5', df2_tmp_5)

    time_elapsed = timer() - start
    return df2_tmp_5, time_elapsed


def main():
    start = time.time()
    app_name = u'## Experimento K-Means - Variação de Núcleos ##'

    spark_options = {
        "driver-library-path": '{}/lib/native/'.format(
            os.environ.get('HADOOP_HOME')),

    }
    builder = SparkSession.builder.appName(app_name)

    spark_session = builder.getOrCreate()
    for option, value in spark_options.iteritems():
        spark_session.conf.set(option, value)

    session_start_time = time.time()
    # spark_session.sparkContext.addPyFile('/tmp/dist.zip')

    # Declares and initializes variabels in order to do not generate NameError.
    # Some tasks may not generate code, but at least one of their outputs is
    # connected to a valid input in a task generating code. This happens when
    # task has a port with multiplicity MANY
    df1, ts_df1 = k_means_clustering_gen_df1(spark_session)
    df3, ts_df3 = data_reader_gen_df3(spark_session)
    df4, models, ts_df4 = feature_indexer_gen_df4_models(spark_session, df3)
    df0, ts_df0 = feature_assembler_gen_df0(spark_session, df4)
    df2, df2_model, ts_df2 = clustering_model_gen_df2_df2_model(
        spark_session, df0, df1)
    df2_tmp_5, ts_df2_tmp_5 = projection_gen_df2_tmp_5(spark_session, df2)

    end = time.time()
    print "{}\t{}".format(end - start, end - session_start_time)
    return {
        '114c207c-5371-4037-9525-9f470a9e15af': (df1, ts_df1),
        '1e5b2f54-cd43-480b-9735-587e0584b283': (df3, ts_df3),
        '0ee7bbbe-c786-4529-9a67-0c1bd8830fc8': (df4, models, ts_df4),
        'dc773e75-f591-4f17-a3ba-6c23fb0324ab': (df0, ts_df0),
        '42380b12-1ddc-40c3-a2a6-c4fdd9e65194': (df2, df2_model, ts_df2),
        'c69dbcc6-71af-4270-bdd3-8f80d14a0b8b': (df2_tmp_5, ts_df2_tmp_5),
    }
