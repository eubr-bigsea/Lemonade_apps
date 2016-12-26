# -*- coding: utf-8 -*-
#!/usr/bin/env python

#
# Auto-generated Spark code from Lemonade Workflow
# (c) Speed Labs - Departamento de Ciência da Computação
#     Universidade Federal de Minas Gerais
# More information about Lemonade to be provided
#
import sys
import string
import json
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
sys.setdefaultencoding("utf-8")

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

def data_reader_gen_df6(spark_session):
    """
    Lista de stop words em portugues
    """
    start = timer()
    schema_df6 = StructType()
    schema_df6.add('stop_word', StringType(), False,
                   {'type': u'TEXT'})

    url_df6 = 'hdfs://spark01.ctweb.inweb.org.br:9000/lemonade/samples/portugues.txt'
    df6 = spark_session.read\
        .option('nullValue', '')\
        .option('treatEmptyValuesAsNulls',
                'true')\
        .csv(url_df6, schema=schema_df6,
             header=False, sep=',',
             inferSchema=False, mode='DROPMALFORMED')
    df6.cache()

    time_elapsed = timer() - start
    return df6, time_elapsed


def lda_clustering_gen_df0(spark_session):
    """
    Configuração do algoritmo LDA
    """
    start = timer()
    df0 = LDA()
    df0.setMaxIter(10)
    df0.setK(10)
    df0.setOptimizer('online')

    time_elapsed = timer() - start
    return df0, time_elapsed


def data_reader_gen_df4(spark_session):
    """
    Corpus

    """
    start = timer()
    schema_df4 = StructType()
    schema_df4.add('id', StringType(), False,
                   {'type': u'TEXT'})
    schema_df4.add('title', StringType(), False,
                   {'type': u'TEXT'})

    url_df4 = 'hdfs://spark01.ctweb.inweb.org.br:9000/lemonade/samples/corpus.txt'
    df4 = spark_session.read\
        .option('nullValue', '')\
        .option('treatEmptyValuesAsNulls',
                'true')\
        .csv(url_df4, schema=schema_df4,
             header=False, sep='\t',
             inferSchema=False, mode='DROPMALFORMED')
    df4.cache()

    time_elapsed = timer() - start
    return df4, time_elapsed


def transformation_gen_df5(spark_session, df4):
    """
    Remove acentos e pontuação
    """
    start = timer()
    df5 = df4.withColumn(
        'title_transformed',
        strip_punctuation(
            strip_accents(
                col('title'))))

    time_elapsed = timer() - start
    return df5, time_elapsed


def tokenizer_gen_df1(spark_session, df5):
    """
    Divide texto em palavras
    """
    start = timer()

    col_alias = [["title_transformed", "words"]]
    tokenizers = [Tokenizer(inputCol=col, outputCol=alias)
                  for col, alias in col_alias]

    # Use Pipeline to process all attributes once
    pipeline = Pipeline(stages=tokenizers)

    df1 = pipeline.fit(df5).transform(df5)

    time_elapsed = timer() - start
    return df1, time_elapsed


def remove_stop_words_gen_df2(spark_session, df1, df6):
    """
    Remove stop words
    """
    start = timer()
    sw = [stop[0].strip() for stop in df6.collect()]
    col_alias = [["words", "words2"]]
    removers = [StopWordsRemover(inputCol=col, outputCol=alias,
                                 stopWords=sw)for col, alias in col_alias]

    # Use Pipeline to process all attributes once
    pipeline = Pipeline(stages=removers)

    df2 = pipeline.fit(df1).transform(df1)

    time_elapsed = timer() - start
    return df2, time_elapsed


def word_to_vector_gen_df3_df7(spark_session, df2):
    """
    Transforma as palavras em vetores
    """
    start = timer()

    col_alias = [["words2", "features"]]
    vectorizers = [CountVectorizer(minTF=1, minDF=5,
                                   vocabSize=10000, binary=False, inputCol=col,
                                   outputCol=alias) for col, alias in col_alias]
    # Use Pipeline to process all attributes once
    pipeline = Pipeline(stages=vectorizers)
    model = pipeline.fit(df2)
    df3 = model.transform(df2)

    df7 = dict([(col_alias[i][1], v.vocabulary)
                for i, v in enumerate(model.stages)])
    juicer_debug(
        'juicer.spark.text_operation.WordToVectorOperation',
        'df3',
        df3)

    time_elapsed = timer() - start
    return df3, df7, time_elapsed


def clustering_model_gen_df8_df9(spark_session, df3, df0):
    start = timer()

    df0.setFeaturesCol('features')
    df9 = df0.fit(df3)
    # There is no way to pass which attribute was used in clustering, so
    # this information will be stored in uid (hack).
    df9.uid += '|features'
    df8 = df9.transform(df3)

    time_elapsed = timer() - start
    return df8, df9, time_elapsed


def topic_report_gen_df7_tmp_9(spark_session, df7, df8, df9):
    start = timer()

    topic_df = df9.describeTopics(maxTermsPerTopic=10)
    # See hack in ClusteringModelOperation
    features = df9.uid.split('|')[1]
    for row in topic_df.collect():
        topic_number = row[0]
        topic_terms = row[1]
        print "Topic: ", topic_number
        print '========================='
        print '\t',
        for inx in topic_terms[:10]:
            try:
                print df7[features][inx],
            except:
                pass
        print
    df7_tmp_9 = df8

    time_elapsed = timer() - start
    return df7_tmp_9, time_elapsed


def main():
    app_name = u'## Clustering example ##'
    spark_session = SparkSession.builder\
        .appName(app_name)\
        .getOrCreate()
    # spark_session.sparkContext.addPyFile('/tmp/dist.zip')

    # Declares and initializes variabels in order to do not generate NameError.
    # Some tasks may not generate code, but at least one of their outputs is
    # connected to a valid input in a task generating code. This happens when
    # task has a port with multiplicity MANY
    df6, ts_df6 = data_reader_gen_df6(spark_session)
    df0, ts_df0 = lda_clustering_gen_df0(spark_session)
    df4, ts_df4 = data_reader_gen_df4(spark_session)
    df5, ts_df5 = transformation_gen_df5(spark_session, df4)
    df1, ts_df1 = tokenizer_gen_df1(spark_session, df5)
    df2, ts_df2 = remove_stop_words_gen_df2(spark_session, df1, df6)
    df3, df7, ts_df3 = word_to_vector_gen_df3_df7(spark_session, df2)
    df8, df9, ts_df8 = clustering_model_gen_df8_df9(spark_session, df3, df0)
    df7_tmp_9, ts_df7_tmp_9 = topic_report_gen_df7_tmp_9(
        spark_session, df7, df8, df9)

    return {
        '65e7bec7-caf3-462b-a0b7-28684d09b90a': (df6, ts_df6),
        'dd56da78-6f69-48b2-b985-162062c4560b': (df0, ts_df0),
        '5033e798-525e-4e17-9bab-00f8da1f037f': (df4, ts_df4),
        '893a4f3b-8d07-4183-a1f3-598395ce4c00': (df5, ts_df5),
        '4bc4c16b-4061-49f8-9584-6e1a1c2ae8f3': (df1, ts_df1),
        '9389e1d3-bc50-4a08-ae31-a63adf6c785d': (df2, ts_df2),
        'ac3d6508-5683-47eb-94db-f6b7a4c296cb': (df3, df7, ts_df3),
        '01b13b01-f9af-4d65-a43d-8915e01c06f7': (df8, df9, ts_df8),
        '521e4586-cada-4d8e-9ef1-c5066864417f': (df7_tmp_9, ts_df7_tmp_9),
    }
main()

