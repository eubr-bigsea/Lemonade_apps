
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.sql.types import *
from pyspark.sql import SparkSession
spark = SparkSession \
    .builder \
    .appName('## Titanic survivers classification k-Fold ##') \
    .getOrCreate()
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

def naive_bayes_classifier_gen_df14(spark_session):
    start = timer()

    param_grid = {
        "labelCol": [
            "survived"
        ],
        "featuresCol": [
            "features"
        ]
    }
    # Output result is the classifier and its parameters. Parameters are
    # need in classification model or cross valitor.
    df14 = (NaiveBayes(), param_grid)

    time_elapsed = timer() - start
    return df14, time_elapsed


def data_reader_gen_df0(spark_session):
    """
    Read Titanic data
    """
    start = timer()
    schema_df0 = StructType()
    schema_df0.add('id', IntegerType(), False,
                   {'type': u'INTEGER'})
    schema_df0.add('class', StringType(), True,
                   {'nullable': True, 'size': 50, 'type': u'CHARACTER'})
    schema_df0.add('survived', DoubleType(), True,
                   {'label': True, 'nullable': True, 'size': 1, 'type': u'DOUBLE'})
    schema_df0.add('name', StringType(), True,
                   {'nullable': True, 'size': 100, 'type': u'CHARACTER'})
    schema_df0.add('sex', StringType(), True,
                   {'enumeration': True,
                    'feature': True,
                    'nullable': True,
                    'size': 40,
                    'type': u'CHARACTER'})
    schema_df0.add('age', FloatType(), True,
                   {'feature': True, 'nullable': True, 'type': u'FLOAT'})
    schema_df0.add('sibsp', IntegerType(), True,
                   {'nullable': True, 'type': u'INTEGER'})
    schema_df0.add('parch', IntegerType(), True,
                   {'nullable': True, 'type': u'INTEGER'})
    schema_df0.add('ticket', StringType(), True,
                   {'nullable': True, 'type': u'CHARACTER'})
    schema_df0.add('fare', FloatType(), True,
                   {'nullable': True, 'type': u'FLOAT'})
    schema_df0.add('cabin', StringType(), True,
                   {'nullable': True, 'size': 10, 'type': u'CHARACTER'})
    schema_df0.add('embarked', StringType(), True,
                   {'feature': True, 'nullable': True, 'size': 100, 'type': u'CHARACTER'})
    schema_df0.add('boat', StringType(), True,
                   {'feature': True, 'nullable': True, 'type': u'CHARACTER'})
    schema_df0.add('body', StringType(), True,
                   {'nullable': True, 'type': u'CHARACTER'})
    schema_df0.add('homedest', StringType(), True,
                   {'feature': True, 'nullable': True, 'size': 100, 'type': u'CHARACTER'})

    url_df0 = 'hdfs://spark01.ctweb.inweb.org.br:9000/lemonade/samples/titanic.csv'
    df0 = spark_session.read\
        .option('nullValue', 'null').option('nullValue', 'NA')\
        .option('treatEmptyValuesAsNulls',
                'true')\
        .csv(url_df0, schema=schema_df0,
             header=True, sep=',',
             inferSchema=False, mode='DROPMALFORMED')
    df0.cache()

    time_elapsed = timer() - start
    return df0, time_elapsed


def transformation_gen_df1(spark_session, df0):
    """
    Extract the title (Mr. Ms., etc) from name
    """
    start = timer()
    df1 = df0.withColumn('title', regexp_extract('name', '.*, (.*?). .*', 1))

    time_elapsed = timer() - start
    return df1, time_elapsed


def decision_tree_classifier_gen_df15(spark_session):
    """

    """
    start = timer()

    param_grid = {
        "labelCol": [
            "survived"
        ],
        "featuresCol": [
            "features"
        ]
    }
    # Output result is the classifier and its parameters. Parameters are
    # need in classification model or cross valitor.
    df15 = (DecisionTreeClassifier(), param_grid)

    time_elapsed = timer() - start
    return df15, time_elapsed


def projection_gen_df2(spark_session, df1):
    start = timer()
    df2 = df1.select(
        "id",
        "survived",
        "sex",
        "age",
        "fare",
        "class",
        "title",
        "embarked")

    time_elapsed = timer() - start
    return df2, time_elapsed


def clean_missing_gen_df3(spark_session, df2):
    """

    """
    start = timer()
    # Computes the ratio of missing values for each attribute
    ratio_df2 = df2.select(
        (count('age') / count('*')).alias('age')).collect()

    attributes_df2 = [c for c in ["age"]
                      if 0.0 <= ratio_df2[0][c] <= 1.0]
    if len(attributes_df2) > 0:
        avg_df2 = df2.select([avg(c).alias(c)
                              for c in attributes_df2]).collect()
        values_df2 = dict([(c, avg_df2[0][c]) for c in attributes_df2])
        df3 = df2.na.fill(value=values_df2)
    else:
        df3 = df2

    time_elapsed = timer() - start
    return df3, time_elapsed


def feature_indexer_gen_df4_models(spark_session, df3):
    start = timer()

    col_alias = dict([["class", "class_indexed"], ["title", "title_indexed"], [
                     "embarked", "embarked_indexed"], ["sex", "sex_indexed"]])
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


def clean_missing_gen_df5(spark_session, df4):
    start = timer()
    # Computes the ratio of missing values for each attribute
    ratio_df4 = df4.select(
        (count('fare') / count('*')).alias('fare')).collect()

    attributes_df4 = [c for c in ["fare"]
                      if 0.0 <= ratio_df4[0][c] <= 1.0]
    if len(attributes_df4) > 0:
        df5 = df4.na.fill(value=0, subset=attributes_df4)
    else:
        df5 = df4

    time_elapsed = timer() - start
    return df5, time_elapsed


def feature_assembler_gen_df6(spark_session, df5):
    start = timer()

    assembler = VectorAssembler(
        inputCols=[
            "embarked_indexed",
            "sex_indexed",
            "class_indexed",
            "title_indexed",
            "age",
            "fare"],
        outputCol="features")
    df5_without_null = df5.na.drop(
        subset=[
            "embarked_indexed",
            "sex_indexed",
            "class_indexed",
            "title_indexed",
            "age",
            "fare"])
    df6 = assembler.transform(df5_without_null)
    juicer_debug('juicer.spark.ml_operation.FeatureAssembler', 'df6', df6)

    time_elapsed = timer() - start
    return df6, time_elapsed


def evaluate_model_gen_df7(spark_session):
    start = timer()

    df7 = MulticlassClassificationEvaluator(predictionCol='prediction',
                                            labelCol='survived', metricName='accuracy')

    time_elapsed = timer() - start
    return df7, time_elapsed


def cross_validation_gen_df9_eval_df9_best_model_df9(
        spark_session, df15, df6, df7):
    start = timer()

    grid_builder = ParamGridBuilder()
    estimator, param_grid = df15
    '''
    if estimator.__class__ == 'LinearRegression':
        param_grid = estimator.maxIter
    elif estimator.__class__  == NaiveBayes:
        pass
    elif estimator.__class__ == DecisionTreeClassifier:
        # param_grid = (estimator.maxDepth, [2,3,4,5,6,7,8,9])
        param_grid = (estimator.impurity, ['gini', 'entropy'])
    elif estimator.__class__ == GBTClassifier:
        pass
    elif estimator.__class__ == RandomForestClassifier:
        param_grid = estimator.maxDepth
    '''
    for param_name, values in param_grid.iteritems():
        param = getattr(estimator, param_name)
        grid_builder.addGrid(param, values)

    evaluator = df7

    grid = grid_builder.build()
    cv = CrossValidator(estimator=estimator, estimatorParamMaps=grid,
                        evaluator=evaluator, numFolds=10)
    cv_model = cv.fit(df6)
    evaluated_data = cv_model.transform(df6)
    best_model_df9 = cv_model.bestModel
    metric_result = evaluator.evaluate(evaluated_data)
    df9 = evaluated_data

    grouped_result = evaluated_data.select(
        evaluator.getLabelCol(), evaluator.getPredictionCol())\
        .groupBy(
        evaluator.getLabelCol(),
        evaluator.getPredictionCol()).count().collect()
    eval_df9 = {
        'metric': {
            'name': evaluator.getMetricName(),
            'value': metric_result
        },
        'estimator': {
            'name': estimator.__class__.__name__,
            'predictionCol': evaluator.getPredictionCol(),
            'labelCol': evaluator.getLabelCol()
        },
        'confusion_matrix': {
            'data': json.dumps(grouped_result)
        },
        'evaluator': evaluator
    }

    time_elapsed = timer() - start
    return df9, eval_df9, best_model_df9, time_elapsed


def cross_validation_gen_df8_eval_df8_best_model_df8(
        spark_session, df14, df6, df7):
    start = timer()

    grid_builder = ParamGridBuilder()
    estimator, param_grid = df14
    '''
    if estimator.__class__ == 'LinearRegression':
        param_grid = estimator.maxIter
    elif estimator.__class__  == NaiveBayes:
        pass
    elif estimator.__class__ == DecisionTreeClassifier:
        # param_grid = (estimator.maxDepth, [2,3,4,5,6,7,8,9])
        param_grid = (estimator.impurity, ['gini', 'entropy'])
    elif estimator.__class__ == GBTClassifier:
        pass
    elif estimator.__class__ == RandomForestClassifier:
        param_grid = estimator.maxDepth
    '''
    for param_name, values in param_grid.iteritems():
        param = getattr(estimator, param_name)
        grid_builder.addGrid(param, values)

    evaluator = df7

    grid = grid_builder.build()
    cv = CrossValidator(estimator=estimator, estimatorParamMaps=grid,
                        evaluator=evaluator, numFolds=10)
    cv_model = cv.fit(df6)
    evaluated_data = cv_model.transform(df6)
    best_model_df8 = cv_model.bestModel
    metric_result = evaluator.evaluate(evaluated_data)
    df8 = evaluated_data

    grouped_result = evaluated_data.select(
        evaluator.getLabelCol(), evaluator.getPredictionCol())\
        .groupBy(
        evaluator.getLabelCol(),
        evaluator.getPredictionCol()).count().collect()
    eval_df8 = {
        'metric': {
            'name': evaluator.getMetricName(),
            'value': metric_result
        },
        'estimator': {
            'name': estimator.__class__.__name__,
            'predictionCol': evaluator.getPredictionCol(),
            'labelCol': evaluator.getLabelCol()
        },
        'confusion_matrix': {
            'data': json.dumps(grouped_result)
        },
        'evaluator': evaluator
    }

    time_elapsed = timer() - start
    return df8, eval_df8, best_model_df8, time_elapsed


def random_forest_classifier_gen_df12(spark_session):
    start = timer()

    param_grid = {
        "labelCol": [
            "survived"
        ],
        "featuresCol": [
            "features"
        ]
    }
    # Output result is the classifier and its parameters. Parameters are
    # need in classification model or cross valitor.
    df12 = (RandomForestClassifier(), param_grid)

    time_elapsed = timer() - start
    return df12, time_elapsed


def cross_validation_gen_df10_eval_df10_best_model_df10(
        spark_session, df12, df6, df7):
    start = timer()

    grid_builder = ParamGridBuilder()
    estimator, param_grid = df12
    '''
    if estimator.__class__ == 'LinearRegression':
        param_grid = estimator.maxIter
    elif estimator.__class__  == NaiveBayes:
        pass
    elif estimator.__class__ == DecisionTreeClassifier:
        # param_grid = (estimator.maxDepth, [2,3,4,5,6,7,8,9])
        param_grid = (estimator.impurity, ['gini', 'entropy'])
    elif estimator.__class__ == GBTClassifier:
        pass
    elif estimator.__class__ == RandomForestClassifier:
        param_grid = estimator.maxDepth
    '''
    for param_name, values in param_grid.iteritems():
        param = getattr(estimator, param_name)
        grid_builder.addGrid(param, values)

    evaluator = df7

    grid = grid_builder.build()
    cv = CrossValidator(estimator=estimator, estimatorParamMaps=grid,
                        evaluator=evaluator, numFolds=10)
    cv_model = cv.fit(df6)
    evaluated_data = cv_model.transform(df6)
    best_model_df10 = cv_model.bestModel
    metric_result = evaluator.evaluate(evaluated_data)
    df10 = evaluated_data

    grouped_result = evaluated_data.select(
        evaluator.getLabelCol(), evaluator.getPredictionCol())\
        .groupBy(
        evaluator.getLabelCol(),
        evaluator.getPredictionCol()).count().collect()
    eval_df10 = {
        'metric': {
            'name': evaluator.getMetricName(),
            'value': metric_result
        },
        'estimator': {
            'name': estimator.__class__.__name__,
            'predictionCol': evaluator.getPredictionCol(),
            'labelCol': evaluator.getLabelCol()
        },
        'confusion_matrix': {
            'data': json.dumps(grouped_result)
        },
        'evaluator': evaluator
    }

    time_elapsed = timer() - start
    return df10, eval_df10, best_model_df10, time_elapsed


def gbt_classifier_gen_df13(spark_session):
    """

    """
    start = timer()

    param_grid = {
        "labelCol": [
            "survived"
        ],
        "featuresCol": [
            "features"
        ]
    }
    # Output result is the classifier and its parameters. Parameters are
    # need in classification model or cross valitor.
    df13 = (GBTClassifier(), param_grid)

    time_elapsed = timer() - start
    return df13, time_elapsed


def cross_validation_gen_df11_eval_df11_best_model_df11(
        spark_session, df13, df6, df7):
    start = timer()

    grid_builder = ParamGridBuilder()
    estimator, param_grid = df13
    '''
    if estimator.__class__ == 'LinearRegression':
        param_grid = estimator.maxIter
    elif estimator.__class__  == NaiveBayes:
        pass
    elif estimator.__class__ == DecisionTreeClassifier:
        # param_grid = (estimator.maxDepth, [2,3,4,5,6,7,8,9])
        param_grid = (estimator.impurity, ['gini', 'entropy'])
    elif estimator.__class__ == GBTClassifier:
        pass
    elif estimator.__class__ == RandomForestClassifier:
        param_grid = estimator.maxDepth
    '''
    for param_name, values in param_grid.iteritems():
        param = getattr(estimator, param_name)
        grid_builder.addGrid(param, values)

    evaluator = df7

    grid = grid_builder.build()
    cv = CrossValidator(estimator=estimator, estimatorParamMaps=grid,
                        evaluator=evaluator, numFolds=10)
    cv_model = cv.fit(df6)
    evaluated_data = cv_model.transform(df6)
    best_model_df11 = cv_model.bestModel
    metric_result = evaluator.evaluate(evaluated_data)
    df11 = evaluated_data

    grouped_result = evaluated_data.select(
        evaluator.getLabelCol(), evaluator.getPredictionCol())\
        .groupBy(
        evaluator.getLabelCol(),
        evaluator.getPredictionCol()).count().collect()
    eval_df11 = {
        'metric': {
            'name': evaluator.getMetricName(),
            'value': metric_result
        },
        'estimator': {
            'name': estimator.__class__.__name__,
            'predictionCol': evaluator.getPredictionCol(),
            'labelCol': evaluator.getLabelCol()
        },
        'confusion_matrix': {
            'data': json.dumps(grouped_result)
        },
        'evaluator': evaluator
    }

    time_elapsed = timer() - start
    return df11, eval_df11, best_model_df11, time_elapsed


def classification_report_gen_df8_tmp_17(spark_session, df8, df9, df10, df11):
    start = timer()

    df8_tmp_17 = "ok"

    time_elapsed = timer() - start
    return df8_tmp_17, time_elapsed


def main():
    app_name = u'## Titanic survivers classification k-Fold ##'
    spark_session = SparkSession.builder\
        .appName(app_name)\
        .getOrCreate()
    # spark_session.sparkContext.addPyFile('/tmp/dist.zip')

    # Declares and initializes variabels in order to do not generate NameError.
    # Some tasks may not generate code, but at least one of their outputs is
    # connected to a valid input in a task generating code. This happens when
    # task has a port with multiplicity MANY
    df8 = df9 = df10 = df11 = None

    df14, ts_df14 = naive_bayes_classifier_gen_df14(spark_session)
    df0, ts_df0 = data_reader_gen_df0(spark_session)
    df1, ts_df1 = transformation_gen_df1(spark_session, df0)
    df15, ts_df15 = decision_tree_classifier_gen_df15(spark_session)
    df2, ts_df2 = projection_gen_df2(spark_session, df1)
    df3, ts_df3 = clean_missing_gen_df3(spark_session, df2)
    df4, models, ts_df4 = feature_indexer_gen_df4_models(spark_session, df3)
    df5, ts_df5 = clean_missing_gen_df5(spark_session, df4)
    df6, ts_df6 = feature_assembler_gen_df6(spark_session, df5)
    df7, ts_df7 = evaluate_model_gen_df7(spark_session)
    df9, eval_df9, best_model_df9, ts_df9 = cross_validation_gen_df9_eval_df9_best_model_df9(
        spark_session, df15, df6, df7)
    df8, eval_df8, best_model_df8, ts_df8 = cross_validation_gen_df8_eval_df8_best_model_df8(
        spark_session, df14, df6, df7)
    df12, ts_df12 = random_forest_classifier_gen_df12(spark_session)
    df10, eval_df10, best_model_df10, ts_df10 = cross_validation_gen_df10_eval_df10_best_model_df10(
        spark_session, df12, df6, df7)
    df13, ts_df13 = gbt_classifier_gen_df13(spark_session)
    df11, eval_df11, best_model_df11, ts_df11 = cross_validation_gen_df11_eval_df11_best_model_df11(
        spark_session, df13, df6, df7)
    df8_tmp_17, ts_df8_tmp_17 = classification_report_gen_df8_tmp_17(
        spark_session, df8, df9, df10, df11)

    return {
        'b5e83305-da49-4079-acd1-33195a4a01ae': (df14, ts_df14),
        'a78f9f81-bcbd-11e6-a154-305a3a9e86de': (df0, ts_df0),
        'a78f9f8c-bcbd-11e6-a154-305a3a9e86de': (df1, ts_df1),
        '773143b7-5c31-4fee-a620-8abde268a3d2': (df15, ts_df15),
        'a78f9f80-bcbd-11e6-a154-305a3a9e86de': (df2, ts_df2),
        'a78f9f89-bcbd-11e6-a154-305a3a9e86de': (df3, ts_df3),
        'a78f9f85-bcbd-11e6-a154-305a3a9e86de': (df4, models, ts_df4),
        'a78f9f88-bcbd-11e6-a154-305a3a9e86de': (df5, ts_df5),
        'a78f9f82-bcbd-11e6-a154-305a3a9e86de': (df6, ts_df6),
        '6c97773f-9eee-4dd1-a88b-2a667cb3c5a4': (df7, ts_df7),
        'f77e186b-77a6-49bf-80a0-428556565fae': (df9, eval_df9, best_model_df9, ts_df9),
        '29dd5a75-7f54-442a-93ff-a03bb3d8f54a': (df8, eval_df8, best_model_df8, ts_df8),
        'a78f9f84-bcbd-11e6-a154-305a3a9e86de': (df12, ts_df12),
        '3109d320-a18f-48fd-bb35-2e009d5c6b94': (df10, eval_df10, best_model_df10, ts_df10),
        'a123cd92-9356-4197-8476-297e05c9b340': (df13, ts_df13),
        '0ce896d0-41b3-4ab7-8711-b9eb6e08e095': (df11, eval_df11, best_model_df11, ts_df11),
        '24605422-630e-423c-ad19-e175f75467a9': (df8_tmp_17, ts_df8_tmp_17),
    }
main()

