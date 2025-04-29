from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.regression import RandomForestRegressor, LinearRegression, DecisionTreeRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


# Run ml pipeline
"""
Arguments:
    csv_path (str): Path to the CSV file.
    feature_cols (list): List of feature column names.
    label_col (str): Name of the label column.
    data_types (dict): Dictionary mapping column names to their data types.
"""
def run_ml_pipeline(csv_path, feature_cols, label_col, data_types):
    # Init Spark
    spark = SparkSession.builder \
        .appName("MLPipelineApp") \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.memory", "2g") \
        .getOrCreate()
    
    # Load data
    # cash data_df for faster access
    data_df = spark.read.csv(csv_path, header=True, inferSchema=True).cache()
    
    # ========== Preprocessing ==========

    # Validate input -> check if all feature columns or label column are in the CSV
    if label_col not in data_df.columns:
        raise ValueError(f"Label column '{label_col}' not found in CSV")
    if not all(col in data_df.columns for col in feature_cols):
        raise ValueError("Some feature columns not found in CSV")
    if not all(col in data_df.columns for col in data_types):
        raise ValueError("Some data type columns not specified correctly")
    
    # init stage as empty
    stages = []

    # Handle categorical features
    categorical_cols = [col for col, dtype in data_types.items() if dtype == "categorical" and col in feature_cols]
    numerical_cols = [col for col, dtype in data_types.items() if dtype == "numerical" and col in feature_cols]
    
    for col in categorical_cols:
        indexer = StringIndexer(inputCol=col, outputCol=f"{col}_indexed")
        stages.append(indexer)
    
    # Assemble features
    feature_cols_processed = numerical_cols + [f"{col}_indexed" for col in categorical_cols]
    if not feature_cols_processed:
        raise ValueError("No valid feature columns after processing")
    assembler = VectorAssembler(inputCols=feature_cols_processed, outputCol="raw_features")
    stages.append(assembler)
    
    # Scale numerical features
    scaler = StandardScaler(inputCol="raw_features", outputCol="features")
    stages.append(scaler)
    
    # ========== Model Training ==========

    # Define models
    lr = LinearRegression(labelCol=label_col, predictionCol="prediction", featuresCol="features")
    rf = RandomForestRegressor(labelCol=label_col, predictionCol="prediction", featuresCol="features")
    dt = DecisionTreeRegressor(labelCol=label_col, predictionCol="prediction", featuresCol="features")
    
    # Create pipelines
    lr_pipeline = Pipeline(stages=stages + [lr])
    rf_pipeline = Pipeline(stages=stages + [rf])
    dt_pipeline = Pipeline(stages=stages + [dt])
    
    # Hyperparameter grids for tuning
    # Note: You can adjust the parameters and values as needed
    lr_param_grid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.1, 0.15]) \
        .addGrid(lr.maxIter, [50, 100]) \
        .build()
    rf_param_grid = ParamGridBuilder() \
        .addGrid(rf.maxDepth, [5, 8]) \
        .addGrid(rf.numTrees, [20, 25]) \
        .build()
    dt_param_grid = ParamGridBuilder() \
        .addGrid(dt.maxDepth, [5, 8]) \
        .build()
    
    # Evaluator
    evaluator = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="rmse")
    
    # Cross-validators for hyperparameter tuning
    # Note: You can adjust the number of folds as needed
    lr_crossval = CrossValidator(
        estimator=lr_pipeline,
        evaluator=evaluator,
        estimatorParamMaps=lr_param_grid,
        numFolds=3
    )
    rf_crossval = CrossValidator(
        estimator=rf_pipeline,
        evaluator=evaluator,
        estimatorParamMaps=rf_param_grid,
        numFolds=3
    )
    dt_crossval = CrossValidator(
        estimator=dt_pipeline,
        evaluator=evaluator,
        estimatorParamMaps=dt_param_grid,
        numFolds=3
    )
    
    # Split data
    #seed = 190088121
    # 80% train, 20% test
    train_df, test_df = data_df.randomSplit([0.8, 0.2])
    train_df.cache()
    test_df.cache()
    
    # Fit models
    lr_model = lr_crossval.fit(train_df)
    rf_model = rf_crossval.fit(train_df)
    dt_model = dt_crossval.fit(train_df)
    
    # Evaluate models
    models = [
        ("Linear Regression", lr_model.bestModel, lr_model.avgMetrics),
        ("Random Forest", rf_model.bestModel, rf_model.avgMetrics),
        ("Decision Tree", dt_model.bestModel, dt_model.avgMetrics)
    ]
    
    results = []
    best_model = None
    best_rmse = float("inf")
    best_model_name = ""
    
    for name, model, cv_metrics in models:
        predictions = model.transform(test_df)
        rmse = evaluator.evaluate(predictions)
        r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})
        results.append({
            "model": name,
            "rmse": rmse,
            "r2": r2,
            "avg_cv_rmse": min(cv_metrics)
        })
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_model_name = name
    
    # Clean up
    spark.stop()
    
    return {
        "best_model_name": best_model_name,
        "best_rmse": best_rmse,
        "results": results,
        "best_model": best_model  # Note: Model is not serialized in JSON response
    }


