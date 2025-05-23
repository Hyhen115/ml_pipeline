from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, Imputer
from pyspark.ml.regression import RandomForestRegressor, LinearRegression, DecisionTreeRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from concurrent.futures import ThreadPoolExecutor
from pyspark.sql.types import StringType as PySparkSQLStringType
from pyspark.ml.regression import GBTRegressor
# from custom_algorithms import WeightedAverageRegressor

import concurrent.futures
import logging
import uuid
from urllib.parse import urlparse
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _train_model(crossval, train_df, name):
    """Helper function to train a single model in parallel."""
    logger.info(f"Training {name} model")
    model = crossval.fit(train_df)
    return (name, model.bestModel, model.avgMetrics)

def parse_s3_path(s3_path):
    """Parse S3 path into bucket and key components."""
    parsed = urlparse(s3_path)
    return parsed.netloc, parsed.path.lstrip('/')

def run_ml_pipeline(s3_input_path, feature_cols, label_col, data_types, train_test_split):
    """
    Run a machine learning pipeline on the provided CSV data.
    
    Args:
        csv_path (str): Path to the CSV file.
        feature_cols (list): List of feature column names.
        label_col (str): Name of the label column.
        data_types (dict): Dictionary mapping column names to their data types 
                          ("numerical" or "categorical").
    
    Returns:
        dict: Results containing best model info and performance metrics.
    """
    logger.info(f"Starting ML pipeline for {s3_input_path}")
    logger.info(f"Features: {feature_cols}")
    logger.info(f"Label: {label_col}")
    logger.info(f"Data types: {data_types}")
    logger.info(f"Train-test split ratio: {train_test_split}")

    # Add these debug prints
    logger.info("DEBUG: AWS Environment Variables:")
    logger.info(f"AWS_ACCESS_KEY_ID exists: {'AWS_ACCESS_KEY_ID' in os.environ}")
    logger.info(f"AWS_SECRET_ACCESS_KEY exists: {'AWS_SECRET_ACCESS_KEY' in os.environ}")
    logger.info(f"AWS_SESSION_TOKEN exists: {'AWS_SESSION_TOKEN' in os.environ}")
    logger.info(f"AWS_REGION: {os.environ.get('AWS_REGION', 'not set')}")
    
    # Get the bucket name from the s3_input_path
    bucket, _ = parse_s3_path(s3_input_path)
    
    # spark with tokens and keys implementation for connecting to s3
    spark = (
        SparkSession.builder
        .appName("MLPipelineApp")
        # .master("local[*]")
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.1")
        # Use the newer AWS credentials provider
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", 
                "com.amazonaws.auth.EnvironmentVariableCredentialsProvider")
        # Additional S3A configurations
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.connection.establish.timeout", "5000")
        .config("spark.hadoop.fs.s3a.connection.timeout", "10000")
        .config("spark.hadoop.fs.s3a.attempts.maximum", "20")
        .config("spark.hadoop.fs.s3a.endpoint", f"s3.{os.environ.get('AWS_REGION', 'us-east-1')}.amazonaws.com")
        .getOrCreate()
    )
    
    try:

        # Load data
        # logger.info(f"Loading data from {csv_path}")
        # data_df = spark.read.csv(csv_path, header=True, inferSchema=True).cache()
        # Load data from S3
        logger.info(f"Loading data from {s3_input_path}")
        data_df = spark.read.csv(s3_input_path, header=True, inferSchema=True).cache()
        
        # Display basic dataset info
        row_count = data_df.count()
        col_count = len(data_df.columns)
        logger.info(f"Dataset loaded: {row_count} rows, {col_count} columns")
        
        # Validate input
        logger.info("Validating input parameters")
        if label_col not in data_df.columns:
            raise ValueError(f"Label column '{label_col}' not found in CSV")
        if not all(col in data_df.columns for col in feature_cols):
            missing_cols = [col for col in feature_cols if col not in data_df.columns]
            raise ValueError(f"Feature columns not found in CSV: {missing_cols}")
        if not all(col in data_df.columns for col in data_types):
            missing_cols = [col for col in data_types if col not in data_df.columns]
            raise ValueError(f"Columns with specified data types not found in CSV: {missing_cols}")
        
        # init stage as empty
        stages = []

        # Handle Label Col (Mod1)
        # The label column for regression models must be numeric.
        # If it's a string, we add a StringIndexer for it and update its name for model use.
        current_label_col = label_col
        if isinstance(data_df.schema[label_col].dataType, PySparkSQLStringType):
          indexed_label_name = f"{label_col}_indexed_label_for_regression"
          label_indexer_stage = StringIndexer(inputCol=label_col,
                                              outputCol=indexed_label_name,
                                              handleInvalid="keep") # Use "error" or "keep" as appropriate
          stages.append(label_indexer_stage)
          current_label_col = indexed_label_name

        # Handle categorical features
        categorical_cols = [col for col, dtype in data_types.items() if dtype == "categorical" and col in feature_cols]
        numerical_cols = [col for col, dtype in data_types.items() if dtype == "numerical" and col in feature_cols]  
        
        logger.info(f"Categorical features: {categorical_cols}")
        logger.info(f"Numerical features: {numerical_cols}")
        
        # Add StringIndexer stages for string categorical columns.
        for col_name in categorical_cols:
          # Check the actual datatype of the column in the DataFrame
          column_actual_datatype = data_df.schema[col_name].dataType
          if isinstance(column_actual_datatype, PySparkSQLStringType):
            # If the categorical column is a string, add an indexer stage for it.
            # The outputCol uses the "_indexed" suffix as in the original code.
            indexer = StringIndexer(inputCol=col_name, outputCol=f"{col_name}_indexed", handleInvalid="keep")
            stages.append(indexer)
        
        # Handle numerical features: Impute missing values
        if numerical_cols:
            imputer = Imputer(
                inputCols=numerical_cols,
                outputCols=[f"{col}_imputed" for col in numerical_cols],
                strategy="mean"
            )
            stages.append(imputer)
            numerical_cols_processed = [f"{col}_imputed" for col in numerical_cols]
        else:
            numerical_cols_processed = []
        
        
        feature_cols_for_assembler = []
        feature_cols_for_assembler.extend(numerical_cols_processed)

        for col_name in categorical_cols:
          column_actual_datatype = data_df.schema[col_name].dataType
          if isinstance(column_actual_datatype, PySparkSQLStringType):
            # If this categorical column was a string (and thus indexed), use its indexed name.
            feature_cols_for_assembler.append(f"{col_name}_indexed")
          else:
            # If this categorical column was not a string (i.e., it's numeric), use its original name.
            feature_cols_for_assembler.append(col_name)
        
        assembler = VectorAssembler(inputCols=feature_cols_for_assembler, outputCol="raw_features")
        stages.append(assembler)
        
        # Scale numerical features (with centering)
        scaler = StandardScaler(inputCol="raw_features", outputCol="features", withMean=True)  # Added withMean=True
        stages.append(scaler)
        
        # Define models
        lr = LinearRegression(labelCol=current_label_col, predictionCol="prediction", featuresCol="features")
        rf = RandomForestRegressor(labelCol=current_label_col, predictionCol="prediction", featuresCol="features")
        dt = DecisionTreeRegressor(labelCol=current_label_col, predictionCol="prediction", featuresCol="features")
        gbt = GBTRegressor(labelCol=current_label_col, predictionCol="prediction", featuresCol="features")
        # wa = WeightedAverageRegressor(labelCol=current_label_col, featuresCol="features")
        # wa = WeightedAverageRegressor(labelCol=current_label_col, predictionCol="prediction", 
        #                     featuresCol="features", regularization=0.01, useInteractions=True)


        # Create pipelines
        lr_pipeline = Pipeline(stages=stages + [lr])
        rf_pipeline = Pipeline(stages=stages + [rf])
        dt_pipeline = Pipeline(stages=stages + [dt])
        gbt_pipeline = Pipeline(stages=stages + [gbt])
        # wa_pipeline = Pipeline(stages=stages + [wa])

        
        # Hyperparameter grids for tuning
        lr_param_grid = ParamGridBuilder() \
            .addGrid(lr.regParam, [0.1, 0.01]) \
            .addGrid(lr.maxIter, [10, 100]) \
            .build()
        
        rf_param_grid = ParamGridBuilder() \
            .addGrid(rf.maxDepth, [5, 10]) \
            .addGrid(rf.numTrees, [10, 20]) \
            .build()
        
        dt_param_grid = ParamGridBuilder() \
            .addGrid(dt.maxDepth, [5, 10]) \
            .build()
        
        gbt_param_grid = ParamGridBuilder() \
            .addGrid(gbt.maxDepth, [5, 8]) \
            .addGrid(gbt.maxIter, [10, 20]) \
            .build()
        
        # wa_param_grid = ParamGridBuilder() \
        #     .addGrid(wa.regularization, [0.01, 0.1]) \
        #     .addGrid(wa.useInteractions, [True, False]) \
        #     .build()
        
        # Evaluator
        evaluator = RegressionEvaluator(labelCol=current_label_col, predictionCol="prediction", metricName="rmse")
        
        # Cross-validators for hyperparameter tuning
        logger.info("Setting up cross-validation")
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

        gbt_crossval = CrossValidator(
            estimator=gbt_pipeline,
            evaluator=evaluator,
            estimatorParamMaps=gbt_param_grid,
            numFolds=3
        )

        # wa_crossval = CrossValidator(
        #     estimator=wa_pipeline,
        #     evaluator=evaluator,
        #     estimatorParamMaps=wa_param_grid,
        #     numFolds=3
        # )
        
        # Split data
        train_df, test_df = data_df.randomSplit([train_test_split, 1 - train_test_split], seed=42)
        train_df.cache()
        test_df.cache()
        
        logger.info(f"Training data: {train_df.count()} rows")
        logger.info(f"Testing data: {test_df.count()} rows")
        
        # Fit models in parallel
        logger.info("Training models in parallel")
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(_train_model, lr_crossval, train_df, "Linear Regression"),
                executor.submit(_train_model, rf_crossval, train_df, "Random Forest"),
                executor.submit(_train_model, dt_crossval, train_df, "Decision Tree"),
                executor.submit(_train_model, gbt_crossval, train_df, "Gradient Boosted Trees"),
                # executor.submit(_train_model, wa_crossval, train_df, "Weighted Average Regressor")
            ]
            
            # Collect results as they complete
            models = []
            for future in concurrent.futures.as_completed(futures):
                name, model, cv_metrics = future.result()
                models.append((name, model, cv_metrics))

        
        # Evaluate models
        models.sort(key=lambda x: x[0])  # Optional: Sort for consistent order
        results = []
        best_model = None
        best_rmse = float("inf")
        best_model_name = ""
        
        for name, model, cv_metrics in models:
            logger.info(f"Evaluating {name} model")
            predictions = model.transform(test_df)
            rmse = evaluator.evaluate(predictions)
            r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})
            
            logger.info(f"{name} - RMSE: {rmse}, R²: {r2}")
            
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
        
        logger.info(f"Best model: {best_model_name} with RMSE: {best_rmse}")

        # Save best model to a temporary directory and zip it
        # with tempfile.TemporaryDirectory() as tmp_dir:
        #     model_path = os.path.join(tmp_dir, "model")
        #     best_model.write().overwrite().save(model_path)
        #
        #     # Zip the model directory
        #     zip_buffer = io.BytesIO()
        #     with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        #         for root, _, files in os.walk(model_path):
        #             for file in files:
        #                 file_path = os.path.join(root, file)
        #                 arcname = os.path.relpath(file_path, start=tmp_dir)
        #                 zipf.write(file_path, arcname=arcname)
        #     zip_buffer.seek(0)
        #     model_bytes = zip_buffer.getvalue()
        #     model_base64 = base64.b64encode(model_bytes).decode('utf-8')

        # Save best model to S3
        # Save model with UUID
        model_id = str(uuid.uuid4())
        s3_model_path = f"models/{model_id}/{best_model_name}"
        #best_model.write().overwrite().save(f"s3a://{os.environ['S3_BUCKET']}/{s3_model_path}")
        best_model.write().overwrite().save(f"s3a://{bucket}/{s3_model_path}")
        logger.info(f"Model saved to S3: {s3_model_path}")

        result = {
            "best_model_name": best_model_name,
            "model_path": s3_model_path,
            "best_rmse": best_rmse,
            "results": results
        }

        logger.info(f"Pipeline complete, returning results: {result}")
        return result

        # return {
        #     "best_model_name": best_model_name,
        #     "best_rmse": best_rmse,
        #     "results": results,
        #     "model_base64": model_base64  # Model as base64 string
        # }
    
    except Exception as e:
        logger.error(f"Error in ML pipeline: {str(e)}", exc_info=True)
        raise
    finally:
        # Stop Spark session
        spark.stop()
        logger.info("Spark session stopped")
