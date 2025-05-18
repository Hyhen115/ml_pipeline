# test_custom_algorithm.py
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from custom_algorithms import WeightedAverageRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# Initialize Spark
print("Initializing Spark...")
spark = SparkSession.builder \
    .appName("TestCustomAlgorithm") \
    .master("local[*]") \
    .getOrCreate()

print("Spark version:", spark.version)

# Create a simple test dataset
print("Creating test data...")
data = [(1.0, 2.0, 3.0), 
        (2.0, 3.0, 5.0), 
        (3.0, 4.0, 7.0), 
        (4.0, 5.0, 9.0), 
        (5.0, 6.0, 11.0)]
columns = ["feature1", "feature2", "label"]
df = spark.createDataFrame(data, columns)

# Prepare features
assembler = VectorAssembler(inputCols=["feature1", "feature2"], outputCol="features")
transformed_df = assembler.transform(df)

# Split into training and test sets
train_df, test_df = transformed_df.randomSplit([0.8, 0.2], seed=42)

# Create and train custom model
print("Training custom model...")
wa = WeightedAverageRegressor(featuresCol="features", labelCol="label", 
                             predictionCol="prediction", regularization=0.01, 
                             useInteractions=True)
model = wa.fit(train_df)

# Output number of features
print(f"Number of features in model: {model.numFeatures()}")

# Make predictions
print("Generating predictions...")
predictions = model.transform(test_df)
predictions.select("features", "label", "prediction").show()

# Evaluate model
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE) on test data = {rmse}")

# Print feature importance
importances = model.featureImportances()
print("Feature importances:", importances)

print("Test completed!")
spark.stop()
