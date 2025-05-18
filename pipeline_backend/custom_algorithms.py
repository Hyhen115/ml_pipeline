from pyspark.ml.param.shared import HasLabelCol, HasFeaturesCol, HasPredictionCol
from pyspark.ml.regression import RegressionModel
from pyspark.ml.util import DefaultParamsWritable, DefaultParamsReadable
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml import Estimator
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType
import numpy as np
from pyspark.ml.linalg import Vectors, VectorUDT

#hello

class WeightedAverageRegressor(Estimator, HasLabelCol, HasFeaturesCol,
                            HasPredictionCol, DefaultParamsWritable):
    """
    A custom regressor that predicts based on weighted feature averages.
    
    This algorithm applies more sophisticated weighting than simple averaging by:
    1. Using correlation-based feature importance
    2. Adding feature interaction terms
    3. Applying regularization to prevent overfitting
    4. Accounting for feature scaling differences
    """
    
    # Define parameters
    regularization = Param(Params._dummy(), "regularization", 
                         "Regularization parameter", TypeConverters.toFloat)
    
    useInteractions = Param(Params._dummy(), "useInteractions",
                          "Whether to include feature interactions", TypeConverters.toBoolean)
    
    def __init__(self, featuresCol="features", labelCol="label", predictionCol="prediction", 
                regularization=0.01, useInteractions=True):
        super(WeightedAverageRegressor, self).__init__()
        
        # Set default values
        self._setDefault(featuresCol=featuresCol, labelCol=labelCol, predictionCol=predictionCol)
        self._setDefault(regularization=0.01, useInteractions=True)
        
        # Set parameter values correctly - use keyword arguments
        self._set(featuresCol=featuresCol)
        self._set(labelCol=labelCol)
        self._set(predictionCol=predictionCol)
        self._set(regularization=regularization)
        self._set(useInteractions=useInteractions)
    
    # Getter and Setter methods
    def getRegularization(self):
        return self.getOrDefault(self.regularization)
    
    def setRegularization(self, value):
        return self._set(regularization=value)
    
    def getUseInteractions(self):
        return self.getOrDefault(self.useInteractions)
    
    def setUseInteractions(self, value):
        return self._set(useInteractions=value)
    
    def _fit(self, dataset):
        """
        Trains a model based on the input dataset.
        
        Parameters:
        -----------
        dataset : pyspark.sql.DataFrame
            The input dataset to train the model.
            
        Returns:
        --------
        WeightedAverageRegressionModel
            The trained model.
        """
        # Extract features and labels
        features_col = self.getFeaturesCol()
        label_col = self.getLabelCol()
        
        # Use getter methods to get parameter values
        regularization = self.getRegularization()
        useInteractions = self.getUseInteractions()
        
        # Calculate feature weights using correlation with label
        feature_data = dataset.select(features_col, label_col).rdd.map(
            lambda row: (row[features_col].toArray(), float(row[label_col]))
        ).cache()
        
        # Get feature dimension
        if feature_data.isEmpty():
            raise ValueError("Empty dataset provided for training")
            
        feature_dim = feature_data.first()[0].size
        
        # Calculate feature statistics for normalization
        feature_means = np.zeros(feature_dim)
        feature_stds = np.zeros(feature_dim)
        
        for i in range(feature_dim):
            values = feature_data.map(lambda row: row[0][i]).collect()
            feature_means[i] = np.mean(values)
            feature_stds[i] = max(np.std(values), 1e-8)  # Avoid division by zero
            
        # Calculate weights using correlation and regularization
        weights = np.zeros(feature_dim)
        
        for i in range(feature_dim):
            feature_label_pairs = feature_data.map(
                lambda row: (row[0][i], row[1])
            ).collect()
            
            features = np.array([f[0] for f in feature_label_pairs])
            labels = np.array([f[1] for f in feature_label_pairs])
            
            # Calculate correlation if there's variation in features
            if np.std(features) > 1e-8:
                # Calculate Pearson correlation
                correlation = np.corrcoef(features, labels)[0, 1]
                # Apply non-linear transformation to emphasize strong correlations
                weights[i] = np.sign(correlation) * (correlation**2)
            else:
                weights[i] = 0
        
        # Apply regularization - shrink weights toward zero
        weights = weights / (1.0 + regularization)
        
        # Calculate interaction terms if enabled
        interaction_indices = []
        interaction_weights = []
        
        if useInteractions and feature_dim > 1:
            # Consider pairwise interactions for highly correlated features
            for i in range(feature_dim):
                for j in range(i+1, feature_dim):
                    if abs(weights[i]) > 0.1 and abs(weights[j]) > 0.1:
                        # Calculate interaction effect
                        interaction_data = feature_data.map(
                            lambda row: (row[0][i] * row[0][j], row[1])
                        ).collect()
                        
                        inter_features = np.array([d[0] for d in interaction_data])
                        inter_labels = np.array([d[1] for d in interaction_data])
                        
                        if np.std(inter_features) > 1e-8:
                            inter_corr = np.corrcoef(inter_features, inter_labels)[0, 1]
                            # Only keep significant interactions
                            if abs(inter_corr) > 0.05:
                                interaction_indices.append((i, j))
                                interaction_weights.append(inter_corr / 2.0)  # Scale down interactions
        
        # Normalize weights
        weight_sum = np.sum(np.abs(weights)) + sum(abs(w) for w in interaction_weights)
        if weight_sum > 1e-8:
            weights = weights / weight_sum
            interaction_weights = [w / weight_sum for w in interaction_weights]
            
        # Calculate bias term (mean of labels)
        bias = feature_data.map(lambda row: row[1]).mean()
        
        # Create and return the model
        return WeightedAverageRegressionModel(
            weights=weights, 
            bias=bias,
            featuresCol=self.getFeaturesCol(),
            predictionCol=self.getPredictionCol(),
            feature_means=feature_means,
            feature_stds=feature_stds,
            interaction_indices=interaction_indices,
            interaction_weights=interaction_weights,
            num_features=feature_dim  # Pass the number of features
        )

class WeightedAverageRegressionModel(RegressionModel, DefaultParamsReadable):
    """
    Model produced by WeightedAverageRegressor.
    
    This model applies the learned weights to make predictions on new data.
    It incorporates:
    1. Feature normalization
    2. Feature interactions
    3. Weighted linear combination
    """
    def __init__(self, weights=None, bias=0.0, featuresCol="features", predictionCol="prediction",
                feature_means=None, feature_stds=None, interaction_indices=None, 
                interaction_weights=None, num_features=None):
        super(WeightedAverageRegressionModel, self).__init__()
        self.weights = weights
        self.bias = bias
        self.feature_means = feature_means
        self.feature_stds = feature_stds
        self.interaction_indices = interaction_indices or []
        self.interaction_weights = interaction_weights or []
        self._num_features = num_features
        self._setDefault(featuresCol=featuresCol, predictionCol=predictionCol)
        
        # Directly set parameters using keyword arguments
        self._set(featuresCol=featuresCol)
        self._set(predictionCol=predictionCol)
    
    def numFeatures(self):
        """
        Returns the number of features the model was trained on.
        """
        return self._num_features
    
    def predict(self, features):
        """
        Predict label for the given features.
        This method is required by the RegressionModel abstract class.
        """
        features_array = features.toArray()
        
        # Apply normalization if available
        if self.feature_means is not None and self.feature_stds is not None:
            normalized_features = (features_array - self.feature_means) / self.feature_stds
        else:
            normalized_features = features_array
            
        # Base prediction using direct features
        prediction = np.dot(normalized_features, self.weights)
        
        # Add interaction terms
        for (i, j), weight in zip(self.interaction_indices, self.interaction_weights):
            interaction_value = normalized_features[i] * normalized_features[j]
            prediction += weight * interaction_value
            
        # Add bias term
        prediction += self.bias
        
        return float(prediction)
        
    def _transform(self, dataset):
        features_col = self.getFeaturesCol()
        prediction_col = self.getPredictionCol()
        
        # Create UDF for prediction
        predict_udf = udf(self.predict, DoubleType())
        
        # Apply prediction
        return dataset.withColumn(prediction_col, predict_udf(col(features_col)))
        
    def featureImportances(self):
        """
        Returns the feature importances for this model.
        
        Returns:
        --------
        importances : pyspark.ml.linalg.Vector
            The feature importances as a Vector (normalized to sum to 1)
        """
        # Start with direct feature weights
        importances = np.abs(self.weights).copy()
        
        # Add importance from interactions
        for (i, j), weight in zip(self.interaction_indices, self.interaction_weights):
            abs_weight = abs(weight)
            importances[i] += abs_weight / 2
            importances[j] += abs_weight / 2
            
        # Normalize
        if np.sum(importances) > 0:
            importances = importances / np.sum(importances)
            
        return Vectors.dense(importances)