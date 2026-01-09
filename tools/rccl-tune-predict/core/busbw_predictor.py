"""BusbwPredictor: ML model for predicting RCCL bus bandwidth."""

import pickle
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error


class BusbwPredictor:
    """ML model to predict bus bandwidth for RCCL configurations.
    
    Uses GradientBoostingRegressor to predict busbw_ip given:
    - collective_encoded, num_nodes, num_gpus, size_bytes (workload params)
    - log_size, gpus_per_node, bytes_per_gpu (derived features)
    - algo_encoded, proto_encoded, nchannels (tuning params)
    """
    
    FEATURE_NAMES = [
        'collective_encoded',
        'num_nodes',
        'num_gpus',
        'size_bytes',
        'log_size',
        'gpus_per_node',
        'bytes_per_gpu',
        'algo_encoded',
        'proto_encoded',
        'nchannels',
    ]
    
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        min_samples_split: int = 5,
        random_state: int = 42,
    ):
        """Initialize the predictor.
        
        Args:
            n_estimators: Number of boosting stages
            max_depth: Maximum depth of individual trees
            learning_rate: Learning rate shrinks contribution of each tree
            min_samples_split: Minimum samples required to split an internal node
            random_state: Random seed for reproducibility
        """
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_samples_split=min_samples_split,
            random_state=random_state,
        )
        self._is_fitted = False
        self.train_metrics: Dict[str, float] = {}
    
    def fit(self, X: np.ndarray, y: np.ndarray, test_split: float = 0.0) -> Dict[str, float]:
        """Train the model.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target vector of shape (n_samples,)
            test_split: Fraction of data to use for testing (0 means train on all)
        
        Returns:
            Dictionary with training/testing metrics (R², MAE)
        """
        metrics = {}
        
        if test_split > 0:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_split, random_state=42
            )
            metrics['train_samples'] = len(X_train)
            metrics['test_samples'] = len(X_test)
        else:
            X_train, y_train = X, y
            X_test, y_test = None, None
            metrics['train_samples'] = len(X_train)
            metrics['test_samples'] = 0
        
        # Fit the model
        self.model.fit(X_train, y_train)
        self._is_fitted = True
        
        # Training metrics
        y_train_pred = self.model.predict(X_train)
        metrics['train_r2'] = r2_score(y_train, y_train_pred)
        metrics['train_mae'] = mean_absolute_error(y_train, y_train_pred)
        
        # Test metrics (if applicable)
        if X_test is not None:
            y_test_pred = self.model.predict(X_test)
            metrics['test_r2'] = r2_score(y_test, y_test_pred)
            metrics['test_mae'] = mean_absolute_error(y_test, y_test_pred)
        
        self.train_metrics = metrics
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict bus bandwidth for given features.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features) or (n_features,)
        
        Returns:
            Predicted busbw values
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        
        # Handle 1D input (single sample)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        return self.model.predict(X)
    
    def predict_single(
        self,
        collective_encoded: int,
        num_nodes: int,
        num_gpus: int,
        size_bytes: int,
        algo_encoded: int,
        proto_encoded: int,
        nchannels: int,
    ) -> float:
        """Predict busbw for a single configuration.
        
        Args:
            collective_encoded: Encoded collective type
            num_nodes: Number of nodes
            num_gpus: Total number of GPUs
            size_bytes: Message size in bytes
            algo_encoded: Encoded algorithm
            proto_encoded: Encoded protocol
            nchannels: Number of channels
        
        Returns:
            Predicted bus bandwidth
        """
        from .utils import build_feature_vector
        
        features = build_feature_vector(
            collective_encoded, num_nodes, num_gpus, size_bytes,
            algo_encoded, proto_encoded, nchannels
        )
        return self.predict(features)[0]
    
    def save(self, path: str) -> None:
        """Save model to a pickle file.
        
        Args:
            path: Path to save the model
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted model. Call fit() first.")
        
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'train_metrics': self.train_metrics,
                'feature_names': self.FEATURE_NAMES,
            }, f)
    
    @classmethod
    def load(cls, path: str) -> 'BusbwPredictor':
        """Load model from a pickle file.
        
        Args:
            path: Path to the saved model
        
        Returns:
            Loaded BusbwPredictor instance
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        predictor = cls()
        predictor.model = data['model']
        predictor.train_metrics = data.get('train_metrics', {})
        predictor._is_fitted = True
        
        return predictor
    
    def get_feature_importances(self) -> Dict[str, float]:
        """Get feature importances from the trained model.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        
        importances = self.model.feature_importances_
        return dict(zip(self.FEATURE_NAMES, importances))


