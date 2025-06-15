"""
Complete Machine Learning Models for Crypto Trading Bot
Includes prediction models, feature engineering, and model management
"""

import numpy as np
import pandas as pd
import logging
import pickle
import json
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import xgboost as xgb

# Try to import TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None

from database import get_database

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    name: str
    model_type: str  # 'classification', 'regression'
    algorithm: str   # 'random_forest', 'xgboost', 'lstm'
    features: List[str]
    target: str
    hyperparameters: Dict[str, Any]
    training_window: int = 1000  # Number of data points for training

@dataclass
class PredictionResult:
    symbol: str
    prediction: float
    confidence: float
    timestamp: datetime
    model_name: str

class FeatureEngineer:
    """Feature engineering for crypto trading data"""
    
    @staticmethod
    def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to price data"""
        try:
            # Simple Moving Averages
            df['sma_5'] = df['price'].rolling(window=5).mean()
            df['sma_10'] = df['price'].rolling(window=10).mean()
            df['sma_20'] = df['price'].rolling(window=20).mean()
            df['sma_50'] = df['price'].rolling(window=50).mean()
            
            # Exponential Moving Averages
            df['ema_12'] = df['price'].ewm(span=12).mean()
            df['ema_26'] = df['price'].ewm(span=26).mean()
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # RSI
            delta = df['price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            bb_window = 20
            bb_std = 2
            df['bb_middle'] = df['price'].rolling(window=bb_window).mean()
            bb_std_val = df['price'].rolling(window=bb_window).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std_val * bb_std)
            df['bb_lower'] = df['bb_middle'] - (bb_std_val * bb_std)
            df['bb_position'] = (df['price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Price features
            df['price_change'] = df['price'].pct_change()
            df['price_change_5'] = df['price'].pct_change(periods=5)
            df['volatility'] = df['price_change'].rolling(window=20).std()
            
            # Volume features (if available)
            if 'volume' in df.columns:
                df['volume_sma'] = df['volume'].rolling(window=10).mean()
                df['volume_ratio'] = df['volume'] / df['volume_sma']
                df['price_volume'] = df['price'] * df['volume']
            
            # Time features
            df['hour'] = pd.to_datetime(df.index).hour
            df['day_of_week'] = pd.to_datetime(df.index).dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            # Momentum indicators
            df['momentum_5'] = df['price'] / df['price'].shift(5) - 1
            df['momentum_10'] = df['price'] / df['price'].shift(10) - 1
            
            # Support/Resistance levels (simplified)
            df['support'] = df['price'].rolling(window=20).min()
            df['resistance'] = df['price'].rolling(window=20).max()
            df['support_distance'] = (df['price'] - df['support']) / df['price']
            df['resistance_distance'] = (df['resistance'] - df['price']) / df['price']
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            return df
    
    @staticmethod
    def create_target_variables(df: pd.DataFrame) -> pd.DataFrame:
        """Create prediction targets"""
        try:
            # Price movement targets
            df['price_next'] = df['price'].shift(-1)
            df['return_next'] = (df['price_next'] / df['price']) - 1
            
            # Classification targets
            df['direction_next'] = (df['return_next'] > 0).astype(int)
            df['strong_move'] = (abs(df['return_next']) > 0.02).astype(int)  # 2% move
            
            # Regression targets
            df['return_1h'] = df['price'].shift(-1) / df['price'] - 1
            df['return_4h'] = df['price'].shift(-4) / df['price'] - 1
            df['return_24h'] = df['price'].shift(-24) / df['price'] - 1
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating target variables: {e}")
            return df

class MLModelManager:
    """Manages machine learning models for trading predictions"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.database = get_database()
        self.feature_engineer = FeatureEngineer()
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        
    def prepare_data(self, symbol: str, hours: int = 720) -> pd.DataFrame:
        """Prepare training data for a symbol"""
        try:
            # Get market data from database
            df = self.database.get_market_data(symbol, hours)
            
            if df.empty:
                logger.warning(f"No data available for {symbol}")
                return pd.DataFrame()
            
            # Add technical indicators
            df = self.feature_engineer.add_technical_indicators(df)
            
            # Create target variables
            df = self.feature_engineer.create_target_variables(df)
            
            # Remove NaN values
            df = df.dropna()
            
            logger.info(f"Prepared data for {symbol}: {len(df)} samples")
            return df
            
        except Exception as e:
            logger.error(f"Error preparing data for {symbol}: {e}")
            return pd.DataFrame()
    
    def train_random_forest_classifier(self, symbol: str, target: str = 'direction_next') -> bool:
        """Train Random Forest classifier"""
        try:
            df = self.prepare_data(symbol)
            if df.empty:
                return False
            
            # Feature selection
            feature_cols = [
                'sma_5', 'sma_10', 'sma_20', 'rsi', 'macd', 'macd_signal',
                'bb_position', 'price_change', 'volatility', 'momentum_5',
                'hour', 'day_of_week', 'support_distance', 'resistance_distance'
            ]
            
            # Filter available features
            available_features = [col for col in feature_cols if col in df.columns]
            
            if not available_features:
                logger.error("No features available for training")
                return False
            
            X = df[available_features].values
            y = df[target].values
            
            # Split data (time series split)
            split_point = int(len(X) * 0.8)
            X_train, X_test = X[:split_point], X[split_point:]
            y_train, y_test = y[:split_point], y[split_point:]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            
            logger.info(f"RF Classifier {symbol} - Train: {train_score:.3f}, Test: {test_score:.3f}")
            
            # Save model and scaler
            model_name = f"rf_classifier_{symbol.replace('/', '_')}"
            model_path = self.models_dir / f"{model_name}.pkl"
            scaler_path = self.models_dir / f"{model_name}_scaler.pkl"
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            
            # Store in memory
            self.models[model_name] = model
            self.scalers[model_name] = scaler
            
            # Save metadata
            metadata = {
                'model_type': 'random_forest_classifier',
                'symbol': symbol,
                'target': target,
                'features': available_features,
                'train_score': train_score,
                'test_score': test_score,
                'trained_at': datetime.now().isoformat(),
                'data_points': len(df)
            }
            
            metadata_path = self.models_dir / f"{model_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Error training RF classifier: {e}")
            return False
    
    def train_xgboost_regressor(self, symbol: str, target: str = 'return_1h') -> bool:
        """Train XGBoost regressor"""
        try:
            df = self.prepare_data(symbol)
            if df.empty:
                return False
            
            # Feature selection
            feature_cols = [
                'sma_5', 'sma_10', 'sma_20', 'rsi', 'macd', 'macd_signal',
                'bb_position', 'price_change', 'volatility', 'momentum_5',
                'hour', 'day_of_week'
            ]
            
            available_features = [col for col in feature_cols if col in df.columns]
            
            if not available_features:
                return False
            
            X = df[available_features].values
            y = df[target].values
            
            # Remove infinite values
            mask = np.isfinite(y)
            X, y = X[mask], y[mask]
            
            if len(X) < 100:
                logger.warning(f"Insufficient data for training: {len(X)} samples")
                return False
            
            # Split data
            split_point = int(len(X) * 0.8)
            X_train, X_test = X[:split_point], X[split_point:]
            y_train, y_test = y[:split_point], y[split_point:]
            
            # Train XGBoost
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            # Evaluate
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            train_mse = mean_squared_error(y_train, train_pred)
            test_mse = mean_squared_error(y_test, test_pred)
            
            logger.info(f"XGBoost {symbol} - Train MSE: {train_mse:.6f}, Test MSE: {test_mse:.6f}")
            
            # Save model
            model_name = f"xgb_regressor_{symbol.replace('/', '_')}"
            model_path = self.models_dir / f"{model_name}.pkl"
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            self.models[model_name] = model
            
            # Save metadata
            metadata = {
                'model_type': 'xgboost_regressor',
                'symbol': symbol,
                'target': target,
                'features': available_features,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'trained_at': datetime.now().isoformat(),
                'data_points': len(df)
            }
            
            metadata_path = self.models_dir / f"{model_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Error training XGBoost regressor: {e}")
            return False
    
    def train_lstm_model(self, symbol: str, target: str = 'return_1h') -> bool:
        """Train LSTM neural network"""
        if not TF_AVAILABLE:
            logger.warning("TensorFlow not available, skipping LSTM training")
            return False
        
        try:
            df = self.prepare_data(symbol, hours=2000)  # More data for LSTM
            if df.empty or len(df) < 500:
                logger.warning(f"Insufficient data for LSTM: {len(df)} samples")
                return False
            
            # Prepare sequences for LSTM
            sequence_length = 60  # 60 time steps
            
            # Feature selection
            feature_cols = ['price', 'volume', 'sma_10', 'rsi', 'macd']
            available_features = [col for col in feature_cols if col in df.columns]
            
            if len(available_features) < 2:
                return False
            
            # Normalize data
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df[available_features].values)
            
            # Create sequences
            X, y = [], []
            for i in range(sequence_length, len(scaled_data)):
                X.append(scaled_data[i-sequence_length:i])
                y.append(df[target].iloc[i])
            
            X, y = np.array(X), np.array(y)
            
            # Remove infinite values
            mask = np.isfinite(y)
            X, y = X[mask], y[mask]
            
            if len(X) < 100:
                return False
            
            # Split data
            split_point = int(len(X) * 0.8)
            X_train, X_test = X[:split_point], X[split_point:]
            y_train, y_test = y[:split_point], y[split_point:]
            
            # Build LSTM model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(sequence_length, len(available_features))),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            # Train model
            history = model.fit(
                X_train, y_train,
                batch_size=32,
                epochs=50,
                validation_data=(X_test, y_test),
                verbose=0
            )
            
            # Evaluate
            train_loss = history.history['loss'][-1]
            val_loss = history.history['val_loss'][-1]
            
            logger.info(f"LSTM {symbol} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Save model
            model_name = f"lstm_{symbol.replace('/', '_')}"
            model_path = self.models_dir / f"{model_name}.h5"
            scaler_path = self.models_dir / f"{model_name}_scaler.pkl"
            
            model.save(model_path)
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            
            self.models[model_name] = model
            self.scalers[model_name] = scaler
            
            # Save metadata
            metadata = {
                'model_type': 'lstm',
                'symbol': symbol,
                'target': target,
                'features': available_features,
                'sequence_length': sequence_length,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'trained_at': datetime.now().isoformat(),
                'data_points': len(df)
            }
            
            metadata_path = self.models_dir / f"{model_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Error training LSTM: {e}")
            return False
    
    def load_model(self, model_name: str) -> bool:
        """Load a trained model"""
        try:
            model_path = self.models_dir / f"{model_name}.pkl"
            h5_path = self.models_dir / f"{model_name}.h5"
            scaler_path = self.models_dir / f"{model_name}_scaler.pkl"
            
            # Load model
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            elif h5_path.exists() and TF_AVAILABLE:
                model = tf.keras.models.load_model(h5_path)
            else:
                logger.error(f"Model file not found: {model_name}")
                return False
            
            self.models[model_name] = model
            
            # Load scaler if exists
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                self.scalers[model_name] = scaler
            
            logger.info(f"Loaded model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return False
    
    def predict(self, symbol: str, model_name: str) -> Optional[PredictionResult]:
        """Make prediction using specified model"""
        try:
            if model_name not in self.models:
                if not self.load_model(model_name):
                    return None
            
            # Get recent data
            df = self.prepare_data(symbol, hours=100)
            if df.empty:
                return None
            
            # Load metadata to get features
            metadata_path = self.models_dir / f"{model_name}_metadata.json"
            if not metadata_path.exists():
                logger.error(f"Metadata not found for {model_name}")
                return None
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            model = self.models[model_name]
            features = metadata['features']
            model_type = metadata['model_type']
            
            # Prepare features
            available_features = [col for col in features if col in df.columns]
            if not available_features:
                return None
            
            X = df[available_features].iloc[-1:].values
            
            # Scale if scaler available
            if model_name in self.scalers:
                X = self.scalers[model_name].transform(X)
            
            # Make prediction
            if model_type == 'lstm':
                # For LSTM, need sequence
                sequence_length = metadata.get('sequence_length', 60)
                if len(df) < sequence_length:
                    return None
                
                X_seq = df[available_features].iloc[-sequence_length:].values
                if model_name in self.scalers:
                    X_seq = self.scalers[model_name].transform(X_seq)
                X_seq = X_seq.reshape(1, sequence_length, len(available_features))
                prediction = model.predict(X_seq)[0][0]
                confidence = 0.5  # Placeholder for LSTM confidence
            else:
                if hasattr(model, 'predict_proba'):
                    # Classifier
                    prediction = model.predict(X)[0]
                    probabilities = model.predict_proba(X)[0]
                    confidence = max(probabilities)
                else:
                    # Regressor
                    prediction = model.predict(X)[0]
                    confidence = 0.5  # Placeholder confidence
            
            return PredictionResult(
                symbol=symbol,
                prediction=float(prediction),
                confidence=float(confidence),
                timestamp=datetime.now(),
                model_name=model_name
            )
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return None
    
    def train_all_models(self, symbol: str) -> Dict[str, bool]:
        """Train all available models for a symbol"""
        results = {}
        
        logger.info(f"Training all models for {symbol}")
        
        # Train Random Forest Classifier
        results['rf_classifier'] = self.train_random_forest_classifier(symbol)
        
        # Train XGBoost Regressor  
        results['xgb_regressor'] = self.train_xgboost_regressor(symbol)
        
        # Train LSTM (if TensorFlow available)
        if TF_AVAILABLE:
            results['lstm'] = self.train_lstm_model(symbol)
        
        logger.info(f"Training completed for {symbol}: {results}")
        return results
    
    def get_model_info(self) -> List[Dict[str, Any]]:
        """Get information about all available models"""
        models_info = []
        
        for file_path in self.models_dir.glob("*_metadata.json"):
            try:
                with open(file_path, 'r') as f:
                    metadata = json.load(f)
                models_info.append(metadata)
            except Exception as e:
                logger.error(f"Error reading metadata {file_path}: {e}")
        
        return models_info

# Global instance
_ml_manager = None

def get_ml_manager() -> MLModelManager:
    """Get singleton ML model manager"""
    global _ml_manager
    if _ml_manager is None:
        _ml_manager = MLModelManager()
    return _ml_manager

# Training functions for backward compatibility
def train_model_by_name(model_name: str, X: np.ndarray, y: np.ndarray):
    """Train model by name (backward compatibility)"""
    try:
        if model_name == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            return model
        elif model_name == "XGBoost":
            model = xgb.XGBRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            return model
        else:
            logger.warning(f"Model {model_name} not implemented")
            return None
    except Exception as e:
        logger.error(f"Error training model {model_name}: {e}")
        return None

def load_pickle_model(file_path: str):
    """Load pickle model (backward compatibility)"""
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading pickle model: {e}")
        return None

def load_tf_model(file_path: str):
    """Load TensorFlow model (backward compatibility)"""
    if not TF_AVAILABLE:
        logger.error("TensorFlow not available")
        return None
    
    try:
        return tf.keras.models.load_model(file_path)
    except Exception as e:
        logger.error(f"Error loading TF model: {e}")
        return None

if __name__ == "__main__":
    # Test the ML system
    manager = get_ml_manager()
    
    # Test training
    symbols = ['BTC/USDT', 'ETH/USDT']
    
    for symbol in symbols:
        print(f"\nTesting {symbol}...")
        results = manager.train_all_models(symbol)
        print(f"Training results: {results}")
        
        # Test predictions
        for model_type in ['rf_classifier', 'xgb_regressor']:
            model_name = f"{model_type}_{symbol.replace('/', '_')}"
            prediction = manager.predict(symbol, model_name)
            if prediction:
                print(f"Prediction from {model_type}: {prediction.prediction:.4f} (confidence: {prediction.confidence:.2f})")
    
    # Get model info
    models_info = manager.get_model_info()
    print(f"\nAvailable models: {len(models_info)}")
    for info in models_info:
        print(f"- {info['model_type']} for {info['symbol']}")