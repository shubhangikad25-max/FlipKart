import random
import os
import numpy as np
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def set_seed(seed=42):
    """Sets random seeds for reproducibility across numpy, random, and environment."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    logging.info(f"Random seed set to {seed} globally.")

def setup_logging(log_file='outputs/pipeline.log'):
    """Sets up unified logging to write logs to both file and standard output."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clean previous handlers to avoid duplicate prints
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    
    logging.info("Logging initialized successfully.")

def decode_geohash(geohash):
    """
    Decodes standard geohash string into latitude and longitude coordinates.
    Pure-Python implementation to avoid external dependencies.
    """
    if not geohash or not isinstance(geohash, str):
        return np.nan, np.nan
        
    base32 = "0123456789bcdefghjkmnpqrstuvwxyz"
    dec_map = {char: i for i, char in enumerate(base32)}
    
    lat_interval = (-90.0, 90.0)
    lon_interval = (-180.0, 180.0)
    is_even = True
    
    try:
        for char in geohash.lower():
            if char not in dec_map:
                return np.nan, np.nan
            val = dec_map[char]
            for mask in [16, 8, 4, 2, 1]:
                if is_even:  # longitude split
                    mid = (lon_interval[0] + lon_interval[1]) / 2.0
                    if val & mask:
                        lon_interval = (mid, lon_interval[1])
                    else:
                        lon_interval = (lon_interval[0], mid)
                else:  # latitude split
                    mid = (lat_interval[0] + lat_interval[1]) / 2.0
                    if val & mask:
                        lat_interval = (mid, lat_interval[1])
                    else:
                        lat_interval = (lat_interval[0], mid)
                is_even = not is_even
                
        lat = (lat_interval[0] + lat_interval[1]) / 2.0
        lon = (lon_interval[0] + lon_interval[1]) / 2.0
        return lat, lon
    except Exception:
        return np.nan, np.nan

def evaluate_regression(y_true, y_pred, model_name="Model"):
    """
    Calculates R2, RMSE, and MAE regression metrics and prints/logs results.
    """
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    logging.info(f"=== {model_name} Performance ===")
    logging.info(f"R² Score : {r2:.6f}")
    logging.info(f"RMSE     : {rmse:.6f}")
    logging.info(f"MAE      : {mae:.6f}")
    logging.info("=================================")
    
    return {"R2": r2, "RMSE": rmse, "MAE": mae}
