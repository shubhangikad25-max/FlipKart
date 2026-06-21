# pyrefly: ignore [missing-import]
import numpy as np
import logging

def compute_blend(predictions_dict, weights_dict=None, clip_range=None):
    """
    Computes a weighted average of prediction vectors.
    
    Parameters:
    -----------
    predictions_dict : dict
        Dictionary mapping model names to prediction arrays, e.g. {'catboost': array, 'lightgbm': array}
    weights_dict : dict, default None
        Dictionary mapping model names to float weights. Must sum to 1.0.
        If None, computes a simple uniform average.
    clip_range : tuple(float, float), optional
        Optional lower and upper bounds for prediction clipping.
        If None, no clipping is applied.
        
    Returns:
    --------
    blended_pred : np.ndarray
        The weighted average prediction array.
    """
    models = list(predictions_dict.keys())
    if len(models) == 0:
        raise ValueError("Predictions dict cannot be empty.")
        
    sample_len = len(predictions_dict[models[0]])
    
    if weights_dict is None:
        uniform_weight = 1.0 / len(models)
        weights_dict = {model: uniform_weight for model in models}
        
    weight_sum = sum(weights_dict.values())
    if not np.isclose(weight_sum, 1.0):
        logging.warning(f"Weights sum to {weight_sum:.4f} instead of 1.0. Normalizing weights...")
        weights_dict = {m: w / weight_sum for m, w in weights_dict.items()}
        
    blended_pred = np.zeros(sample_len)
    for model in models:
        if model not in predictions_dict:
            raise KeyError(f"Model '{model}' predictions missing from input dict.")
        weight = weights_dict.get(model, 0.0)
        blended_pred += predictions_dict[model] * weight
        logging.info(f"Blended model '{model}' with weight: {weight:.4f}")
        
    if clip_range is not None:
        blended_pred = np.clip(blended_pred, clip_range[0], clip_range[1])
    return blended_pred
