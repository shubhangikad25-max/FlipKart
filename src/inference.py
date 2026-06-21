import os
import pickle
import logging

import numpy as np
import pandas as pd
from catboost import Pool

from src.preprocessing import preprocess_data
from src.feature_engineering import extract_features
from src.train_model import prepare_data_for_boosters


def run_inference():
    """
    Runs the final inference pipeline and generates the test submission.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('Starting inference pipeline...')

    metadata = {
        'medians': 'models/train_medians.pkl',
        'feature_stats': 'models/feature_stats.pkl',
        'category_maps': 'models/category_maps.pkl'
    }

    for name, path in metadata.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f'Expected metadata file missing: {path}')

    with open(metadata['medians'], 'rb') as f:
        train_medians = pickle.load(f)
    with open(metadata['feature_stats'], 'rb') as f:
        feature_stats = pickle.load(f)
    with open(metadata['category_maps'], 'rb') as f:
        category_maps = pickle.load(f)

    model_paths = {
        'catboost': 'models/catboost_model.pkl',
        'lightgbm': 'models/lightgbm_model.pkl',
        'xgboost': 'models/xgboost_model.pkl'
    }

    models = {}
    for name, path in model_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f'Expected model file missing: {path}')
        with open(path, 'rb') as f:
            models[name] = pickle.load(f)

    logging.info('Loading test dataset...')
    test_raw = pd.read_csv('data/test.csv')
    logging.info(f'Loaded test.csv shape: {test_raw.shape}')

    test_clean, _ = preprocess_data(test_raw, is_train=False, train_medians=train_medians)

    categorical_cols = [
        'RoadType', 'LargeVehicles', 'Landmarks', 'Weather',
        'time_block', 'road_lane_interaction', 'road_weather'
    ]
    cols_to_drop = [
        'Index', 'timestamp',
        'geohash', 'geo_prefix_5', 'geo_prefix_4', 'geo_prefix_3',
        'geo_hour', 'road_hour'
    ]

    logging.info('Extracting test features...')
    test_feat, _ = extract_features(test_clean, stats=feature_stats, is_train=False)
    X_test_cat = test_feat.drop(columns=cols_to_drop).fillna(0)

    for col in categorical_cols:
        if col in X_test_cat.columns:
            X_test_cat[col] = X_test_cat[col].astype(str)

    logging.info('Predicting with CatBoost...')
    cat_features_indices = [
        X_test_cat.columns.get_loc(col)
        for col in categorical_cols
        if col in X_test_cat.columns
    ]
    test_pool = Pool(X_test_cat, cat_features=cat_features_indices)
    preds_cat = models['catboost'].predict(test_pool)

    logging.info('Encoding test features for LightGBM/XGBoost...')
    X_test_encoded, _ = prepare_data_for_boosters(
        test_feat.drop(columns=cols_to_drop), categorical_cols, category_maps=category_maps
    )
    X_test_encoded = X_test_encoded.fillna(0)

    xgb_features = models['xgboost'].get_booster().feature_names
    for col in xgb_features:
        if col not in X_test_encoded.columns:
            X_test_encoded[col] = 0
    X_test_encoded = X_test_encoded[xgb_features]

    logging.info('Predicting with LightGBM...')
    preds_lgb = models['lightgbm'].predict(X_test_encoded)

    logging.info('Predicting with XGBoost...')
    preds_xgb = models['xgboost'].predict(X_test_encoded)

    final_blend = 0.6 * preds_cat + 0.4 * preds_lgb
    final_trio = 0.5 * preds_cat + 0.3 * preds_lgb + 0.2 * preds_xgb

    os.makedirs('outputs', exist_ok=True)

    primary_submission = pd.DataFrame({'Index': test_clean['Index'], 'demand': final_blend})
    primary_submission.to_csv('outputs/submission.csv', index=False)
    logging.info('Saved primary submission to outputs/submission.csv')

    trio_submission = pd.DataFrame({'Index': test_clean['Index'], 'demand': final_trio})
    trio_submission.to_csv('outputs/submission_trio.csv', index=False)
    logging.info('Saved trio submission to outputs/submission_trio.csv')

    if len(primary_submission) != 41778:
        logging.warning(f'Expected 41778 rows, got {len(primary_submission)}. Verify the test file and split.')
    if list(primary_submission.columns) != ['Index', 'demand']:
        raise ValueError(f'Unexpected submission columns: {list(primary_submission.columns)}')
    if primary_submission['demand'].isna().sum() > 0:
        raise ValueError('Submission contains missing demand predictions.')

    logging.info('Inference completed successfully.')


if __name__ == '__main__':
    run_inference()
