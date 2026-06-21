"""
Recovery script: Trains only the final models on all data.
Run this after the 5-fold CV has already completed.
"""
import os
import pickle
import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool

from src.preprocessing import preprocess_data
from src.feature_engineering import fit_feature_engineering_stats, extract_features
from src.train_model import prepare_data_for_boosters, save_pickle
from src.utils import set_seed, setup_logging


def run_final_training():
    setup_logging()
    set_seed(42)

    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs/plots', exist_ok=True)

    logging.info('Loading training dataset...')
    train_raw = pd.read_csv('data/train.csv')

    logging.info('Preprocessing...')
    with open('models/train_medians.pkl', 'rb') as f:
        medians = pickle.load(f)
    train_clean, _ = preprocess_data(train_raw, is_train=False, train_medians=medians)

    categorical_cols = [
        'RoadType', 'LargeVehicles', 'Landmarks', 'Weather',
        'time_block', 'road_lane_interaction', 'road_weather'
    ]
    cols_to_drop = [
        'Index', 'timestamp', 'demand',
        'geohash', 'geo_prefix_5', 'geo_prefix_4', 'geo_prefix_3',
        'geo_hour', 'road_hour'
    ]

    logging.info('Fitting feature engineering stats on all training data...')
    final_stats = fit_feature_engineering_stats(train_clean)
    final_train_feat, _ = extract_features(train_clean, stats=final_stats, is_train=True)

    final_X_cat = final_train_feat.drop(columns=cols_to_drop).fillna(0)
    final_y = final_train_feat['demand']

    for col in categorical_cols:
        if col in final_X_cat.columns:
            final_X_cat[col] = final_X_cat[col].astype(str)

    final_cat_features = [
        final_X_cat.columns.get_loc(col)
        for col in categorical_cols
        if col in final_X_cat.columns
    ]

    logging.info(f'Training final CatBoost on {len(final_X_cat)} rows, {len(final_X_cat.columns)} features...')
    final_cb_model = CatBoostRegressor(
        iterations=1200,
        learning_rate=0.05,
        depth=7,
        eval_metric='R2',
        random_seed=42,
        verbose=100
    )
    final_train_pool = Pool(final_X_cat, final_y, cat_features=final_cat_features)
    final_cb_model.fit(final_train_pool)
    save_pickle(final_cb_model, 'models/catboost_model.pkl')
    logging.info('Saved models/catboost_model.pkl')

    # Encode for LightGBM / XGBoost
    final_X_encoded, final_category_maps = prepare_data_for_boosters(
        final_train_feat.drop(columns=cols_to_drop), categorical_cols
    )
    final_X_encoded = final_X_encoded.fillna(0)
    save_pickle(final_category_maps, 'models/category_maps.pkl')
    save_pickle(final_stats, 'models/feature_stats.pkl')
    logging.info('Saved models/category_maps.pkl and models/feature_stats.pkl')

    logging.info('Training final LightGBM...')
    final_lgb_train = lgb.Dataset(final_X_encoded, final_y)
    final_lgb_model = lgb.train(
        {
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.05,
            'num_leaves': 63,
            'max_depth': 7,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'random_state': 42,
            'verbose': -1
        },
        final_lgb_train,
        num_boost_round=1200,
        callbacks=[lgb.log_evaluation(period=100)]
    )
    save_pickle(final_lgb_model, 'models/lightgbm_model.pkl')
    logging.info('Saved models/lightgbm_model.pkl')

    logging.info('Training final XGBoost...')
    final_xgb_model = xgb.XGBRegressor(
        n_estimators=1200,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='rmse'
    )
    final_xgb_model.fit(final_X_encoded, final_y, verbose=100)
    save_pickle(final_xgb_model, 'models/xgboost_model.pkl')
    logging.info('Saved models/xgboost_model.pkl')

    # Generate plots
    try:
        from src.visualization import generate_all_plots
        logging.info('Generating EDA plots...')
        generate_all_plots(final_train_feat, final_cb_model, list(final_X_cat.columns))
    except Exception as e:
        logging.error(f'Plot generation failed: {e}')

    logging.info('Final model training completed successfully.')


if __name__ == '__main__':
    run_final_training()
