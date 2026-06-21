import os
import json
import pickle
import logging

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold

from src.preprocessing import preprocess_data
from src.feature_engineering import fit_feature_engineering_stats, extract_features
from src.utils import set_seed, setup_logging, evaluate_regression


def prepare_data_for_boosters(df, categorical_cols, category_maps=None):
    """
    Prepares dataset for LightGBM and XGBoost by label encoding categorical columns.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with raw categorical fields.
    categorical_cols : list
        Columns that should be transformed into integer codes.
    category_maps : dict or None
        If provided, use this mapping to align categories between train and future sets.

    Returns:
    --------
    df_encoded : pd.DataFrame
        The encoded dataframe ready for tree boosters.
    category_maps : dict
        The category mapping used for each categorical column.
    """
    df_encoded = df.copy()
    category_maps = {} if category_maps is None else dict(category_maps)

    for col in categorical_cols:
        if col not in df_encoded.columns:
            continue

        df_encoded[col] = df_encoded[col].astype(str).fillna('Unknown')

        if col in category_maps:
            categories = category_maps[col]
        else:
            categories = pd.Categorical(df_encoded[col]).categories.tolist()

        df_encoded[col] = pd.Categorical(df_encoded[col], categories=categories)
        df_encoded[col + '_code'] = df_encoded[col].cat.codes
        df_encoded = df_encoded.drop(columns=[col])
        category_maps[col] = categories

    return df_encoded, category_maps


def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def train_and_evaluate():
    """
    Executes the entire training pipeline with time-aware validation and final ensemble model serialization.
    """
    setup_logging()
    set_seed(42)

    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs/plots', exist_ok=True)

    logging.info('Loading training dataset...')
    train_raw = pd.read_csv('data/train.csv')
    logging.info(f'Loaded train.csv shape: {train_raw.shape}')

    logging.info('Preprocessing training dataset...')
    train_clean, medians = preprocess_data(train_raw, is_train=True)
    save_pickle(medians, 'models/train_medians.pkl')
    logging.info('Saved training medians to models/train_medians.pkl')

    day48_df = train_clean[train_clean['day'] == 48].reset_index(drop=True)
    day49_holdout = train_clean[train_clean['day'] == 49].reset_index(drop=True)

    if day49_holdout.empty:
        raise ValueError('Day 49 holdout data is missing from the training file. Cannot perform time-aware validation.')

    logging.info(f'Day 48 training rows: {len(day48_df)}, Day 49 holdout rows: {len(day49_holdout)}')

    categorical_cols = [
        'RoadType', 'LargeVehicles', 'Landmarks', 'Weather',
        'time_block', 'road_lane_interaction', 'road_weather'
    ]

    cols_to_drop = [
        'Index', 'timestamp', 'demand',
        'geohash', 'geo_prefix_5', 'geo_prefix_4', 'geo_prefix_3',
        'geo_hour', 'road_hour'
    ]

    fold_stats_list = []
    category_maps_master = {}

    # OOF predictions for time-aware holdout and day 48 validation
    day48_oof_cat = np.zeros(len(day48_df))
    day48_oof_lgb = np.zeros(len(day48_df))
    day48_oof_xgb = np.zeros(len(day48_df))

    day49_holdout_cat = np.zeros(len(day49_holdout))
    day49_holdout_lgb = np.zeros(len(day49_holdout))
    day49_holdout_xgb = np.zeros(len(day49_holdout))

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    logging.info('Starting 5-Fold Cross Validation on Day 48 data...')

    for fold, (train_idx, val_idx) in enumerate(kf.split(day48_df)):
        logging.info(f'\n==================== FOLD {fold+1} / 5 ====================')

        fold_train_raw = day48_df.iloc[train_idx].reset_index(drop=True)
        fold_val_raw = day48_df.iloc[val_idx].reset_index(drop=True)

        fold_stats = fit_feature_engineering_stats(fold_train_raw)
        fold_stats_list.append(fold_stats)

        fold_train_feat, _ = extract_features(fold_train_raw, stats=fold_stats, is_train=True)
        fold_val_feat, _ = extract_features(fold_val_raw, stats=fold_stats, is_train=False)
        fold_holdout_feat, _ = extract_features(day49_holdout, stats=fold_stats, is_train=False)

        X_train_cat = fold_train_feat.drop(columns=cols_to_drop)
        y_train = fold_train_feat['demand']
        X_val_cat = fold_val_feat.drop(columns=cols_to_drop)
        y_val = fold_val_feat['demand']
        X_hold_cat = fold_holdout_feat.drop(columns=cols_to_drop)
        y_hold = fold_holdout_feat['demand']

        X_train_cat = X_train_cat.fillna(0)
        X_val_cat = X_val_cat.fillna(0)
        X_hold_cat = X_hold_cat.fillna(0)

        logging.info(f'Training features count: {X_train_cat.shape[1]}')

        cat_features_indices = [
            X_train_cat.columns.get_loc(col)
            for col in categorical_cols
            if col in X_train_cat.columns
        ]

        for col in categorical_cols:
            if col in X_train_cat.columns:
                X_train_cat[col] = X_train_cat[col].astype(str)
                X_val_cat[col] = X_val_cat[col].astype(str)
                X_hold_cat[col] = X_hold_cat[col].astype(str)

        train_pool = Pool(X_train_cat, y_train, cat_features=cat_features_indices)
        val_pool = Pool(X_val_cat, y_val, cat_features=cat_features_indices)

        cb_model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=7,
            eval_metric='R2',
            random_seed=42,
            early_stopping_rounds=50,
            verbose=100
        )
        cb_model.fit(train_pool, eval_set=val_pool, use_best_model=True)

        save_pickle(cb_model, f'models/catboost_fold_{fold+1}.pkl')

        fold_val_pred_cat = cb_model.predict(X_val_cat)
        day48_oof_cat[val_idx] = fold_val_pred_cat
        evaluate_regression(y_val, fold_val_pred_cat, f'CatBoost Fold {fold+1} (Day48 val)')

        fold_holdout_pred_cat = cb_model.predict(X_hold_cat)
        day49_holdout_cat += fold_holdout_pred_cat / kf.n_splits

        X_train_encoded, category_maps = prepare_data_for_boosters(
            fold_train_feat.drop(columns=cols_to_drop), categorical_cols, category_maps=None
        )
        X_val_encoded, _ = prepare_data_for_boosters(
            fold_val_feat.drop(columns=cols_to_drop), categorical_cols, category_maps=category_maps
        )
        X_hold_encoded, _ = prepare_data_for_boosters(
            fold_holdout_feat.drop(columns=cols_to_drop), categorical_cols, category_maps=category_maps
        )

        X_train_encoded = X_train_encoded.fillna(0)
        X_val_encoded = X_val_encoded.fillna(0)
        X_hold_encoded = X_hold_encoded.fillna(0)

        category_maps_master = category_maps

        missing_cols = set(X_train_encoded.columns) - set(X_val_encoded.columns)
        for col in missing_cols:
            X_val_encoded[col] = 0
        X_val_encoded = X_val_encoded[X_train_encoded.columns]

        missing_cols = set(X_train_encoded.columns) - set(X_hold_encoded.columns)
        for col in missing_cols:
            X_hold_encoded[col] = 0
        X_hold_encoded = X_hold_encoded[X_train_encoded.columns]

        logging.info('Training LightGBMRegressor...')
        lgb_train = lgb.Dataset(X_train_encoded, y_train)
        lgb_val = lgb.Dataset(X_val_encoded, y_val, reference=lgb_train)

        lgb_params = {
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
        }

        lgb_model = lgb.train(
            lgb_params,
            lgb_train,
            num_boost_round=1000,
            valid_sets=[lgb_train, lgb_val],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=100)
            ]
        )

        save_pickle(lgb_model, f'models/lightgbm_fold_{fold+1}.pkl')

        fold_val_pred_lgb = lgb_model.predict(X_val_encoded)
        day48_oof_lgb[val_idx] = fold_val_pred_lgb
        evaluate_regression(y_val, fold_val_pred_lgb, f'LightGBM Fold {fold+1} (Day48 val)')

        fold_holdout_pred_lgb = lgb_model.predict(X_hold_encoded)
        day49_holdout_lgb += fold_holdout_pred_lgb / kf.n_splits

        logging.info('Training XGBoostRegressor...')
        xgb_model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            early_stopping_rounds=50,
            eval_metric='rmse'
        )
        xgb_model.fit(
            X_train_encoded, y_train,
            eval_set=[(X_val_encoded, y_val)],
            verbose=100
        )

        save_pickle(xgb_model, f'models/xgboost_fold_{fold+1}.pkl')

        fold_val_pred_xgb = xgb_model.predict(X_val_encoded)
        day48_oof_xgb[val_idx] = fold_val_pred_xgb
        evaluate_regression(y_val, fold_val_pred_xgb, f'XGBoost Fold {fold+1} (Day48 val)')

        fold_holdout_pred_xgb = xgb_model.predict(X_hold_encoded)
        day49_holdout_xgb += fold_holdout_pred_xgb / kf.n_splits

    save_pickle(fold_stats_list, 'models/fold_stats_list.pkl')
    save_pickle(category_maps_master, 'models/category_maps.pkl')
    logging.info('Saved fitted fold statistics and category maps for inference.')

    day48_true = day48_df['demand'].values
    logging.info('\n==================== DAY 48 OUT-OF-FOLD PERFORMANCE ====================')
    evaluate_regression(day48_true, day48_oof_cat, 'CatBoost Day48 OOF')
    evaluate_regression(day48_true, day48_oof_lgb, 'LightGBM Day48 OOF')
    evaluate_regression(day48_true, day48_oof_xgb, 'XGBoost Day48 OOF')

    day49_true = day49_holdout['demand'].values
    logging.info('\n==================== DAY 49 HOLDOUT PERFORMANCE ====================')
    evaluate_regression(day49_true, day49_holdout_cat, 'CatBoost Day49 Holdout')
    evaluate_regression(day49_true, day49_holdout_lgb, 'LightGBM Day49 Holdout')
    evaluate_regression(day49_true, day49_holdout_xgb, 'XGBoost Day49 Holdout')

    save_pickle({
        'day48_oof_cat': day48_oof_cat.tolist(),
        'day48_oof_lgb': day48_oof_lgb.tolist(),
        'day48_oof_xgb': day48_oof_xgb.tolist(),
        'day49_holdout_cat': day49_holdout_cat.tolist(),
        'day49_holdout_lgb': day49_holdout_lgb.tolist(),
        'day49_holdout_xgb': day49_holdout_xgb.tolist()
    }, 'outputs/oof_predictions.pkl')

    oof_df = pd.DataFrame({
        'Index': day48_df['Index'],
        'demand_true': day48_true,
        'pred_cat': day48_oof_cat,
        'pred_lgb': day48_oof_lgb,
        'pred_xgb': day48_oof_xgb,
        'pred_ensemble_blend': 0.6 * day48_oof_cat + 0.4 * day48_oof_lgb,
        'pred_ensemble_trio': 0.5 * day48_oof_cat + 0.3 * day48_oof_lgb + 0.2 * day48_oof_xgb
    })
    oof_df.to_csv('outputs/oof_predictions.csv', index=False)
    logging.info('Saved out-of-fold predictions to outputs/oof_predictions.csv')

    # Train final models on all available data before inference
    logging.info('\nTraining final models on all available training data...')
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

    final_X_encoded, final_category_maps = prepare_data_for_boosters(final_train_feat.drop(columns=cols_to_drop), categorical_cols)
    final_X_encoded = final_X_encoded.fillna(0)
    save_pickle(final_category_maps, 'models/category_maps.pkl')
    save_pickle(final_stats, 'models/feature_stats.pkl')

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
        callbacks=[
            lgb.log_evaluation(period=100)
        ]
    )
    save_pickle(final_lgb_model, 'models/lightgbm_model.pkl')

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

    logging.info('Saved final trained models to models/catboost_model.pkl, models/lightgbm_model.pkl, models/xgboost_model.pkl')
    
    # Generate all premium diagnostic and EDA plots
    try:
        from src.visualization import generate_all_plots
        logging.info('Generating diagnostic and EDA plots...')
        generate_all_plots(final_train_feat, final_cb_model, list(final_X_cat.columns))
    except Exception as e:
        logging.error(f'Failed to generate plots: {str(e)}')
        
    logging.info('Model training pipeline successfully completed.')


if __name__ == '__main__':
    train_and_evaluate()
