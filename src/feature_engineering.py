import pandas as pd
import numpy as np
import logging
from src.utils import decode_geohash

def fit_feature_engineering_stats(train_df):
    """
    Fits feature engineering statistics (like geohash target encodings) purely on the training set
    to avoid temporal data leakage.
    """
    stats = {}
    
    # 1. Base Target Statistics by Geohash
    geo_stats = train_df.groupby('geohash')['demand'].agg(['mean', 'std', 'median', 'count']).reset_index()
    geo_stats.columns = ['geohash', 'geo_mean_demand', 'geo_std_demand', 'geo_median_demand', 'geo_freq']
    # Fill standard deviation for single occurrences
    geo_stats['geo_std_demand'] = geo_stats['geo_std_demand'].fillna(0.0)
    stats['geohash_stats'] = geo_stats
    
    # 2. Overall training global mean for ultimate fallback
    global_mean = train_df['demand'].mean()
    global_median = train_df['demand'].median()
    stats['global_mean'] = global_mean
    stats['global_median'] = global_median
    logging.info(f"Global training demand mean: {global_mean:.4f}, median: {global_median:.4f}")
    
    # 3. Create hierarchical spatial prefixes to use as fallback levels for unseen geohashes
    train_df_copy = train_df.copy()
    train_df_copy['geo_prefix_5'] = train_df_copy['geohash'].str.slice(0, 5)
    train_df_copy['geo_prefix_4'] = train_df_copy['geohash'].str.slice(0, 4)
    train_df_copy['geo_prefix_3'] = train_df_copy['geohash'].str.slice(0, 3)
    
    stats['prefix_5_stats'] = train_df_copy.groupby('geo_prefix_5')['demand'].mean().to_dict()
    stats['prefix_4_stats'] = train_df_copy.groupby('geo_prefix_4')['demand'].mean().to_dict()
    stats['prefix_3_stats'] = train_df_copy.groupby('geo_prefix_3')['demand'].mean().to_dict()
    
    # 4. Geohash Rankings
    geo_stats_sorted = geo_stats.sort_values(by='geo_mean_demand', ascending=False).reset_index(drop=True)
    geo_stats_sorted['geo_rank'] = geo_stats_sorted.index + 1
    stats['geohash_rank_map'] = geo_stats_sorted.set_index('geohash')['geo_rank'].to_dict()
    
    logging.info(f"Fitted feature engineering stats for {len(geo_stats)} unique training geohashes.")
    return stats

def extract_features(df, stats=None, is_train=True):
    """
    Extracts advanced spatial, temporal, road, weather, and interaction features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input cleaned dataframe.
    stats : dict, default None
        Fitted statistics from fit_feature_engineering_stats (required for consistent mappings).
    is_train : bool, default True
        If True, statistics will be fitted if stats is None.
        
    Returns:
    --------
    featured_df : pd.DataFrame
        The dataframe enriched with engineered features.
    """
    df_feat = df.copy()
    
    # 1. Parsing Timestamps
    # Split timestamp string (e.g. '13:45') into hour and minute
    df_feat[['hour_str', 'minute_str']] = df_feat['timestamp'].str.split(':', expand=True)
    df_feat['hour'] = df_feat['hour_str'].astype(int)
    df_feat['minute'] = df_feat['minute_str'].astype(int)
    df_feat['total_minutes'] = df_feat['hour'] * 60 + df_feat['minute']
    df_feat = df_feat.drop(columns=['hour_str', 'minute_str'])
    
    # Cyclical hour encoding
    df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['hour'] / 24.0)
    df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['hour'] / 24.0)
    
    # Peak Hour (7-10 AM and 5-8 PM)
    df_feat['is_peak_hour'] = df_feat['hour'].apply(lambda h: 1 if h in [7, 8, 9, 17, 18, 19] else 0)
    
    # Weekend Mapping (Assume day % 7 in [5, 6] represents weekends)
    df_feat['day_of_week'] = df_feat['day'] % 7
    df_feat['is_weekend'] = df_feat['day_of_week'].apply(lambda d: 1 if d in [5, 6] else 0)
    
    # Time Blocks
    def get_time_block(h):
        if 5 <= h < 12:
            return 'Morning'
        elif 12 <= h < 17:
            return 'Afternoon'
        elif 17 <= h < 21:
            return 'Evening'
        else:
            return 'Night'
    df_feat['time_block'] = df_feat['hour'].apply(get_time_block)
    
    # 2. Geohash Spatial Coordinate Decoding
    # Decode geohash to exact continuous latitude and longitude
    coordinates = df_feat['geohash'].apply(decode_geohash)
    df_feat['latitude'] = [coords[0] for coords in coordinates]
    df_feat['longitude'] = [coords[1] for coords in coordinates]
    
    # Hierarchical subdivisions
    df_feat['geo_prefix_5'] = df_feat['geohash'].str.slice(0, 5)
    df_feat['geo_prefix_4'] = df_feat['geohash'].str.slice(0, 4)
    df_feat['geo_prefix_3'] = df_feat['geohash'].str.slice(0, 3)
    
    # 3. Spatial Target Encodings and Densities
    if is_train and stats is None:
        stats = fit_feature_engineering_stats(df_feat)
        
    if stats is not None:
        # Merge basic geohash statistics
        df_feat = pd.merge(df_feat, stats['geohash_stats'], on='geohash', how='left')
        
        # Handle missing geohash values (unseen in training set)
        unseen_mask = df_feat['geo_mean_demand'].isna()
        unseen_count = unseen_mask.sum()
        
        if unseen_count > 0:
            logging.info(f"Imputing target statistics for {unseen_count} unseen geohash rows using prefix groupings.")
            
            # Map values based on prefix-5 group mean
            p5_map = stats['prefix_5_stats']
            df_feat.loc[unseen_mask, 'geo_mean_demand'] = df_feat.loc[unseen_mask, 'geo_prefix_5'].map(p5_map)
            
            # For remaining NaNs, map prefix-4 group mean
            unseen_mask = df_feat['geo_mean_demand'].isna()
            p4_map = stats['prefix_4_stats']
            df_feat.loc[unseen_mask, 'geo_mean_demand'] = df_feat.loc[unseen_mask, 'geo_prefix_4'].map(p4_map)
            
            # For remaining NaNs, map prefix-3 group mean
            unseen_mask = df_feat['geo_mean_demand'].isna()
            p3_map = stats['prefix_3_stats']
            df_feat.loc[unseen_mask, 'geo_mean_demand'] = df_feat.loc[unseen_mask, 'geo_prefix_3'].map(p3_map)
            
            # Complete fallback to global mean
            df_feat['geo_mean_demand'] = df_feat['geo_mean_demand'].fillna(stats['global_mean'])
            df_feat['geo_median_demand'] = df_feat['geo_median_demand'].fillna(stats['global_median'])
            df_feat['geo_std_demand'] = df_feat['geo_std_demand'].fillna(0.0)
            df_feat['geo_freq'] = df_feat['geo_freq'].fillna(1.0)
            
        # Map Geohash Rankings
        rank_map = stats['geohash_rank_map']
        df_feat['geo_rank'] = df_feat['geohash'].map(rank_map)
        
        # Unseen geohashes get worst rank (len(rank_map) + 1)
        worst_rank = len(rank_map) + 1
        df_feat['geo_rank'] = df_feat['geo_rank'].fillna(worst_rank).astype(int)
        
        # Density feature
        df_feat['geo_density'] = df_feat['geo_freq'] / df_feat['geo_freq'].max()
    else:
        # Dummy placeholders just in case stats are not passed
        df_feat['geo_mean_demand'] = 0.0
        df_feat['geo_std_demand'] = 0.0
        df_feat['geo_median_demand'] = 0.0
        df_feat['geo_freq'] = 1.0
        df_feat['geo_rank'] = 1
        df_feat['geo_density'] = 1.0
        
    # 4. Road Capacity Features
    df_feat['lane_capacity'] = df_feat['NumberofLanes'].astype(float)
    df_feat['heavy_vehicle_score'] = df_feat['LargeVehicles'].apply(lambda x: 1 if x == 'Allowed' else 0)
    df_feat['road_lane_interaction'] = df_feat['RoadType'] + "_" + df_feat['NumberofLanes'].astype(str)
    
    # Avoid zero-division when calculating congestion ratios
    df_feat['congestion_capacity_ratio'] = df_feat['geo_mean_demand'] / df_feat['lane_capacity'].clip(lower=1.0)
    
    # 5. Weather Features
    weather_severity_map = {
        'Sunny': 1.0,
        'Cloudy': 2.0,
        'Unknown': 2.0,
        'Rainy': 3.5,
        'Foggy': 4.0,
        'Snowy': 5.0
    }
    df_feat['weather_severity'] = df_feat['Weather'].map(weather_severity_map).fillna(2.0)
    df_feat['temp_weather_interaction'] = df_feat['Temperature'] * df_feat['weather_severity']
    
    # 6. Interaction Features
    df_feat['temp_hour'] = df_feat['Temperature'] * df_feat['hour']
    df_feat['lane_temp'] = df_feat['NumberofLanes'] * df_feat['Temperature']
    df_feat['weather_hour'] = df_feat['weather_severity'] * df_feat['hour']
    df_feat['road_weather'] = df_feat['RoadType'] + "_" + df_feat['Weather']
    df_feat['road_hour'] = df_feat['RoadType'] + "_" + df_feat['hour'].astype(str)
    df_feat['lane_vehicle'] = df_feat['NumberofLanes'] * df_feat['heavy_vehicle_score']
    df_feat['geo_hour'] = df_feat['geo_prefix_5'] + "_" + df_feat['hour'].astype(str)
    df_feat['peak_weather'] = df_feat['is_peak_hour'] * df_feat['weather_severity']
    
    logging.info(f"Feature engineering pipeline completed. Total columns: {len(df_feat.columns)}")
    return df_feat, stats
