import pandas as pd
import numpy as np
import logging
import os

def preprocess_data(df, is_train=True, train_medians=None):
    """
    Cleans and standardizes raw traffic dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input train or test dataframe.
    is_train : bool, default True
        If True, drops duplicates and computes medians.
    train_medians : dict, default None
        A dictionary containing historical training medians (required when is_train=False).
        
    Returns:
    --------
    cleaned_df : pd.DataFrame
        The cleaned dataframe.
    medians : dict
        A dictionary of medians computed on training set.
    """
    df_cleaned = df.copy()
    
    # 1. Deduplication (only on train to avoid losing validation/test rows)
    if is_train:
        initial_len = len(df_cleaned)
        df_cleaned = df_cleaned.drop_duplicates()
        dropped_count = initial_len - len(df_cleaned)
        if dropped_count > 0:
            logging.info(f"Removed {dropped_count} duplicate rows from training set.")
            
    # 2. Impute Numerical Columns
    medians = {}
    if is_train:
        temp_median = df_cleaned['Temperature'].median()
        lanes_median = df_cleaned['NumberofLanes'].median()
        if np.isnan(lanes_median):
            lanes_median = 1
        medians['Temperature'] = temp_median
        medians['NumberofLanes'] = int(round(lanes_median))
        logging.info(f"Calculated Temperature median: {temp_median:.4f}")
        logging.info(f"Calculated NumberofLanes median: {medians['NumberofLanes']}")
    else:
        if train_medians is not None:
            temp_median = train_medians.get('Temperature', df_cleaned['Temperature'].median())
            lanes_median = train_medians.get('NumberofLanes', 1)
        else:
            temp_median = df_cleaned['Temperature'].median()
            lanes_median = 1
            logging.warning("Training medians were not provided for testing. Falling back to local medians.")
    
    df_cleaned['Temperature'] = pd.to_numeric(df_cleaned['Temperature'], errors='coerce').fillna(temp_median)
    df_cleaned['NumberofLanes'] = pd.to_numeric(df_cleaned['NumberofLanes'], errors='coerce').fillna(lanes_median).astype(int)
    
    # 3. Impute Categorical Columns
    categorical_cols = ['RoadType', 'Weather', 'LargeVehicles', 'Landmarks']
    for col in categorical_cols:
        if col in df_cleaned.columns:
            missing_count = df_cleaned[col].isna().sum()
            df_cleaned[col] = df_cleaned[col].fillna('Unknown')
            if missing_count > 0:
                logging.info(f"Imputed {missing_count} missing values in '{col}' with 'Unknown'.")
                
    # 4. Standardize Categorical Strings (strip trailing/leading spaces and match casing)
    str_cols = ['geohash', 'timestamp', 'RoadType', 'LargeVehicles', 'Landmarks', 'Weather']
    for col in str_cols:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].astype(str).str.strip()
            
    # 5. Sanity Checks & Logging
    nan_remaining = df_cleaned.isna().sum().sum()
    if nan_remaining > 0:
        logging.warning(f"Sanity Check Warning: {nan_remaining} missing values still remain in the dataset!")
    else:
        logging.info("Sanity Check Passed: Zero missing values remain in the cleaned dataset.")
        
    return df_cleaned, medians
