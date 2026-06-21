import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import logging

def set_premium_style():
    """Applies modern dark-theme styles for stunning scientific visualizations."""
    plt.style.use('dark_background')
    plt.rcParams['figure.facecolor'] = '#121212'
    plt.rcParams['axes.facecolor'] = '#1e1e1e'
    plt.rcParams['grid.color'] = '#333333'
    plt.rcParams['text.color'] = '#e0e0e0'
    plt.rcParams['axes.labelcolor'] = '#e0e0e0'
    plt.rcParams['xtick.color'] = '#a0a0a0'
    plt.rcParams['ytick.color'] = '#a0a0a0'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Inter', 'DejaVu Sans', 'Arial']

def generate_all_plots(df_feat, cb_model=None, feature_cols=None):
    """
    Generates and saves all requested exploratory data analysis and model performance plots.
    """
    logging.info("Generating EDA and model explainability plots...")
    set_premium_style()
    
    plot_dir = 'outputs/plots'
    importance_dir = 'outputs/feature_importance'
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(importance_dir, exist_ok=True)
    
    # Pre-select cyberpunk palette
    palette = sns.color_palette("cool", as_cmap=False)
    accent_color = '#00f2fe' # Neon cyan
    secondary_color = '#4facfe' # Neon blue
    warn_color = '#ff0844' # Neon rose
    
    # 1. Demand Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df_feat['demand'], kde=True, color=accent_color, bins=50, alpha=0.7)
    plt.title('Traffic Demand Density Distribution', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Traffic Demand Target', fontsize=12)
    plt.ylabel('Frequency (Count)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/demand_distribution.png", dpi=150, facecolor='#121212')
    plt.close()
    
    # 2. Hourly Demand Curve
    plt.figure(figsize=(11, 6))
    hourly_demand = df_feat.groupby('hour')['demand'].mean().reset_index()
    plt.plot(hourly_demand['hour'], hourly_demand['demand'], marker='o', linewidth=3, color=accent_color, markersize=8)
    # Highlight Peak Hours (7-10 AM and 5-8 PM)
    for start, end in [(7, 10), (17, 20)]:
        plt.axvspan(start, end, color=warn_color, alpha=0.15, label='Peak Rush Hours' if start==7 else "")
    plt.title('Hourly Congestion Pattern Analysis', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Hour of Day (24h Scale)', fontsize=12)
    plt.ylabel('Average Traffic Demand', fontsize=12)
    plt.xticks(range(24))
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(frameon=True, facecolor='#1e1e1e')
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/hourly_demand.png", dpi=150, facecolor='#121212')
    plt.close()
    
    # 3. Weather vs Demand
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Weather', y='demand', data=df_feat, palette='cool', linewidth=1.5)
    plt.title('Weather Condition Impact on Traffic Demand', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Weather Type', fontsize=12)
    plt.ylabel('Traffic Demand Level', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/weather_impact.png", dpi=150, facecolor='#121212')
    plt.close()
    
    # 4. Temperature vs Demand
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Temperature', y='demand', data=df_feat.sample(min(5000, len(df_feat))), color=secondary_color, alpha=0.4, edgecolor='none')
    sns.regplot(x='Temperature', y='demand', data=df_feat, scatter=False, color=warn_color, line_kws={"linewidth": 3})
    plt.title('Ambient Temperature Correlation with Travel Demand', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Temperature (°C)', fontsize=12)
    plt.ylabel('Traffic Demand Level', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/temperature_impact.png", dpi=150, facecolor='#121212')
    plt.close()
    
    # 5. RoadType vs Demand
    plt.figure(figsize=(10, 6))
    road_demand = df_feat.groupby('RoadType')['demand'].mean().sort_values(ascending=False).reset_index()
    sns.barplot(x='demand', y='RoadType', data=road_demand, palette='cool')
    plt.title('Road Network Demand Capacity Utilization', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Average Traffic Demand', fontsize=12)
    plt.ylabel('Road Type Classification', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/road_impact.png", dpi=150, facecolor='#121212')
    plt.close()
    
    # 6. Peak Hour Analysis
    plt.figure(figsize=(8, 6))
    peak_demand = df_feat.groupby('is_peak_hour')['demand'].mean().reset_index()
    peak_demand['Label'] = peak_demand['is_peak_hour'].map({0: 'Off-Peak Hours', 1: 'Peak Rush Hours'})
    sns.barplot(x='Label', y='demand', data=peak_demand, palette=[secondary_color, warn_color])
    plt.title('Traffic Density: Peak vs Off-Peak Contrast', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('', fontsize=12)
    plt.ylabel('Average Traffic Demand', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/peak_hour_analysis.png", dpi=150, facecolor='#121212')
    plt.close()
    
    # 7. Spatial Geohash Demands
    plt.figure(figsize=(11, 8))
    # We sample for cleaner scatter representation
    sample_df = df_feat.sample(min(15000, len(df_feat)), random_state=42)
    scatter = plt.scatter(
        sample_df['longitude'], 
        sample_df['latitude'], 
        c=sample_df['demand'], 
        cmap='coolwarm', 
        alpha=0.6, 
        s=15, 
        edgecolor='none'
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label('Traffic Congestion Score (demand)', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    plt.title('UrbanFlow Smart City Geographic Traffic Congestion Heatmap', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Decoded Longitude Coordinates', fontsize=12)
    plt.ylabel('Decoded Latitude Coordinates', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/spatial_heatmap.png", dpi=150, facecolor='#121212')
    plt.close()
    
    # 8. Correlation Heatmap of Numeric Features
    plt.figure(figsize=(12, 10))
    numeric_cols = [
        'demand', 'hour', 'minute', 'total_minutes', 'latitude', 'longitude',
        'NumberofLanes', 'Temperature', 'weather_severity', 'lane_capacity', 
        'geo_freq', 'geo_mean_demand', 'geo_density', 'geo_rank', 
        'congestion_capacity_ratio', 'temp_weather_interaction'
    ]
    numeric_cols_present = [col for col in numeric_cols if col in df_feat.columns]
    corr_matrix = df_feat[numeric_cols_present].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, annot_kws={"size": 8}, alpha=0.95)
    plt.title('Core Spatial-Temporal Feature Correlations', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/correlation_heatmap.png", dpi=150, facecolor='#121212')
    plt.close()
    
    # 9. Model Feature Importance
    if cb_model is not None and feature_cols is not None:
        plt.figure(figsize=(10, 8))
        importances = cb_model.get_feature_importance()
        importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False).head(20)
        
        sns.barplot(x='Importance', y='Feature', data=importance_df, palette='cool')
        plt.title('Top 20 Predictive Features (CatBoost Importance)', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Feature Relative Importance (%)', fontsize=12)
        plt.ylabel('Engineered Parameter Name', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{importance_dir}/feature_importance_ranking.png", dpi=150, facecolor='#121212')
        plt.savefig(f"{plot_dir}/feature_importance.png", dpi=150, facecolor='#121212')
        plt.close()
        
    logging.info("Visualizations successfully saved to outputs/plots/ and outputs/feature_importance/")
