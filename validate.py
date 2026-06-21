import os
import pandas as pd

# Check models
models = ['catboost_model.pkl', 'lightgbm_model.pkl', 'xgboost_model.pkl',
          'feature_stats.pkl', 'category_maps.pkl', 'train_medians.pkl']
print('=== MODEL FILES ===')
for m in models:
    path = f'models/{m}'
    exists = os.path.exists(path)
    size = os.path.getsize(path) // 1024 if exists else 0
    status = 'OK' if exists else 'MISSING'
    print(f'  {m}: {status} ({size} KB)')

# Check submission
print('\n=== SUBMISSION VALIDATION ===')
df = pd.read_csv('outputs/submission.csv')
row_pass = 'PASS' if len(df) == 41778 else 'FAIL'
col_pass = 'PASS' if list(df.columns) == ['Index', 'demand'] else 'FAIL'
nan_pass = 'PASS' if df.isna().sum().sum() == 0 else 'FAIL'
print(f'  Rows: {len(df)} (expected 41778) - {row_pass}')
print(f'  Columns: {list(df.columns)} - {col_pass}')
print(f'  NaN values: {df.isna().sum().sum()} - {nan_pass}')
print(f'  Demand range: [{df.demand.min():.4f}, {df.demand.max():.4f}]')

# Check plots
print('\n=== GENERATED PLOTS ===')
plots = os.listdir('outputs/plots')
for p in sorted(plots):
    print(f'  {p}')

print('\n=== ALL CHECKS COMPLETE ===')
