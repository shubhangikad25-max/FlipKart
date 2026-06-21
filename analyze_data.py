import pandas as pd
import numpy as np

def analyze_data():
    try:
        train = pd.read_csv('train.csv')
        print("--- train.csv target (demand) stats ---")
        if 'demand' in train.columns:
            print(f"min: {train['demand'].min()}")
            print(f"max: {train['demand'].max()}")
            print(f"mean: {train['demand'].mean()}")
            print(f"median: {train['demand'].median()}")
            print(f"std: {train['demand'].std()}")
        else:
            print("No 'demand' column in train.csv")
    except Exception as e:
        print(f"Error reading train.csv: {e}")

    try:
        sub = pd.read_csv('submission.csv')
        print("\n--- submission.csv target (demand) stats ---")
        if 'demand' in sub.columns:
            print(f"min: {sub['demand'].min()}")
            print(f"max: {sub['demand'].max()}")
            print(f"mean: {sub['demand'].mean()}")
            print(f"median: {sub['demand'].median()}")
            print(f"std: {sub['demand'].std()}")
        else:
            print("No 'demand' column in submission.csv")
            
        print("\n--- submission.csv verification ---")
        print(f"rows: {len(sub)}")
        print(f"columns: {len(sub.columns)}")
        print(f"column names: {list(sub.columns)}")
        
    except Exception as e:
        print(f"Error reading submission.csv: {e}")

if __name__ == '__main__':
    analyze_data()
