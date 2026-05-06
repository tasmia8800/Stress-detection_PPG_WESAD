import pandas as pd
import os

def extract_s2_snapshot():
    csv_path = '27_features_ppg_test/bi/ens/3/subject_feature_bp_time_ens120/S2_feats_4.csv'
    output_path = 'result/intermediate/feature_snapshot_PROPOSED.csv'
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path, index_col=0)
    
    print("Columns:", df.columns.tolist())
    
    if 'label' not in df.columns:
        cols = df.columns
        if '0' in cols and '1' in cols:
             df['label'] = df['1'] 
             pass
        elif '0.0' in cols and '1.0' in cols:
             df['label'] = df['1.0']
             
    print(f"Saving to {output_path}...")
    df.to_csv(output_path)
    print("Done.")

if __name__ == "__main__":
    extract_s2_snapshot()
