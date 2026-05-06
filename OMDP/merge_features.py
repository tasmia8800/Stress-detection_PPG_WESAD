import pandas as pd
import os

savePath = '27_features_ppg_test/bi/ens/3'
subject_feature_path = '/subject_feature_bp_time_ens120'
merged_path = '/data_merged_bp_time_ens.csv'
subjects = [2, 3, 4]

df_list = []
for s in subjects:
    df = pd.read_csv(f'{savePath}{subject_feature_path}/S{s}_feats_4.csv', index_col=0)
    df['subject'] = s
    df_list.append(df)

df = pd.concat(df_list)

if '0' in df.columns and '1' in df.columns:
    df['label'] = df['1'].astype(int)  # Binary: 0 or 1
    df.drop(['0', '1'], axis=1, inplace=True)

df.reset_index(drop=True, inplace=True)

df.to_csv(savePath + merged_path)

print(f"Merged data saved to: {savePath + merged_path}")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nLabel distribution:")
print(df['label'].value_counts())
