import pandas as pd
import os

def merge_task(task_type, save_path, feature_path, merged_filename, subjects, num_classes):
    print(f"Merging {task_type}...")
    df_list = []
    for s in subjects:
        file_path = f'{save_path}{feature_path}/S{s}_feats_4.csv'
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, index_col=0)
            df['subject'] = s
            df_list.append(df)
        else:
            print(f"Warning: Missing data for S{s} at {file_path}")

    if not df_list:
        print(f"No data found for {task_type}")
        return

    df = pd.concat(df_list)

    label_cols = [str(i) for i in range(num_classes)]
    if all(col in df.columns for col in label_cols):
        df['label'] = df[label_cols].astype(str).apply(lambda x: "".join(x).find('1'), axis=1)
        df.drop(label_cols, axis=1, inplace=True)
    elif 'label' not in df.columns:
         if '0' in df.columns and '1' in df.columns:
             df['label'] = df['1'].astype(int)
             df.drop(['0', '1'], axis=1, inplace=True)

    df.reset_index(drop=True, inplace=True)
    out_path = os.path.join(save_path, merged_filename)
    df.to_csv(out_path)

    print(f"Merged {task_type} saved to: {out_path}")
    print(f"Shape: {df.shape}")
    print(f"Label distribution:")
    print(df['label'].value_counts())
    print("-" * 30)

subjects = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]

merge_task("Binary", 
           "27_features_ppg_test/bi/ens/3", 
           "/subject_feature_bp_time_ens120", 
           "data_merged_bp_time_ens.csv", 
           subjects, 2)

merge_task("Tri-class", 
           "27_features_ppg_test_3/LMM", 
           "/subject_feature_bp_time_ens120", 
           "data_merged_bp_time_ens120.csv", 
           subjects, 3)

merge_task("Quad-class", 
           "27_features_ppg_test_4/LMM", 
           "/subject_feature_bp_time_ens120", 
           "data_merged_bp_time_ens120.csv", 
           subjects, 4)
