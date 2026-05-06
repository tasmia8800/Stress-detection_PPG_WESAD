# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import pandas as pd
import random
from preprocessing_tool.feature_extraction import *

np.random.seed(42)
random.seed(42)
import data_utils

WINDOW_IN_SECONDS = 120
NOISE = ['bp_time_ens']

label_dict = {'baseline': 1, 'stress': 2, 'amusement': 0}
int_to_label = {1: 'baseline', 2: 'stress', 0: 'amusement'}

def seperator_tri(df):
    grouped = df.groupby('label')
    try:
        baseline = grouped.get_group(1)
    except KeyError:
        baseline = pd.DataFrame()
        
    try:
        stress = grouped.get_group(2)
    except KeyError:
        stress = pd.DataFrame()
        
    try:
        amusement = grouped.get_group(3)
    except KeyError:
        amusement = pd.DataFrame()
        
    return baseline, stress, amusement



def get_samples(data, label, ma_usage):
    if data.empty:
        return pd.DataFrame()

    samples = []
    fs_bvp = data_utils.fs_dict['BVP']
    window_len = fs_bvp * WINDOW_IN_SECONDS
    sliding_window_len = int(fs_bvp * WINDOW_IN_SECONDS * 0.25)
    
    i = 0
    bvp_data = data['BVP'].tolist()
    
    while sliding_window_len * i <= len(bvp_data) - window_len:
        w = bvp_data[sliding_window_len * i : (sliding_window_len * i) + window_len]
        
        wstats = get_window_stats_27_features(ppg_seg=w, window_length=window_len, label=label, ensemble=ENSEMBLE, ma_usage=ma_usage)
        
        if wstats:
            samples.append(pd.DataFrame(wstats, index=[i]))
        i += 1
        
    return pd.concat(samples) if samples else pd.DataFrame()

def make_patient_data(subject_id, ma_usage):
    global savePath
    
    temp_ths = [1.0,2.0,1.8,1.5]
    clean_df = pd.read_csv('clean_signal_by_rate.csv', index_col=0)
    cycle = 15
    
    subject = data_utils.SubjectData(data_utils.MAIN_PATH, subject_id)
    e4_data_dict = subject.get_wrist_data()
    
    df = data_utils.extract_ppg_data(e4_data_dict, subject.labels, norm_type='std')
    
    
    df_BVP = df.BVP.tolist()
    fs = data_utils.fs_dict['BVP']
    bp_bvp = butter_bandpassfilter(df_BVP, 0.5, 10, fs, order=2)
    
    if BP:
        df['BVP'] = bp_bvp
        
    if FREQ:
         sec = 12
         N = fs * sec
         overlap = int(np.round(N * 0.02))
         overlap = overlap if overlap%2 ==0 else overlap+1
         
         signal_one_percent = int(len(df_BVP) * 0.01)
         cutoff = get_cutoff(df_BVP[:signal_one_percent], fs)
         freq_signal = compute_and_reconstruction_dft(df_BVP, fs, sec, overlap, cutoff)
         df['BVP'] = freq_signal
         
         
    if TIME:
        fwd = moving_average(bp_bvp, size=3)
        bwd = moving_average(bp_bvp[::-1], size=3)
        bp_bvp_smooth = np.mean(np.vstack((fwd,bwd[::-1])), axis=0)
        df['BVP'] = bp_bvp_smooth
        
        signal_01_percent = int(len(df_BVP) * 0.001)
        clean_idx = int(clean_df.loc[subject_id]['index'])
        clean_signal = df_BVP[clean_idx : clean_idx + signal_01_percent]
        
        ths = statistic_threshold(clean_signal, fs, temp_ths)
        _, _, time_signal_index = eliminate_noise_in_time(df['BVP'].to_numpy(), fs, ths, cycle)
        
        df = df.iloc[time_signal_index, :].reset_index(drop=True)

    baseline, stress, amusement = seperator_tri(df)
    
    baseline_samples = get_samples(baseline, 1, ma_usage) 
    stress_samples = get_samples(stress, 2, ma_usage)   
    amusement_samples = get_samples(amusement, 0, ma_usage)
    
    print("bas: ", len(baseline_samples))
    print("st: ", len(stress_samples))
    print("Am: ", len(amusement_samples))
    
    window_len_count = len(baseline_samples) + len(stress_samples) + len(amusement_samples)
    
    all_samples = pd.concat([baseline_samples, stress_samples, amusement_samples])
    
    if all_samples.empty:
        print(f"Warning: No samples for S{subject_id}")
        return window_len_count

    # One-hot encoding
    all_samples = pd.concat([all_samples.drop('label', axis=1), pd.get_dummies(all_samples['label'])], axis=1)
    
    all_samples.to_csv(f'{savePath}{subject_feature_path}/S{subject_id}_feats_4.csv')
    return window_len_count

def combine_files(subjects):
    df_list = []
    for s in subjects:
        df = pd.read_csv(f'{savePath}{subject_feature_path}/S{s}_feats_4.csv', index_col=0)
        df['subject'] = s
        df_list.append(df)

    df = pd.concat(df_list)
    
    
    try:
        df['label'] = (df['0'].astype(str) + df['1'].astype(str) + df['2'].astype(str)).apply(lambda x: x.index('1'))
        df.drop(['0', '1', '2'], axis=1, inplace=True)
    except KeyError as e:
        print(f"Warning: Key error in combining {e}. Columns found: {df.columns}")
        if '0.0' in df.columns:
             df['label'] = (df['0.0'].astype(str) + df['1.0'].astype(str) + df['2.0'].astype(str)).apply(lambda x: x.index('1'))
             df.drop(['0.0', '1.0', '2.0'], axis=1, inplace=True)

    df.reset_index(drop=True, inplace=True)
    df.to_csv(savePath + merged_path)

    counts = df['label'].value_counts()
    print('Number of samples per class:')
    for label, number in zip(counts.index, counts.values):
        print(f'{int_to_label.get(label, label)}: {number}')

savePath = '27_features_ppg_test_3/LMM/'
subject_ids = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]

if os.environ.get('WESAD_FAST') == 'True':
    print("[INFO] FAST MODE: Processing subset (Run 1: S2, S3, S4)")
    subject_ids = [2, 3, 4]
BP, FREQ, TIME, ENSEMBLE = False, False, False, False

if not os.path.exists(savePath):
    os.makedirs(savePath)

for n in NOISE:
    flags = n.split('_')
    BP = 'bp' in flags
    TIME = 'time' in flags
    ENSEMBLE = 'ens' in flags

    subject_feature_path = '/subject_feature_' + n + str(WINDOW_IN_SECONDS)
    merged_path = '/data_merged_' + n + str(WINDOW_IN_SECONDS) + '.csv'
    
    if not os.path.exists(savePath + subject_feature_path):
        os.makedirs(savePath + subject_feature_path)
        
    total_window_len = 0
    for patient in subject_ids:
        print(f'Processing data for S{patient}...')
        total_window_len += make_patient_data(patient, BP)

    combine_files(subject_ids)
    print('total_Window_len: ', total_window_len)
    print('Processing complete.', n)

