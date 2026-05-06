import os
import pickle
import numpy as np
import pandas as pd
from preprocessing_tool.feature_extraction import *

fs_dict = {'ACC': 32, 'BVP': 64, 'EDA': 4, 'TEMP': 4, 'label': 700, 'Resp': 700}
MAIN_PATH = 'WESAD/'

class SubjectData:
    def __init__(self, main_path, subject_number):
        self.name = f'S{subject_number}'
        self.subject_keys = ['signal', 'label', 'subject']
        self.signal_keys = ['chest', 'wrist']
        self.chest_keys = ['ACC', 'ECG', 'EMG', 'EDA', 'Temp', 'Resp']
        self.wrist_keys = ['ACC', 'BVP', 'EDA', 'TEMP']
        
        pkl_path = os.path.join(main_path, self.name, f'{self.name}.pkl')
        with open(pkl_path, 'rb') as file:
            self.data = pickle.load(file, encoding='latin1')
        self.labels = self.data['label']



    def get_wrist_data(self):
        data = self.data['signal']['wrist']
        data.update({'Resp': self.data['signal']['chest']['Resp']})
        return data

    def get_chest_data(self):
        return self.data['signal']['chest']

    def extract_features(self):  # only wrist
        results = {
            key: get_statistics(self.get_wrist_data()[key].flatten(), self.labels, key)
            for key in self.wrist_keys
        }
        return results


def extract_ppg_data(e4_data_dict, labels, norm_type=None):
    df = pd.DataFrame(e4_data_dict['BVP'], columns=['BVP'])
    label_df = pd.DataFrame(labels, columns=['label'])
    
    df.index = [(1 / fs_dict['BVP']) * i for i in range(len(df))]
    label_df.index = [(1 / fs_dict['label']) * i for i in range(len(label_df))]

    df.index = pd.to_datetime(df.index, unit='s')
    label_df.index = pd.to_datetime(label_df.index, unit='s')

    df = df.join(label_df, how='outer')
    
    df['label'] = df['label'].fillna(method='bfill')
    
    df.reset_index(drop=True, inplace=True)
    
    if norm_type == 'std':
        df['BVP'] = (df['BVP'] - df['BVP'].mean()) / df['BVP'].std()
            
    elif norm_type == 'minmax':
        df = (df - df.min()) / (df.max() - df.min())

    df = df.dropna(axis=0) # nan인 행 제거
    
    return df

def preprocess_signal(df, subject_id, clean_df, bp_bvp, fs_bvp, config):
    """
    Applies signal processing based on config flags (BP, TIME, FREQ).
    Returns processed df (reduced if TIME logic applies).
    """
    
    if config['BP']:
        df['BVP'] = bp_bvp
        
    if config.get('FREQ', False):
        sec = config['sec']
        overlap = config['overlap']
        df_BVP = df['BVP'].tolist() # Use current state of BVP
        
        signal_one_percent = int(len(df_BVP) * 0.01)
        print(signal_one_percent)
        cutoff = get_cutoff(df_BVP[:signal_one_percent], fs_bvp)
        freq_signal = compute_and_reconstruction_dft(df_BVP, fs_bvp, sec, overlap, cutoff)
        df['BVP'] = freq_signal

    if config['TIME']:
        fwd = moving_average(bp_bvp, size=3)
        bwd = moving_average(bp_bvp[::-1], size=3)
        bp_bvp_mean = np.mean(np.vstack((fwd, bwd[::-1])), axis=0)
        df['BVP'] = bp_bvp_mean
        
        signal_01_percent = int(len(bp_bvp) * 0.001)
        clean_idx = int(clean_df.loc[subject_id]['index'])
        
        
        
        pass # The logic inside make_patient_data for TIME is quite coupled with local variables.
             
    return df
