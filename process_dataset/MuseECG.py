import os
from tqdm import tqdm 
import pandas as pd
from tools import *
import pickle

def get_Muse(in_path,sampling_rate, num_heartbeats, shift_N, len_seq, store_aux,out_path):

    mapping = {
        'AFIB': 'AFIB',
        'AF': 'AFIB',
        'SVT': 'GSVT',
        'AT': 'GSVT',
        'SAAWR': 'GSVT',
        'ST': 'GSVT',
        'AVNRT': 'GSVT',
        'AVRT': 'GSVT',
        'SB': 'SB',
        'SR': 'SR',
        'SA': 'SR'
    }
    df = pd.read_excel(f'{in_path}/Diagnostics.xlsx')
    filename = df['FileName']
    rhythm = df['Rhythm']
    idx = 0
    ecg_list, sub_list, class_list, aux_list = [], [], [], [] 
    for f, r in tqdm(zip(filename, rhythm)):
        path = os.path.join(f'{in_path}/ECGDataDenoised',f'{f}.csv') 
        data = pd.read_csv(path)
        data_array = data.values  
        data_array = np.array(data_array, dtype=np.float32)
        lead1_ecg = data_array[:,0]
        if lead1_ecg.max() != lead1_ecg.min():
            superclass = mapping[r]
            resample_ecg, r_peaks_ecg = resample_signal_with_rpeaks(lead1_ecg, original_fs=500, new_fs=sampling_rate)
            segments_list_ecg , subjects_list, min_max = segment_ecg_by_rpeaks(resample_ecg, r_peaks_ecg , fs=sampling_rate, window_heartbeats=num_heartbeats, shift_N=shift_N, subject=int(idx))
            cla_list = [superclass] * len(subjects_list)
            segments_list_ecg, len_info = to_fixed_length(segments_list_ecg, L=len_seq)  
            for e in len_info:
                aux_info = dict()
                aux_info['original_fs'] = 500
                aux_info['new_fs'] = sampling_rate
                aux_info['min_max'] = min_max
                aux_info['len_info'] = e
                aux_list.append(aux_info)
            ecg_list.extend(segments_list_ecg)
            sub_list.extend(subjects_list)
            class_list.extend(cla_list)
            idx+=1
    from sklearn.model_selection import train_test_split

    def split_data(class_list, sub_list):
       
        class_list = np.array(class_list)
        sub_list = np.array(sub_list)
        unique_subjects = np.unique(sub_list)
        split_list = np.empty(len(class_list), dtype=object)
        train_subjects, test_subjects = train_test_split(unique_subjects, test_size=0.2, random_state=42)
        split_list = np.empty(len(class_list), dtype=object) 
        for i, subject in enumerate(sub_list):
            if subject in train_subjects:
                split_list[i] = 'train'
            else:
                split_list[i] = 'test'
        
        return split_list
    split_list = split_data(class_list,sub_list)
    if not os.path.exists(out_path):
            os.makedirs(out_path)
    if store_aux:
        data_pair = (ecg_list, sub_list, class_list, split_list, aux_list)
        with open(os.path.join(out_path, f'MuseECG_aux.pkl'), "wb") as output_file:
            pickle.dump(data_pair, output_file)   
        
    else:
        data_pair = (ecg_list, sub_list, class_list, split_list)
        with open(os.path.join(out_path, f'MuseECG.pkl'), "wb") as output_file:
            pickle.dump(data_pair, output_file)   
    return 