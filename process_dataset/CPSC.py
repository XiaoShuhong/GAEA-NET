import os
import pickle
from tools import *
from tqdm import tqdm 
import pandas as pd
import wfdb



def get_CPSC(in_path,sampling_rate,num_heartbeats,shift_N,len_seq,store_aux,out_path):
    ref = os.path.join(in_path,'REFERENCE.csv')
    df = pd.read_csv(ref, header=None, names=["record", "label"])
    ecg_list, sub_list, class_list, aux_list = [], [], [], [] 
    for index, row in tqdm(df.iterrows()):
        record_id = row['record']
        label = row['label']
        if label != '~':
            data_path = f'{in_path}/{record_id}'
            ecg_signal = wfdb.rdsamp(data_path)[0].reshape(-1)
            butterworth_ecg = butterworth_lowpass_filter(ecg_signal, fs=300, passband=50, stopband=60, 
                              pass_ripple=1.0, stop_atten=2.5)
            nlm_ecg = butterworth_ecg
            resampl_ecg, r_peaks_ecg = resample_signal_with_rpeaks(nlm_ecg, original_fs=300, new_fs=sampling_rate)
            segments_list_ecg , subjects_list, min_max = segment_ecg_by_rpeaks(resampl_ecg, r_peaks_ecg, fs=sampling_rate, window_heartbeats=num_heartbeats, shift_N=shift_N, subject=index)
            cla_list = [label] * len(subjects_list)
            segments_list_ecg, len_info = to_fixed_length(segments_list_ecg, L=len_seq)   
            for e in len_info:
                aux_info = dict()
                aux_info['original_fs'] = 300
                aux_info['new_fs'] = sampling_rate
                aux_info['min_max'] = min_max
                aux_info['len_info'] = e
                aux_list.append(aux_info)   
            ecg_list.extend(segments_list_ecg)
            sub_list.extend(subjects_list)
            class_list.extend(cla_list)
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
        with open(os.path.join(out_path, f'CPSC_aux.pkl'), "wb") as output_file:
            pickle.dump(data_pair, output_file)   
    else:
        data_pair = (ecg_list, sub_list, class_list, split_list)
        with open(os.path.join(out_path, f'CPSC.pkl'), "wb") as output_file:
            pickle.dump(data_pair, output_file)   
    return 


 