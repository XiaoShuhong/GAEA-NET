import glob
import os
from tqdm import tqdm 
import pandas as pd
from tools import *
import pickle
import wfdb

def get_PTB(in_path,sampling_rate, num_heartbeats, shift_N,len_seq,store_aux,out_path):  

    ecg_list, sub_list, class_list, aux_list = [], [], [], [] 
    mapping ={
        'Healthy control': 'NORM',
        'Myocardial infarction' :'MI',
        'Dysrhythmia': 'STTC',
        'Bundle branch block' : 'CD',
        'Cardiomyopathy':'CD',
        'Hypertrophy': 'HYP'
    }
    for p in tqdm(os.listdir(in_path)):
        p_path = os.path.join(in_path, p)
        hea_files = glob.glob(p_path + '/*.hea') 
        for hea in hea_files:
            ecg_path = os.path.join(p_path, hea.split('.hea')[0])
            ecg_signal = wfdb.rdsamp(ecg_path)
            ecg = ecg_signal[0][:,0]
            fs = ecg_signal[1]['fs']
            label = ecg_signal[1]['comments'][4].split('Reason for admission: ')[-1]
            if label in mapping.keys():
                supperclass = mapping[label]
                patient_id = int(p.split('/')[0].split('patient')[-1])
                butterworth_ecg = butterworth_lowpass_filter(ecg, fs=300, passband=50, stopband=60, 
                                    pass_ripple=1.0, stop_atten=2.5)
                resampl_ecg, r_peaks_ecg = resample_signal_with_rpeaks(butterworth_ecg, original_fs=300, new_fs=sampling_rate)
                segments_list_ecg, subjects_list, min_max = segment_ecg_by_rpeaks(resampl_ecg, r_peaks_ecg , fs=sampling_rate, window_heartbeats=num_heartbeats, shift_N=shift_N, subject=int(patient_id))
                
                cla_list = [supperclass] * len(subjects_list)
                segments_list_ecg, len_info = to_fixed_length(segments_list_ecg, L=len_seq)
                for e in len_info:
                    aux_info = dict()
                    aux_info['original_fs'] = fs
                    aux_info['new_fs'] = 300
                    aux_info['min_max'] = min_max
                    aux_info['len_info'] = e
                    aux_list.append(aux_info)

                ecg_list.extend(segments_list_ecg)
                sub_list.extend(subjects_list)
                class_list.extend(cla_list)
    if store_aux:
        data_pair = (ecg_list, sub_list, class_list, aux_list)
        with open(os.path.join(out_path, f'PTB_aux.pkl'), "wb") as output_file:
            pickle.dump(data_pair, output_file)   
    else:
        data_pair = (ecg_list, sub_list, class_list)
        with open(os.path.join(out_path, f'PTB.pkl'), "wb") as output_file:
            pickle.dump(data_pair, output_file)
    return