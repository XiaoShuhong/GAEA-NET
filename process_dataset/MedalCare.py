import os
from tqdm import tqdm 
import pandas as pd
from tools import *
import pickle

def get_MedalCare(in_path,sampling_rate,num_heartbeats,shift_N,len_seq,store_aux,out_path):
    label = ['avblock','fam','iab','lae','lbbb','mi/LAD_0.3','mi/LAD_1.0','mi/LCX_0.3_ant','mi/LCX_0.3_post','mi/LCX_1.0_ant','mi/LCX_1.0_post','mi/RCA_0.3','mi/RCA_1.0','rbbb','sinus']
    ecg_list, sub_list, class_list, aux_list = [], [], [], [] 
    idx = 0
    for split in ['train', 'test']:
        for l in label:
            base_dir = os.path.join(in_path,l, split)
            for p in tqdm(os.listdir(base_dir)):
                p_path = os.path.join(base_dir, p)
                for run in os.listdir(p_path):
                    if "filtered" in run:
                        r_path = os.path.join(p_path, run)
                        ecg = pd.read_csv(r_path,header=None).T[0]
                        resample_ecg, r_peaks_ecg = resample_signal_with_rpeaks(ecg, original_fs=500, new_fs=sampling_rate)
                        segments_list_ecg , subjects_list, min_max = segment_ecg_by_rpeaks(resample_ecg, r_peaks_ecg , fs=sampling_rate, window_heartbeats=num_heartbeats, shift_N=shift_N, subject=int(idx))
                        if "mi" in l:
                            cla_list = ['mi'] * len(subjects_list)
                        else:
                            cla_list = [l] * len(subjects_list)
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
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        if store_aux:
            data_pair = (ecg_list, sub_list, class_list, aux_list)
            with open(os.path.join(out_path, f'MedalCare_{split}_aux.pkl'), "wb") as output_file:
                pickle.dump(data_pair, output_file)  
        else:
            data_pair = (ecg_list, sub_list, class_list)
            with open(os.path.join(out_path, f'MedalCare_{split}.pkl'), "wb") as output_file:
                pickle.dump(data_pair, output_file)   
    return 

