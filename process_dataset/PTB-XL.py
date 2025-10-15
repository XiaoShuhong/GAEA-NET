import os
from tqdm import tqdm 
import pandas as pd
from tools import *
import pickle
import wfdb

def get_PTB_xl(in_path, sampling_rate, num_heartbeats, shift_N, len_seq, store_aux, out_path):
    
    def get_ecg_path(ecg_id):
        folder_number = (ecg_id // 1000) * 1000
        path = f"./ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/records500/{folder_number:05d}/{ecg_id:05d}_hr"
        return path
    for subset in ['Train_sclc_X', 'Patient_Selected_291_sclc']:
        ecg_list, sub_list, class_list, aux_list = [], [], [], [] 
        ref_info_path = os.path.join(in_path,f'{subset}.csv')
        ref_info = pd.read_csv(ref_info_path) 
        for row in tqdm(ref_info.itertuples(index=True, name='Pandas')):
            ecg_id = row.ecg_id
            superclass = row.detail_superclass
            patient_id = row.patient_id
            ecg_signal = wfdb.rdsamp(get_ecg_path(ecg_id))[0][:,0] 
            butterworth_ecg = butterworth_lowpass_filter(ecg_signal, fs=500, passband=50, stopband=60, 
                                pass_ripple=1.0, stop_atten=2.5)
            nlm_ecg = butterworth_ecg
            resampl_ecg, r_peaks_ecg = resample_signal_with_rpeaks(nlm_ecg, original_fs=500, new_fs=sampling_rate)
            segments_list_ecg , subjects_list, min_max = segment_ecg_by_rpeaks(resampl_ecg, r_peaks_ecg , fs=sampling_rate, window_heartbeats=num_heartbeats, shift_N=shift_N, subject=int(patient_id))
            cla_list = [superclass] * len(subjects_list)
            segments_list_ecg, len_info = to_fixed_length(segments_list_ecg, L=len_seq)

            for e in len_info:
                aux_info = dict()
                aux_info['original_fs'] = 500
                aux_info['new_fs'] = 300
                aux_info['min_max'] = min_max
                aux_info['len_info'] = e
                aux_list.append(aux_info)
            ecg_list.extend(segments_list_ecg)
            sub_list.extend(subjects_list)
            class_list.extend(cla_list)

        if not os.path.exists(out_path):
            os.makedirs(out_path)
        if store_aux:
            data_pair = (ecg_list, sub_list, class_list, aux_list)
            with open(os.path.join(out_path, f'PTB-XL_{subset}_aux.pkl'), "wb") as output_file:
                pickle.dump(data_pair, output_file)   
        else:
            data_pair = (ecg_list, sub_list, class_list)
            with open(os.path.join(out_path, f'PTB-XL_{subset}.pkl'), "wb") as output_file:
                pickle.dump(data_pair, output_file)   
        return