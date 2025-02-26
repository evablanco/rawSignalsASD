import pandas as pd
from torch.utils.data import DataLoader, Dataset
import os
import torch
import numpy as np

def load_data_to_dict(path):
    df = pd.read_csv(path, dtype={'SubjectID': str, 'SessionID': str})
    #df = df[['SubjectID', 'SessionID', 'Timestamp','EDA', 'ACC_X', 'ACC_Y', 'ACC_Z', 'BVP', 'AGG', 'ED', 'SIB', 'Condition']]
    df = df[['SubjectID', 'SessionID', 'Timestamp', 'EDA', 'ACC_X', 'ACC_Y', 'ACC_Z', 'BVP', 'Condition']]
    data_dict = {}
    for (subject_id, session_id), group_df in df.groupby(['SubjectID', 'SessionID']):
        #group_df = group_df.reset_index(drop=True)
        group_df['Timestamp'] = pd.to_datetime(group_df['Timestamp'], unit='ms')
        group_df = group_df.set_index('Timestamp')
        if subject_id not in data_dict:
            data_dict[subject_id] = {}
        #data_dict[subject_id][session_id] = group_df[['EDA', 'ACC_X', 'ACC_Y', 'ACC_Z', 'BVP', 'AGG', 'ED', 'SIB', 'Condition']]
        data_dict[subject_id][session_id] = group_df[['EDA', 'ACC_X', 'ACC_Y', 'ACC_Z', 'BVP', 'Condition']]
    return data_dict


def load_data_to_dict_multi(path):
    df = pd.read_csv(path, dtype={'SubjectID': str, 'SessionID': str})
    df = df[['SubjectID', 'SessionID', 'Timestamp','EDA', 'ACC_X', 'ACC_Y', 'ACC_Z', 'BVP', 'AGG', 'ED', 'SIB', 'Condition']]
    data_dict = {}
    for (subject_id, session_id), group_df in df.groupby(['SubjectID', 'SessionID']):
        group_df['Timestamp'] = pd.to_datetime(group_df['Timestamp'], unit='ms')
        group_df = group_df.set_index('Timestamp')
        if subject_id not in data_dict:
            data_dict[subject_id] = {}
        data_dict[subject_id][session_id] = group_df[['EDA', 'ACC_X', 'ACC_Y', 'ACC_Z', 'BVP', 'AGG', 'ED', 'SIB']]
    return data_dict


def load_data_to_dict_ACCNorm(path):
    df = pd.read_csv(path, dtype={'SubjectID': str, 'SessionID': str})
    #df = df[['SubjectID', 'SessionID', 'Timestamp','EDA', 'ACC_X', 'ACC_Y', 'ACC_Z', 'BVP', 'AGG', 'ED', 'SIB', 'Condition']]
    df = df[['SubjectID', 'SessionID', 'Timestamp', 'EDA', 'ACC_X', 'ACC_Y', 'ACC_Z', 'BVP', 'Condition']]
    data_dict = {}
    for (subject_id, session_id), group_df in df.groupby(['SubjectID', 'SessionID']):
        #group_df = group_df.reset_index(drop=True)
        group_df['Timestamp'] = pd.to_datetime(group_df['Timestamp'], unit='ms')
        group_df = group_df.set_index('Timestamp')
        if subject_id not in data_dict:
            data_dict[subject_id] = {}
        #data_dict[subject_id][session_id] = group_df[['EDA', 'ACC_X', 'ACC_Y', 'ACC_Z', 'BVP', 'AGG', 'ED', 'SIB', 'Condition']]
        acc_data = (group_df[['ACC_X', 'ACC_Y', 'ACC_Z']]).to_numpy().astype(float)
        group_df['ACCNorm'] = np.linalg.norm(acc_data, axis=1)
        data_dict[subject_id][session_id] = group_df[['EDA', 'BVP', 'ACCNorm', 'Condition']]
    return data_dict


def load_data_to_dict_noACC(path):
    df = pd.read_csv(path, dtype={'SubjectID': str, 'SessionID': str})
    #df = df[['SubjectID', 'SessionID', 'Timestamp','EDA', 'ACC_X', 'ACC_Y', 'ACC_Z', 'BVP', 'AGG', 'ED', 'SIB', 'Condition']]
    df = df[['SubjectID', 'SessionID', 'Timestamp', 'EDA', 'BVP', 'Condition']]
    data_dict = {}
    for (subject_id, session_id), group_df in df.groupby(['SubjectID', 'SessionID']):
        #group_df = group_df.reset_index(drop=True)
        group_df['Timestamp'] = pd.to_datetime(group_df['Timestamp'], unit='ms')
        group_df = group_df.set_index('Timestamp')
        if subject_id not in data_dict:
            data_dict[subject_id] = {}
        data_dict[subject_id][session_id] = group_df[['EDA', 'BVP', 'Condition']]
    return data_dict


def load_data_to_dict_noEDA(path):
    df = pd.read_csv(path, dtype={'SubjectID': str, 'SessionID': str})
    #df = df[['SubjectID', 'SessionID', 'Timestamp','EDA', 'ACC_X', 'ACC_Y', 'ACC_Z', 'BVP', 'AGG', 'ED', 'SIB', 'Condition']]
    df = df[['SubjectID', 'SessionID', 'Timestamp', 'ACC_X', 'ACC_Y', 'ACC_Z', 'BVP', 'Condition']]
    data_dict = {}
    for (subject_id, session_id), group_df in df.groupby(['SubjectID', 'SessionID']):
        #group_df = group_df.reset_index(drop=True)
        group_df['Timestamp'] = pd.to_datetime(group_df['Timestamp'], unit='ms')
        group_df = group_df.set_index('Timestamp')
        if subject_id not in data_dict:
            data_dict[subject_id] = {}
        #acc_data = (group_df[['ACC_X', 'ACC_Y', 'ACC_Z']]).to_numpy().astype(float)
        #group_df['ACCNorm'] = np.linalg.norm(acc_data, axis=1)
        data_dict[subject_id][session_id] = group_df[['BVP', 'ACC_X', 'ACC_Y', 'ACC_Z', 'Condition']]
    return data_dict


def load_data_to_dict_ACC(path):
    df = pd.read_csv(path, dtype={'SubjectID': str, 'SessionID': str})
    #df = df[['SubjectID', 'SessionID', 'Timestamp','EDA', 'ACC_X', 'ACC_Y', 'ACC_Z', 'BVP', 'AGG', 'ED', 'SIB', 'Condition']]
    df = df[['SubjectID', 'SessionID', 'Timestamp', 'ACC_X', 'ACC_Y', 'ACC_Z', 'Condition']]
    data_dict = {}
    for (subject_id, session_id), group_df in df.groupby(['SubjectID', 'SessionID']):
        #group_df = group_df.reset_index(drop=True)
        group_df['Timestamp'] = pd.to_datetime(group_df['Timestamp'], unit='ms')
        group_df = group_df.set_index('Timestamp')
        if subject_id not in data_dict:
            data_dict[subject_id] = {}
        #acc_data = (group_df[['ACC_X', 'ACC_Y', 'ACC_Z']]).to_numpy().astype(float)
        #group_df['ACCNorm'] = np.linalg.norm(acc_data, axis=1)
        data_dict[subject_id][session_id] = group_df[['ACC_X', 'ACC_Y', 'ACC_Z', 'Condition']]
    return data_dict



def load_data_to_dict_BVP(path):
    df = pd.read_csv(path, dtype={'SubjectID': str, 'SessionID': str})
    #df = df[['SubjectID', 'SessionID', 'Timestamp','EDA', 'ACC_X', 'ACC_Y', 'ACC_Z', 'BVP', 'AGG', 'ED', 'SIB', 'Condition']]
    df = df[['SubjectID', 'SessionID', 'Timestamp', 'BVP', 'Condition']]
    data_dict = {}
    for (subject_id, session_id), group_df in df.groupby(['SubjectID', 'SessionID']):
        #group_df = group_df.reset_index(drop=True)
        group_df['Timestamp'] = pd.to_datetime(group_df['Timestamp'], unit='ms')
        group_df = group_df.set_index('Timestamp')
        if subject_id not in data_dict:
            data_dict[subject_id] = {}
        #acc_data = (group_df[['ACC_X', 'ACC_Y', 'ACC_Z']]).to_numpy().astype(float)
        #group_df['ACCNorm'] = np.linalg.norm(acc_data, axis=1)
        data_dict[subject_id][session_id] = group_df[['BVP', 'Condition']]
    return data_dict


def load_data_to_dict_EDA(path):
    df = pd.read_csv(path, dtype={'SubjectID': str, 'SessionID': str})
    #df = df[['SubjectID', 'SessionID', 'Timestamp','EDA', 'ACC_X', 'ACC_Y', 'ACC_Z', 'BVP', 'AGG', 'ED', 'SIB', 'Condition']]
    df = df[['SubjectID', 'SessionID', 'Timestamp', 'EDA', 'Condition']]
    data_dict = {}
    for (subject_id, session_id), group_df in df.groupby(['SubjectID', 'SessionID']):
        #group_df = group_df.reset_index(drop=True)
        group_df['Timestamp'] = pd.to_datetime(group_df['Timestamp'], unit='ms')
        group_df = group_df.set_index('Timestamp')
        if subject_id not in data_dict:
            data_dict[subject_id] = {}
        #acc_data = (group_df[['ACC_X', 'ACC_Y', 'ACC_Z']]).to_numpy().astype(float)
        #group_df['ACCNorm'] = np.linalg.norm(acc_data, axis=1)
        data_dict[subject_id][session_id] = group_df[['EDA', 'Condition']]
    return data_dict


def load_data_to_dict_noBVP(path):
    df = pd.read_csv(path, dtype={'SubjectID': str, 'SessionID': str})
    df = df[['SubjectID', 'SessionID', 'Timestamp', 'EDA', 'ACC_X', 'ACC_Y', 'ACC_Z', 'Condition']]
    data_dict = {}
    for (subject_id, session_id), group_df in df.groupby(['SubjectID', 'SessionID']):
        #group_df = group_df.reset_index(drop=True)
        group_df['Timestamp'] = pd.to_datetime(group_df['Timestamp'], unit='ms')
        group_df = group_df.set_index('Timestamp')
        if subject_id not in data_dict:
            data_dict[subject_id] = {}
        #acc_data = (group_df[['ACC_X', 'ACC_Y', 'ACC_Z']]).to_numpy().astype(float)
        #group_df['ACCNorm'] = np.linalg.norm(acc_data, axis=1)
        data_dict[subject_id][session_id] = group_df[['EDA', 'ACC_X', 'ACC_Y', 'ACC_Z', 'Condition']] # 'ACCNorm',
    return data_dict



def load_orig_data_dict(base_path = './dataset'):
    subject_data = {}
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                parts = file.split('_')
                subject_id = parts[0]  # Identificación del sujeto (xxxx.xx)
                session_id = parts[1].split('.')[0]  # Identificación de la sesión (xxxx.xx_xx)
                signal_type = parts[2].split('.')[0]  # Tipo de señal (ACC, BVP, EDA)
                data = pd.read_csv(file_path)
                if subject_id not in subject_data:
                    subject_data[subject_id] = {}
                if session_id not in subject_data[subject_id]:
                    subject_data[subject_id][session_id] = {}
                subject_data[subject_id][session_id][signal_type] = data
    return subject_data


def get_features_from_dic_prevLabels_bins(data_dict, tp=60, tf=180, bin_size=15, freq=32):
    win_size = bin_size * freq
    label_win_size = tf * freq
    output_dict = {}
    for main_key, sub_dict in data_dict.items():
        if main_key not in output_dict:
            output_dict[main_key] = {}
        for sub_key, df in sub_dict.items():
            if sub_key not in output_dict[main_key]:
                output_dict[main_key][sub_key] = {}
            df_subject = data_dict[main_key][sub_key]
            #print(f"key: {main_key}-{sub_key}, num. values in session: {len(df_subject)}")
            counter = 0
            end_win_l = 0
            windows_list = []
            labels_list = []
            prev_labels_list = [0] # primer bin no tiene prev_label
            next_limit = counter * win_size + win_size + label_win_size
            while (next_limit < len(df_subject)):
                init_win_f = (counter * win_size)
                end_win_f = init_win_f + win_size
                init_win_l = end_win_f
                end_win_l = init_win_l + label_win_size
                window_data = df_subject.iloc[init_win_f:end_win_f]
                win_features = window_data.drop(columns=['Condition'])
                windows_list.append(win_features)
                labels_df = df_subject.iloc[init_win_l:end_win_l]['Condition'].max() # se puede cambiar, ej. avg.
                labels_list.append(labels_df)
                prev_labels_list.append(labels_df)
                counter += 1
                next_limit = counter * win_size + win_size + label_win_size
                #print(f"Window: {counter}, added win_f: [{init_win_f}, {end_win_f}], win_l: [{init_win_l}, {end_win_l}]")
            #print(f"Total windows: {len(windows_list)}, num_labels: {len(labels_list)}")
            #print('')
            output_dict[main_key][sub_key] = {'features': windows_list, 'labels_prev': prev_labels_list[:-1], 'labels': labels_list}
    #print('done...')
    return output_dict


def get_features_from_dict_bins(data_dict, tp=60, tf=180, bin_size=15, freq=32):
    # Calculate the size of one observation bin and the prediction window in terms of datapoints
    win_size = bin_size * freq  # Observation bin size in number of datapoints
    output_dict = {}
    for main_key, sub_dict in data_dict.items():
        if main_key not in output_dict:
            output_dict[main_key] = {}
        for sub_key, df in sub_dict.items():
            if sub_key not in output_dict[main_key]:
                output_dict[main_key][sub_key] = {}
            df_subject = data_dict[main_key][sub_key]
            #print(f"key: {main_key}-{sub_key}, num. values in session: {len(df_subject)}")
            counter = 0 # Counter to keep track of the current bin
            windows_list = []
            labels_list = []
            agg_observed_list = []
            # Calculate the endpoint of the next window
            next_limit = counter * win_size + win_size
            # Loop to extract bins until the end of the session
            while (next_limit < len(df_subject)):
                init_win_f = (counter * win_size)  # Start of the current observation bin
                end_win_f = init_win_f + win_size  # End of the current observation bin
                init_win_l = end_win_f  # Start of the prediction window
                end_win_l = init_win_l + win_size  # End of the prediction window
                # Extract data for the observation bin
                window_data = df_subject.iloc[init_win_f:end_win_f]
                # Determine the maximum aggression value in the observation bin
                agg_observed = window_data.Condition.max()
                agg_observed_list.append(agg_observed)
                win_features = window_data.drop(columns=['Condition'])
                windows_list.append(win_features)
                # Increment the bin counter and update the endpoint of the next window
                counter += 1
                next_limit = counter * win_size + win_size
                #print(f"Window: {counter}, added win_f: [{init_win_f}, {end_win_f}], win_l: [{init_win_l}, {end_win_l}]")
            #print(f"Total windows: {len(windows_list)}, num_labels: {len(labels_list)}")
            #print('')
            output_dict[main_key][sub_key] = {'features': windows_list, 'labels': agg_observed_list}
    #print('done...')
    return output_dict


def get_features_from_dict_bins_multi(data_dict, bin_size=15, freq=32):
    # Calculate the size of one observation bin and the prediction window in terms of datapoints
    win_size = bin_size * freq  # Observation bin size in number of datapoints
    output_dict = {}
    for main_key, sub_dict in data_dict.items():
        if main_key not in output_dict:
            output_dict[main_key] = {}
        for sub_key, df in sub_dict.items():
            if sub_key not in output_dict[main_key]:
                output_dict[main_key][sub_key] = {}
            df_subject = data_dict[main_key][sub_key]
            #print(f"key: {main_key}-{sub_key}, num. values in session: {len(df_subject)}")
            counter = 0 # Counter to keep track of the current bin
            windows_list = []
            labels_sib_list = []
            labels_agg_list = []
            labels_ed_list = []
            # Calculate the endpoint of the next window
            next_limit = counter * win_size + win_size
            # Loop to extract bins until the end of the session
            while (next_limit <= len(df_subject)):
                init_win_f = (counter * win_size)  # Start of the current observation bin
                end_win_f = init_win_f + win_size  # End of the current observation bin
                # Extract data for the observation bin
                window_data = df_subject.iloc[init_win_f:end_win_f]
                # Determine the maximum aggression value in the observation bin for each label
                sib_label = window_data['SIB'].max()
                agg_label = window_data['AGG'].max()
                ed_label = window_data['ED'].max()
                labels_sib_list.append(sib_label)
                labels_agg_list.append(agg_label)
                labels_ed_list.append(ed_label)
                #agg_observed = window_data.Condition.max()
                #agg_observed_list.append(agg_observed)
                win_features = window_data.drop(columns=['SIB', 'AGG', 'ED'])
                windows_list.append(win_features)
                # Increment the bin counter and update the endpoint of the next window
                counter += 1
                next_limit = counter * win_size + win_size
                #print(f"Window: {counter}, added win_f: [{init_win_f}, {end_win_f}], win_l: [{init_win_l}, {end_win_l}]")
            #print(f"Total windows: {len(windows_list)}, num_labels: {len(labels_list)}")
            #print('')
            output_dict[main_key][sub_key] = {
                'features': windows_list,
                'SIB': labels_sib_list,
                'AGG': labels_agg_list,
                'ED': labels_ed_list
            }
    #print('done...')
    return output_dict


def get_features_from_dic_aggBehavior_wins(data_dict, tp=60, tf=180, stride=15, freq=32):
    win_size = tp * freq
    label_size = tf * freq
    stride_size = stride * freq
    output_dict = {}
    for main_key, sub_dict in data_dict.items():
        if main_key not in output_dict:
            output_dict[main_key] = {}
        for sub_key, df in sub_dict.items():
            if sub_key not in output_dict[main_key]:
                output_dict[main_key][sub_key] = {}
            df_subject = data_dict[main_key][sub_key]
            #print(f"key: {main_key}-{sub_key}, num. values in session: {len(df_subject)}")
            counter = 0
            windows_list = []
            labels_list = []
            final_labels_list = []
            agg_observed_list = [] # si en el la ventana hubo un episodio agresivo, se concatena con la siguiente ventana
            agg_observed_list.append(0) # no hay para la primera ventana, la segunda empieza con la etiqueta de la primera
            next_limit = (counter * stride_size) + win_size + label_size
            while (next_limit < len(df_subject)):
                init_win_f = (counter * stride_size)
                end_win_f = init_win_f + win_size
                init_win_l = end_win_f
                end_win_l = init_win_l + label_size
                window_data = df_subject.iloc[init_win_f:end_win_f]
                agg_observed = window_data.Condition.max()
                agg_observed_list.append(agg_observed)
                win_features = window_data.drop(columns=['Condition'])
                windows_list.append(win_features)
                labels_list.append(df_subject.iloc[init_win_l:end_win_l])
                labels_df = df_subject.iloc[init_win_l:end_win_l]['Condition'].max()
                final_labels_list.append(labels_df)
                counter += 1
                next_limit = (counter * stride_size) + win_size + label_size
                #print(f"Window: {counter}, added win_f: [{init_win_f}, {end_win_f}], win_l: [{init_win_l}, {end_win_l}]")
            #print(f"Total windows: {len(windows_list)}, num_labels: {len(final_labels_list)}")
            #print('')
            output_dict[main_key][sub_key] = {'features': windows_list, 'labels': final_labels_list, 'aggObserved': agg_observed_list[0:-1]}

    #print('done...')
    return output_dict


def get_features_from_dic_aggBehavior_wins_fixed(data_dict, tp=60, tf=180, stride=15, freq=32):
    # Calculate the size of the observation window and the prediction window
    win_size = tp * freq # Observation window size in datapoints
    label_size = tf * freq # Prediction window size in datapoints
    # Calculate the stride size in datapoints
    stride_size = stride * freq
    output_dict = {}
    for main_key, sub_dict in data_dict.items():
        if main_key not in output_dict:
            output_dict[main_key] = {}
        for sub_key, df in sub_dict.items():
            if sub_key not in output_dict[main_key]:
                output_dict[main_key][sub_key] = {}
            df_subject = data_dict[main_key][sub_key]
            #print(f"key: {main_key}-{sub_key}, num. values in session: {len(df_subject)}")
            counter = 0 # Counter to keep track of the current window
            windows_list = []
            final_labels_list = []
            agg_observed_list = []
            # Calculate the endpoint of the next prediction window
            next_limit = (counter * stride_size) + win_size + label_size
            while (next_limit < len(df_subject)):
                init_win_f = (counter * stride_size)  # Start of the observation window
                end_win_f = init_win_f + win_size  # End of the observation window
                init_win_l = end_win_f  # Start of the prediction window
                end_win_l = init_win_l + label_size  # End of the prediction window
                window_data = df_subject.iloc[init_win_f:end_win_f]
                # Determine is there was an aggresive episode in the observation window
                agg_observed = window_data.Condition.max()
                agg_observed_list.append(agg_observed)
                win_features = window_data.drop(columns=['Condition'])
                windows_list.append(win_features)
                # Determine is there was an aggresive episode in the prediction window
                labels_df = df_subject.iloc[init_win_l:end_win_l]['Condition'].max()
                final_labels_list.append(labels_df)
                # Increment the window counter and update the endpoint of the next prediction window
                counter += 1
                next_limit = (counter * stride_size) + win_size + label_size
                #print(f"Window: {counter}, added win_f: [{init_win_f}, {end_win_f}], win_l: [{init_win_l}, {end_win_l}]")
            #print(f"Total windows: {len(windows_list)}, num_labels: {len(final_labels_list)}")
            #print('')
            output_dict[main_key][sub_key] = {'features': windows_list, 'labels': final_labels_list, 'aggObserved': agg_observed_list}

    #print('done...')
    return output_dict


def split_data_per_session_aggObserved(data_dict, train_ratio=0.8):
    train_dict = {}
    test_dict = {}
    for user, sessions in data_dict.items():
        train_dict[user] = {}
        test_dict[user] = {}
        for session, data in sessions.items():
            features = data['features']
            labels = data['labels']
            split_idx = int(len(features) * train_ratio)
            train_features = features[:split_idx]
            train_labels = labels[:split_idx]
            test_features = features[split_idx:]
            test_labels = labels[split_idx:]
            train_dict[user][session] = {'features': train_features, 'labels': train_labels}
            test_dict[user][session] = {'features': test_features, 'labels': test_labels}
    return train_dict, test_dict


def split_data_per_session(data_dict, train_ratio=0.8):
    train_dict = {}
    test_dict = {}
    for user, sessions in data_dict.items():
        train_dict[user] = {}
        test_dict[user] = {}
        for session, data in sessions.items():
            features = data['features']
            labels = data['labels']
            split_idx = int(len(features) * train_ratio)
            train_features = features[:split_idx]
            train_labels = labels[:split_idx]
            test_features = features[split_idx:]
            test_labels = labels[split_idx:]
            train_dict[user][session] = {'features': train_features, 'labels': train_labels}
            test_dict[user][session] = {'features': test_features, 'labels': test_labels}
    return train_dict, test_dict


def new_split_data_per_session(data_dict, train_ratio=0.8):
    # data_dict (dict): Diccionario con datos organizados por usuario y sesión.
    # train_ratio (float): Proporción de datos para el conjunto de entrenamiento
    train_dict = {}
    test_dict = {}
    for user, sessions in data_dict.items():
        train_dict[user] = {}
        test_dict[user] = {}
        for session, data in sessions.items():
            split_idx = int(len(data) * train_ratio)
            train_data = data.iloc[:split_idx]
            test_data = data.iloc[split_idx:]
            train_dict[user][session] = train_data
            test_dict[user][session] = test_data
    return train_dict, test_dict


def new_split_data_full_session(data_dict, train_ratio=0.8):
    train_dict = {}
    test_dict = {}
    for user, sessions in data_dict.items():
        session_keys = list(sessions.keys())
        num_sessions = len(session_keys)
        # Si el 20% de las sesiones es menor que 1, todas las sesiones van a train
        if num_sessions * (1 - train_ratio) < 1:
            train_dict[user] = {session: sessions[session] for session in session_keys}
            test_dict[user] = {}  # No hay sesiones en test
        else:
            train_dict[user] = {}
            test_dict[user] = {}
            split_idx = int(num_sessions * train_ratio)
            train_sessions = session_keys[:split_idx]
            test_sessions = session_keys[split_idx:]
            for session in train_sessions:
                train_dict[user][session] = sessions[session]
            for session in test_sessions:
                test_dict[user][session] = sessions[session]
    return train_dict, test_dict


class AggressiveBehaviorDatasetBinLabels(Dataset):
    def __init__(self, data_dict, tp=60, bin_size=15):
        self.data = []
        self.prev_labels = []
        self.labels = []
        self.sequence_size = (tp//bin_size)
        # For each user and session, group each sample of size bin_size into N sequences, where N = (tp / bin_size)
        for user, sessions in data_dict.items():
            for session, session_data in sessions.items():
                features = session_data['features']
                prev_labels = session_data['labels_prev']
                labels = session_data['labels']
                for i in range(len(features) - self.sequence_size + 1):
                    windows = features[i:i+self.sequence_size]
                    win_prev_labels = prev_labels[i:i+self.sequence_size]
                    win_labels = labels[i:i+self.sequence_size]
                    data_tensor = torch.stack([torch.tensor(window.values.T, dtype=torch.float32) for window in windows])
                    prev_labels_tensor = torch.stack([torch.tensor(prev_label, dtype=torch.float32) for prev_label in win_prev_labels])
                    # Select the label of the last bin as the label of the sequence
                    label_tensor = torch.tensor(win_labels[-1], dtype=torch.float32)
                    self.data.append(data_tensor)
                    self.prev_labels.append(prev_labels_tensor.unsqueeze(1))
                    self.labels.append(label_tensor)
        print('')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.prev_labels[idx], self.labels[idx]


class AggressiveBehaviorDatasetBin(Dataset):
    def __init__(self, data_dict, tp=15, bin_size=15):
        self.data = []
        self.labels = []
        self.sequence_size = (tp//bin_size)
        # For each user and session, group each sample of size bin_size into N sequences, where N = (tp / bin_size)
        for user, sessions in data_dict.items():
            for session, session_data in sessions.items():
                features = session_data['features']
                labels = session_data['labels']
                for i in range(len(features) - self.sequence_size + 1):
                    windows = features[i:i+self.sequence_size]
                    win_labels = labels[i:i+self.sequence_size]
                    data_tensor = torch.stack([torch.tensor(window.values.T, dtype=torch.float32) for window in windows])
                    label_tensor = torch.stack([torch.tensor(win_l, dtype=torch.float32) for win_l in win_labels])
                    self.data.append(data_tensor)
                    self.labels.append(label_tensor)
        print('')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]




class New_AggressiveBehaviorDatasetBin(Dataset):
    def __init__(self, data_dict, tp=15, tf=15, bin_size=15):
        # tp: Observation time (s.)
        # tf: Prediction time (s.)
        # bin_size: bin size (s.)
        # self.data contains past windows (sequences of bins) with the signal data associated with each bin in the range (t-tp, t)
        # self.labels contains the label obtained from the prediction window associated with each past window,
        # indicating the occurrence of an aggressive eppisode in the range (t, t+tf).
        # sliding windows of size bin_size are used
        self.data = []
        self.labels = []
        # number of bins in the past window
        self.sequence_size = (tp//bin_size)
        # number of bins in the prediction window
        self.prediction_sequence_size = (tf // bin_size)
        # For each user and session, group each sample of size bin_size into N sequences, where N = (tp / bin_size)
        for user, sessions in data_dict.items():
            for session, session_data in sessions.items():
                features = session_data['features']
                labels = session_data['labels']
                for i in range(len(features) - self.sequence_size - self.prediction_sequence_size + 1):
                    # Get the signal values for each bin within the observation window (t-tp, t)
                    windows = features[i:i+self.sequence_size]
                    data_tensor = torch.stack([torch.tensor(window.values.T, dtype=torch.float32) for window in windows])
                    self.data.append(data_tensor)
                    # check if there was an aggressive episode (aggObs) in the prediction interval (t, t+tf)
                    prediction_label = np.max(labels[i + self.sequence_size: i + self.sequence_size + self.prediction_sequence_size])
                    label_tensor = torch.tensor(prediction_label, dtype=torch.float32)
                    self.labels.append(label_tensor)
                    # win_labels = labels[i:i + self.sequence_size] # AGGObserved, not used
                    # labels_bins_tensor = torch.stack([torch.tensor(win_l, dtype=torch.float32) for win_l in win_labels]) # AGGObserved, not used
        print('')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]



class New_AggressiveBehaviorDatasetBin_multi(Dataset):
    def __init__(self, data_dict, label_key, tp=15, tf=15, bin_size=15):
        # tp: Observation time (s.)
        # tf: Prediction time (s.)
        # bin_size: bin size (s.)
        # self.data contains past windows (sequences of bins) with the signal data associated with each bin in the range (t-tp, t)
        # self.labels contains the label obtained from the prediction window associated with each past window,
        # indicating the occurrence of an aggressive eppisode in the range (t, t+tf).
        # sliding windows of size bin_size are used
        self.data = []
        self.labels = []
        # number of bins in the past window
        self.sequence_size = (tp//bin_size)
        # number of bins in the prediction window
        self.prediction_sequence_size = (tf // bin_size)
        # For each user and session, group each sample of size bin_size into N sequences, where N = (tp / bin_size)
        for user, sessions in data_dict.items():
            for session, session_data in sessions.items():
                features = session_data['features']
                labels = session_data[label_key]
                for i in range(len(features) - self.sequence_size - self.prediction_sequence_size + 1):
                    # Get the signal values for each bin within the observation window (t-tp, t)
                    windows = features[i:i+self.sequence_size]
                    data_tensor = torch.stack([torch.tensor(window.values.T, dtype=torch.float32) for window in windows])
                    self.data.append(data_tensor)
                    # check if there was an aggressive episode (aggObs) in the prediction interval (t, t+tf)
                    prediction_label = np.max(labels[i + self.sequence_size: i + self.sequence_size + self.prediction_sequence_size])
                    label_tensor = torch.tensor(prediction_label, dtype=torch.float32)
                    self.labels.append(label_tensor)
                    # win_labels = labels[i:i + self.sequence_size] # AGGObserved, not used
                    # labels_bins_tensor = torch.stack([torch.tensor(win_l, dtype=torch.float32) for win_l in win_labels]) # AGGObserved, not used
        print('')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class New_AggressiveBehaviorDatasetBin_AGGObserved(Dataset):
    def __init__(self, data_dict, tp=15, tf=15, bin_size=15):
        # tp: Observation time (s.)
        # tf: Prediction time (s.)
        # bin_size: bin size (s.)
        # self.data contains past windows (sequences of bins) with the signal data associated with each bin in the range (t-tp, t)
        # self.aggObs indicates the occurrence of an aggressive episode within each bin in the range (t-tp, t)
        # self.labels contains the label obtained from the prediction window associated with each past window,
        # indicating the occurrence of an aggressive eppisode in the range (t, t+tf).
        # sliding windows of size bin_size are used
        self.data = []
        self.labels = []
        self.aggObs = []
        self.sequence_size = (tp//bin_size) # no. of past bins
        self.prediction_bins = (tf//bin_size) # no. of prediction bins
        # For each user and session, group each sample of size bin_size into N sequences, where N = (tp / bin_size)
        for user, sessions in data_dict.items():
            for session, session_data in sessions.items():
                features = session_data['features']
                labels = session_data['labels']
                for i in range(len(features) - self.sequence_size - self.prediction_bins + 1):
                    # get the signal values for each bin within the observation window (t-tp, t)
                    windows = features[i:i+self.sequence_size]
                    data_tensor = torch.stack(
                        [torch.tensor(window.values.T, dtype=torch.float32) for window in windows])
                    self.data.append(data_tensor)
                    # check if there was an aggresive eppisode within the bins (aggObs)
                    win_labels = labels[i:i+self.sequence_size]
                    aggObsr_tensor = torch.stack([torch.tensor(win_l, dtype=torch.float32) for win_l in win_labels])
                    self.aggObs.append(aggObsr_tensor)
                    # check if there was an aggressive episode (aggObs) in the prediction interval (t, t+tf)
                    prediction_labels = labels[i+self.sequence_size:i+self.sequence_size+self.prediction_bins]
                    label_tensor = torch.tensor(np.max(prediction_labels), dtype=torch.float32)
                    self.labels.append(label_tensor)
        print('')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.aggObs[idx], self.labels[idx]


class New_AggressiveBehaviorDatasetBin_AGGObserved_multi(Dataset):
    def __init__(self, data_dict, label_key, tp=15, tf=15, bin_size=15):
        # tp: Observation time (s.)
        # tf: Prediction time (s.)
        # bin_size: bin size (s.)
        # self.data contains past windows (sequences of bins) with the signal data associated with each bin in the range (t-tp, t)
        # self.aggObs indicates the occurrence of an aggressive episode within each bin in the range (t-tp, t)
        # self.labels contains the label obtained from the prediction window associated with each past window,
        # indicating the occurrence of an aggressive eppisode in the range (t, t+tf).
        # sliding windows of size bin_size are used
        self.data = []
        self.labels = []
        self.aggObs = []
        self.sequence_size = (tp//bin_size) # no. of past bins
        self.prediction_bins = (tf//bin_size) # no. of prediction bins
        # For each user and session, group each sample of size bin_size into N sequences, where N = (tp / bin_size)
        for user, sessions in data_dict.items():
            for session, session_data in sessions.items():
                features = session_data['features']
                labels = session_data[label_key]
                for i in range(len(features) - self.sequence_size - self.prediction_bins + 1):
                    # get the signal values for each bin within the observation window (t-tp, t)
                    windows = features[i:i+self.sequence_size]
                    data_tensor = torch.stack(
                        [torch.tensor(window.values.T, dtype=torch.float32) for window in windows])
                    self.data.append(data_tensor)
                    # check if there was an aggresive episode within the bins (aggObs)
                    win_labels = labels[i:i+self.sequence_size]
                    aggObsr_tensor = torch.stack([torch.tensor(win_l, dtype=torch.float32) for win_l in win_labels])
                    self.aggObs.append(aggObsr_tensor)
                    # check if there was an aggressive episode (label_key) in the prediction interval (t, t+tf)
                    prediction_labels = labels[i+self.sequence_size:i+self.sequence_size+self.prediction_bins]
                    label_tensor = torch.tensor(np.max(prediction_labels), dtype=torch.float32)
                    self.labels.append(label_tensor)
        print('')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.aggObs[idx], self.labels[idx]



class AggressiveBehaviorDatasetwinLabels(Dataset):
    def __init__(self, data_dict, tp=60, bin_size=15):
        self.data = []
        self.labels = []
        self.aggObser = []
        for user, sessions in data_dict.items():
            for session, session_data in sessions.items():
                features = session_data['features']
                aggObs = session_data['aggObserved']
                labels = session_data['labels']
                for i in range(len(features)):
                    window = features[i]
                    aggObserved = aggObs[i]
                    label = labels[i]
                    self.data.append(torch.tensor(window.values.T, dtype=torch.float32))
                    self.labels.append(torch.tensor(aggObserved, dtype=torch.long))
                    self.aggObser.append(torch.tensor(label, dtype=torch.long))
        print('')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.aggObser[idx], self.labels[idx]


