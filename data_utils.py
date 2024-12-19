import pandas as pd
from torch.utils.data import DataLoader, Dataset
import os
import torch


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


def get_features_from_dic_aggObserved_bins(data_dict, tp=60, tf=180, bin_size=15, freq=32):
    win_size = bin_size * freq
    label_win_size = tf * freq
    output_dict = {}
    agg_observed_list = []  # si en el la ventana hubo un episodio agresivo, se concatena con la siguiente ventana
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
            next_limit = counter * win_size + win_size + label_win_size
            while (next_limit < len(df_subject)):
                init_win_f = (counter * win_size)
                end_win_f = init_win_f + win_size
                init_win_l = end_win_f
                end_win_l = init_win_l + label_win_size
                window_data = df_subject.iloc[init_win_f:end_win_f]
                agg_observed = window_data.Condition.max()
                agg_observed_list.append(agg_observed)
                win_features = window_data.drop(columns=['Condition'])
                windows_list.append(win_features)
                labels_df = df_subject.iloc[init_win_l:end_win_l]['Condition'].max() # se puede cambiar, ej. avg.
                labels_list.append(labels_df)
                counter += 1
                next_limit = counter * win_size + win_size + label_win_size
                #print(f"Window: {counter}, added win_f: [{init_win_f}, {end_win_f}], win_l: [{init_win_l}, {end_win_l}]")
            #print(f"Total windows: {len(windows_list)}, num_labels: {len(labels_list)}")
            #print('')
            output_dict[main_key][sub_key] = {'features': windows_list, 'aggObserved': agg_observed_list, 'labels': labels_list}
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
            aggObserved = data['aggObserved']
            split_idx = int(len(features) * train_ratio)
            train_features = features[:split_idx]
            train_labels = labels[:split_idx]
            train_aggObs = aggObserved[:split_idx]
            test_features = features[split_idx:]
            test_labels = labels[split_idx:]
            test_aggObs = aggObserved[split_idx:]
            train_dict[user][session] = {'features': train_features, 'labels': train_labels, 'aggObserved': train_aggObs}
            test_dict[user][session] = {'features': test_features, 'labels': test_labels, 'aggObserved': test_aggObs}
    return train_dict, test_dict


def split_data_per_session_prevLabel(data_dict, train_ratio=0.8):
    train_dict = {}
    test_dict = {}
    for user, sessions in data_dict.items():
        train_dict[user] = {}
        test_dict[user] = {}
        for session, data in sessions.items():
            features = data['features']
            labels = data['labels']
            labels_prev = data['labels_prev']
            split_idx = int(len(features) * train_ratio)
            train_features = features[:split_idx]
            train_labels = labels[:split_idx]
            train_labels_prev = labels_prev[:split_idx]
            test_features = features[split_idx:]
            test_labels = labels[split_idx:]
            test_labels_prev = labels_prev[split_idx:]
            train_dict[user][session] = {'features': train_features, 'labels': train_labels, 'labels_prev': train_labels_prev}
            test_dict[user][session] = {'features': test_features, 'labels': test_labels, 'labels_prev': test_labels_prev}
    return train_dict, test_dict



class AggressiveBehaviorDatasetBinLabels(Dataset):
    def __init__(self, data_dict, tp=60, bin_size=15):
        self.data = []
        self.prev_labels = []
        self.labels = []
        self.sequence_size = (tp//bin_size)
        #self.stride = stride//bin_size # modificar si en algun momento se cambia el valor de la slide window
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
                    #label_tensor = torch.stack([torch.tensor(label, dtype=torch.float32) for label in win_labels])
                    label_tensor = torch.tensor(win_labels[-1], dtype=torch.float32)
                    self.data.append(data_tensor)
                    self.prev_labels.append(prev_labels_tensor.unsqueeze(1))
                    self.labels.append(label_tensor)
        print('')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.prev_labels[idx], self.labels[idx]


class AggressiveBehaviorDatasetBinAGGobserved(Dataset):
    def __init__(self, data_dict, tp=60, bin_size=15):
        self.data = []
        self.aggObs = []
        self.labels = []
        self.sequence_size = (tp//bin_size)
        #self.stride = stride//bin_size # modificar si en algun momento se cambia el valor de la slide window
        for user, sessions in data_dict.items():
            for session, session_data in sessions.items():
                features = session_data['features']
                aggObserved = session_data['aggObserved']
                labels = session_data['labels']
                for i in range(len(features) - self.sequence_size + 1):
                    windows = features[i:i+self.sequence_size]
                    win_aggObser = aggObserved[i:i+self.sequence_size]
                    win_labels = labels[i:i+self.sequence_size]
                    data_tensor = torch.stack([torch.tensor(window.values.T, dtype=torch.float32) for window in windows])
                    aggObs_tensor = torch.stack([torch.tensor(aggObs, dtype=torch.float32) for aggObs in win_aggObser])
                    #label_tensor = torch.stack([torch.tensor(label, dtype=torch.float32) for label in win_labels])
                    label_tensor = torch.tensor(win_labels[-1], dtype=torch.float32)
                    self.data.append(data_tensor)
                    self.aggObs.append(aggObs_tensor.unsqueeze(1))
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


