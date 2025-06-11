import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random


CALM, PRE_ATTACK, ATTACK = 0, 1, 2


def load_data_to_dict(path, selected_columns=None):
    df = pd.read_csv(path, dtype={'SubjectID': str, 'SessionID': str})
    if selected_columns is None:  # default features: all, labels: combined
        selected_columns = ['EDA', 'ACC_X', 'ACC_Y', 'ACC_Z', 'BVP']
    base_columns = ['Timestamp', 'SubjectID', 'SessionID', 'Condition']
    all_columns = base_columns + selected_columns
    final_columns = selected_columns + ['Condition']
    df = df[all_columns]
    data_dict = {}
    for (subject_id, session_id), group_df in df.groupby(['SubjectID', 'SessionID']):
        group_df['Timestamp'] = pd.to_datetime(group_df['Timestamp'], unit='ms')
        group_df = group_df.set_index('Timestamp')
        if subject_id not in data_dict:
            data_dict[subject_id] = {}
        data_dict[subject_id][session_id] = group_df[final_columns]
    return data_dict


def split_data_full_sessions(data_dict, train_ratio=0.8):
    train_dict = {}
    test_dict = {}
    for user, sessions in data_dict.items():
        session_keys = list(sessions.keys())
        num_sessions = len(session_keys)
        if num_sessions == 1:
            train_dict[user] = {session_keys[0]: sessions[session_keys[0]]}
            test_dict[user] = {}
        else:
            train_dict[user] = {}
            test_dict[user] = {}
            split_idx = max(1, int(num_sessions * train_ratio))  # al menos 1 en test
            train_sessions = session_keys[:split_idx]
            test_sessions = session_keys[split_idx:]
            for session in train_sessions:
                train_dict[user][session] = sessions[session]
            for session in test_sessions:
                test_dict[user][session] = sessions[session]
    return train_dict, test_dict


def get_distribution_labels(dataset):
    labels = [label.item() for label in dataset.labels]
    count_dict = Counter(labels)
    class_map = {
        CALM: "Calm",
        PRE_ATTACK: "Pre-attack",
        ATTACK: "Attack"
        #3: "Post-attack"
    }
    print("Class distribution in the dataset:")
    for class_id, count in sorted(count_dict.items()):
        print(f"  {class_map[class_id]} ({class_id}): {count} samples")

    plt.figure(figsize=(6, 4))
    plt.bar(
        [class_map[k] for k in count_dict.keys()],
        [count_dict[k] for k in count_dict.keys()],
        color="steelblue"
    )
    plt.title("Class Distribution in Aggressive Behavior Dataset", fontsize=12)
    plt.xlabel("Class", fontsize=10)
    plt.ylabel("Number of Samples", fontsize=10)
    plt.xticks(rotation=15)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def compute_normalization_stats(train_dict, bin_size=15, freq=32):
    all_data = []
    win_size = bin_size * freq
    for user_sessions in train_dict.values():
        for df in user_sessions.values():
            for i in range(0, len(df) - win_size, win_size):
                window = df.iloc[i:i + win_size].drop(columns=["Condition"])
                all_data.append(window.to_numpy())  # (win_size, num_features)
    all_data = np.concatenate(all_data, axis=0)  # (total_samples, num_features)
    mean = all_data.mean(axis=0)
    std = all_data.std(axis=0) + 1e-8  # Para evitar 0 div
    return mean, std


def get_features_from_dict(data_dict, bin_size=15, freq=32, mean=None, std=None):
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
                if mean is not None and std is not None:
                    win_features = (win_features - mean) / std
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



class AggressiveBehaviorDataset(Dataset):
    def __init__(self, data_dict, tp=15, tf=15, bin_size=15, return_aggObsr=False):
        # tp: Observation time (s.)
        # tf: Prediction time (s.)
        # bin_size: bin size (s.)
        # self.data contains past windows (sequences of bins) with the signal data associated with each bin in the range (t-tp, t)
        # self.aggObs indicates the occurrence of an aggressive episode within each bin in the range (t-tp, t)
        # self.labels contains the label obtained from the prediction window associated with each past window,
        # indicating the occurrence of an aggressive eppisode in the range (t, t+tf).
        # sliding windows of size bin_size are used
        self.return_aggObsr = return_aggObsr
        self.return_onset = False
        self.data = []
        self.aggObs = []
        self.labels = []
        self.time_to_onset = []
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
                    # Get the signal values for each bin within the observation window [t-tp, t)
                    windows = features[i:i+self.sequence_size]
                    data_tensor = torch.stack([torch.tensor(window.values.T, dtype=torch.float32) for window in windows])
                    self.data.append(data_tensor)
                    # Labels from the past [t−tp, t): check if there was an aggresive eppisode within the bins (aggObs)
                    win_labels = labels[i:i + self.sequence_size]
                    aggObsr_tensor = torch.stack([torch.tensor(win_l, dtype=torch.float32) for win_l in win_labels])
                    self.aggObs.append(aggObsr_tensor)

                    ##### new version: multi-class classification:
                    # - Attack: si hay ataque activo en t (last bin de aggObsr == 1)
                    # - Pre-attack: si no hay ataque en t y hay uno en [t, t+tf] (last aggObsr == 0 y sum prediction_labels != 0)
                    # - Post-attack: si no hay ataque en t ni futuro, pero uno terminó antes de t (last bin aggObsr == 0, sum prediction_labels != 0 y sum aggObrs != 0)
                    # - Calm: si no hay ataque en [t−tp, t+tf] (cualquier otro caso, es decir, sum aggObsr == 0 y sum prediction_bins == 0
                    # else... print wtf!
                    prediction_labels = labels[i + self.sequence_size:i + self.sequence_size + self.prediction_sequence_size]
                    last_bin = aggObsr_tensor[-1]
                    last_sum = last_bin.sum()
                    past_sum = aggObsr_tensor.sum()
                    future_sum = sum(torch.tensor(l, dtype=torch.float32).sum() for l in prediction_labels)
                    '''
                    # si 4 clases
                    if last_sum > 0:
                        label = 2  # ATAQUE
                    elif future_sum > 0:
                        label = 1  # PRE-ATAQUE
                    elif past_sum > 0:
                        label = 3  # POST-ATAQUE
                    elif past_sum+future_sum == 0:
                        label = 0  # NORMAL
                    else:
                        print("WTF condition — check")
                        print(f"last_sum: {last_sum}, past_sum: {past_sum}, future_sum: {future_sum}")  
                    '''
                    # si 3 clases
                    if last_sum > 0:
                        label = ATTACK  # ATAQUE
                    elif future_sum > 0:
                        label = PRE_ATTACK  # PRE-ATAQUE
                    else:
                        label = CALM  # NORMAL

                    label_tensor = torch.tensor(label, dtype=torch.long)
                    self.labels.append(label_tensor)

                    '''
                    ##### previous version: binary classification, agg. episode prediction.
                    # check if there was an aggressive episode (aggObs) in the prediction interval (t, t+tf)
                    prediction_label = np.max(labels[i + self.sequence_size: i + self.sequence_size + self.prediction_sequence_size])
                    label_tensor = torch.tensor(prediction_label, dtype=torch.float32)
                    self.labels.append(label_tensor)
                    '''
                    # Calcular el tiempo hasta el siguiente ataque (onset)
                    time_to_onset = -1  # -1 = no hay episodio futuro
                    for j, future_label in enumerate(prediction_labels):
                        if torch.tensor(future_label).sum() > 0:
                            time_to_onset = j * bin_size  # segundos hasta el ataque
                            break
                    self.time_to_onset.append(time_to_onset)
        print('done.')


    def __setaggobsr__(self, flag):
        self.return_aggObsr = flag

    def __setonset__(self, flag):
        self.return_onset = flag

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.return_aggObsr:
            return self.data[idx], self.aggObs[idx], self.labels[idx]
        elif self.return_onset:
            return self.data[idx], self.time_to_onset[idx], self.labels[idx]
        else:
            return self.data[idx], self.labels[idx]



def undersample_binary_classes(data_list, label_list, seed=42):
    data_array = np.array(data_list)
    label_array = np.array(label_list)

    idx_pos = np.where(label_array == 1)[0]
    idx_neg = np.where(label_array == 0)[0]

    np.random.seed(seed)
    sampled_neg = np.random.choice(idx_neg, size=len(idx_pos), replace=False)

    balanced_idx = np.concatenate([idx_pos, sampled_neg])
    np.random.shuffle(balanced_idx)

    balanced_data = [data_array[i] for i in balanced_idx]
    balanced_labels = [torch.tensor(label_array[i], dtype=torch.long) for i in balanced_idx]

    print(f"Undersampling applied: {len(idx_pos)} pos / {len(sampled_neg)} neg samples")
    return balanced_data, balanced_labels



### Only calm and attack samples for pretrain feature extractor
class AggressiveBehaviorDatasetExtreme(Dataset):
    def __init__(self, data_dict, tp=15, tf=15, bin_size=15, balanced=True):
        # tp: Observation time (s.)
        # tf: Prediction time (s.)
        # bin_size: bin size (s.)
        # self.data contains past windows (sequences of bins) with the signal data associated with each bin in the range (t-tp, t)
        # self.labels contains the label obtained from the prediction window associated with each past window,
        # indicating the occurrence of an aggressive episode in the range (t, t+tf).
        # sliding windows of size bin_size are used
        self.data = []
        self.labels = []
        # number of bins in the past window
        self.sequence_size = (tp//bin_size)
        # number of bins in the prediction window
        self.prediction_sequence_size = (tf // bin_size)
        raw_data = []
        raw_labels = []
        # For each user and session, group each sample of size bin_size into N sequences, where N = (tp / bin_size)
        for user, sessions in data_dict.items():
            for session, session_data in sessions.items():
                features = session_data['features']
                labels = session_data['labels']
                for i in range(len(features) - self.sequence_size - self.prediction_sequence_size + 1):
                    # Get the signal values for each bin within the observation window [t-tp, t)
                    windows = features[i:i+self.sequence_size]
                    # Labels from the past [t−tp, t): check if there was an aggresive eppisode within the bins (aggObs)
                    win_labels = labels[i:i + self.sequence_size]
                    ##### new version: multi-class classification:
                    # - Attack: si hay ataque activo en t (last bin de aggObsr == 1)
                    # - Pre-attack: si no hay ataque en t y hay uno en [t, t+tf] (last aggObsr == 0 y sum prediction_labels != 0)
                    # - Post-attack: si no hay ataque en t ni futuro, pero uno terminó antes de t (last bin aggObsr == 0, sum prediction_labels != 0 y sum aggObrs != 0)
                    # - Calm: si no hay ataque en [t−tp, t+tf] (cualquier otro caso, es decir, sum aggObsr == 0 y sum prediction_bins == 0
                    # else... print wtf!
                    prediction_labels = labels[i + self.sequence_size:i + self.sequence_size + self.prediction_sequence_size]
                    aggObsr_tensor = torch.stack([torch.tensor(win_l, dtype=torch.float32) for win_l in win_labels])
                    last_bin = aggObsr_tensor[-1]
                    last_sum = last_bin.sum()
                    future_sum = sum(torch.tensor(l, dtype=torch.float32).sum() for l in prediction_labels)
                    '''
                    # si 4 clases
                    if last_sum > 0:
                        label = 2  # ATAQUE
                    elif future_sum > 0:
                        label = 1  # PRE-ATAQUE
                    elif past_sum > 0:
                        label = 3  # POST-ATAQUE
                    elif past_sum+future_sum == 0:
                        label = 0  # NORMAL
                    else:
                        print("WTF condition — check")
                        print(f"last_sum: {last_sum}, past_sum: {past_sum}, future_sum: {future_sum}")  
                    '''
                    # si 3 clases
                    if last_sum > 0:
                        label = ATTACK  # ATAQUE
                    elif future_sum > 0:
                        label = PRE_ATTACK  # PRE-ATAQUE
                    else:
                        label = CALM  # NORMAL

                    if label == ATTACK or label == CALM:
                        if label == ATTACK: label = 1 # binary, pos class
                        label_tensor = torch.tensor(label, dtype=torch.long)
                        #self.labels.append(label_tensor)
                        raw_labels.append(label)
                        data_tensor = torch.stack([torch.tensor(window.values.T, dtype=torch.float32) for window in windows])
                        #self.data.append(data_tensor)
                        raw_data.append(data_tensor)

        if balanced:
            self.data, self.labels = undersample_binary_classes(raw_data, raw_labels)
        else:
            self.data = raw_data
            self.labels = [torch.tensor(l, dtype=torch.long) for l in raw_labels]
            print(f"No undersampling applied. Total samples: {len(self.labels)}")

        print('done.')
        print('done.')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]




def plot_windows_from_dataset(dataset, class_names, save_path, n_samples_per_class=5, tp=300, channel_names=None,
                              highlight_secs=15):
    print("Visualizando señales por clase...")

    if channel_names is None:
        channel_names = [f"Ch {i}" for i in range(dataset[0][0].shape[1])]

    class_indices = {i: [] for i in range(len(class_names))}
    for idx in range(len(dataset)):
        data, label = dataset[idx]
        label = int(label.item()) if isinstance(label, torch.Tensor) else int(label)
        if label in class_indices and len(class_indices[label]) < n_samples_per_class:
            class_indices[label].append(idx)

    samples_per_class = {
        c: [dataset[i][0] for i in idxs]
        for c, idxs in class_indices.items() if len(idxs) > 0
    }

    fig, axs = plt.subplots(len(class_names), n_samples_per_class,
                            figsize=(n_samples_per_class * 4, len(class_names) * 2.5), sharex=True, sharey='row')
    if len(class_names) == 1:
        axs = np.expand_dims(axs, axis=0)

    for row_idx, (class_idx, samples) in enumerate(samples_per_class.items()):
        for col_idx, sample in enumerate(samples):
            if isinstance(sample, torch.Tensor):
                sample = sample.numpy()
            ax = axs[row_idx, col_idx]

            # Time axis (t=-tp to t=0)
            n_subwindows = sample.shape[0]
            step = tp // n_subwindows
            time_axis = np.arange(-tp, 0, step)

            # Highlight first and last highlight_secs (optional)
            ax.axvspan(-tp, -tp + highlight_secs, color='gray', alpha=0.1)
            ax.axvspan(-highlight_secs, 0, color='gray', alpha=0.1)

            for ch in range(sample.shape[1]):
                ax.plot(time_axis, sample[:, ch], label=channel_names[ch], linewidth=0.8, alpha=0.8)

            if row_idx == 0:
                ax.set_title(f"Sample {col_idx + 1}")
            if col_idx == 0:
                ax.set_ylabel(class_names[class_idx])


    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.01, 0.5), title="Canales", fontsize="small")
    fig.suptitle("Señales por clase", fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.98, 0.95])
    plt.savefig(save_path)
    plt.close()
    print(f"Saved en {save_path}")


