import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import torch.nn as nn
import pandas as pd
from torch.optim import Adam
from sklearn.metrics import roc_curve, f1_score, roc_auc_score
import models
import data_utils
from torch.utils.data import DataLoader


def evaluate_val_auc_model(model, test_dataloader, device):
    model.eval()
    all_probs = []  # AUC-ROC
    all_labels = []
    with torch.no_grad():
        for batch_features, batch_aggObs, batch_labels in test_dataloader:
            batch_features = batch_features.to(device)
            batch_aggObs = batch_aggObs.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(batch_features, batch_aggObs)
            probs = torch.sigmoid(logits).squeeze()
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    return roc_auc_score(all_labels, all_probs)


def evaluate_model(model, test_dataloader, device):
    model.eval()
    all_probs = []  # AUC-ROC
    all_labels = []
    with torch.no_grad():
        for batch_features, batch_aggObs, batch_labels in test_dataloader:
            batch_features = batch_features.to(device)
            batch_aggObs = batch_aggObs.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(batch_features, batch_aggObs)
            probs = torch.sigmoid(logits).squeeze()
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    best_f1 = 0
    best_threshold = thresholds[0]
    for threshold in thresholds:
        predictions = (all_probs >= threshold).astype(int)
        current_f1 = f1_score(all_labels, predictions)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold

    threshold = 0.5
    all_predictions = (np.array(all_probs) > threshold).astype(int)
    f1_s = f1_score(all_labels, all_predictions)
    try:
        auc_roc_s = roc_auc_score(all_labels, all_probs)
    except:
        auc_roc_s = -1
    best_th = best_threshold
    best_f1 = best_f1
    return f1_s, best_th, best_f1, auc_roc_s



def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, device):
    best_auc = 0
    patience = 5
    counter_patience = 0
    train_losses = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_features, batch_AGGObs, batch_labels in train_dataloader:
            batch_features = batch_features.to(device)
            batch_AGGObs = batch_AGGObs.to(device).float()
            batch_labels = batch_labels.to(device).float()
            optimizer.zero_grad()
            predictions = model(batch_features, batch_AGGObs).squeeze()  # Remover dimensión extra si es necesario
            loss = criterion(predictions, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        auc_val = evaluate_val_auc_model(model, val_dataloader, device)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_dataloader):.4f}, Val auc: {auc_val}")
        train_losses.append(total_loss / len(train_dataloader))
        if auc_val > best_auc:
            best_auc = auc_val
            # torch.save(model.state_dict(), path_to_save_model)
            counter_patience = 0
        else:
            print('Val auc does not improve...')
            counter_patience += 1
        if counter_patience == patience:
            break

    print("Train Done...")
    if counter_patience == patience:
        train_losses = train_losses[: -patience]
        final_epochs = len(train_losses)
    else:
        final_epochs = num_epochs
    return final_epochs


def invalid_data(dataloader):
    labels_list = []
    for _, _, labels in dataloader:
        labels_flat = labels.numpy().flatten()
        labels_list.extend(labels_flat)
    labels_array = np.array(labels_list)
    print(f'Total samples: {len(labels_array)}, 0: {len(labels_array) - np.sum(labels_array)}, 1:{np.sum(labels_array)}')
    return (len(np.unique(labels_array)) == 1)


def retrain_model(model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs, device, exp_name, path_model):
    print('Retraining model...')
    train_losses = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_features, batch_AGGObs, batch_labels in train_dataloader:
            batch_features = batch_features.to(device)
            batch_AGGObs = batch_AGGObs.to(device).float()
            batch_labels = batch_labels.to(device).float()
            optimizer.zero_grad()
            predictions = model(batch_features, batch_AGGObs).squeeze()  # Remover dimensión extra si es necesario
            loss = criterion(predictions, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_dataloader):.4f}")
        train_losses.append(total_loss / len(train_dataloader))

    torch.save(model.state_dict(), path_model)
    f1_s, best_th, best_f1, auc_roc_s = evaluate_model(model, test_dataloader, device)
    print(f'Test results: f1_s: {f1_s}, best_th: {best_th}, best_f1: {best_f1}, auc: {auc_roc_s}.')
    print('')
    results = {
        "Experiment": exp_name,
        "F1-Score": f1_s,
        'Best_F1-score': best_f1,
        'Best_th': best_th,
        "AUC-ROC": auc_roc_s,
        "Num_epochs": num_epochs
    }
    return results



def set_model(model_code):
    if model_code == 0:
        model_fun = models.EEGNetLSTMwinLabels
        features_fun = data_utils.get_features_from_dic_aggBehavior_wins
        dataloader_fun = data_utils.AggressiveBehaviorDatasetwinLabels
        split_fun = data_utils.split_data_per_session_aggObserved
    elif model_code == 1:
        model_fun = models.EEGNetLSTM_v1
        features_fun = data_utils.get_features_from_dic_prevLabels_bins
        dataloader_fun = data_utils.AggressiveBehaviorDatasetBinLabels
        split_fun = data_utils.split_data_per_session_prevLabel
    else: # model_code == 2, still on going
        model_fun = models.EEGNetLSTM_v1
        features_fun = data_utils.get_features_from_dic_aggObserved_bins
        dataloader_fun = data_utils.AggressiveBehaviorDatasetBinAGGobserved
        split_fun = data_utils.split_data_per_session_aggObserved
    return model_fun, features_fun, dataloader_fun, split_fun


def create_dataloader(dataloader_fun, output_dict, tp, bin_size, batch_size, shuffle=False):
    dataset = dataloader_fun(output_dict, tp, bin_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    return dataloader


def start_exps_PM(tp, tf, freq, data_path_resampled, path_results, path_models, model_code, seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds_path = data_path_resampled + "dataset_" + str(freq) + "Hz.csv"
    data_dict = data_utils.load_data_to_dict(ds_path)
    #data_dict = dict(list(data_dict.items())[:1])
    EEG_channels = 5
    model_fun, features_fun, dataloader_fun, split_fun = set_model(model_code)
    num_exps = 10 # TO-DO: configurar particones para cada exp!
    tp, tf, stride, bin_size = tp, tf, 15, 15
    print(f'tp: {tp}, tf: {tf}, stride: {stride}, bin_size: {bin_size}.')
    final_results = []
    for exp in range(0, num_exps):
        # TO-DO: configurar particones para cada exp!
        output_dict = features_fun(data_dict, tp, tf, bin_size, freq)
        train_dict_, test_dict = split_fun(output_dict, train_ratio=0.8)
        train_dict, val_dict = split_fun(train_dict_, train_ratio=0.8)

        batch_size = 128
        dataloader = create_dataloader(dataloader_fun, train_dict, tp, bin_size, batch_size, shuffle=True)
        dataloader_val = create_dataloader(dataloader_fun, val_dict, tp, bin_size, batch_size, shuffle=False)
        dataloader_test = create_dataloader(dataloader_fun, test_dict, tp, bin_size, batch_size, shuffle=False)

        for batch_features, batch_AGGObs, batch_labels in dataloader:
            print(f"Features: {batch_features.shape}")
            print(f"Labels: {batch_labels.shape}")
            break

        # compute class weights
        labels_list = []
        for _, _, labels in dataloader:
            labels_flat = labels.numpy().flatten()
            labels_list.extend(labels_flat)
        labels_array = np.array(labels_list)
        print(
            f'Total samples: {len(labels_array)}, 0: {len(labels_array) - np.sum(labels_array)}, 1:{np.sum(labels_array)}')
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(labels_list),
            y=labels_list
        )
        class_weights_dict = dict(zip(np.unique(labels_list), class_weights))
        print('class_weights_dict: ', class_weights_dict)
        positive_class_weight = class_weights_dict[1]
        pos_weight_tensor = torch.tensor([positive_class_weight]).to(device)

        # create model
        num_sequences = tp // bin_size  # tp//bin_size en EEGNetLSTMwinLabelsSeqs_binS

        bin_size = 15
        eegnet_params = {
            'num_electrodes': EEG_channels,  # (EDA, ACC_X, ACC_Y, ACC_Z, BVP) | (EDA, ACC_Norm, BVP) | ....
            'chunk_size': tp * freq // num_sequences  # muestras en cada ventana
        }
        lstm_hidden_dim = 64
        num_classes = 1
        model = model_fun(eegnet_params, lstm_hidden_dim, num_classes).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        optimizer = Adam(model.parameters(), lr=0.001)
        num_epochs = 1

        best_num_epochs = train_model(model, dataloader, dataloader_val, criterion,
                                      optimizer, num_epochs, device)

        final_model = model_fun(eegnet_params, lstm_hidden_dim, num_classes).to(device)
        dataloader_final = create_dataloader(dataloader_fun, train_dict_, tp, bin_size, batch_size, shuffle=True)

        labels_list = []
        for _, _, labels in dataloader_final:
            labels_flat = labels.numpy().flatten()
            labels_list.extend(labels_flat)
        labels_array = np.array(labels_list)
        print(
            f'Total samples: {len(labels_array)}, 0: {len(labels_array) - np.sum(labels_array)}, 1:{np.sum(labels_array)}')
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(labels_list),
            y=labels_list
        )
        class_weights_dict = dict(zip(np.unique(labels_list), class_weights))
        print('class_weights_dict: ', class_weights_dict)
        positive_class_weight = class_weights_dict[1]
        pos_weight_tensor = torch.tensor([positive_class_weight]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        optimizer = Adam(final_model.parameters(), lr=0.001)
        exp_name = 'tp_' + str(tp) + '_tf' + str(tf)
        path_model = path_models + 'tf' + str(tf) + '_tp' + str(tp) +  '_exp_' + str(exp) + '_model.pth'
        results = retrain_model(final_model, dataloader_final, dataloader_test, criterion, optimizer, best_num_epochs,
                                device, exp, path_model)
        final_results.append(results)

    results_df = pd.DataFrame(final_results)
    avg_metrics = results_df[['F1-Score', 'Best_F1-score', 'Best_th', 'AUC-ROC', 'Num_epochs']].mean()
    std_metrics = results_df[['F1-Score', 'Best_F1-score', 'Best_th', 'AUC-ROC', 'Num_epochs']].std()
    summary_df = pd.DataFrame({
        "Experiment": ["Avg.", "Std."],
        'F1-Score': [avg_metrics['F1-Score'], std_metrics['F1-Score']],
        'Best_F1-score': [avg_metrics['Best_F1-score'], std_metrics['Best_F1-score']],
        'Best_th': [avg_metrics['Best_th'], std_metrics['Best_th']],
        'AUC-ROC': [avg_metrics['AUC-ROC'], std_metrics['AUC-ROC']],
        'Num_epochs': [avg_metrics['Num_epochs'], std_metrics['Num_epochs']]
    })
    final_results_df = pd.concat([results_df, summary_df], ignore_index=True)
    path_to_save_results = f'{path_results}PM_SS_model{model_code}_tf{tf}_tp{tp}_all_experiments_results.csv'
    final_results_df.to_csv(path_to_save_results, index=False)
    print("Results saved successfully.")


def start_exps_PDM(tp, tf, freq, data_path_resampled, path_results, path_models, model_code, seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds_path = data_path_resampled + "dataset_" + str(freq) + "Hz.csv"
    data_dict = data_utils.load_data_to_dict(ds_path)
    data_dict = dict(list(data_dict.items())[:2])
    EEG_channels = 5
    model_fun, features_fun, dataloader_fun, split_fun = set_model(model_code)
    num_exps = 10 # TO-DO: configurar particones para cada exp!
    tp, tf, stride, bin_size = tp, tf, 15, 15
    print(f'tp: {tp}, tf: {tf}, stride: {stride}, bin_size: {bin_size}.')
    final_results = []
    num_users = len(list(data_dict.items()))
    invalid_users = []
    for u in range(0, num_users):
        first_item = dict(list(data_dict.items())[:1])
        key_subject = list(first_item.keys())[0]
        print('userID: ', key_subject)

        output_dict = features_fun(first_item, tf, tp, bin_size, freq)
        train_dict_, test_dict = split_fun(output_dict, train_ratio=0.8)
        train_dict, val_dict = split_fun(train_dict_, train_ratio=0.8)

        batch_size = 16
        dataloader = create_dataloader(dataloader_fun, train_dict, tp, bin_size, batch_size, shuffle=True)
        dataloader_val = create_dataloader(dataloader_fun, val_dict, tp, bin_size, batch_size, shuffle=False)
        dataloader_test = create_dataloader(dataloader_fun, test_dict, tp, bin_size, batch_size, shuffle=False)

        for batch_features, batch_AGGObs, batch_labels in dataloader:
            print(f"Features: {batch_features.shape}")
            print(f"Labels: {batch_labels.shape}")
            break

        if invalid_data(dataloader) or invalid_data(dataloader_test) or invalid_data(dataloader_val):
            invalid_users.append(key_subject)
        else:
            # compute class weights
            labels_list = []
            for _, _, labels in dataloader:
                labels_flat = labels.numpy().flatten()
                labels_list.extend(labels_flat)
            labels_array = np.array(labels_list)
            print(
                f'Total samples: {len(labels_array)}, 0: {len(labels_array) - np.sum(labels_array)}, 1:{np.sum(labels_array)}')
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(labels_list),
                y=labels_list
            )
            class_weights_dict = dict(zip(np.unique(labels_list), class_weights))
            print('class_weights_dict: ', class_weights_dict)
            positive_class_weight = class_weights_dict[1]
            pos_weight_tensor = torch.tensor([positive_class_weight]).to(device)

            # create model
            num_sequences = tp // bin_size  # tp//bin_size en EEGNetLSTMwinLabelsSeqs_binS

            bin_size = 15
            eegnet_params = {
                'num_electrodes': EEG_channels,  # (EDA, ACC_X, ACC_Y, ACC_Z, BVP) | (EDA, ACC_Norm, BVP) | ....
                'chunk_size': tp * freq // num_sequences  # muestras en cada ventana
            }
            lstm_hidden_dim = 64
            num_classes = 1
            model = model_fun(eegnet_params, lstm_hidden_dim, num_classes).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
            optimizer = Adam(model.parameters(), lr=0.001)
            num_epochs = 1

            best_num_epochs = train_model(model, dataloader, dataloader_val, criterion,
                                          optimizer, num_epochs, device)

            final_model = model_fun(eegnet_params, lstm_hidden_dim, num_classes).to(device)
            dataloader_final = create_dataloader(dataloader_fun, train_dict_, tp, bin_size, batch_size, shuffle=True)

            labels_list = []
            for _, _, labels in dataloader_final:
                labels_flat = labels.numpy().flatten()
                labels_list.extend(labels_flat)
            labels_array = np.array(labels_list)
            print(
                f'Total samples: {len(labels_array)}, 0: {len(labels_array) - np.sum(labels_array)}, 1:{np.sum(labels_array)}')
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(labels_list),
                y=labels_list
            )
            class_weights_dict = dict(zip(np.unique(labels_list), class_weights))
            print('class_weights_dict: ', class_weights_dict)
            positive_class_weight = class_weights_dict[1]
            pos_weight_tensor = torch.tensor([positive_class_weight]).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
            optimizer = Adam(final_model.parameters(), lr=0.001)
            exp_name = 'tp_' + str(tp) + '_tf' + str(tf)
            path_model = path_models + 'tf' + str(tf) + '_tp' + str(tp) +  '_exp_' + key_subject + '_model.pth'
            results = retrain_model(final_model, dataloader_final, dataloader_test, criterion, optimizer, best_num_epochs,
                                    device, key_subject, path_model)
            final_results.append(results)
        data_dict.pop(next(iter(first_item)))

    results_df = pd.DataFrame(final_results)
    avg_metrics = results_df[['F1-Score', 'Best_F1-score', 'Best_th', 'AUC-ROC', 'Num_epochs']].mean()
    std_metrics = results_df[['F1-Score', 'Best_F1-score', 'Best_th', 'AUC-ROC', 'Num_epochs']].std()
    summary_df = pd.DataFrame({
        "Experiment": ["Avg.", "Std."],
        'F1-Score': [avg_metrics['F1-Score'], std_metrics['F1-Score']],
        'Best_F1-score': [avg_metrics['Best_F1-score'], std_metrics['Best_F1-score']],
        'Best_th': [avg_metrics['Best_th'], std_metrics['Best_th']],
        'AUC-ROC': [avg_metrics['AUC-ROC'], std_metrics['AUC-ROC']],
        'Num_epochs': [avg_metrics['Num_epochs'], std_metrics['Num_epochs']]
    })
    final_results_df = pd.concat([results_df, summary_df], ignore_index=True)
    path_to_save_results = f'{path_results}PDM_SS_model{model_code}_tf{tf}_tp{tp}_all_experiments_results.csv'
    final_results_df.to_csv(path_to_save_results, index=False)
    print("Results saved successfully.")