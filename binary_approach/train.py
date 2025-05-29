import time

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
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.dummy import DummyClassifier


def evaluate_val_auc_model(model, test_dataloader, device):
    model.eval()
    all_probs = []  # AUC-ROC
    all_labels = []
    with torch.no_grad():
        for batch_features, batch_labels in test_dataloader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(batch_features)
            probs = torch.sigmoid(logits).squeeze()
            if probs.dim() == 0:
                probs = probs.unsqueeze(0)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    return roc_auc_score(all_labels, all_probs)

def evaluate_val_auc_model_aGGObs(model, test_dataloader, device):
    model.eval()
    all_probs = []  # AUC-ROC
    all_labels = []
    with torch.no_grad():
        for batch_features, batch_aGGObs, batch_labels in test_dataloader:
            batch_features = batch_features.to(device)
            batch_aGGObs = batch_aGGObs.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(batch_features, batch_aGGObs)
            probs = torch.sigmoid(logits).squeeze()
            if probs.dim() == 0:
                probs = probs.unsqueeze(0)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    #neg_samples = np.sum(all_labels)
    return roc_auc_score(all_labels, all_probs)


def evaluate_model(model, test_dataloader, device):
    model.eval()
    all_probs = []  # AUC-ROC
    all_labels = []
    with torch.no_grad():
        for batch_features, batch_labels in test_dataloader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(batch_features)
            probs = torch.sigmoid(logits).squeeze()
            if probs.dim() == 0:
                probs = probs.unsqueeze(0)
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


def evaluate_model_aGGObs(model, test_dataloader, device):
    model.eval()
    all_probs = []  # AUC-ROC
    all_labels = []
    with torch.no_grad():
        for batch_features, batch_aGGObs, batch_labels in test_dataloader:
            batch_features = batch_features.to(device)
            batch_aGGObs = batch_aGGObs.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(batch_features, batch_aGGObs)
            probs = torch.sigmoid(logits).squeeze()
            if probs.dim() == 0:
                probs = probs.unsqueeze(0)
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



def train_model(model, train_dataloader, val_dataloader, num_epochs, cw_type, device, test_dataloader=None):
    for batch_features, batch_labels in train_dataloader:
        print(f"Features: {batch_features.shape}")
        print(f"Labels: {batch_labels.shape}")
        break
    '''
    # compute class weights
    labels_list = []
    for _, labels in train_dataloader:
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
    '''
    start_time = time.time()
    class_weights_dict = compute_class_weights(train_dataloader, cw_type)
    total_time = time.time() - start_time
    print('Compute cw time... : ', total_time)
    print('class_weights_dict: ', class_weights_dict)
    positive_class_weight = class_weights_dict[1]
    pos_weight_tensor = torch.tensor([positive_class_weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = Adam(model.parameters(), lr=0.001)
    best_auc = 0
    patience = 10
    counter_patience = 0
    train_losses = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_features, batch_labels in train_dataloader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device).float()
            optimizer.zero_grad()
            predictions = model(batch_features)
            loss = criterion(predictions, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        auc_val = evaluate_val_auc_model(model, val_dataloader, device)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_dataloader):.4f}, Val auc: {auc_val}")
        train_losses.append(total_loss / len(train_dataloader))
        if auc_val > best_auc:
            best_auc = auc_val
            counter_patience = 0
        else:
            print('Val auc does not improve...')
            counter_patience += 1
        if counter_patience == patience:
            break

    print("Train Done...")
    if test_dataloader != None:
        print('testing model at last epoch reached...')
        auc_test = evaluate_val_auc_model(model, test_dataloader, device)
        print('AUC test: ', auc_test)
    if counter_patience == patience:
        train_losses = train_losses[: -patience]
        final_epochs = len(train_losses)
    else:
        final_epochs = num_epochs
    return final_epochs


def compute_class_weights_balanced(train_dataloader):
    """
    Computes class weights efficiently without loading all data into memory.
    """

    positive_count = 0
    negative_count = 0

    first_batch = next(iter(train_dataloader))
    if len(first_batch) != 3:
        for _, labels in train_dataloader:
            labels_flat = labels.numpy().flatten().astype(int)
            positive_count += np.sum(labels_flat)
            negative_count += len(labels_flat) - np.sum(labels_flat)
    else:
        for _, _, labels in train_dataloader:
            labels_flat = labels.numpy().flatten().astype(int)
            positive_count += np.sum(labels_flat)
            negative_count += len(labels_flat) - np.sum(labels_flat)

    total_samples = positive_count + negative_count
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array([0, 1]),
        #y=np.array([0] * negative_count + [1] * positive_count)
        y=np.concatenate(([0] * negative_count, [1] * positive_count))
    )

    class_weights_dict = {0: class_weights[0], 1: class_weights[1]}

    print(f'Total samples analyzed: {total_samples}')
    print('Computed class weights:', class_weights_dict)

    return class_weights_dict



def compute_class_weights_onset(train_dataloader):
    """
    Computes class weights using only 'new' positive samples (where aggression indicator a == 0 and label y == 1)
    and treating all other samples as negatives.
    Assumes each batch from train_dataloader returns (data, aggression, labels) where 'agg' has a shape of
    (batch_size, Nbins) and 'labels' has a shape of (batch_size,).

    The strategy is to use the aggression indicator from the first bin for each sample.
    """
    positive_count = 0
    negative_count = 0

    for data, agg, labels in train_dataloader:
        labels_flat = labels.cpu().numpy().flatten().astype(int)
        agg_array = agg.cpu().numpy().reshape(labels.size(0), -1).astype(int)
        agg_sample = agg_array[:, 0]

        new_positive = np.sum((labels_flat == 1) & (agg_sample == 0))
        batch_size = len(labels_flat)
        positive_count += new_positive
        negative_count += (batch_size - new_positive)

    total_samples = positive_count + negative_count

    y_combined = np.concatenate(([0] * negative_count, [1] * positive_count))

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array([0, 1]),
        y=y_combined
    )
    class_weights_dict = {0: class_weights[0], 1: class_weights[1]}

    print(f'Total samples analyzed: {total_samples}')
    print('Computed class weights (applied only to new positives):', class_weights_dict)

    return class_weights_dict


def compute_class_weights(train_dataloader, cw_type):
    print('cw_type: ', cw_type)
    if cw_type == 0 or cw_type == 1:
        class_weights_dict = compute_class_weights_balanced(train_dataloader)
    elif cw_type == 2: ### solo con model_version >= 2, sino falla...
        class_weights_dict = compute_class_weights_onset(train_dataloader)
    return class_weights_dict


def train_model_AGGObs_prev(model, train_dataloader, val_dataloader, num_epochs, cw_type, device, test_dataloader=None):
    for batch_features, batch_aGGOBs, batch_labels in train_dataloader:
        print(f"Features: {batch_features.shape}")
        print(f"aGGObsr: {batch_aGGOBs.shape}")
        print(f"Labels: {batch_labels.shape}")
        break
    '''
    # compute class weights
    labels_list = []
    for _, _, labels in train_dataloader:
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
    '''
    print(model)
    print(model.eeg_feature_dim)
    start_time = time.time()
    class_weights_dict = compute_class_weights(train_dataloader, cw_type)
    total_time = time.time() - start_time
    print('Compute cw time... : ', total_time)
    print('class_weights_dict: ', class_weights_dict)
    positive_class_weight = class_weights_dict[1]
    pos_weight_tensor = torch.tensor([positive_class_weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = Adam(model.parameters(), lr=0.001)
    best_auc = 0
    patience = 5 #### antes 10!
    counter_patience = 0
    train_losses = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_features, batch_aGGObs, batch_labels in train_dataloader:
            batch_features = batch_features.to(device)
            batch_aGGObs = batch_aGGObs.to(device)
            batch_labels = batch_labels.to(device).float()
            optimizer.zero_grad()
            predictions = model(batch_features, batch_aGGObs)
            loss = criterion(predictions, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        auc_val = evaluate_val_auc_model_aGGObs(model, val_dataloader, device)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_dataloader):.4f}, Val auc: {auc_val}")
        train_losses.append(total_loss / len(train_dataloader))
        if auc_val > best_auc:
            best_auc = auc_val
            counter_patience = 0
        else:
            print('Val auc does not improve...')
            counter_patience += 1
        if counter_patience == patience:
            break

    print("Train Done...")
    if test_dataloader != None:
        print('testing model at last epoch reached...')
        auc_test = evaluate_val_auc_model_aGGObs(model, test_dataloader, device)
        print('AUC test: ', auc_test)
    if counter_patience == patience:
        train_losses = train_losses[: -patience]
        final_epochs = len(train_losses)
    else:
        final_epochs = num_epochs
    return final_epochs


def train_model_AGGObs(model, train_dataloader, val_dataloader, num_epochs, cw_type, device, test_dataloader=None):
    for batch_features, batch_aGGOBs, batch_labels in train_dataloader:
        print(f"Features: {batch_features.shape}")
        print(f"aGGObsr: {batch_aGGOBs.shape}")
        print(f"Labels: {batch_labels.shape}")
        break
    print(model)
    print(model.eeg_feature_dim)
    start_time = time.time()
    class_weights_dict = compute_class_weights(train_dataloader, cw_type)
    total_time = time.time() - start_time
    print('Compute cw time... : ', total_time)
    print('class_weights_dict: ', class_weights_dict)
    positive_class_weight = class_weights_dict[1]
    pos_weight_tensor = torch.tensor([positive_class_weight]).to(device)

    if cw_type <= 1:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    else:
        criterion = nn.BCEWithLogitsLoss(reduction='none')

    optimizer = Adam(model.parameters(), lr=0.001)
    best_auc = 0
    patience = 5 #### antes 10!
    counter_patience = 0
    train_losses = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_features, batch_aGGObs, batch_labels in train_dataloader:
            batch_features = batch_features.to(device)
            batch_aGGObs = batch_aGGObs.to(device)
            batch_labels = batch_labels.to(device).float()
            optimizer.zero_grad()
            predictions = model(batch_features, batch_aGGObs)
            if cw_type > 1:
                loss_per_sample = criterion(predictions, batch_labels)
                agg_sample = batch_aGGObs[:, -1]  # use the aggression indicator from the last bin for each sample
                weights = torch.ones_like(batch_labels)
                weights[(batch_labels == 1) & (agg_sample == 0)] = positive_class_weight
                loss = (loss_per_sample * weights).mean()
            else:
                loss = criterion(predictions, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        auc_val = evaluate_val_auc_model_aGGObs(model, val_dataloader, device)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_dataloader):.4f}, Val auc: {auc_val}")
        train_losses.append(total_loss / len(train_dataloader))
        if auc_val > best_auc:
            best_auc = auc_val
            counter_patience = 0
        else:
            print('Val auc does not improve...')
            counter_patience += 1
        if counter_patience == patience:
            break

    print("Train Done...")
    if test_dataloader != None:
        print('testing model at last epoch reached...')
        auc_test = evaluate_val_auc_model_aGGObs(model, test_dataloader, device)
        print('AUC test: ', auc_test)
    if counter_patience == patience:
        train_losses = train_losses[: -patience]
        final_epochs = len(train_losses)
    else:
        final_epochs = num_epochs
    return final_epochs



def invalid_data(dataloader):
    labels_list = []
    for _, labels in dataloader:
        labels_flat = labels.numpy().flatten()
        labels_list.extend(labels_flat)
    labels_array = np.array(labels_list)
    print(f'Total samples: {len(labels_array)}, 0: {len(labels_array) - np.sum(labels_array)}, 1:{np.sum(labels_array)}')
    return (len(np.unique(labels_array)) == 1)

def invalid_data_PDM(dataloader):
    labels_list = []
    for batch in dataloader:
        labels = batch[-1]
        labels_flat = labels.numpy().flatten()
        labels_list.extend(labels_flat)
    labels_array = np.array(labels_list)
    print(f'Total samples: {len(labels_array)}, 0: {len(labels_array) - np.sum(labels_array)}, 1:{np.sum(labels_array)}')
    #neg_samples = np.sum(labels_array)
    #classes = np.unique(labels_array)
    return (len(np.unique(labels_array)) != 2)


def retrain_model(model, train_dataloader, test_dataloader, num_epochs, cw_type, device, exp_name, path_model):
    print('Retraining model...')
    '''
    labels_list = []
    for _, labels in train_dataloader:
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
    '''
    start_time = time.time()
    class_weights_dict = compute_class_weights(train_dataloader, cw_type)
    total_time = time.time() - start_time
    print('Compute cw time... : ', total_time)
    print('class_weights_dict: ', class_weights_dict)
    positive_class_weight = class_weights_dict[1]
    pos_weight_tensor = torch.tensor([positive_class_weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = Adam(model.parameters(), lr=0.001)
    train_losses = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_features, batch_labels in train_dataloader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device).float()
            optimizer.zero_grad()
            predictions = model(batch_features)
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
        "Fold": exp_name,
        "F1-Score": f1_s,
        'Best_F1-score': best_f1,
        'Best_th': best_th,
        "AUC-ROC": auc_roc_s,
        "Num_epochs": num_epochs
    }
    return results


def retrain_model_aGGObs(model, train_dataloader, test_dataloader, num_epochs, cw_type, device, exp_name, path_model):
    print('Retraining model...')
    for batch_features, batch_aGGOBs, batch_labels in train_dataloader:
        print(f"Features: {batch_features.shape}")
        print(f"aGGObsr: {batch_aGGOBs.shape}")
        print(f"Labels: {batch_labels.shape}")
        break
    print(model)
    print(model.eeg_feature_dim)
    start_time = time.time()
    class_weights_dict = compute_class_weights(train_dataloader, cw_type)
    total_time = time.time() - start_time
    print('Compute cw time... : ', total_time)
    print('class_weights_dict: ', class_weights_dict)
    positive_class_weight = class_weights_dict[1]
    pos_weight_tensor = torch.tensor([positive_class_weight]).to(device)

    if cw_type <= 1:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    else:
        criterion = nn.BCEWithLogitsLoss(reduction='none')

    optimizer = Adam(model.parameters(), lr=0.001)
    train_losses = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_features, batch_aGGObs, batch_labels in train_dataloader:
            batch_features = batch_features.to(device)
            batch_aGGObs = batch_aGGObs.to(device)
            batch_labels = batch_labels.to(device).float()
            optimizer.zero_grad()
            predictions = model(batch_features, batch_aGGObs)

            if cw_type > 1:
                loss_per_sample = criterion(predictions, batch_labels)
                agg_sample = batch_aGGObs[:, -1]  # use the aggression indicator from the last bin for each sample
                weights = torch.ones_like(batch_labels)
                weights[(batch_labels == 1) & (agg_sample == 0)] = positive_class_weight
                loss = (loss_per_sample * weights).mean()
            else:
                loss = criterion(predictions, batch_labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_dataloader):.4f}")
        train_losses.append(total_loss / len(train_dataloader))

    torch.save(model.state_dict(), path_model)
    f1_s, best_th, best_f1, auc_roc_s = evaluate_model_aGGObs(model, test_dataloader, device)
    print(f'Test results: f1_s: {f1_s}, best_th: {best_th}, best_f1: {best_f1}, auc: {auc_roc_s}.')
    print('')
    results = {
        "Fold": exp_name,
        "F1-Score": f1_s,
        'Best_F1-score': best_f1,
        'Best_th': best_th,
        "AUC-ROC": auc_roc_s,
        "Num_epochs": num_epochs
    }
    return results


def test_trained_model_aGGObs(model, test_dataloader, num_epochs, device, exp_name):
    print('Testing trained model...')
    f1_s, best_th, best_f1, auc_roc_s = evaluate_model_aGGObs(model, test_dataloader, device)
    print(f'Test results: f1_s: {f1_s}, best_th: {best_th}, best_f1: {best_f1}, auc: {auc_roc_s}.')
    print('')
    results = {
        "Fold": exp_name,
        "F1-Score": f1_s,
        'Best_F1-score': best_f1,
        'Best_th': best_th,
        "AUC-ROC": auc_roc_s,
        "Num_epochs": num_epochs
    }
    return results


def set_model(model_code):
    if model_code == 0:
        model_fun = models.EEGNetLSTM
        features_fun = data_utils.get_features_from_dict_bins
        dataloader_fun = data_utils.AggressiveBehaviorDatasetBin
        train_fun = train_model
        retrain_fun = retrain_model
    elif model_code == 1:
        model_fun = models.New_EEGNetLSTM
        features_fun = data_utils.get_features_from_dict_bins
        dataloader_fun = data_utils.New_AggressiveBehaviorDatasetBin
        ### label only as positive onset: New_AggressiveBehaviorDatasetBin_only_onset
        train_fun = train_model
        retrain_fun = retrain_model
    elif model_code == 2:
        model_fun = models.New_EEGNetLSTM_AGGObsr
        features_fun = data_utils.get_features_from_dict_bins
        dataloader_fun = data_utils.New_AggressiveBehaviorDatasetBin_AGGObserved #data_utils.New_AggressiveBehaviorDatasetBin_AGGObserved
        # No optimizado: data_utils.New_AggressiveBehaviorDatasetBin_AGGObserved
        # Optimizado par menores bin sizes: data_utils.New_AggressiveBehaviorDatasetBin_AGGObserved_optimized
        # label only as positive onset: New_AggressiveBehaviorDatasetBin_AGGObserved_only_onset
        train_fun = train_model_AGGObs
        retrain_fun = retrain_model_aGGObs
    elif model_code == 3: ### to-do: unificar, ahora quick testing...
        model_fun = models.New_EEGNetLSTM_AGGObsr_Simple_1
        features_fun = data_utils.get_features_from_dict_bins
        dataloader_fun = data_utils.New_AggressiveBehaviorDatasetBin_AGGObserved  # data_utils.New_AggressiveBehaviorDatasetBin_AGGObserved
        # No optimizado: data_utils.New_AggressiveBehaviorDatasetBin_AGGObserved
        # Optimizado par menores bin sizes: data_utils.New_AggressiveBehaviorDatasetBin_AGGObserved_optimized
        # label only as positive onset: New_AggressiveBehaviorDatasetBin_AGGObserved_only_onset
        train_fun = train_model_AGGObs
        retrain_fun = retrain_model_aGGObs
    elif model_code == 4: ### to-do: unificar, ahora quick testing...
        model_fun = models.New_EEGNetLSTM_AGGObsr_Simple_2
        features_fun = data_utils.get_features_from_dict_bins
        dataloader_fun = data_utils.New_AggressiveBehaviorDatasetBin_AGGObserved  # data_utils.New_AggressiveBehaviorDatasetBin_AGGObserved
        # No optimizado: data_utils.New_AggressiveBehaviorDatasetBin_AGGObserved
        # Optimizado par menores bin sizes: data_utils.New_AggressiveBehaviorDatasetBin_AGGObserved_optimized
        # label only as positive onset: New_AggressiveBehaviorDatasetBin_AGGObserved_only_onset
        train_fun = train_model_AGGObs
        retrain_fun = retrain_model_aGGObs
    elif model_code == 6:
        model_fun = models.New_EEGNetLSTM
        features_fun = data_utils.get_features_from_dict_bins
        dataloader_fun = data_utils.New_AggressiveBehaviorDatasetBin_only_onset
        ### label only as positive onset: New_AggressiveBehaviorDatasetBin_only_onset
        train_fun = train_model
        retrain_fun = retrain_model
    elif model_code == 5: ### to-do: unificar, ahora quick testing... ONLY ONSET SAMPLES AS POSITIVES!!!!!
        model_fun = models.New_EEGNetLSTM_AGGObsr
        features_fun = data_utils.get_features_from_dict_bins
        dataloader_fun = data_utils.New_AggressiveBehaviorDatasetBin_AGGObserved_only_onset  # data_utils.New_AggressiveBehaviorDatasetBin_AGGObserved
        # No optimizado: data_utils.New_AggressiveBehaviorDatasetBin_AGGObserved
        # Optimizado par menores bin sizes: data_utils.New_AggressiveBehaviorDatasetBin_AGGObserved_optimized
        # label only as positive onset: New_AggressiveBehaviorDatasetBin_AGGObserved_only_onset
        train_fun = train_model_AGGObs
        retrain_fun = retrain_model_aGGObs
    else:
        print('Not supported yet.')
        model_fun, features_fun, dataloader_fun, train_fun, retrain_fun = None, None, None, None, None
    return model_fun, features_fun, dataloader_fun, train_fun, retrain_fun


def create_dataloader(dataloader_fun, output_dict, tp, tf, bin_size, batch_size, shuffle=False):
    dataset = dataloader_fun(output_dict, tp, tf, bin_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    return dataloader


def generate_user_kfolds(data_dict, k=5):
    uids = list(data_dict.keys())
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    folds = []
    for train_idx, test_idx in kf.split(uids):
        train_uids = [uids[i] for i in train_idx]
        test_uids = [uids[i] for i in test_idx]
        folds.append((train_uids, test_uids))
    return folds


def get_split_fun(split_code, PDM=False):
    if split_code == 0:
        if PDM:
            split_fun = data_utils.split_data_per_session
        else:
            split_fun = data_utils.new_split_data_per_session
    elif split_code == 1:
        split_fun = data_utils.new_split_data_full_session
    else:
        split_fun = None
        print('Split function not supported...')
    return split_fun


def get_partitions_from_fold(data_dict, train_uids, test_uids, split_code, PDM, seed):
    split_fun = get_split_fun(split_code, PDM)
    # obtener diccionario de los usuarios de train
    train_dict = {uid: data_dict[uid] for uid in train_uids}
    # obtener diccionario de los usuarios de test
    test_dict = {uid: data_dict[uid] for uid in test_uids}
    # Dividir los usuarios de entrenamiento en entrenamiento y validación
    train_uids_split, val_uids_split = train_test_split(train_uids, test_size=0.2, random_state=seed)
    train_dict_split = {uid: train_dict[uid] for uid in train_uids_split}
    val_dict_split = {uid: train_dict[uid] for uid in val_uids_split}
    print(f"  Train UIDs (después de apartar validación): {train_uids_split}")
    print(f"  Val UIDs: {val_uids_split}")
    # Dividir diccionario de usuarios de val, 80% inicial de la sesion a train y el 20% final a val
    train_dict_val, test_dict_val = split_fun(val_dict_split, train_ratio=0.8)
    # Dividir diccionario de usuarios de test, 80% inicial de la sesion a train y el 20% final a test
    train_dict_test, test_dict_test = split_fun(test_dict, train_ratio=0.8)
    # Añadir al diccionario de train los datos de train_dict_test y train_dict_val
    for uid, data in {**train_dict_test, **train_dict_val}.items():
        if uid not in train_dict_split:
            train_dict_split[uid] = {}
        train_dict_split[uid].update(data)
    print(f"  Train UIDs (final): {list(train_dict_split.keys())}")
    # Añadir al diccionario de train (con el conjunto de val incluido) los datos de train_dict_test
    for uid, data in train_dict_test.items():
        if uid not in train_dict:
            train_dict[uid] = {}
        train_dict[uid].update(data)
    print(f"  Train UIDs (final): {list(train_dict_split.keys())}")
    # todos los usuarios en el conjunto de train, pero los usuarios del conjunto de test/validacion
    # solo tienen el 80% inicial de sus sesiones
    return train_dict_split, test_dict_val, test_dict_test, train_dict


def set_features(feats_code, model_fun):
    if feats_code == 0: # All,: ACC x, y, z, BVP, EDA, AGGObs
        EEG_channels = 5
        load_data_fun = data_utils.load_data_to_dict
    elif feats_code == 1: # All, ACC Norm
        EEG_channels = 3
        load_data_fun = data_utils.load_data_to_dict_ACCNorm
    elif feats_code == 2: # Solo aGGObsr
        EEG_channels = None
        load_data_fun = data_utils.load_data_to_dict_ACCNorm # da igual cual, porque solo se usa aggObs, pero cuantos menos datos mejor
        model_fun = models.New_LSTM_AGGObsr
    elif feats_code == 3: # No ACC
        EEG_channels = 2
        load_data_fun = data_utils.load_data_to_dict_noACC
    elif feats_code == 4: # No BVP
        EEG_channels = 4
        load_data_fun = data_utils.load_data_to_dict_noBVP
    elif feats_code == 5: # No EDA
        EEG_channels = 4
        load_data_fun = data_utils.load_data_to_dict_noEDA
    elif feats_code == 6: # Only BVP
        EEG_channels = 1
        load_data_fun = data_utils.load_data_to_dict_BVP
    elif feats_code == 7: # Only ACC
        EEG_channels = 3
        load_data_fun = data_utils.load_data_to_dict_ACC
    elif feats_code == 8:  # Only EDA
        EEG_channels = 1
        load_data_fun = data_utils.load_data_to_dict_EDA
    return load_data_fun, model_fun, EEG_channels


def start_exps_PM(tp, tf, freq, data_path_resampled, path_results, path_models, model_code, feats_code, split_code, b_size, cw_type, seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_fun, features_fun, dataloader_fun, train_fun, retrain_fun = set_model(model_code)
    load_data_fun, model_fun, EEG_channels = set_features(feats_code, model_fun)
    ds_path = data_path_resampled + "dataset_" + str(freq) + "Hz.csv"
    data_dict = load_data_fun(ds_path)
    #data_dict = dict(list(data_dict.items())[:20])
    num_folds = 5
    folds = generate_user_kfolds(data_dict, k=num_folds)
    tp, tf, stride, bin_size = tp, tf, b_size, b_size
    print(f'tp: {tp}, tf: {tf}, stride: {stride}, bin_size: {bin_size}.')
    final_results = []
    for fold_idx, (train_uids, test_uids) in enumerate(folds):
        print(f"Fold {fold_idx + 1}:")
        print(f"  Train UIDs: {train_uids}")
        print(f"  Test UIDs: {test_uids}")
        # Obtener particiones de train, val y test en funcion de los ids del fold
        train_dict, val_dict, test_dict, all_train_dict = get_partitions_from_fold(data_dict, train_uids, test_uids, split_code, False, seed)
        # Dividir las sesiones de cada usuario en bins de 15 segundos y almacenar raw signals y etiquetas
        # (variable binaria que indica la ocurrencia de un episodio agresivo en el bin)
        train_data = features_fun(train_dict, tp, tf, bin_size, freq)
        val_data = features_fun(val_dict, tp, tf, bin_size, freq)
        test_data = features_fun(test_dict, tp, tf, bin_size, freq)

        batch_size = 128
        dataloader = create_dataloader(dataloader_fun, train_data, tp, tf, bin_size, batch_size, shuffle=True)
        dataloader_val = create_dataloader(dataloader_fun, val_data, tp, tf, bin_size, batch_size, shuffle=False)
        dataloader_test = create_dataloader(dataloader_fun, test_data, tp, tf, bin_size, batch_size, shuffle=False)

        # create model
        num_sequences = tp // bin_size
        eegnet_params = {
            'num_electrodes': EEG_channels,  # (EDA, ACC_X, ACC_Y, ACC_Z, BVP) | (EDA, ACC_Norm, BVP) | ....
            'chunk_size': tp * freq // num_sequences  # muestras en cada ventana
        }
        lstm_hidden_dim = 64
        num_classes = 1
        model = model_fun(eegnet_params, lstm_hidden_dim, num_classes).to(device)
        num_epochs = 100

        best_num_epochs = train_fun(model, dataloader, dataloader_val, num_epochs, cw_type, device, test_dataloader=dataloader_test)

        final_model = model_fun(eegnet_params, lstm_hidden_dim, num_classes).to(device)
        all_train_data = features_fun(all_train_dict, tp, tf, bin_size, freq)
        dataloader_final = create_dataloader(dataloader_fun, all_train_data, tp, tf, bin_size, batch_size, shuffle=True)

        path_model = f"{path_models}mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_sc{split_code}_cw{cw_type}_fold{fold_idx}_model.pth"
        results = retrain_fun(final_model, dataloader_final, dataloader_test, best_num_epochs, cw_type, device, fold_idx,
                              path_model)
        final_results.append(results)

    results_df = pd.DataFrame(final_results)
    avg_metrics = results_df[['F1-Score', 'Best_F1-score', 'Best_th', 'AUC-ROC', 'Num_epochs']].mean()
    std_metrics = results_df[['F1-Score', 'Best_F1-score', 'Best_th', 'AUC-ROC', 'Num_epochs']].std()
    summary_df = pd.DataFrame({
        "Fold": ["Avg.", "Std."],
        'F1-Score': [avg_metrics['F1-Score'], std_metrics['F1-Score']],
        'Best_F1-score': [avg_metrics['Best_F1-score'], std_metrics['Best_F1-score']],
        'Best_th': [avg_metrics['Best_th'], std_metrics['Best_th']],
        'AUC-ROC': [avg_metrics['AUC-ROC'], std_metrics['AUC-ROC']],
        'Num_epochs': [avg_metrics['Num_epochs'], std_metrics['Num_epochs']]
    })
    final_results_df = pd.concat([results_df, summary_df], ignore_index=True)
    path_to_save_results = f'{path_results}PM_SS_mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_sc{split_code}_cw{cw_type}_all_experiments_results_5cv.csv'
    final_results_df.to_csv(path_to_save_results, index=False)
    print("Results saved successfully.")


def start_exps_PDM(tp, tf, freq, data_path_resampled, path_results, path_models, model_code, feats_code, split_code, b_size, cw_type, seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_fun, features_fun, dataloader_fun, train_fun, retrain_fun = set_model(model_code)
    load_data_fun, model_fun, EEG_channels = set_features(feats_code, model_fun)
    split_fun = get_split_fun(split_code, True)
    ds_path = data_path_resampled + "dataset_" + str(freq) + "Hz.csv"
    data_dict = load_data_fun(ds_path)
    #data_dict = dict(list(data_dict.items())[:2])
    ##############################################
    num_exps = 10 # TO-DO: configurar particones para cada exp!
    tp, tf, stride, bin_size = tp, tf, b_size, b_size
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
        dataloader = create_dataloader(dataloader_fun, train_dict, tp, tf, bin_size, batch_size, shuffle=True)
        dataloader_val = create_dataloader(dataloader_fun, val_dict, tp, tf, bin_size, batch_size, shuffle=False)
        dataloader_test = create_dataloader(dataloader_fun, test_dict, tp, tf, bin_size, batch_size, shuffle=False)
        dataloader_final = create_dataloader(dataloader_fun, train_dict_, tp, tf, bin_size, batch_size, shuffle=True)

        if invalid_data_PDM(dataloader) or invalid_data_PDM(dataloader_test) or invalid_data_PDM(dataloader_val) or invalid_data_PDM(dataloader_final):
            invalid_users.append(key_subject)
        else:
            # create model
            num_sequences = tp // bin_size  # tp//bin_size en EEGNetLSTMwinLabelsSeqs_binS
            eegnet_params = {
                'num_electrodes': EEG_channels,  # (EDA, ACC_X, ACC_Y, ACC_Z, BVP) | (EDA, ACC_Norm, BVP) | ....
                'chunk_size': tp * freq // num_sequences  # muestras en cada ventana
            }
            lstm_hidden_dim = 64
            num_classes = 1
            model = model_fun(eegnet_params, lstm_hidden_dim, num_classes).to(device)
            num_epochs = 100

            best_num_epochs = train_fun(model, dataloader, dataloader_val, num_epochs, cw_type, device)

            final_model = model_fun(eegnet_params, lstm_hidden_dim, num_classes).to(device)
            #dataloader_final = create_dataloader(dataloader_fun, train_dict_, tp, bin_size, batch_size, shuffle=True)

            path_model = f"{path_models}mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_sc{split_code}_cw{cw_type}_subj{key_subject}_model.pth"
            results = retrain_fun(final_model, dataloader_final, dataloader_test, best_num_epochs, cw_type,
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
    path_to_save_results = f'{path_results}PDM_mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_sc{split_code}_cw{cw_type}_all_experiments_results.csv'
    final_results_df.to_csv(path_to_save_results, index=False)
    print("Results saved successfully.")


def pretrain_model_for_HM(tp, tf, freq, data_dict, path_results, path_models, model_code, feats_code, split_code, b_size, cw_type, seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_fun, features_fun, dataloader_fun, train_fun, retrain_fun = set_model(model_code)
    load_data_fun, model_fun, EEG_channels = set_features(feats_code, model_fun)
    tp, tf, stride, bin_size = tp, tf, b_size, b_size
    print(f'tp: {tp}, tf: {tf}, stride: {stride}, bin_size: {bin_size}.')
    split_fun = get_split_fun(split_code, True)
    output_dict = features_fun(data_dict, tf, tp, bin_size, freq)
    train_dict_, test_dict = split_fun(output_dict, train_ratio=0.8)
    train_dict, val_dict = split_fun(train_dict_, train_ratio=0.8)
    batch_size = 128
    dataloader = create_dataloader(dataloader_fun, train_dict, tp, tf, bin_size, batch_size, shuffle=True)
    dataloader_val = create_dataloader(dataloader_fun, val_dict, tp, tf, bin_size, batch_size, shuffle=False)
    dataloader_test = create_dataloader(dataloader_fun, test_dict, tp, tf, bin_size, batch_size, shuffle=False)
    dataloader_final = create_dataloader(dataloader_fun, train_dict_, tp, tf, bin_size, batch_size, shuffle=True)
    # create model
    num_sequences = tp // bin_size
    eegnet_params = {
        'num_electrodes': EEG_channels,  # (EDA, ACC_X, ACC_Y, ACC_Z, BVP) | (EDA, ACC_Norm, BVP) | ....
        'chunk_size': tp * freq // num_sequences  # muestras en cada ventana
    }
    lstm_hidden_dim = 64
    num_classes = 1
    model = model_fun(eegnet_params, lstm_hidden_dim, num_classes).to(device)
    num_epochs = 100
    best_num_epochs = train_fun(model, dataloader, dataloader_val, num_epochs, cw_type, device, test_dataloader=dataloader_test)
    final_model = model_fun(eegnet_params, lstm_hidden_dim, num_classes).to(device)
    path_model = f"{path_models}temp_model.pth"
    _ = retrain_fun(final_model, dataloader_final, dataloader_test, best_num_epochs, cw_type, device, 0, path_model)
    return path_model


def start_exps_HM(tp, tf, freq, data_path_resampled, path_results, path_models, model_code, feats_code, split_code, b_size, cw_type, seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_fun, features_fun, dataloader_fun, train_fun, retrain_fun = set_model(model_code)
    load_data_fun, model_fun, EEG_channels = set_features(feats_code, model_fun)
    split_fun = get_split_fun(split_code, True)
    ds_path = data_path_resampled + "dataset_" + str(freq) + "Hz.csv"
    data_dict = load_data_fun(ds_path)
    pretrain_data_dict = load_data_fun(ds_path)
    tp, tf, stride, bin_size = tp, tf, b_size, b_size
    print(f'tp: {tp}, tf: {tf}, stride: {stride}, bin_size: {bin_size}.')
    final_results = []
    num_users = len(list(data_dict.items()))
    invalid_users = []
    for u in range(0, num_users):
        first_item = dict(list(data_dict.items())[:1])
        key_subject = list(first_item.keys())[0]
        print('userID: ', key_subject)
        ### Target set: key_subject data
        output_dict = features_fun(first_item, tf, tp, bin_size, freq)
        train_dict_, test_dict = split_fun(output_dict, train_ratio=0.8)
        train_dict, val_dict = split_fun(train_dict_, train_ratio=0.8)
        batch_size = 16
        dataloader = create_dataloader(dataloader_fun, train_dict, tp, tf, bin_size, batch_size, shuffle=True)
        dataloader_val = create_dataloader(dataloader_fun, val_dict, tp, tf, bin_size, batch_size, shuffle=False)
        dataloader_test = create_dataloader(dataloader_fun, test_dict, tp, tf, bin_size, batch_size, shuffle=False)
        dataloader_final = create_dataloader(dataloader_fun, train_dict_, tp, tf, bin_size, batch_size, shuffle=True)
        if invalid_data_PDM(dataloader) or invalid_data_PDM(dataloader_test) or invalid_data_PDM(dataloader_val) or invalid_data_PDM(dataloader_final):
            invalid_users.append(key_subject)
        else:
            try:
                # source_set = coger todos los datos de los usuarios menos los suyos de pretrain_data_dict
                source_data_dict = {k: v for k, v in pretrain_data_dict.items() if k != key_subject}
                source_data_dict_ids = [k for k, v in pretrain_data_dict.items() if k != key_subject] # test
                print('source_data_dict subjects: ', source_data_dict_ids)
                # entrenar modelo con source
                path_pretrained_model = pretrain_model_for_HM(tp, tf, freq, source_data_dict, path_results, path_models,
                                                              model_code, feats_code, split_code,
                                                              b_size, cw_type, seed=1)
                # load pretrained model (de path_pretrained_model), importante no en eval()
                num_sequences = tp // bin_size  # tp//bin_size en EEGNetLSTMwinLabelsSeqs_binS
                eegnet_params = {
                    'num_electrodes': EEG_channels,  # (EDA, ACC_X, ACC_Y, ACC_Z, BVP) | (EDA, ACC_Norm, BVP) | ....
                    'chunk_size': tp * freq // num_sequences  # muestras en cada ventana
                }
                lstm_hidden_dim = 64
                num_classes = 1
                model = model_fun(eegnet_params, lstm_hidden_dim, num_classes).to(device)
                model.load_state_dict(torch.load(path_pretrained_model, map_location=device, weights_only=True))
                num_epochs = 100
                best_num_epochs = train_fun(model, dataloader, dataloader_val, num_epochs, cw_type, device)
                final_model = model_fun(eegnet_params, lstm_hidden_dim, num_classes).to(device)
                final_model.load_state_dict(torch.load(path_pretrained_model, map_location=device, weights_only=True))
                path_model = f"{path_models}mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_sc{split_code}_subj{key_subject}_cw{cw_type}_hybrid_model.pth"
                results = retrain_fun(final_model, dataloader_final, dataloader_test, best_num_epochs, cw_type,
                                      device, key_subject, path_model)
                final_results.append(results)
                results_df_temp = pd.DataFrame(final_results)
                path_to_save_results_temp = f'{path_results}HM_mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_sc{split_code}_cw{cw_type}_all_experiments_results_temp.csv'
                results_df_temp.to_csv(path_to_save_results_temp, index=False)
            except Exception as e:
                print(f"❌ Error con usuario {key_subject}: {e}")
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
    path_to_save_results = f'{path_results}HM_mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_sc{split_code}_cw{cw_type}_all_experiments_results.csv'
    final_results_df.to_csv(path_to_save_results, index=False)
    print("Results saved successfully.")


def start_exps_PM_v2(tp, tf, freq, data_path_resampled, path_results, path_models, model_code, feats_code, split_code, b_size, cw_type, seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_fun, features_fun, dataloader_fun, train_fun, retrain_fun = set_model(model_code)
    load_data_fun, model_fun, EEG_channels = set_features(feats_code, model_fun)
    split_fun = get_split_fun(split_code, True)
    ds_path = data_path_resampled + "dataset_" + str(freq) + "Hz.csv"
    data_dict = load_data_fun(ds_path)
    pretrain_data_dict = load_data_fun(ds_path)
    tp, tf, stride, bin_size = tp, tf, b_size, b_size
    print(f'tp: {tp}, tf: {tf}, stride: {stride}, bin_size: {bin_size}.')
    final_results = []
    num_users = len(list(data_dict.items()))
    invalid_users = []
    for u in range(0, num_users):
        first_item = dict(list(data_dict.items())[:1])
        key_subject = list(first_item.keys())[0]
        print('userID: ', key_subject)
        ### Target set: key_subject data
        output_dict = features_fun(first_item, tf, tp, bin_size, freq)
        train_dict_, test_dict = split_fun(output_dict, train_ratio=0.8)
        train_dict, val_dict = split_fun(train_dict_, train_ratio=0.8)
        batch_size = 16
        dataloader = create_dataloader(dataloader_fun, train_dict, tp, tf, bin_size, batch_size, shuffle=True)
        dataloader_val = create_dataloader(dataloader_fun, val_dict, tp, tf, bin_size, batch_size, shuffle=False)
        dataloader_test = create_dataloader(dataloader_fun, test_dict, tp, tf, bin_size, batch_size, shuffle=False)
        dataloader_final = create_dataloader(dataloader_fun, train_dict_, tp, tf, bin_size, batch_size, shuffle=True)
        if invalid_data_PDM(dataloader) or invalid_data_PDM(dataloader_test) or invalid_data_PDM(dataloader_val) or invalid_data_PDM(dataloader_final):
            invalid_users.append(key_subject)
        else:
            try:
                # source_set = coger todos los datos de los usuarios menos los suyos de pretrain_data_dict
                source_data_dict = {k: v for k, v in pretrain_data_dict.items() if k != key_subject}
                source_data_dict_ids = [k for k, v in pretrain_data_dict.items() if k != key_subject] # test
                print('source_data_dict subjects: ', source_data_dict_ids)
                # entrenar modelo con source
                path_pretrained_model = pretrain_model_for_HM(tp, tf, freq, source_data_dict, path_results, path_models,
                                                              model_code, feats_code, split_code,
                                                              b_size, cw_type, seed=1)
                # load pretrained model (de path_pretrained_model), importante no en eval()
                num_sequences = tp // bin_size  # tp//bin_size en EEGNetLSTMwinLabelsSeqs_binS
                eegnet_params = {
                    'num_electrodes': EEG_channels,  # (EDA, ACC_X, ACC_Y, ACC_Z, BVP) | (EDA, ACC_Norm, BVP) | ....
                    'chunk_size': tp * freq // num_sequences  # muestras en cada ventana
                }
                lstm_hidden_dim = 64
                num_classes = 1
                model = model_fun(eegnet_params, lstm_hidden_dim, num_classes).to(device)
                model.load_state_dict(torch.load(path_pretrained_model, map_location=device, weights_only=True))

                path_model = f"{path_models}mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_sc{split_code}_subj{key_subject}_cw{cw_type}_pretrained_model.pth"
                torch.save(model.state_dict(), path_model)
                results = test_trained_model_aGGObs(model, dataloader_test, 0, device, key_subject)
                final_results.append(results)
                results_df_temp = pd.DataFrame(final_results)
                path_to_save_results_temp = f'{path_results}PM_v2_mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_sc{split_code}_cw{cw_type}_all_experiments_results_temp.csv'
                results_df_temp.to_csv(path_to_save_results_temp, index=False)
            except Exception as e:
                print(f"❌ Error con usuario {key_subject}: {e}")
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
    path_to_save_results = f'{path_results}PM_v2_mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_sc{split_code}_cw{cw_type}_all_experiments_results.csv'
    final_results_df.to_csv(path_to_save_results, index=False)
    print("Results saved successfully.")


def set_model_multi(model_code):
    if model_code == 0:
        pass
    elif model_code == 1:
        model_fun = models.New_EEGNetLSTM
        features_fun = data_utils.get_features_from_dict_bins_multi
        dataloader_fun = data_utils.New_AggressiveBehaviorDatasetBin_multi
        train_fun = train_model
        retrain_fun = retrain_model
    elif model_code == 2:
        model_fun = models.New_EEGNetLSTM_AGGObsr
        features_fun = data_utils.get_features_from_dict_bins_multi
        dataloader_fun = data_utils.New_AggressiveBehaviorDatasetBin_AGGObserved_multi
        train_fun = train_model_AGGObs
        retrain_fun = retrain_model_aGGObs
    else:
        print('Not supported yet.')
        model_fun, features_fun, dataloader_fun, train_fun, retrain_fun = None, None, None, None, None
    return model_fun, features_fun, dataloader_fun, train_fun, retrain_fun


def set_features_multi(feats_code, model_fun):
    if feats_code == 0: # All,: ACC x, y, z, BVP, EDA, AGGObs
        EEG_channels = 5
        load_data_fun = data_utils.load_data_to_dict_multi
    else:
        print('Not supported yet... using all features.')
        EEG_channels = 5
        load_data_fun = data_utils.load_data_to_dict_multi
    return load_data_fun, model_fun, EEG_channels


def create_dataloader_multi(dataloader_fun, output_dict, label_key, tp, tf, bin_size, batch_size, shuffle=False):
    dataset = dataloader_fun(output_dict, label_key, tp, tf, bin_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    return dataloader


def evaluate_ensemble_results_multi(all_labels, probs_sib, probs_agg, probs_ed):
    # max. de probabilidades (soft voting)
    final_probs = np.maximum.reduce([probs_sib, probs_agg, probs_ed])
    auc_score = roc_auc_score(all_labels, final_probs)
    fpr, tpr, thresholds = roc_curve(all_labels, final_probs)

    best_threshold_auc_idx = np.argmax(tpr - fpr)
    best_threshold_auc = thresholds[best_threshold_auc_idx]

    best_f1, best_threshold_f1 = 0, 0
    for threshold in thresholds:
        predictions = (final_probs >= threshold).astype(int)
        f1 = f1_score(all_labels, predictions)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold_f1 = threshold

    return {
        "AUC-ROC": auc_score,
        "Best F1-Score": best_f1,
        "Best Threshold (AUC)": best_threshold_auc,
        "Best Threshold (F1)": best_threshold_f1
    }


def start_exps_PM_multi(tp, tf, freq, data_path_resampled, path_results, path_models, model_code, feats_code, split_code, b_size, cw_type, seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    PDM = False # variable de control para splits, pq diferente PM de PDM (cambiar en el futuro....)
    model_fun, features_fun, dataloader_fun, train_fun, retrain_fun = set_model_multi(model_code)
    load_data_fun, model_fun, EEG_channels = set_features_multi(feats_code, model_fun)

    ds_path = data_path_resampled + "dataset_" + str(freq) + "Hz.csv"
    data_dict = load_data_fun(ds_path)
    #data_dict = dict(list(data_dict.items())[:20])

    num_folds = 5
    folds = generate_user_kfolds(data_dict, k=num_folds)
    tp, tf, stride, bin_size = tp, tf, 15, 15
    print(f'tp: {tp}, tf: {tf}, stride: {stride}, bin_size: {bin_size}.')

    labels_list = ['SIB', 'AGG', 'ED']
    final_results = {"SIB": [], "AGG": [], "ED": []}
    ensemble_results_list = []
    for fold_idx, (train_uids, test_uids) in enumerate(folds):
        print(f"Fold {fold_idx + 1}:")
        print(f"  Train UIDs: {train_uids}")
        print(f"  Test UIDs: {test_uids}")

        # Obtener particiones de train, val y test en funcion de los ids del fold
        train_dict, val_dict, test_dict, all_train_dict = get_partitions_from_fold(data_dict, train_uids, test_uids, split_code, PDM, seed)
        # Dividir las sesiones de cada usuario en bins de 15 segundos y almacenar raw signals y etiquetas
        # (variable binaria que indica la ocurrencia de un episodio agresivo en el bin)
        train_data = features_fun(train_dict, bin_size, freq)
        val_data = features_fun(val_dict, bin_size, freq)
        test_data = features_fun(test_dict, bin_size, freq)
        batch_size = 128

        for label_type in labels_list:
            print(f"Training model for {label_type}")

            dataloader = create_dataloader_multi(dataloader_fun, train_data, label_type, tp, tf, bin_size, batch_size, shuffle=True)
            dataloader_val = create_dataloader_multi(dataloader_fun, val_data, label_type, tp, tf, bin_size, batch_size, shuffle=False)
            dataloader_test = create_dataloader_multi(dataloader_fun, test_data, label_type, tp, tf, bin_size, batch_size, shuffle=False)

            # create model
            num_sequences = tp // bin_size
            bin_size = 15
            eegnet_params = {
                'num_electrodes': EEG_channels,  # (EDA, ACC_X, ACC_Y, ACC_Z, BVP) | (EDA, ACC_Norm, BVP) | ....
                'chunk_size': tp * freq // num_sequences  # muestras en cada ventana
            }
            lstm_hidden_dim = 64
            num_classes = 1
            model = model_fun(eegnet_params, lstm_hidden_dim, num_classes).to(device)
            num_epochs = 100

            best_num_epochs = train_fun(model, dataloader, dataloader_val, num_epochs, cw_type, device)

            final_model = model_fun(eegnet_params, lstm_hidden_dim, num_classes).to(device)
            all_train_data = features_fun(all_train_dict, bin_size, freq)
            dataloader_final = create_dataloader_multi(dataloader_fun, all_train_data, label_type, tp, tf, bin_size, batch_size, shuffle=True)

            path_model = f"{path_models}mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_sc{split_code}_fold{fold_idx}_label{label_type}__cw{cw_type}_model.pth"
            print('eegnet params: ', eegnet_params)
            results = retrain_fun(final_model, dataloader_final, dataloader_test, best_num_epochs, cw_type, device, fold_idx,
                                  path_model)
            final_results[label_type].append(results)

    for label_type in labels_list:
        results_df = pd.DataFrame(final_results[label_type])
        avg_metrics = results_df[['F1-Score', 'Best_F1-score', 'Best_th', 'AUC-ROC', 'Num_epochs']].mean()
        std_metrics = results_df[['F1-Score', 'Best_F1-score', 'Best_th', 'AUC-ROC', 'Num_epochs']].std()
        summary_df = pd.DataFrame({
            "Fold": ["Avg.", "Std."],
            'F1-Score': [avg_metrics['F1-Score'], std_metrics['F1-Score']],
            'Best_F1-score': [avg_metrics['Best_F1-score'], std_metrics['Best_F1-score']],
            'Best_th': [avg_metrics['Best_th'], std_metrics['Best_th']],
            'AUC-ROC': [avg_metrics['AUC-ROC'], std_metrics['AUC-ROC']],
            'Num_epochs': [avg_metrics['Num_epochs'], std_metrics['Num_epochs']]
        })
        final_results_df = pd.concat([results_df, summary_df], ignore_index=True)
        path_to_save_results = f'{path_results}PM_multi_{label_type}_mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_sc{split_code}_cw{cw_type}_all_experiments_results_5cv.csv'
        final_results_df.to_csv(path_to_save_results, index=False)
        print(f"Results for {label_type} saved successfully.")
    print("Results saved successfully.")

