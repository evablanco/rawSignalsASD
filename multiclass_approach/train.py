import numpy as np
import pandas as pd
import torch
import os
import models, data_utils
import time
from collections import Counter
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
import evaluation_utils
from evaluation_utils import (
    evaluate_auc_pre_attack,
    evaluate_f1_selected_classes,
    evaluate_f1_score_model,
    evaluate_macro_auc,
    report_model,
    model_evaluation
)
from sklearn.metrics import roc_auc_score, roc_curve, f1_score



# Subject states (classes)
#CALM, PRE_ATTACK, ATTACK, POST_ATTACK = 0, 1, 2, 3
CALM, PRE_ATTACK, ATTACK = 0, 1, 2


# set default model
def set_lstm(num_classes=3):
    lstm_hidden_dim = 64
    lstm_num_layers = 1
    num_classes = num_classes
    print(f"*** Setting LSTM: lstm_hidden_dim={lstm_hidden_dim}, lstm_num_layers={lstm_num_layers}, num_classes={num_classes}")
    return lstm_hidden_dim, lstm_num_layers, num_classes


# weights = get_custom_class_weights({1: 3.0}, num_classes=4)
def get_custom_class_weights(weight_dict, num_classes=3):
    class_weights_dict = {i: weight_dict.get(i, 1.0) for i in range(num_classes)}
    print("Custom class weights:", class_weights_dict)
    return class_weights_dict


def get_uniform_class_weights(num_classes=3):
    class_weights_dict = {i: 1.0 for i in range(num_classes)}
    print("Using uniform class weights:", class_weights_dict)
    return class_weights_dict


def compute_class_weights_balanced_multiclass(train_dataloader, num_classes=3):
    class_counter = Counter()
    first_batch = next(iter(train_dataloader))
    dataloader_iter = iter(train_dataloader)
    if len(first_batch) == 3:
        for _, _, labels in dataloader_iter:
            class_counter.update(labels.numpy().flatten().astype(int))
    else:
        for _, labels in dataloader_iter:
            class_counter.update(labels.numpy().flatten().astype(int))

    all_labels_expanded = []
    for class_id in range(num_classes):
        all_labels_expanded.extend([class_id] * class_counter[class_id])

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=np.array(all_labels_expanded)
    )
    class_weights_dict = {i: w for i, w in enumerate(class_weights)}
    print(f"Total samples analyzed: {sum(class_counter.values())}")
    print("Class distribution:", dict(class_counter))
    print("Computed class weights (multiclass):", class_weights_dict)
    return class_weights_dict


def compute_class_weights_balanced_binary(train_dataloader):
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

    print(f'Total samples analyzed: {total_samples}, {positive_count} positive and {negative_count} negative.')
    print('Computed class weights:', class_weights_dict)

    return class_weights_dict



def compute_class_weights(train_dataloader, cw_type, num_classes=3):
    if cw_type == 0:  # no cw
        class_weights_dict = get_uniform_class_weights(num_classes)
    elif cw_type == 1:  # balanced cw
        if num_classes == 1:  ### binary case
            class_weights_dict = compute_class_weights_balanced_binary(train_dataloader)
        else:
            class_weights_dict = compute_class_weights_balanced_multiclass(train_dataloader, num_classes)
    elif cw_type == 2:  # custom cw
        if num_classes != 1:
            class_weights_dict = get_custom_class_weights({1: 2.0}, num_classes) # ej. pre-attack peso x2
        else:  # binary case in intraTL, no custom en esta fase
            class_weights_dict = compute_class_weights_balanced_binary(train_dataloader)
    else:
        print('cw_type not supported.')  # no cw
        class_weights_dict = get_uniform_class_weights(num_classes)
    return class_weights_dict


def train_model_multiclass(model, train_dataloader, val_dataloader, num_epochs, cw_type, device, test_dataloader=None, pretrained=False):
    class_weights_dict = compute_class_weights(train_dataloader, cw_type)  # if cw==0, devolver 1,1
    print('class_weights_dict: ', class_weights_dict)
    weights_list = [class_weights_dict[i] for i in sorted(class_weights_dict.keys())]
    class_weights_tensor = torch.tensor(weights_list, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    if pretrained:
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    else:
        optimizer = Adam(model.parameters(), lr=0.001)
    for name, param in model.named_parameters():
        print(f"{name}: {'trainable' if param.requires_grad else 'frozen'}")
    best_val_score = 0
    patience = 10
    counter_patience = 0
    train_losses = []
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_features, batch_labels in train_dataloader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device).long()
            optimizer.zero_grad()
            predictions = model(batch_features)
            loss = criterion(predictions, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        val_score_pre = evaluate_f1_selected_classes(model, val_dataloader, device, [CALM])   ### probar con recall de calm.... o f1
        #val_score_att = evaluate_f1_selected_classes(model, val_dataloader, device, [ATTACK])
        val_score = val_score_pre #+ 0.25 * val_score_att
        val_score_gen = evaluate_f1_score_model(model, val_dataloader, device)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_dataloader):.4f}, Val sc. (f1-CALM): {val_score}, F1-gen: {val_score_gen}")
        train_losses.append(total_loss / len(train_dataloader))
        if val_score > best_val_score:
            best_val_score = val_score
            counter_patience = 0
        else:
            print('Val score does not improve (f1-score)')
            counter_patience += 1
        if counter_patience == patience:
            break
    total_time = time.time() - start_time
    print('Training finished. Total train time... : ', total_time)
    if test_dataloader is not None:
        print('Testing model at last epoch reached...')
        test_score = evaluate_f1_score_model(model, test_dataloader, device)
        test_score_pre = evaluate_f1_selected_classes(model, test_dataloader, device, [PRE_ATTACK])
        print('\t test score -> F1-Score macro: ', test_score, 'F1-Score class ', PRE_ATTACK, ': ', test_score_pre)

    if counter_patience == patience:
        train_losses = train_losses[: -patience]
        final_epochs = len(train_losses)
    else:
        final_epochs = num_epochs
    return final_epochs


def retrain_model_multiclass(model, train_dataloader, test_dataloader, num_epochs, cw_type, device, exp_name, path_model, pretrained=False):
    print('Retraining model...')
    print(model)
    class_weights_dict = compute_class_weights(train_dataloader, cw_type)  # if cw==0, devolver 1,1
    print('class_weights_dict: ', class_weights_dict)
    weights_list = [class_weights_dict[i] for i in sorted(class_weights_dict.keys())]
    class_weights_tensor = torch.tensor(weights_list, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    if pretrained:
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    else:
        optimizer = Adam(model.parameters(), lr=0.001)
    for name, param in model.named_parameters():
        print(f"{name}: {'trainable' if param.requires_grad else 'frozen'}")
    train_losses = []
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_features, batch_labels in train_dataloader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device).long()
            optimizer.zero_grad()
            predictions = model(batch_features)
            loss = criterion(predictions, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_dataloader):.4f}")
        train_losses.append(total_loss / len(train_dataloader))
    total_time = time.time() - start_time
    print('Training finished. Real Total train time... : ', total_time)
    torch.save(model.state_dict(), path_model)
    test_time_init = time.time()
    f1_macro, f1_weighted, auc_macro, acc, f1_per_class = evaluation_utils.model_evaluation(model, test_dataloader, device)
    test_time = time.time() - test_time_init
    print(f'Test results: f1_macro: {f1_macro}, f1_weighted: {f1_weighted}, auc_macro: {auc_macro}, auc: {acc}.')
    print(f'              f1_per_class: {f1_per_class}. (test time: {test_time})')
    print('')
    results = {
        "Exp_name": exp_name,
        "F1_macro": round(f1_macro, 4),
        "F1_weighted": round(f1_weighted, 4),
        "AUC_macro": round(auc_macro, 4),
        "Accuracy": round(acc, 4),
        "Per_class_F1": f1_per_class,
        "Training_time": round(total_time, 2),
        "Testing_time": round(test_time, 2),
        "Num_epochs": num_epochs,
        "Model_path": path_model
    }
    return results


def create_dataloader(dataloader_fun, output_dict, tp, tf, bin_size, batch_size, shuffle=False):
    dataset = dataloader_fun(output_dict, tp, tf, bin_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    return dataloader


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



def evaluate_model_val_f1(model, test_dataloader, device):
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
    threshold = 0.5
    all_predictions = (np.array(all_probs) > threshold).astype(int)
    f1_s = f1_score(all_labels, all_predictions)
    return f1_s


def train_model_binary(model, train_dataloader, val_dataloader, num_epochs, cw_type, device, test_dataloader=None):
    for batch_features, batch_labels in train_dataloader:
        print(f"Features: {batch_features.shape}")
        print(f"Labels: {batch_labels.shape}")
        break
    start_time = time.time()
    class_weights_dict = compute_class_weights(train_dataloader, cw_type, num_classes=1)
    total_time = time.time() - start_time
    print('Compute cw time... : ', total_time)
    print('class_weights_dict: ', class_weights_dict)
    positive_class_weight = class_weights_dict[1]
    pos_weight_tensor = torch.tensor([positive_class_weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = Adam(model.parameters(), lr=0.001)
    best_val = 0
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
            loss = criterion(predictions.view(-1), batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        #auc_val = evaluate_val_auc_model(model, val_dataloader, device)
        f1_val = evaluate_model_val_f1(model, val_dataloader, device)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_dataloader):.4f}, Val f1-score: {f1_val}")
        train_losses.append(total_loss / len(train_dataloader))
        if f1_val > best_val:
            best_val = f1_val
            counter_patience = 0
        else:
            print('Val f1-score does not improve...')
            counter_patience += 1
        if counter_patience == patience:
            break

    print("Train Done...")
    if test_dataloader != None:
        print('testing model at last epoch reached...')
        auc_test = evaluate_val_auc_model(model, test_dataloader, device)
        f1_test = evaluate_model_val_f1(model, test_dataloader, device)
        print('AUC test: ', auc_test, ', f1-score test: ', f1_test)
    if counter_patience == patience:
        train_losses = train_losses[: -patience]
        final_epochs = len(train_losses)
    else:
        final_epochs = num_epochs
    return final_epochs


def evaluate_model_binary(model, test_dataloader, device):
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


def retrain_model_binary(model, train_dataloader, test_dataloader, num_epochs, cw_type, device, exp_name, path_model):
    print('Retraining model with extremes...')
    start_time = time.time()
    class_weights_dict = compute_class_weights(train_dataloader, cw_type, num_classes=1)
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
            loss = criterion(predictions.view(-1), batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_dataloader):.4f}")
        train_losses.append(total_loss / len(train_dataloader))

    torch.save(model.state_dict(), path_model)
    f1_s, best_th, best_f1, auc_roc_s = evaluate_model_binary(model, test_dataloader, device)
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
    return results, model


def set_model(model_code):
    if model_code == 0: ### for pre-train with extremes...
        model_fun = models.EEGNetLSTM
        features_fun = data_utils.get_features_from_dict
        dataloader_fun = data_utils.AggressiveBehaviorDatasetExtreme
        train_fun = train_model_binary
        retrain_fun = retrain_model_binary ## needed? yep
    elif model_code == 1:
        model_fun = models.EEGNetLSTM
        features_fun = data_utils.get_features_from_dict
        dataloader_fun = data_utils.AggressiveBehaviorDataset
        train_fun = train_model_multiclass
        retrain_fun = retrain_model_multiclass
    else:
        print('Not supported yet.')
        model_fun, features_fun, dataloader_fun, train_fun, retrain_fun = None, None, None, None, None
    return model_fun, features_fun, dataloader_fun, train_fun, retrain_fun


def set_features(feats_code):
    feature_sets = {
        0: ['EDA', 'ACC_X', 'ACC_Y', 'ACC_Z', 'BVP'],  # all sensors
        1: ['EDA', 'ACC_X', 'ACC_Y', 'ACC_Z'],  # leave one out
        2: ['ACC_X', 'ACC_Y', 'ACC_Z', 'BVP'],
        3: ['EDA', 'BVP'],
        4: ['BVP'],  # only one source
        5: ['EDA'],
        6: ['ACC_X', 'ACC_Y', 'ACC_Z']
    }
    selected_columns = feature_sets.get(feats_code, feature_sets[0])  # default to 0
    if feats_code not in feature_sets:
        print("Not implemented... using default: all features")
    EEG_channels = len(selected_columns)
    return selected_columns, EEG_channels


def get_split_fun_PDM(split_code):
    if split_code == 1:
        split_fun = data_utils.split_data_full_sessions
    else:
        split_fun = None
        print('Split function not supported...')
    return split_fun


# PM
def generate_subject_kfolds(data_dict, k=5):
    uids = list(data_dict.keys())
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    folds = []
    for train_idx, test_idx in kf.split(uids):
        train_uids = [uids[i] for i in train_idx]
        test_uids = [uids[i] for i in test_idx]
        folds.append((train_uids, test_uids))
    return folds


# PDM
def generate_leave_sessions_out_kfolds(data_dict, k=5, seed=42):
    folds_per_user = {}
    for user, sessions in data_dict.items():
        session_keys = list(sessions.keys())
        if len(session_keys) < k: # k, minimun sessions
            continue
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)
        folds = []
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(session_keys)):
            train_sessions = [session_keys[i] for i in train_idx]
            test_sessions = [session_keys[i] for i in test_idx]
            folds.append((train_sessions, test_sessions))
            folds.append({
                "fold": fold_idx,
                "train_sessions": train_sessions,
                "test_sessions": test_sessions
            })
        folds_per_user[user] = folds
    return folds_per_user


def generate_hybrid_model_splits(data_dict, k=5, seed=42):
    hybrid_folds = {}
    for user, sessions in data_dict.items():
        session_keys = list(sessions.keys())
        if len(session_keys) < k:
            continue
        # Sessions for this user (PDM part)
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)
        pdm_folds = []
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(session_keys)):
            train_sessions = [session_keys[i] for i in train_idx]
            test_sessions = [session_keys[i] for i in test_idx]
            # PM part: use all other users
            other_users = [u for u in data_dict if u != user]
            pm_sessions = []
            for u in other_users:
                for s in data_dict[u]:
                    pm_sessions.append((u, s))
            pdm_folds.append({
                "fold": fold_idx,
                "user": user,
                "pdm_train_sessions": train_sessions,
                "pdm_test_sessions": test_sessions,
                "pm_train_sessions": pm_sessions
            })
        hybrid_folds[user] = pdm_folds
    return hybrid_folds


def get_partitions_from_fold(data_dict, train_uids, test_uids, split_code, seed):
    # obtener diccionario de los usuarios de train
    train_dict = {uid: data_dict[uid] for uid in train_uids}
    # obtener diccionario de los usuarios de test
    test_dict = {uid: data_dict[uid] for uid in test_uids}
    # Dividir los usuarios de entrenamiento en train y val
    train_uids_split, val_uids_split = train_test_split(train_uids, test_size=0.2, random_state=seed)
    train_dict_split = {uid: train_dict[uid] for uid in train_uids_split}
    val_dict_split = {uid: train_dict[uid] for uid in val_uids_split}
    print(f"  Train UIDs (sin val): {train_uids_split}")
    print(f"  Val UIDs: {val_uids_split}")
    if split_code: # PM-SS
        # Dividir diccionario de usuarios de test/val, 80% de las sesiones a train y el 20% a test/val
        # (al menos 1 en test/val si tiene al menos 2)
        dict_test_80, dict_test_20 = data_utils.split_data_full_sessions(test_dict, train_ratio=0.8)
        dict_val_80, dict_val_20 = data_utils.split_data_full_sessions(val_dict_split, train_ratio=0.8)
        # Añadir al diccionario de train los datos del 80% de las sesiones de los sujetos de test y val
        for uid, data in {**dict_test_80, **dict_val_80}.items():
            if uid not in train_dict_split:
                train_dict_split[uid] = {}
            train_dict_split[uid].update(data)
        print(f"  Train UIDs (final): {list(train_dict_split.keys())}")
        # Añadir al diccionario de train+val los datos del 80% de las sesiones de los sujetos de test
        for uid, data in dict_test_80.items():
            if uid not in train_dict:
                train_dict[uid] = {}
            train_dict[uid].update(data)
        print(f"  Train UIDs (final): {list(train_dict_split.keys())}")
        test_dict = dict_test_20
        val_dict_split = dict_val_20
    return train_dict_split, val_dict_split, test_dict, train_dict



def start_exps_PM(tp, tf, freq, data_path_resampled, path_results, path_models, model_code, feats_code, split_code, b_size, cw_type, seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set exp. config
    model_fun, features_fun, dataloader_fun, train_fun, retrain_fun = set_model(model_code)
    selected_features, EEG_channels = set_features(feats_code)
    tp, tf, bin_size = tp, tf, b_size
    print(f'tp: {tp}, tf: {tf}, bin_size: {bin_size}, model_code: {model_code}, split_code: {split_code}, cw_type: {cw_type}.')
    ds_path = data_path_resampled + "dataset_" + str(freq) + "Hz.csv"
    data_dict = data_utils.load_data_to_dict(ds_path, selected_features)
    # data_dict = dict(list(data_dict.items())[:20])
    # generate folds
    num_folds = 5
    folds = generate_subject_kfolds(data_dict, k=num_folds)
    final_results = []
    for fold_idx, (train_uids, test_uids) in enumerate(folds):
        print(f"Fold {fold_idx + 1}:")
        print(f"  Train UIDs: {train_uids}")
        print(f"  Test UIDs: {test_uids}")
        # Obtener particiones de train, val y test en funcion de los ids del fold
        train_dict, val_dict, test_dict, all_train_dict = get_partitions_from_fold(data_dict, train_uids, test_uids,
                                                                                   split_code, seed)
        # Dividir las sesiones de cada usuario en bin_size segundos y almacenar raw signals y etiquetas
        # (variable binaria que indica la ocurrencia de un episodio agresivo en el bin)
        mean, std = data_utils.compute_normalization_stats(all_train_dict, bin_size, freq)
        #mean, std = None, None
        train_data = features_fun(train_dict, bin_size, freq, mean, std)
        val_data = features_fun(val_dict, bin_size, freq, mean, std)
        test_data = features_fun(test_dict, bin_size, freq, mean, std)

        batch_size = 128
        dataloader = create_dataloader(dataloader_fun, train_data, tp, tf, bin_size, batch_size, shuffle=True)
        dataloader_val = create_dataloader(dataloader_fun, val_data, tp, tf, bin_size, batch_size, shuffle=False)
        dataloader_test = create_dataloader(dataloader_fun, test_data, tp, tf, bin_size, batch_size, shuffle=False)

        '''
        # plot class examples, still work in progress...
        dataset_train = data_utils.AggressiveBehaviorDataset(train_data, tp=tp, tf=tf, bin_size=bin_size)
        class_names = ["Calm", "Pre-attack", "Attack", "Post-attack"]
        # Visualizar muestras del dataset ya limpio
        path_img_results = f'{path_results}PM_mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_sc{split_code}_cw{cw_type}_fold{fold_idx}_classes_sample.png'
        data_utils.plot_windows_from_dataset(dataset_train, class_names, path_img_results, n_samples_per_class=5, tp=180, channel_names=selected_features)
        '''

        num_sequences = tp // bin_size
        eegnet_params = {
            'num_electrodes': EEG_channels,  # (EDA, ACC_X, ACC_Y, ACC_Z, BVP) | (EDA, ACC_Norm, BVP) | ....
            'chunk_size': tp * freq // num_sequences  # muestras en cada ventana
        }
        lstm_hidden_dim, lstm_num_layers, num_classes = set_lstm()
        model = model_fun(eegnet_params, lstm_hidden_dim, lstm_num_layers, num_classes).to(device)
        num_epochs = 100

        best_num_epochs = train_fun(model, dataloader, dataloader_val, num_epochs, cw_type, device,
                                    test_dataloader=dataloader_test)
        final_model = model_fun(eegnet_params, lstm_hidden_dim, lstm_num_layers, num_classes).to(device)
        all_train_data = features_fun(all_train_dict, bin_size, freq, mean, std)
        dataloader_final = create_dataloader(dataloader_fun, all_train_data, tp, tf, bin_size, batch_size, shuffle=True)

        path_model = f"{path_models}PM_sc{split_code}_mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}_fold{fold_idx}_model.pth"
        results = retrain_fun(final_model, dataloader_final, dataloader_test, best_num_epochs, cw_type, device,
                              fold_idx, path_model)
        final_results.append(results)

    results_df = pd.DataFrame(final_results)
    avg_metrics = results_df[['F1_macro', 'F1_weighted', 'AUC_macro', 'Accuracy', 'Training_time', 'Testing_time',
                              'Num_epochs']].mean()
    std_metrics = results_df[['F1_macro', 'F1_weighted', 'AUC_macro', 'Accuracy', 'Training_time', 'Testing_time',
                              'Num_epochs' ]].std()

    summary_df = pd.DataFrame({
        "Fold": ["Avg.", "Std."],
        'F1_macro': [avg_metrics['F1_macro'], std_metrics['F1_macro']],
        'F1_weighted': [avg_metrics['F1_weighted'], std_metrics['F1_weighted']],
        'AUC_macro': [avg_metrics['AUC_macro'], std_metrics['AUC_macro']],
        'Accuracy': [avg_metrics['Accuracy'], std_metrics['Accuracy']],
        'Training_time': [avg_metrics['Training_time'], std_metrics['Training_time']],
        'Testing_time': [avg_metrics['Testing_time'], std_metrics['Testing_time']],
        'Num_epochs': [avg_metrics['Num_epochs'], std_metrics['Num_epochs']]
    })

    #class_names = ["Calm", "Pre-attack", "Attack", "Post-attack"]
    class_names = ["Calm", "Pre-attack", "Attack"]
    per_class_f1_all_folds = [
        [res["Per_class_F1"][cls] for cls in class_names]
        for res in final_results
    ]
    per_class_f1_array = np.array(per_class_f1_all_folds)
    # media y std por clase
    per_class_f1_mean = np.mean(per_class_f1_array, axis=0)
    per_class_f1_std = np.std(per_class_f1_array, axis=0)

    for i, cls in enumerate(class_names):
        results_df[f"F1_{cls}"] = per_class_f1_array[:, i]

    for i, cls in enumerate(class_names):
        summary_df[f"F1_{cls}"] = [per_class_f1_mean[i], per_class_f1_std[i]]

    if 'Per_class_F1' in results_df.columns:
        results_df.drop(['Per_class_F1'], axis=1, inplace=True)
    final_results_df = pd.concat([results_df, summary_df], ignore_index=True)
    path_to_save_results = f'{path_results}PM_mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_sc{split_code}_cw{cw_type}_all_experiments_results_5cv.csv'
    final_results_df.to_csv(path_to_save_results, index=False)
    print("Results saved successfully.")


def load_pretrained_eegnet_weights(model, pretrained_path):
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    eegnet_dict = {k.replace("eegnet.", ""): v for k, v in checkpoint.items() if k.startswith("eegnet.")}
    model.eegnet.load_state_dict(eegnet_dict)
    for param in model.eegnet.parameters():
        param.requires_grad = False
    return model



def start_exps_PM_intraTL(tp, tf, freq, data_path_resampled, path_results, path_models, model_code, feats_code, split_code, b_size, cw_type, seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set exp. config
    model_fun, features_fun, dataloader_fun, train_fun, retrain_fun = set_model(model_code)
    _, _, dataloader_fun_extr, train_fun_extr, retrain_fun_extr = set_model(0) ## pre-train with extremes
    selected_features, EEG_channels = set_features(feats_code)
    tp, tf, bin_size = tp, tf, b_size
    print(f'tp: {tp}, tf: {tf}, bin_size: {bin_size}, model_code: {model_code}, split_code: {split_code}, cw_type: {cw_type}.')
    ds_path = data_path_resampled + "dataset_" + str(freq) + "Hz.csv"
    data_dict = data_utils.load_data_to_dict(ds_path, selected_features)
    # data_dict = dict(list(data_dict.items())[:20])
    # generate folds
    num_folds = 5
    folds = generate_subject_kfolds(data_dict, k=num_folds)
    final_results = []
    for fold_idx, (train_uids, test_uids) in enumerate(folds):
        print(f"Fold {fold_idx + 1}:")
        print(f"  Train UIDs: {train_uids}")
        print(f"  Test UIDs: {test_uids}")
        # Obtener particiones de train, val y test en funcion de los ids del fold
        train_dict, val_dict, test_dict, all_train_dict = get_partitions_from_fold(data_dict, train_uids, test_uids,
                                                                                   split_code, seed)
        mean, std = data_utils.compute_normalization_stats(train_dict, bin_size, freq)
        #mean, std = None, None
        # Dividir las sesiones de cada usuario en bin_size segundos y almacenar raw signals y etiquetas
        # (variable binaria que indica la ocurrencia de un episodio agresivo en el bin)
        train_data = features_fun(train_dict, bin_size, freq, mean, std)
        val_data = features_fun(val_dict, bin_size, freq, mean, std)
        test_data = features_fun(test_dict, bin_size, freq, mean, std)

        ######## Pre-training with extremes
        num_sequences = tp // bin_size
        eegnet_params = {
            'num_electrodes': EEG_channels,  # (EDA, ACC_X, ACC_Y, ACC_Z, BVP) | (EDA, ACC_Norm, BVP) | ....
            'chunk_size': tp * freq // num_sequences  # muestras en cada ventana
        }
        lstm_hidden_dim, lstm_num_layers, num_classes = set_lstm(num_classes=1)  # técnicamente 2.. pero es la last FC, así que 1... corregir
        model_extr = model_fun(eegnet_params, lstm_hidden_dim, lstm_num_layers, num_classes=1).to(device)
        path_model_extr = f"{path_models}PMTL_sc{split_code}_mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}_fold{fold_idx}_model_extr.pth"
        ##### comprobar antes si ya existe.. por si tarda too much en slurm...
        if os.path.exists(path_model_extr):
            model_extr.load_state_dict(torch.load(path_model_extr, map_location=device, weights_only=True))
        else:
            all_conditions = []
            for user_id, sessions in val_dict.items():
                for session_id, session_data in sessions.items():
                    df = session_data
                    if "Condition" in df.columns:
                        all_conditions.extend(df["Condition"].values.tolist())
            unique_vals, counts = np.unique(all_conditions, return_counts=True)
            for val, count in zip(unique_vals, counts):
                print(f"Condition = {val}: {count} veces")

            batch_size = 128
            dataloader_extr = create_dataloader(dataloader_fun_extr, train_data, tp, tf, bin_size, batch_size, shuffle=True)
            dataloader_val_extr = create_dataloader(dataloader_fun_extr, val_data, tp, tf, bin_size, batch_size, shuffle=False)
            dataloader_test_extr = create_dataloader(dataloader_fun_extr, test_data, tp, tf, bin_size, batch_size, shuffle=False)
            num_epochs = 100
            best_num_epochs_extr = train_fun_extr(model_extr, dataloader_extr, dataloader_val_extr, num_epochs, cw_type, device,
                                        test_dataloader=dataloader_test_extr)
            final_model_extr = model_fun(eegnet_params, lstm_hidden_dim, lstm_num_layers, num_classes=1).to(device)
            all_train_data = features_fun(all_train_dict, bin_size, freq, mean, std)
            dataloader_final = create_dataloader(dataloader_fun_extr, all_train_data, tp, tf, bin_size, batch_size, shuffle=True)
            results_extr, _ = retrain_fun_extr(final_model_extr, dataloader_final, dataloader_test_extr, best_num_epochs_extr, cw_type, device,
                                  fold_idx, path_model_extr)
            print('Extreme model results: ')
            print(results_extr)

        #################################
        batch_size = 128
        dataloader = create_dataloader(dataloader_fun, train_data, tp, tf, bin_size, batch_size, shuffle=True)
        dataloader_val = create_dataloader(dataloader_fun, val_data, tp, tf, bin_size, batch_size, shuffle=False)
        dataloader_test = create_dataloader(dataloader_fun, test_data, tp, tf, bin_size, batch_size, shuffle=False)

        lstm_hidden_dim, lstm_num_layers, num_classes = set_lstm()
        model = model_fun(eegnet_params, lstm_hidden_dim, lstm_num_layers, num_classes).to(device)
        model = load_pretrained_eegnet_weights(model, path_model_extr)
        num_epochs = 100
        best_num_epochs = train_fun(model, dataloader, dataloader_val, num_epochs, cw_type, device,
                                    test_dataloader=dataloader_test, pretrained=True)
        final_model = model_fun(eegnet_params, lstm_hidden_dim, lstm_num_layers, num_classes).to(device)
        final_model = load_pretrained_eegnet_weights(final_model, path_model_extr)
        all_train_data = features_fun(all_train_dict, bin_size, freq, mean, std)
        dataloader_final = create_dataloader(dataloader_fun, all_train_data, tp, tf, bin_size, batch_size, shuffle=True)

        path_model = f"{path_models}PMTL_sc{split_code}_mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}_fold{fold_idx}_model.pth"
        results = retrain_fun(final_model, dataloader_final, dataloader_test, best_num_epochs, cw_type, device,
                              fold_idx, path_model, pretrained=True)
        final_results.append(results)

    results_df = pd.DataFrame(final_results)
    avg_metrics = results_df[['F1_macro', 'F1_weighted', 'AUC_macro', 'Accuracy', 'Training_time', 'Testing_time',
                              'Num_epochs']].mean()
    std_metrics = results_df[['F1_macro', 'F1_weighted', 'AUC_macro', 'Accuracy', 'Training_time', 'Testing_time',
                              'Num_epochs' ]].std()

    summary_df = pd.DataFrame({
        "Fold": ["Avg.", "Std."],
        'F1_macro': [avg_metrics['F1_macro'], std_metrics['F1_macro']],
        'F1_weighted': [avg_metrics['F1_weighted'], std_metrics['F1_weighted']],
        'AUC_macro': [avg_metrics['AUC_macro'], std_metrics['AUC_macro']],
        'Accuracy': [avg_metrics['Accuracy'], std_metrics['Accuracy']],
        'Training_time': [avg_metrics['Training_time'], std_metrics['Training_time']],
        'Testing_time': [avg_metrics['Testing_time'], std_metrics['Testing_time']],
        'Num_epochs': [avg_metrics['Num_epochs'], std_metrics['Num_epochs']]
    })

    #class_names = ["Calm", "Pre-attack", "Attack", "Post-attack"]
    class_names = ["Calm", "Pre-attack", "Attack"]
    per_class_f1_all_folds = [
        [res["Per_class_F1"][cls] for cls in class_names]
        for res in final_results
    ]
    per_class_f1_array = np.array(per_class_f1_all_folds)
    # media y std por clase
    per_class_f1_mean = np.mean(per_class_f1_array, axis=0)
    per_class_f1_std = np.std(per_class_f1_array, axis=0)

    for i, cls in enumerate(class_names):
        results_df[f"F1_{cls}"] = per_class_f1_array[:, i]

    for i, cls in enumerate(class_names):
        summary_df[f"F1_{cls}"] = [per_class_f1_mean[i], per_class_f1_std[i]]

    if 'Per_class_F1' in results_df.columns:
        results_df.drop(['Per_class_F1'], axis=1, inplace=True)
    final_results_df = pd.concat([results_df, summary_df], ignore_index=True)
    path_to_save_results = f'{path_results}PMTL_mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_sc{split_code}_cw{cw_type}_all_experiments_results_5cv.csv'
    final_results_df.to_csv(path_to_save_results, index=False)
    print("Results saved successfully.")



def invalid_data_multiclass(dataloader, expected_classes=(0, 1, 2)):
    labels_list = []
    for batch in dataloader:
        labels = batch[-1]
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        labels_flat = labels.flatten()
        labels_list.extend(labels_flat)
    labels_array = np.array(labels_list)
    unique_present = np.unique(labels_array)
    print(f'Total samples: {len(labels_array)}, Clase presentes: {unique_present}')
    return not set(expected_classes).issubset(set(unique_present))


def start_exps_PDM(tp, tf, freq, data_path_resampled, path_results, path_models, model_code, feats_code, split_code,
                   b_size, cw_type, seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set exp. config
    model_fun, features_fun, dataloader_fun, train_fun, retrain_fun = set_model(model_code)
    selected_features, EEG_channels = set_features(feats_code)
    tp, tf, bin_size = tp, tf, b_size
    print(
        f'tp: {tp}, tf: {tf}, bin_size: {bin_size}, model_code: {model_code}, split_code: {split_code}, cw_type: {cw_type}.')
    ds_path = data_path_resampled + "dataset_" + str(freq) + "Hz.csv"
    data_dict = data_utils.load_data_to_dict(ds_path, selected_features)
    num_users = len(list(data_dict.items()))
    invalid_users = []
    final_results = []
    for u in range(0, num_users):
        first_item = dict(list(data_dict.items())[:1])
        key_subject = list(first_item.keys())[0]
        print('userID: ', key_subject)
        split_fun = get_split_fun_PDM(split_code)

        train_data, _ = split_fun(first_item, train_ratio=0.8)
        mean, std = data_utils.compute_normalization_stats(train_data, bin_size, freq)
        #mean, std = None, None

        output_dict = features_fun(first_item, bin_size, freq, mean, std)
        train_dict_, test_dict = split_fun(output_dict, train_ratio=0.8)
        train_dict, val_dict = split_fun(train_dict_, train_ratio=0.8)

        batch_size = 128
        dataloader = create_dataloader(dataloader_fun, train_dict, tp, tf, bin_size, batch_size, shuffle=True)
        dataloader_val = create_dataloader(dataloader_fun, val_dict, tp, tf, bin_size, batch_size, shuffle=False)
        dataloader_test = create_dataloader(dataloader_fun, test_dict, tp, tf, bin_size, batch_size, shuffle=False)
        dataloader_final = create_dataloader(dataloader_fun, train_dict_, tp, tf, bin_size, batch_size, shuffle=False)

        if invalid_data_multiclass(dataloader) or invalid_data_multiclass(dataloader_test) or invalid_data_multiclass(
                dataloader_val) or invalid_data_multiclass(dataloader_final):
            invalid_users.append(key_subject)
        else:
            num_sequences = tp // bin_size
            eegnet_params = {
                'num_electrodes': EEG_channels,  # (EDA, ACC_X, ACC_Y, ACC_Z, BVP) | (EDA, ACC_Norm, BVP) | ....
                'chunk_size': tp * freq // num_sequences  # muestras en cada ventana
            }
            lstm_hidden_dim, lstm_num_layers, num_classes = set_lstm()
            model = model_fun(eegnet_params, lstm_hidden_dim, lstm_num_layers, num_classes).to(device)
            num_epochs = 100

            best_num_epochs = train_fun(model, dataloader, dataloader_val, num_epochs, cw_type, device,
                                        test_dataloader=dataloader_test)
            final_model = model_fun(eegnet_params, lstm_hidden_dim, lstm_num_layers, num_classes).to(device)
            path_model = f"{path_models}PDM_sc{split_code}_mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}_fold{key_subject}_model.pth"
            results = retrain_fun(final_model, dataloader_final, dataloader_test, best_num_epochs, cw_type, device,
                                  key_subject, path_model)
            final_results.append(results)
        data_dict.pop(next(iter(first_item)))

    results_df = pd.DataFrame(final_results)
    avg_metrics = results_df[['F1_macro', 'F1_weighted', 'AUC_macro', 'Accuracy', 'Training_time', 'Testing_time',
                              'Num_epochs']].mean()
    std_metrics = results_df[['F1_macro', 'F1_weighted', 'AUC_macro', 'Accuracy', 'Training_time', 'Testing_time',
                              'Num_epochs']].std()

    summary_df = pd.DataFrame({
        "Fold": ["Avg.", "Std."],
        'F1_macro': [avg_metrics['F1_macro'], std_metrics['F1_macro']],
        'F1_weighted': [avg_metrics['F1_weighted'], std_metrics['F1_weighted']],
        'AUC_macro': [avg_metrics['AUC_macro'], std_metrics['AUC_macro']],
        'Accuracy': [avg_metrics['Accuracy'], std_metrics['Accuracy']],
        'Training_time': [avg_metrics['Training_time'], std_metrics['Training_time']],
        'Testing_time': [avg_metrics['Testing_time'], std_metrics['Testing_time']],
        'Num_epochs': [avg_metrics['Num_epochs'], std_metrics['Num_epochs']]
    })

    # class_names = ["Calm", "Pre-attack", "Attack", "Post-attack"]
    class_names = ["Calm", "Pre-attack", "Attack"]
    per_class_f1_all_folds = [
        [res["Per_class_F1"][cls] for cls in class_names]
        for res in final_results
    ]
    per_class_f1_array = np.array(per_class_f1_all_folds)
    # media y std por clase
    per_class_f1_mean = np.mean(per_class_f1_array, axis=0)
    per_class_f1_std = np.std(per_class_f1_array, axis=0)

    for i, cls in enumerate(class_names):
        results_df[f"F1_{cls}"] = per_class_f1_array[:, i]

    for i, cls in enumerate(class_names):
        summary_df[f"F1_{cls}"] = [per_class_f1_mean[i], per_class_f1_std[i]]

    if 'Per_class_F1' in results_df.columns:
        results_df.drop(['Per_class_F1'], axis=1, inplace=True)

    final_results_df = pd.concat([results_df, summary_df], ignore_index=True)
    path_to_save_results = f'{path_results}PDM_mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_sc{split_code}_cw{cw_type}_all_experiments_results_5cv.csv'
    final_results_df.to_csv(path_to_save_results, index=False)
    print("Results saved successfully.")