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
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
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



def train_model(model, train_dataloader, val_dataloader, num_epochs, device):
    for batch_features, batch_labels in train_dataloader:
        print(f"Features: {batch_features.shape}")
        print(f"Labels: {batch_labels.shape}")
        break
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
    if counter_patience == patience:
        train_losses = train_losses[: -patience]
        final_epochs = len(train_losses)
    else:
        final_epochs = num_epochs
    return final_epochs



def train_model_AGGObs(model, train_dataloader, val_dataloader, num_epochs, device):
    for batch_features, batch_aGGOBs, batch_labels in train_dataloader:
        print(f"Features: {batch_features.shape}")
        print(f"aGGObsr: {batch_aGGOBs.shape}")
        print(f"Labels: {batch_labels.shape}")
        break
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


def retrain_model(model, train_dataloader, test_dataloader, num_epochs, device, exp_name, path_model):
    print('Retraining model...')
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


def retrain_model_aGGObs(model, train_dataloader, test_dataloader, num_epochs, device, exp_name, path_model):
    print('Retraining model...')
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
    positive_class_weight = class_weights_dict[1]
    pos_weight_tensor = torch.tensor([positive_class_weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
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
        train_fun = train_model
        retrain_fun = retrain_model
    elif model_code == 2:
        model_fun = models.New_EEGNetLSTM_AGGObsr
        features_fun = data_utils.get_features_from_dict_bins
        dataloader_fun = data_utils.New_AggressiveBehaviorDatasetBin_AGGObserved
        train_fun = train_model_AGGObs
        retrain_fun = retrain_model_aGGObs
    else:
        print('Not supported yet.')
        model_fun, features_fun, dataloader_fun, train_fun, retrain_fun = None, None, None, None, None
    return model_fun, features_fun, dataloader_fun, train_fun, retrain_fun


def create_dataloader(dataloader_fun, output_dict, tp, bin_size, batch_size, shuffle=False):
    dataset = dataloader_fun(output_dict, tp, bin_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    return dataloader


def generate_user_kfolds(data_dict, k=5):
    # Genera k particiones de los IDs de sujetos utilizando KFold
    #Args:
    #    data_dict (dict): Diccionario de datos organizado por usuario y sesión.
    #    k (int): Número de pliegues para la validación cruzada.
    #Returns: list: Lista de tuplas, donde cada tupla contiene dos listas (train_uids, test_uids) para cada fold.

    # Extraer todos los SubjectIDs (uids) unicos del diccionario
    uids = list(data_dict.keys())
    # Inicializar KFold
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    # Generar particiones
    folds = []
    for train_idx, test_idx in kf.split(uids):
        train_uids = [uids[i] for i in train_idx]
        test_uids = [uids[i] for i in test_idx]
        folds.append((train_uids, test_uids))
    return folds


def get_partitions_from_fold(data_dict, train_uids, test_uids, seed):
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
    train_dict_val, test_dict_val = data_utils.new_split_data_per_session(val_dict_split, train_ratio=0.8)
    # Dividir diccionario de usuarios de test, 80% inicial de la sesion a train y el 20% final a test
    train_dict_test, test_dict_test = data_utils.new_split_data_per_session(test_dict, train_ratio=0.8)
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



def start_exps_PM(tp, tf, freq, data_path_resampled, path_results, path_models, model_code, seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds_path = data_path_resampled + "dataset_" + str(freq) + "Hz.csv"
    data_dict = data_utils.load_data_to_dict(ds_path)
    data_dict = dict(list(data_dict.items())[:20])
    num_folds = 5
    folds = generate_user_kfolds(data_dict, k=num_folds)
    EEG_channels = 5
    model_fun, features_fun, dataloader_fun, train_fun, retrain_fun = set_model(model_code)
    tp, tf, stride, bin_size = tp, tf, 15, 15
    print(f'tp: {tp}, tf: {tf}, stride: {stride}, bin_size: {bin_size}.')
    final_results = []
    for fold_idx, (train_uids, test_uids) in enumerate(folds):
        print(f"Fold {fold_idx + 1}:")
        print(f"  Train UIDs: {train_uids}")
        print(f"  Test UIDs: {test_uids}")
        # Obtener particiones de train, val y test en funcion de los ids del fold
        train_dict, val_dict, test_dict, all_train_dict = get_partitions_from_fold(data_dict, train_uids, test_uids, seed)
        # Dividir las sesiones de cada usuario en bins de 15 segundos y almacenar raw signals y etiquetas
        # (variable binaria que indica la ocurrencia de un episodio agresivo en el bin)
        train_data = features_fun(train_dict, tp, tf, bin_size, freq)
        val_data = features_fun(val_dict, tp, tf, bin_size, freq)
        test_data = features_fun(test_dict, tp, tf, bin_size, freq)

        batch_size = 128
        dataloader = create_dataloader(dataloader_fun, train_data, tp, bin_size, batch_size, shuffle=True)
        dataloader_val = create_dataloader(dataloader_fun, val_data, tp, bin_size, batch_size, shuffle=False)
        dataloader_test = create_dataloader(dataloader_fun, test_data, tp, bin_size, batch_size, shuffle=False)

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

        best_num_epochs = train_fun(model, dataloader, dataloader_val, num_epochs, device)

        final_model = model_fun(eegnet_params, lstm_hidden_dim, num_classes).to(device)
        all_train_data = features_fun(all_train_dict, tp, tf, bin_size, freq)
        dataloader_final = create_dataloader(dataloader_fun, all_train_data, tp, bin_size, batch_size, shuffle=True)


        path_model = path_models + 'tf' + str(tf) + '_tp' + str(tp) +  '_fold_' + str(fold_idx) + '_model.pth'
        results = retrain_fun(final_model, dataloader_final, dataloader_test, best_num_epochs, device, fold_idx,
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
    path_to_save_results = f'{path_results}PM_SS_model{model_code}_tf{tf}_tp{tp}_all_experiments_results_5cv.csv'
    final_results_df.to_csv(path_to_save_results, index=False)
    print("Results saved successfully.")



def start_exps_PDM(tp, tf, freq, data_path_resampled, path_results, path_models, model_code, seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds_path = data_path_resampled + "dataset_" + str(freq) + "Hz.csv"
    data_dict = data_utils.load_data_to_dict(ds_path)
    #data_dict = dict(list(data_dict.items())[:2])
    EEG_channels = 5
    model_fun, features_fun, dataloader_fun, _, _ = set_model(model_code)
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
        train_dict_, test_dict = data_utils.split_data_per_session(output_dict, train_ratio=0.8)
        train_dict, val_dict = data_utils.split_data_per_session(train_dict_, train_ratio=0.8)

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
            if model_code == 0 or model_code == 3:
                num_sequences = 1
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
            num_epochs = 100

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