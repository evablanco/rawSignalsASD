# explain.py

import torch
import train
import data_utils
from models import EEGNetLSTM
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
import os

RESULTS_BASE = './results_analysis/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
freq = 32
batch_size = 32

## TO-DO.. usar model_version, que ahora solo v1 a pincho...

def load_model_and_data(path_models, feats_code, bin_size, tp, tf, split_code, cw_type, fold_idx):
    selected_features, EEG_channels = train.set_features(feats_code)
    data_dict = data_utils.load_data_to_dict('./dataset_resampled/dataset_32Hz.csv', selected_features)
    folds = train.generate_subject_kfolds(data_dict, k=5)
    train_uids, test_uids = folds[fold_idx]
    _, _, test_dict, _ = train.get_partitions_from_fold(data_dict, train_uids, test_uids, split_code, seed=42)
    test_data = data_utils.get_features_from_dict(test_dict, bin_size, freq)
    test_loader = train.create_dataloader(
        data_utils.AggressiveBehaviorDataset, test_data, tp, tf, bin_size,
        batch_size=batch_size, shuffle=False
    )
    lstm_hidden_dim, lstm_num_layers, num_classes = train.set_lstm()
    num_sequences = tp // bin_size
    eegnet_params = {'num_electrodes': EEG_channels, 'chunk_size': tp * freq // num_sequences}
    model = EEGNetLSTM(eegnet_params, lstm_hidden_dim, lstm_num_layers, num_classes).to(device)
    model_path = f"{path_models}PM_sc{split_code}_mv1_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}_fold{fold_idx}_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    model.train()
    return model, test_loader, selected_features


# Análisis temporal
def evaluate_tp_model(path_models, model_version, feats_code, tf, tps, bin_size, split_code, cw_type, seed=1):  # TO-DO: especificar clase en params, ahora default pre-attack
    tps = [tps] if isinstance(tps, int) else tps
    rows = 1 if len(tps) <= 5 else 2
    cols = (len(tps) + rows - 1) // rows
    fig, axs = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), squeeze=False)
    for idx, tp in enumerate(tps):
        fold_attrs = []
        for fold_idx in range(5):
            model, test_loader, _ = load_model_and_data(path_models, feats_code, bin_size, tp, tf, split_code, cw_type, fold_idx)
            ig = IntegratedGradients(model)
            attrs_fold = [ig.attribute(inputs.to(device), target=1).cpu().detach().numpy().mean(axis=(2,3)) for inputs, _ in test_loader]
            fold_attrs.append(np.concatenate(attrs_fold).mean(axis=0))
        mean_attr = np.mean(fold_attrs, axis=0)
        axs[idx // cols, idx % cols].plot(np.arange(-tp, 0, bin_size), mean_attr)
        axs[idx // cols, idx % cols].set_title(f'TP={tp}s')
        axs[idx // cols, idx % cols].set_xlabel('Time (sec)')
        axs[idx // cols, idx % cols].grid()
    plt.tight_layout()
    save_path = f"{RESULTS_BASE}PM_sc{split_code}_mv1_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}_fold{fold_idx}_model_grads_tps.png"
    plt.savefig(save_path)
    plt.show()


# Análisis temporal por señal
def evaluate_tp_model_signals(path_models, model_version, feats_code, tf, tps, bin_size, split_code, cw_type, seed=1):
    tps = [tps] if isinstance(tps, int) else tps
    rows = 1 if len(tps) <= 5 else 2
    cols = (len(tps) + rows - 1) // rows
    fig, axs = plt.subplots(rows, cols, figsize=(6*cols, 4*rows), squeeze=False)
    for idx, tp in enumerate(tps):
        fold_attrs = []
        for fold_idx in range(5):
            model, test_loader, selected_features = load_model_and_data(path_models, feats_code, bin_size, tp, tf, split_code, cw_type, fold_idx)
            ig = IntegratedGradients(model)
            attrs_fold = [ig.attribute(inputs.to(device), target=1).cpu().detach().numpy().mean(axis=3) for inputs, _ in test_loader]
            fold_attrs.append(np.concatenate(attrs_fold).mean(axis=0))
        mean_attr = np.mean(fold_attrs, axis=0)
        for i, channel in enumerate(selected_features):
            axs[idx // cols, idx % cols].plot(np.arange(-tp, 0, bin_size), mean_attr[:, i], label=channel)
        axs[idx // cols, idx % cols].set_title(f'TP={tp}s')
        axs[idx // cols, idx % cols].set_xlabel('Time (sec)')
        axs[idx // cols, idx % cols].legend()
        axs[idx // cols, idx % cols].grid()
    plt.tight_layout()
    save_path = f"{RESULTS_BASE}PM_sc{split_code}_mv1_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}_fold{fold_idx}_model_grads_tps_signals.png"
    plt.savefig(save_path)
    #plt.show()


# Comparativa entre clases
def evaluate_classes_model(path_models, model_version, feats_code, tf, tp, bin_size, split_code, cw_type, seed=1):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for target_class in range(3):
        fold_attrs = []
        for fold_idx in range(5):
            model, test_loader, _ = load_model_and_data(path_models, feats_code, bin_size, tp, tf, split_code, cw_type, fold_idx)
            ig = IntegratedGradients(model)
            attrs_fold = [ig.attribute(inputs.to(device), target=target_class).cpu().detach().numpy().mean(axis=(2,3)) for inputs, _ in test_loader]
            fold_attrs.append(np.concatenate(attrs_fold).mean(axis=0))
        mean_attr = np.mean(fold_attrs, axis=0)
        axs[target_class].plot(np.arange(-tp, 0, bin_size), mean_attr)
        axs[target_class].set_title(f'Class {target_class}')
        axs[target_class].set_xlabel('Time (sec)')
        axs[target_class].grid()
    plt.tight_layout()
    save_path = f"{RESULTS_BASE}PM_sc{split_code}_mv1_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}_fold{fold_idx}_model_grads_classes.png"
    plt.savefig(save_path)
    #plt.show()


# Comparativa entre clases por señal
def evaluate_classes_model_signals(path_models, model_version, feats_code, tf, tp, bin_size, split_code, cw_type, seed=1):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    for target_class in range(3):
        fold_attrs = []
        for fold_idx in range(5):
            model, test_loader, selected_features = load_model_and_data(path_models, feats_code, bin_size, tp, tf, split_code, cw_type, fold_idx)
            ig = IntegratedGradients(model)
            attrs_fold = [ig.attribute(inputs.to(device), target=target_class).cpu().detach().numpy().mean(axis=3) for inputs, _ in test_loader]
            fold_attrs.append(np.concatenate(attrs_fold).mean(axis=0))
        mean_attr = np.mean(fold_attrs, axis=0)
        for i, channel in enumerate(selected_features):
            axs[target_class].plot(np.arange(-tp, 0, bin_size), mean_attr[:, i], label=channel)

        axs[target_class].set_title(f'Class {target_class}')
        axs[target_class].set_xlabel('Time (sec)')
        axs[target_class].legend()
        axs[target_class].grid()

    plt.tight_layout()
    save_path = f"{RESULTS_BASE}PM_sc{split_code}_mv1_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}_fold{fold_idx}_model_grads_classes_signals.png"
    plt.savefig(save_path)


# Análisis específico de errores
def evaluate_errors_model(path_models, model_version, feats_code, tf, tp, bin_size, split_code, cw_type, seed=1):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for target_class in range(3):
        fold_attrs = []
        for fold_idx in range(5):
            model, test_loader, _ = load_model_and_data(path_models, feats_code, bin_size, tp, tf, split_code, cw_type, fold_idx)
            ig = IntegratedGradients(model)
            attrs_fold = []
            for inputs, labels in test_loader:
                preds = model(inputs.to(device)).argmax(dim=1).cpu()
                incorrect = preds != labels
                if incorrect.any():
                    attrs = ig.attribute(inputs[incorrect].to(device), target=target_class).cpu().detach().numpy().mean(axis=(2,3))
                    attrs_fold.append(attrs)
            if attrs_fold:
                fold_attrs.append(np.concatenate(attrs_fold).mean(axis=0))
        mean_attr = np.mean(fold_attrs, axis=0)
        axs[target_class].plot(np.arange(-tp, 0, bin_size), mean_attr)
        axs[target_class].set_title(f'Class {target_class}')
        axs[target_class].set_xlabel('Time (sec)')
        axs[target_class].grid()

    plt.tight_layout()
    save_path = f"{RESULTS_BASE}PM_sc{split_code}_mv1_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}_fold{fold_idx}_model_grads_errors.png"
    plt.savefig(save_path)
    #plt.show()


# Análisis específico de errores por señal
def evaluate_errors_model_signals(path_models, model_version, feats_code, tf, tp, bin_size, split_code, cw_type, seed=1):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    for target_class in range(3):
        fold_attrs = []
        for fold_idx in range(5):
            model, test_loader, selected_features = load_model_and_data(path_models, feats_code, bin_size, tp, tf, split_code, cw_type, fold_idx)
            ig = IntegratedGradients(model)
            attrs_fold = []
            for inputs, labels in test_loader:
                preds = model(inputs.to(device)).argmax(dim=1).cpu()
                incorrect = preds != labels
                if incorrect.any():
                    attrs = ig.attribute(inputs[incorrect].to(device), target=target_class).cpu().detach().numpy().mean(axis=3)
                    attrs_fold.append(attrs)
            if attrs_fold:
                fold_attrs.append(np.concatenate(attrs_fold).mean(axis=0))
        mean_attr = np.mean(fold_attrs, axis=0)
        for i, channel in enumerate(selected_features):
            axs[target_class].plot(np.arange(-tp, 0, bin_size), mean_attr[:, i], label=channel)
        axs[target_class].set_title(f'Class {target_class} - Errores')
        axs[target_class].set_xlabel('Time (sec)')
        axs[target_class].legend()
        axs[target_class].grid()

    plt.tight_layout()
    save_path = f"{RESULTS_BASE}PM_sc{split_code}_mv1_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}_fold{fold_idx}_model_grads_errors_signals.png"
    plt.savefig(save_path)


# Reducción de dimensionalidad y visualización de attribute scores (TO-DO!!! por señal)
def evaluate_visualization_model(path_models, model_version, feats_code, tf, tp, bin_size, split_code, cw_type, seed=1):
    reducers = [PCA(n_components=2), TSNE(n_components=2, perplexity=30), umap.UMAP(n_components=2)]
    reducer_names = ['PCA', 't-SNE', 'UMAP']
    fig, axs = plt.subplots(3, 3, figsize=(18, 15))
    for i, reducer in enumerate(reducers):
        for target_class in range(3):
            all_attrs = []
            for fold_idx in range(5):
                model, test_loader, _ = load_model_and_data(path_models, feats_code, bin_size, tp, tf, split_code, cw_type, fold_idx)
                ig = IntegratedGradients(model)
                attrs_fold = [ig.attribute(inputs.to(device), target=target_class).cpu().detach().numpy().reshape(inputs.size(0), -1) for inputs, _ in test_loader]
                all_attrs.extend(np.concatenate(attrs_fold, axis=0))
            all_attrs = np.array(all_attrs)
            reduced = reducer.fit_transform(all_attrs)
            axs[i, target_class].scatter(reduced[:, 0], reduced[:, 1], alpha=0.5)
            axs[i, target_class].set_title(f'{reducer_names[i]} - Class {target_class}')
            axs[i, target_class].grid()

    plt.tight_layout()
    save_path = f"{RESULTS_BASE}PM_sc{split_code}_mv1_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}_fold{fold_idx}_model_grads_visual.png"
    plt.savefig(save_path)
    plt.show()


def plot_folds_attributions(models_base_path, ds_path, feats_code, bin_size, tp, tf, split_code, cw_type, num_folds=5,
                            target_class=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    freq = 32
    batch_size = 32
    selected_features, EEG_channels = train.set_features(feats_code)
    data_dict = data_utils.load_data_to_dict(ds_path, selected_features)
    folds = train.generate_subject_kfolds(data_dict, k=num_folds)
    lstm_hidden_dim, lstm_num_layers, num_classes = train.set_lstm()
    num_sequences = tp // bin_size
    eegnet_params = {'num_electrodes': EEG_channels, 'chunk_size': tp * freq // num_sequences}

    fig, axs = plt.subplots(1, num_folds, figsize=(4 * num_folds, 4), sharey=True)
    for fold_idx, (train_uids, test_uids) in enumerate(folds):
        _, _, test_dict, _ = train.get_partitions_from_fold(data_dict, train_uids, test_uids, split_code, seed=42)
        test_data = data_utils.get_features_from_dict(test_dict, bin_size, freq)
        test_loader = train.create_dataloader(
            data_utils.AggressiveBehaviorDataset, test_data, tp, tf, bin_size,
            batch_size=batch_size, shuffle=False
        )
        model = EEGNetLSTM(eegnet_params, lstm_hidden_dim, lstm_num_layers, num_classes).to(device)
        model_path = f"{models_base_path}PM_sc{split_code}_mv1_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}_fold{fold_idx}_model.pth"
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
        model.train()  # modo entrenamiento por captum, no cambia gradientes

        ig = IntegratedGradients(model)
        fold_attributions = []
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            inputs.requires_grad = True
            attrs = ig.attribute(inputs, target=target_class).cpu().detach().numpy()
            fold_attributions.append(attrs)
        fold_attributions = np.concatenate(fold_attributions, axis=0) #  Promedio por fold
        mean_attributions = fold_attributions.mean(axis=0)

        time_bins = np.arange(-tp, 0, bin_size)
        channels = selected_features
        for i, channel in enumerate(channels):
            channel_attr = mean_attributions[:, i, :].mean(axis=1)
            axs[fold_idx].plot(time_bins, channel_attr, label=channel)
        axs[fold_idx].set_title(f'Fold {fold_idx + 1}')
        axs[fold_idx].set_xlabel('Time (sec)')
        if fold_idx == 0:
            axs[fold_idx].set_ylabel('Mean Attribution')
        axs[fold_idx].grid(True)

    axs[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.suptitle(f'Mean Integrated Gradients Attribution for class {target_class}')
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    save_path = f"{RESULTS_BASE}PM_sc{split_code}_mv1_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}_fold{fold_idx}_model_grads.png"
    plt.savefig(save_path)
    plt.show()


def plot_folds_attributions_avg(models_base_path, ds_path, feats_code, bin_size, tp, tf, split_code, cw_type, num_folds=5, target_class=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    freq = 32
    batch_size = 32
    selected_features, EEG_channels = train.set_features(feats_code)
    data_dict = data_utils.load_data_to_dict(ds_path, selected_features)
    folds = train.generate_subject_kfolds(data_dict, k=num_folds)
    lstm_hidden_dim, lstm_num_layers, num_classes = train.set_lstm()
    num_sequences = tp // bin_size
    eegnet_params = {'num_electrodes': EEG_channels, 'chunk_size': tp * freq // num_sequences}
    fig, axs = plt.subplots(1, num_folds + 1, figsize=(4*(num_folds + 1), 4), sharey=True)
    all_fold_attrs = []
    for fold_idx, (train_uids, test_uids) in enumerate(folds):
        _, _, test_dict, _ = train.get_partitions_from_fold(data_dict, train_uids, test_uids, split_code, seed=42)
        test_data = data_utils.get_features_from_dict(test_dict, bin_size, freq)
        test_loader = train.create_dataloader(
            data_utils.AggressiveBehaviorDataset, test_data, tp, tf, bin_size,
            batch_size=batch_size, shuffle=False
        )
        model = EEGNetLSTM(eegnet_params, lstm_hidden_dim, lstm_num_layers, num_classes).to(device)
        model_path = f"{models_base_path}PM_sc{split_code}_mv1_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}_fold{fold_idx}_model.pth"
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
        model.train()

        ig = IntegratedGradients(model)
        fold_attrs = [ig.attribute(inputs.to(device), target=target_class).cpu().detach().numpy() for inputs, _ in test_loader]
        fold_attrs = np.concatenate(fold_attrs, axis=0)
        mean_attributions = fold_attrs.mean(axis=0)
        all_fold_attrs.append(mean_attributions)
        time_bins = np.arange(-tp, 0, bin_size)
        channels = selected_features
        for i, channel in enumerate(channels):
            channel_attr = mean_attributions[:, i, :].mean(axis=1)
            axs[fold_idx].plot(time_bins, channel_attr, label=channel)
        axs[fold_idx].set_title(f'Fold {fold_idx + 1}')
        axs[fold_idx].set_xlabel('Time (sec)')
        axs[fold_idx].grid(True)

    mean_all_folds = np.mean(all_fold_attrs, axis=0)
    for i, channel in enumerate(channels):
        channel_attr = mean_all_folds[:, i, :].mean(axis=1)
        axs[-1].plot(time_bins, channel_attr, label=channel)

    axs[-1].set_title('Mean Across All Folds')
    axs[-1].set_xlabel('Time (sec)')
    axs[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axs[-1].grid(True)
    plt.tight_layout()
    save_path = f"{RESULTS_BASE}PM_sc{split_code}_mv1_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}_fold{fold_idx}_model_grads_avg.png"
    plt.savefig(save_path)
    #plt.show()


def feature_importance_analysis(path_models, ds_path, feats_code, bin_size, tp, tf, split_code, cw_type, fold_idx=0, target_class=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_features, EEG_channels = train.set_features(feats_code)
    data_dict = data_utils.load_data_to_dict(ds_path, selected_features)
    _, _, test_dict, _ = train.get_partitions_from_fold(data_dict, *train.generate_subject_kfolds(data_dict)[fold_idx], split_code, seed=42)
    test_data = data_utils.get_features_from_dict(test_dict, bin_size, 32)
    test_loader = train.create_dataloader(data_utils.AggressiveBehaviorDataset, test_data, tp, tf, bin_size, batch_size=32, shuffle=False)

    lstm_hidden_dim, lstm_num_layers, num_classes = train.set_lstm()
    num_sequences = tp // bin_size
    eegnet_params = {'num_electrodes': EEG_channels, 'chunk_size': tp * 32 // num_sequences}
    model = EEGNetLSTM(eegnet_params, lstm_hidden_dim, lstm_num_layers, num_classes).to(device)
    model.load_state_dict(torch.load(f"{path_models}PM_sc{split_code}_mv1_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}_fold{fold_idx}_model.pth", map_location=device), strict=True)
    model.train()

    ig = IntegratedGradients(model)
    all_attrs = []
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        attributions = ig.attribute(inputs, target=target_class)
        all_attrs.append(attributions.cpu().detach().numpy())

    all_attrs = np.concatenate(all_attrs, axis=0)  # (N, chunks, features, time)
    mean_attr = all_attrs.mean(axis=(0, 3))        # (chunks, features)
    mean_attr_per_feature = mean_attr.mean(axis=0) # (features,)

    assert len(mean_attr_per_feature) == len(selected_features), "Dimension mismatch!"
    plt.figure(figsize=(10, 5))
    plt.bar(selected_features, mean_attr_per_feature)
    plt.xlabel('Features')
    plt.ylabel('Mean Attribution')
    plt.title('Feature Importance Analysis')
    plt.grid(True)
    save_path = f"{RESULTS_BASE}PM_sc{split_code}_mv1_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}_fold{fold_idx}_model_grads_feature_importance.png"
    plt.savefig(save_path)


def feature_importance_analysis_Nsamples(path_models, ds_path, feats_code, bin_size, tp, tf, split_code, cw_type, fold_idx=0, target_class=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_features, EEG_channels = train.set_features(feats_code)
    data_dict = data_utils.load_data_to_dict(ds_path, selected_features)
    _, _, test_dict, _ = train.get_partitions_from_fold(
        data_dict, *train.generate_subject_kfolds(data_dict)[fold_idx], split_code, seed=42)
    test_data = data_utils.get_features_from_dict(test_dict, bin_size, 32)
    test_loader = train.create_dataloader(
        data_utils.AggressiveBehaviorDataset, test_data, tp, tf, bin_size, batch_size=32, shuffle=False)

    lstm_hidden_dim, lstm_num_layers, num_classes = train.set_lstm()
    num_sequences = tp // bin_size
    eegnet_params = {
        'num_electrodes': EEG_channels,
        'chunk_size': tp * 32 // num_sequences
    }
    model = EEGNetLSTM(eegnet_params, lstm_hidden_dim, lstm_num_layers, num_classes).to(device)
    model.load_state_dict(
        torch.load(f"{path_models}PM_sc{split_code}_mv1_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}_fold{fold_idx}_model.pth",
                   map_location=device), strict=True)
    model.train()

    ig = IntegratedGradients(model)
    all_attrs = []
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        attributions = ig.attribute(inputs, target=target_class)
        all_attrs.append(attributions.cpu().detach().numpy())
    all_attrs = np.concatenate(all_attrs, axis=0)  # (N, chunks, features, time)
    mean_attr = all_attrs.mean(axis=(0, 3))        # (chunks, features)
    mean_attr_per_feature = mean_attr.mean(axis=0) # (features,)

    assert len(mean_attr_per_feature) == len(selected_features), "Dimension mismatch!!!"
    plt.figure(figsize=(10, 5))
    plt.bar(selected_features, mean_attr_per_feature)
    plt.xlabel('Features')
    plt.ylabel('Mean Attribution')
    plt.title('Feature Importance Analysis')
    plt.grid(True)
    save_path = f"{RESULTS_BASE}PM_sc{split_code}_mv1_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}_fold{fold_idx}_model_grads_feature_importance_Nsamples.png"
    plt.savefig(save_path)


def feature_importance_stacked(path_models, ds_path, feats_code, bin_size, tp, tf, split_code, cw_type,
                               fold_idx=0, target_class=1, max_samples=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_features, EEG_channels = train.set_features(feats_code)
    data_dict = data_utils.load_data_to_dict(ds_path, selected_features)
    _, _, test_dict, _ = train.get_partitions_from_fold(
        data_dict, *train.generate_subject_kfolds(data_dict)[fold_idx], split_code, seed=42)
    test_data = data_utils.get_features_from_dict(test_dict, bin_size, 32)
    test_loader = train.create_dataloader(
        data_utils.AggressiveBehaviorDataset, test_data, tp, tf, bin_size, batch_size=1, shuffle=False)

    lstm_hidden_dim, lstm_num_layers, num_classes = train.set_lstm()
    num_sequences = tp // bin_size
    eegnet_params = {
        'num_electrodes': EEG_channels,
        'chunk_size': tp * 32 // num_sequences
    }
    model = EEGNetLSTM(eegnet_params, lstm_hidden_dim, lstm_num_layers, num_classes).to(device)
    model_path = f"{path_models}PM_sc{split_code}_mv1_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}_fold{fold_idx}_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    model.train()

    ig = IntegratedGradients(model)
    feature_attr_matrix = []
    count = 0
    for inputs, _ in test_loader:
        if count >= max_samples:
            break
        inputs = inputs.to(device)
        attributions = ig.attribute(inputs, target=target_class)
        attrs = attributions.cpu().detach().numpy()  # (1, chunks, features, time)
        attrs = attrs.mean(axis=(1, 3))  # (1, features)
        feature_attr_matrix.append(attrs.squeeze())
        count += 1
    feature_attr_matrix = np.array(feature_attr_matrix)  # (samples, features)

    x = np.arange(len(selected_features))
    fig, ax = plt.subplots(figsize=(12, 6))
    bottoms = np.zeros_like(x, dtype=np.float32)
    colors = plt.cm.tab10(np.linspace(0, 1, max_samples))
    for i in range(feature_attr_matrix.shape[0]):
        ax.bar(x, feature_attr_matrix[i], bottom=bottoms, label=f'Sample{i+1}', color=colors[i])
        bottoms += feature_attr_matrix[i]

    ax.set_xticks(x)
    ax.set_xticklabels(selected_features, rotation=45)
    ax.set_ylabel('Attributions')
    ax.set_xlabel('Input Features')
    ax.set_title(f'Feature importances for {max_samples} samples (Fold {fold_idx})')
    ax.legend()
    plt.tight_layout()
    save_path = f"{RESULTS_BASE}PM_sc{split_code}_mv1_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}_fold{fold_idx}_stacked_feature_importance.png"
    plt.savefig(save_path)
    plt.close()



# Activaciones de la primera capa convolucional
def visualize_first_conv_activations(path_models, ds_path, feats_code, bin_size, tp, tf, split_code, cw_type, fold_idx=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_features, EEG_channels = train.set_features(feats_code)
    data_dict = data_utils.load_data_to_dict(ds_path, selected_features)
    _, _, test_dict, _ = train.get_partitions_from_fold(
        data_dict, *train.generate_subject_kfolds(data_dict)[fold_idx], split_code, seed=42)
    test_data = data_utils.get_features_from_dict(test_dict, bin_size, 32)
    test_loader = train.create_dataloader(
        data_utils.AggressiveBehaviorDataset, test_data, tp, tf, bin_size, batch_size=1, shuffle=False)

    lstm_hidden_dim, lstm_num_layers, num_classes = train.set_lstm()
    num_sequences = tp // bin_size
    eegnet_params = {
        'num_electrodes': EEG_channels,
        'chunk_size': tp * 32 // num_sequences
    }
    model = EEGNetLSTM(eegnet_params, lstm_hidden_dim, lstm_num_layers, num_classes).to(device)
    model_path = f"{path_models}PM_sc{split_code}_mv1_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}_fold{fold_idx}_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    model.eval()

    inputs, _ = next(iter(test_loader))
    inputs = inputs.to(device)
    if inputs.dim() == 4 and inputs.size(1) != 1:
        inputs = inputs.permute(0, 2, 1, 3) #(B, features, chunks, time)
        inputs = inputs.reshape(inputs.shape[0], 1, EEG_channels, -1)  # (B, 1, C, T)

    with torch.no_grad():
        conv_output = model.eegnet.block1[0](inputs)
    conv_output = conv_output.cpu().numpy().squeeze() #(filters, height, width)
    num_filters = conv_output.shape[0]
    fig, axs = plt.subplots(num_filters, 1, figsize=(12, 2 * num_filters), squeeze=False)
    for i in range(num_filters):
        axs[i, 0].imshow(conv_output[i], aspect='auto', cmap='viridis')
        axs[i, 0].set_title(f'Filtro {i+1} - Activaciones')
        axs[i, 0].set_ylabel('Espacial')
        axs[i, 0].set_xlabel('Tiempo')
        axs[i, 0].grid(False)

    plt.tight_layout()
    save_path = f"{RESULTS_BASE}PM_sc{split_code}_mv1_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}_fold{fold_idx}_first_conv_activ.png"
    plt.savefig(save_path)


def visualize_first_conv_activations_grid(path_models, ds_path, feats_code, bin_size, tp, tf, split_code, cw_type, fold_idx=0, max_samples=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_features, EEG_channels = train.set_features(feats_code)
    data_dict = data_utils.load_data_to_dict(ds_path, selected_features)
    _, _, test_dict, _ = train.get_partitions_from_fold(
        data_dict, *train.generate_subject_kfolds(data_dict)[fold_idx], split_code, seed=42)
    test_data = data_utils.get_features_from_dict(test_dict, bin_size, 32)
    test_loader = train.create_dataloader(
        data_utils.AggressiveBehaviorDataset, test_data, tp, tf, bin_size, batch_size=1, shuffle=False)

    lstm_hidden_dim, lstm_num_layers, num_classes = train.set_lstm()
    num_sequences = tp // bin_size
    eegnet_params = {
        'num_electrodes': EEG_channels,
        'chunk_size': tp * 32 // num_sequences
    }
    model = EEGNetLSTM(eegnet_params, lstm_hidden_dim, lstm_num_layers, num_classes).to(device)
    model_path = f"{path_models}PM_sc{split_code}_mv1_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}_fold{fold_idx}_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    model.eval()

    all_activations = []
    count = 0
    for inputs, _ in test_loader:
        if count >= max_samples:
            break
        inputs = inputs.to(device)
        if inputs.dim() == 4 and inputs.size(1) != 1:
            inputs = inputs.permute(0, 2, 1, 3)
            inputs = inputs.reshape(inputs.shape[0], 1, EEG_channels, -1)
        with torch.no_grad():
            conv_output = model.eegnet.block1[0](inputs) #(1, F, H, W)
        all_activations.append(conv_output.squeeze(0).cpu().numpy()) #(F, H, W)
        count += 1

    num_samples = len(all_activations)
    num_filters = all_activations[0].shape[0]
    fig, axs = plt.subplots(num_samples, num_filters, figsize=(3 * num_filters, 2 * num_samples), squeeze=False)
    for i in range(num_samples):
        for j in range(num_filters):
            axs[i, j].imshow(all_activations[i][j], aspect='auto', cmap='viridis')
            if i == 0:
                axs[i, j].set_title(f'Filter {j + 1}')
            if j == 0:
                axs[i, j].set_ylabel(f'Sample {i + 1}')
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])

    plt.suptitle('Activations first convolutional layer', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = f"{RESULTS_BASE}PM_sc{split_code}_mv1_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}_fold{fold_idx}_conv_grid.png"
    plt.savefig(save_path)
    plt.close()


def visualize_conv_activations_by_class(path_models, ds_path, feats_code, bin_size, tp, tf, split_code, cw_type,
                                        fold_idx=0, max_samples_per_class=3, target_classes=[0, 1, 2]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_features, EEG_channels = train.set_features(feats_code)
    data_dict = data_utils.load_data_to_dict(ds_path, selected_features)
    _, _, test_dict, _ = train.get_partitions_from_fold(
        data_dict, *train.generate_subject_kfolds(data_dict)[fold_idx], split_code, seed=42)

    test_data = data_utils.get_features_from_dict(test_dict, bin_size, 32)
    test_loader = train.create_dataloader(
        data_utils.AggressiveBehaviorDataset, test_data, tp, tf, bin_size, batch_size=1, shuffle=False)

    lstm_hidden_dim, lstm_num_layers, num_classes = train.set_lstm()
    num_sequences = tp // bin_size
    eegnet_params = {
        'num_electrodes': EEG_channels,
        'chunk_size': tp * 32 // num_sequences
    }

    model = EEGNetLSTM(eegnet_params, lstm_hidden_dim, lstm_num_layers, num_classes).to(device)
    model_path = f"{path_models}PM_sc{split_code}_mv1_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}_fold{fold_idx}_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    model.eval()

    # Almacena activaciones por clase
    class_activations = {cls: [] for cls in target_classes}

    for inputs, labels in test_loader:
        label = labels.item()
        if label in target_classes and len(class_activations[label]) < max_samples_per_class:
            inputs = inputs.to(device)
            if inputs.dim() == 4 and inputs.size(1) != 1:
                inputs = inputs.permute(0, 2, 1, 3)
                inputs = inputs.reshape(inputs.shape[0], 1, EEG_channels, -1)
            with torch.no_grad():
                conv_output = model.eegnet.block1[0](inputs) #(1, F, H, W)
            class_activations[label].append(conv_output.squeeze(0).cpu().numpy())  #(F, H, W)

        if all(len(class_activations[cls]) >= max_samples_per_class for cls in target_classes):
            break

    # Plot por clase...
    for cls in target_classes:
        samples = class_activations[cls]
        num_samples = len(samples)
        num_filters = samples[0].shape[0]

        fig, axs = plt.subplots(num_samples, num_filters, figsize=(3 * num_filters, 2 * num_samples), squeeze=False)
        for i in range(num_samples):
            for j in range(num_filters):
                axs[i, j].imshow(samples[i][j], aspect='auto', cmap='viridis')
                if i == 0:
                    axs[i, j].set_title(f'Filter {j + 1}')
                if j == 0:
                    axs[i, j].set_ylabel(f'Sample {i + 1}')
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])

        plt.suptitle(f'Activations - Class {cls}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = f"{RESULTS_BASE}PM_sc{split_code}_mv1_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}_fold{fold_idx}_class{cls}_conv_grid.png"
        plt.savefig(save_path)
        plt.close()



def visualize_conv_activations_by_class_secs(path_models, ds_path, feats_code, bin_size, tp, tf, split_code, cw_type,
                                        fold_idx=0, max_samples_per_class=3, target_classes=[0, 1, 2]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_features, EEG_channels = train.set_features(feats_code)
    data_dict = data_utils.load_data_to_dict(ds_path, selected_features)

    _, _, test_dict, _ = train.get_partitions_from_fold(
        data_dict, *train.generate_subject_kfolds(data_dict)[fold_idx], split_code, seed=42)
    test_data = data_utils.get_features_from_dict(test_dict, bin_size, 32)
    test_loader = train.create_dataloader(
        data_utils.AggressiveBehaviorDataset, test_data, tp, tf, bin_size, batch_size=1, shuffle=False)

    lstm_hidden_dim, lstm_num_layers, num_classes = train.set_lstm()
    num_sequences = tp // bin_size
    eegnet_params = {
        'num_electrodes': EEG_channels,
        'chunk_size': tp * 32 // num_sequences
    }
    model = EEGNetLSTM(eegnet_params, lstm_hidden_dim, lstm_num_layers, num_classes).to(device)
    model_path = f"{path_models}PM_sc{split_code}_mv1_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}_fold{fold_idx}_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    model.eval()

    class_activations = {cls: [] for cls in target_classes}
    fs = 32
    for inputs, labels in test_loader:
        label = labels.item()
        if label in target_classes and len(class_activations[label]) < max_samples_per_class:
            inputs = inputs.to(device)
            if inputs.dim() == 4 and inputs.size(1) != 1:
                inputs = inputs.permute(0, 2, 1, 3)
                inputs = inputs.reshape(inputs.shape[0], 1, EEG_channels, -1)

            with torch.no_grad():
                conv_output = model.eegnet.block1[0](inputs)  # Output primera capa conv

            class_activations[label].append(conv_output.squeeze(0).cpu().numpy())  #(F, H, W)

        if all(len(class_activations[cls]) >= max_samples_per_class for cls in target_classes):
            break

    for cls in target_classes:
        samples = class_activations[cls]
        num_samples = len(samples)
        num_filters = samples[0].shape[0]
        fig, axs = plt.subplots(num_samples, num_filters, figsize=(3 * num_filters, 2 * num_samples), squeeze=False)
        for i in range(num_samples):
            for j in range(num_filters):
                activation = samples[i][j]
                time_len = activation.shape[1]
                # eje temporal en segundos
                xtick_positions = np.linspace(0, time_len - 1, 5)
                xtick_labels = np.round(xtick_positions / fs).astype(int)
                axs[i, j].imshow(activation, aspect='auto', cmap='viridis')
                axs[i, j].set_xticks(xtick_positions)
                axs[i, j].set_xticklabels(xtick_labels)
                axs[i, j].set_xlabel("Time (s)")
                axs[i, j].set_ylabel("Spatial")
                if i == 0:
                    axs[i, j].set_title(f'Filter {j + 1}')

        plt.suptitle(f'Activations - Class {cls}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = f"{RESULTS_BASE}PM_sc{split_code}_mv1_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}_fold{fold_idx}_class{cls}_conv_grid_secs.png"
        plt.savefig(save_path)
        plt.close()


def visualize_f2_activations_by_class(path_models, ds_path, feats_code, bin_size, tp, tf, split_code, cw_type,
                                      fold_idx=0, max_samples_per_class=3, target_classes=[0, 1, 2]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_features, EEG_channels = train.set_features(feats_code)
    data_dict = data_utils.load_data_to_dict(ds_path, selected_features)
    _, _, test_dict, _ = train.get_partitions_from_fold(
        data_dict, *train.generate_subject_kfolds(data_dict)[fold_idx], split_code, seed=42)
    test_data = data_utils.get_features_from_dict(test_dict, bin_size, 32)
    test_loader = train.create_dataloader(
        data_utils.AggressiveBehaviorDataset, test_data, tp, tf, bin_size, batch_size=1, shuffle=False)

    lstm_hidden_dim, lstm_num_layers, num_classes = train.set_lstm()
    num_sequences = tp // bin_size
    eegnet_params = {
        'num_electrodes': EEG_channels,
        'chunk_size': tp * 32 // num_sequences
    }
    model = EEGNetLSTM(eegnet_params, lstm_hidden_dim, lstm_num_layers, num_classes).to(device)
    model_path = f"{path_models}PM_sc{split_code}_mv1_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}_fold{fold_idx}_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    model.eval()

    class_activations = {cls: [] for cls in target_classes}
    for inputs, labels in test_loader:
        label = labels.item()
        if label in target_classes and len(class_activations[label]) < max_samples_per_class:
            inputs = inputs.to(device)
            with torch.no_grad():
                inputs = inputs.squeeze(0)  #[12, 5, 480]
                inputs = inputs.permute(1, 0, 2)  #[5, 12, 480]
                inputs = inputs.reshape(1, 1, -1, 480)  #[1, 1, 60, 480] (5*12=60 features combinadas)
                x = model.eegnet.block1(inputs)
                x = model.eegnet.block2[0](x)  # F2 activs (pointwise conv)
            class_activations[label].append(x.squeeze(0).cpu().numpy())  #(F2, H, W)
        if all(len(class_activations[cls]) >= max_samples_per_class for cls in target_classes):
            break

    for cls in target_classes:
        samples = class_activations[cls]
        num_samples = len(samples)
        num_filters = samples[0].shape[0]
        fig, axs = plt.subplots(num_samples, num_filters, figsize=(3 * num_filters, 2 * num_samples), squeeze=False)
        for i in range(num_samples):
            for j in range(num_filters):
                axs[i, j].imshow(samples[i][j], aspect='auto', cmap='viridis')
                if i == 0:
                    axs[i, j].set_title(f'Filter F2-{j + 1}')
                if j == 0:
                    axs[i, j].set_ylabel(f'Sample {i + 1}')
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])
        plt.suptitle(f'Activations - Filters F2 - Class {cls}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = f"{RESULTS_BASE}PM_sc{split_code}_mv1_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}_fold{fold_idx}_class{cls}_F2_activations.png"
        plt.savefig(save_path)
        plt.close()


def visualize_f2_weights(path_models, feats_code, tf, tp, bin_size, split_code, cw_type, fold_idx=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_features, EEG_channels = train.set_features(feats_code)
    lstm_hidden_dim, lstm_num_layers, num_classes = train.set_lstm()
    num_sequences = tp // bin_size
    eegnet_params = {'num_electrodes': EEG_channels, 'chunk_size': tp * 32 // num_sequences}
    model = EEGNetLSTM(eegnet_params, lstm_hidden_dim, lstm_num_layers, num_classes).to(device)
    model_path = f"{path_models}PM_sc{split_code}_mv1_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}_fold{fold_idx}_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    model.eval()

    # pesos de la segunda conv2d de block2 (pointwise conv que genera F2)
    weights = model.eegnet.block2[1].weight.data.cpu().numpy()
    print("Shape original pesos F2:", weights.shape)
    if weights.ndim != 4 or weights.shape[2:] != (1, 1):
        raise ValueError(f"Unexpected shape for F2 weights: {weights.shape}. Expected (F2, F1*D, 1, 1)")
    weights = weights.squeeze(-1).squeeze(-1)
    # Normalizar para ver mejor...
    weights_norm = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
    num_filters = weights.shape[0]
    fig, axs = plt.subplots(1, num_filters, figsize=(2.5 * num_filters, 3), squeeze=False)
    for i in range(num_filters):
        ax = axs[0, i]
        ax.imshow(weights_norm[i][np.newaxis, :], cmap='viridis', aspect='auto')
        ax.set_title(f'Weight F2-{i+1}')
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Weight Visualization F2 Filters (Pointwise Conv)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_path = f"{RESULTS_BASE}PM_sc{split_code}_mv1_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}_fold{fold_idx}_F2_weights_fixed.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def usage_example():
    MODEL_PATH = './models/PM_sc1_mv1_f0_tf180_tp180_bs15_cw2_fold0_model.pth'
    DS_PATH = './dataset_resampled/dataset_32Hz.csv'
    FEATS_CODE = 0
    BIN_SIZE = 15
    TP, TF = 180, 180
    FREQ = 32
    SPLIT_CODE = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_features, EEG_channels = train.set_features(FEATS_CODE)
    data_dict = data_utils.load_data_to_dict(DS_PATH, selected_features)

    folds = train.generate_subject_kfolds(data_dict, k=5)
    train_uids, test_uids = folds[0]  # fold 0 como ejemplo
    _, _, test_dict, _ = train.get_partitions_from_fold(data_dict, train_uids, test_uids, SPLIT_CODE, seed=42)
    features_fun = data_utils.get_features_from_dict
    dataloader_fun = data_utils.AggressiveBehaviorDataset
    test_data = features_fun(test_dict, BIN_SIZE, FREQ)
    test_loader = train.create_dataloader(dataloader_fun, test_data, TP, TF, BIN_SIZE, batch_size=1, shuffle=False)

    lstm_hidden_dim, lstm_num_layers, num_classes = train.set_lstm()
    num_sequences = TP // BIN_SIZE
    eegnet_params = {'num_electrodes': EEG_channels, 'chunk_size': TP * FREQ // num_sequences}
    model = EEGNetLSTM(eegnet_params, lstm_hidden_dim, lstm_num_layers, num_classes).to(device)
    #model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=True)
    #model.eval()
    model.train()  # Importante: para que cudnn RNN backward funcione con Captum, pero no modifica grandientes...
    ig = IntegratedGradients(model)

    '''
    # Sample de prueba
    inputs, label = next(iter(test_loader))
    inputs = inputs.to(device)
    inputs.requires_grad = True
    # Calcular feature importance para la clase "Pre-episode" (target=1)
    attributions, delta = ig.attribute(inputs, target=1, return_convergence_delta=True)
    attributions = attributions.cpu().detach().numpy().squeeze()
    '''

    fold_attributions = []
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        inputs.requires_grad = True
        attrs = ig.attribute(inputs, target=1).cpu().detach().numpy()
        fold_attributions.append(attrs)

    fold_attributions = np.concatenate(fold_attributions, axis=0)
    attributions = fold_attributions.mean(axis=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    time_bins = np.arange(-TP, 0, BIN_SIZE)
    channels = selected_features

    for i, channel in enumerate(channels):
        channel_attr = attributions[:, i, :].mean(axis=1)
        ax.plot(time_bins, channel_attr, label=channel)

    ax.axhline(0, color='black', lw=0.5)
    ax.set_xlabel('Time (seconds before prediction)')
    ax.set_ylabel('Attribution Score')
    ax.set_title('Feature Attribution using Integrated Gradients for class "Pre-episode"')
    ax.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./evaluate_grads_preatt_fold0.png')



models_base_path = "./models/"
ds_path = "./dataset_resampled/dataset_32Hz.csv"
feats_code = 0
model_version = 1
bin_size = 15
tp, tf = 300, 300
split_code = 1
cw_type = 1
num_folds = 5
target_class = 1  # "Pre-episode"
#usage_example()
#plot_folds_attributions(models_base_path, ds_path, feats_code, bin_size, tp, tf, split_code, cw_type, num_folds, target_class)
plot_folds_attributions_avg(models_base_path, ds_path, feats_code, bin_size, tp, tf, split_code, cw_type, num_folds, target_class)
#feature_importance_analysis(models_base_path, ds_path, feats_code, bin_size, tp, tf, split_code, cw_type)
feature_importance_analysis_Nsamples(models_base_path, ds_path, feats_code, bin_size, tp, tf, split_code, cw_type)
feature_importance_stacked(models_base_path, ds_path, feats_code, bin_size, tp, tf, split_code, cw_type, fold_idx=0, target_class=1, max_samples=5)
#visualize_first_conv_activations(models_base_path, ds_path, feats_code, bin_size, tp, tf, split_code, cw_type, fold_idx=0)
#visualize_first_conv_activations_grid(models_base_path, ds_path, feats_code, bin_size, tp, tf, split_code, cw_type, fold_idx=0, max_samples=5)
visualize_conv_activations_by_class_secs(models_base_path, ds_path, feats_code, bin_size, tp, tf, split_code, cw_type, fold_idx=0, max_samples_per_class=3, target_classes=[0, 1, 2])
visualize_f2_activations_by_class(models_base_path, ds_path, feats_code, bin_size, tp, tf, split_code, cw_type, fold_idx=0, max_samples_per_class=3, target_classes=[0, 1, 2])
#visualize_f2_weights(models_base_path, feats_code, tf, tp, bin_size, split_code, cw_type, fold_idx=0)
#evaluate_classes_model(models_base_path, model_version, feats_code, tf, tp, bin_size, split_code, cw_type, seed=1)
evaluate_classes_model_signals(models_base_path, model_version, feats_code, tf, tp, bin_size, split_code, cw_type, seed=1)
#evaluate_errors_model(models_base_path, model_version, feats_code, tf, tp, bin_size, split_code, cw_type, seed=1)
evaluate_errors_model_signals(models_base_path, model_version, feats_code, tf, tp, bin_size, split_code, cw_type, seed=1)
#evaluate_visualization_model(models_base_path, model_version, feats_code, tf, tp, bin_size, split_code, cw_type, seed=1)
tps=[60,180,300]
#evaluate_tp_model(models_base_path, model_version, feats_code, tf, tps, bin_size, split_code, cw_type, seed=1)
evaluate_tp_model_signals(models_base_path, model_version, feats_code, tf, tps, bin_size, split_code, cw_type, seed=1)