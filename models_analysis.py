# model_path_example = mv1_f0_tf180_tp180_fold0_model.pth
import torch
import numpy as np
import data_utils
import models_utils
import train
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, confusion_matrix


def evaluate_all_results(all_labels, all_probs):
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    auc_score = roc_auc_score(all_labels, all_probs)
    best_f1, best_threshold_f1 = 0, 0
    f1_scores = []
    tnr = []
    fnr = []

    for threshold in thresholds:
        predictions = (np.array(all_probs) >= threshold).astype(int)
        f1 = f1_score(all_labels, predictions)
        f1_scores.append(f1)

        tn, fp, fn, tp = confusion_matrix(all_labels, predictions).ravel()
        total_positive = tp + fn
        total_negative = tn + fp

        false_negative_rate = fn / total_positive if total_positive > 0 else 0
        true_negative_rate = tn / total_negative if total_negative > 0 else 0

        fnr.append(false_negative_rate)
        tnr.append(true_negative_rate)

        if f1 > best_f1:
            best_f1, best_threshold_f1 = f1, threshold

    best_tpr_idx = np.argmax(tpr - fpr)
    best_threshold_roc = thresholds[best_tpr_idx]
    best_fpr_at_auc = fpr[best_tpr_idx]
    best_tpr_at_auc = tpr[best_tpr_idx]

    return auc_score, fpr, tpr, tnr, fnr, best_f1, best_threshold_f1, thresholds, f1_scores, best_threshold_roc, best_fpr_at_auc, best_tpr_at_auc


def plot_five_folds_metrics(folds_metrics):
    """
    Genera una figura con 5 subgráficos, uno por cada fold, mostrando TPR y FNR en función de los thresholds.

    Args:
        folds_metrics (list): Lista de metricas para cada fold. Cada elemento debe ser un diccionario con las claves:
                              'fpr', 'tpr', 'fnr', 'thresholds', 'auc_score', 'fold_idx'.
    """
    fig, axes = plt.subplots(1, 5, figsize=(25, 5), sharey=True)

    for metrics, ax in zip(folds_metrics, axes):
        ax.plot(metrics['thresholds'], metrics['tpr'], label='TPR', linestyle='-', color='orange')
        ax.plot(metrics['thresholds'], metrics['fnr'], label='FNR', linestyle='--', color='blue')
        ax.set_title(f"Fold {metrics['fold_idx']}\nAUC: {metrics['auc_score']:.4f}")
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Rate')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()



def plot_five_folds_rates(folds_metrics):
    """
    TPR, FNR y FPR para cada fold en función de los thresholds.

    Args:
        folds_metrics (list): Lista de metricas para cada fold. Cada elemento debe ser un diccionario con las claves:
                              'fpr', 'tpr', 'fnr', 'thresholds', 'auc_score', 'fold_idx'.
    """
    fig, axes = plt.subplots(1, 5, figsize=(25, 5), sharey=True)

    for metrics, ax in zip(folds_metrics, axes):
        ax.plot(metrics['thresholds'], metrics['tpr'], label='TPR (Episodes Detected)', linestyle='-', color='green')
        ax.plot(metrics['thresholds'], metrics['fnr'], label='FNR (Episodes Missed)', linestyle='--', color='red')
        ax.plot(metrics['thresholds'], metrics['fpr'], label='FPR (False Alarms)', linestyle='-.', color='blue')
        ax.set_title(f"Fold {metrics['fold_idx']}\nAUC: {metrics['auc_score']:.4f}")
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Rate')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()



def plot_f1_scores(folds_metrics):
    """
    Valores de F1-Score en función de los thresholds.

    Args:
        folds_metrics (list): Lista de metricas para cada fold. Cada elemento debe ser un diccionario con las claves:
                              'f1_scores', 'thresholds', 'auc_score', 'fold_idx'.
    """
    fig, axes = plt.subplots(1, 5, figsize=(25, 5), sharey=True)

    for metrics, ax in zip(folds_metrics, axes):
        ax.plot(metrics['thresholds'], metrics['f1_scores'], label='F1-Score', color='purple', linestyle='-')
        ax.set_title(f"Fold {metrics['fold_idx']}\nAUC: {metrics['auc_score']:.4f}")
        ax.set_xlabel('Threshold')
        ax.set_ylabel('F1-Score')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()



def plot_combined_metrics(folds_metrics, path_results):
    """
    TPR, FNR, FPR y F1-Score para cada fold en función de los thresholds.

    Args:
        folds_metrics (list): Lista de metricas para cada fold. Cada elemento debe ser un diccionario con las claves:
                              'fpr', 'tpr', 'fnr', 'thresholds', 'auc_score', 'fold_idx', 'f1_scores'.
    """
    fig, axes = plt.subplots(1, 5, figsize=(25, 5), sharey=True)

    for metrics, ax in zip(folds_metrics, axes):
        if 'f1_scores' not in metrics or 'thresholds' not in metrics:
            print(f"Error: Missing keys in fold metrics for Fold {metrics.get('fold_idx', 'Unknown')}")
            continue

        # Plot TPR, FNR, and FPR
        ax.plot(metrics['thresholds'], metrics['tpr'], label='TPR (Episodes Detected)', linestyle='-', color='green')
        ax.plot(metrics['thresholds'], metrics['fnr'], label='FNR (Episodes Missed)', linestyle='--', color='red')
        ax.plot(metrics['thresholds'], metrics['fpr'], label='FPR (False Alarms)', linestyle='-.', color='blue')

        # Plot F1-Score
        ax.plot(metrics['thresholds'], metrics['f1_scores'], label='F1-Score', color='purple', linestyle='-')

        # Title and labels
        ax.set_title(f"Fold {metrics['fold_idx']}\nAUC: {metrics['auc_score']:.4f}")
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Rate / F1-Score')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path_results)
    plt.show()


def plot_distribution_all_folds(folds_metrics, path_to_save):
    """
    Distribución de probabilidades por cada fold.
    """
    fig, axes = plt.subplots(1, 5, figsize=(25, 5), sharey=True)

    for metrics, ax in zip(folds_metrics, axes):
        all_labels = metrics['all_labels']
        all_probs = metrics['all_probs']
        fold_idx = metrics['fold_idx']

        positive_probs = [p for p, label in zip(all_probs, all_labels) if label == 1]
        negative_probs = [p for p, label in zip(all_probs, all_labels) if label == 0]

        ax.hist(positive_probs, bins=20, alpha=0.6, color='green', label='Positive (Label=1)')
        ax.hist(negative_probs, bins=20, alpha=0.6, color='red', label='Negative (Label=0)')
        ax.set_title(f"Fold {metrics['fold_idx']}\nAUC: {metrics['auc_score']:.4f}")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path_to_save)
    plt.show()


def plot_tpr_vs_fpr_all_folds(folds_metrics, path_to_save):
    """
    TPR vs FPR para cada fold.
    """
    fig, axes = plt.subplots(1, 5, figsize=(25, 5), sharey=True)

    for metrics, ax in zip(folds_metrics, axes):
        fpr = metrics['fpr']
        tpr = metrics['tpr']
        auc_score = metrics['auc_score']
        fold_idx = metrics['fold_idx']

        ax.plot(fpr, tpr, marker='o', linestyle='-', color='blue', label=f"AUC={auc_score:.4f}")
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random Classifier")
        ax.set_title(f"Fold {fold_idx} TPR vs FPR")
        ax.set_xlabel("False Positive Rate (FPR)")
        ax.set_ylabel("True Positive Rate (TPR)")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path_to_save)
    plt.show()


model_exp = 0 # PM
model_versions = [2]
feat_code = 2
tp, tf = 180, 180
num_folds = 5
folds = np.arange(num_folds)

freq = 32
data_path_resampled = './dataset_resampled/'
ds_path = data_path_resampled + "dataset_" + str(freq) + "Hz.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eegnet_params = {"chunk_size": 151, "num_electrodes": 60, "F1": 8, "F2": 16, "D": 2, "num_classes": 2, "kernel_1": 64, "kernel_2": 16, "dropout": 0.25}
lstm_hidden_dim = 64

for mv in model_versions:
    folds_metrics = []
    # Pasos
    # 1) Cargar conjunto de test
    # 2) Cargar modelo
    # 3) Obtener métricas: auc_roc score, tpr, tfr, f1_score
    # 4) Obtener plots de distribución de métricas en diferentes thresholds (marcados por los que devuelve la función de scipy del auc-score)

    load_data_fun = data_utils.load_data_to_dict  # All,: ACC x, y, z, BVP, EDA, AGGObs, si mv 1 no aggObsr
    data_dict = load_data_fun(ds_path)
    num_folds = 5
    folds = train.generate_user_kfolds(data_dict, k=num_folds)
    tp, tf, stride, bin_size = tp, tf, 15, 15
    print(f'tp: {tp}, tf: {tf}, stride: {stride}, bin_size: {bin_size}.')
    for fold_idx, (train_uids, test_uids) in enumerate(folds):
        print(f"Fold {fold_idx + 1}:")
        print(f"  Train UIDs: {train_uids}")
        print(f"  Test UIDs: {test_uids}")

        # Obtener particiones de test en funcion de los ids del fold
        _, _, test_dict, _ = train.get_partitions_from_fold(data_dict, train_uids, test_uids, seed=1)  # same seed as in train
        features_fun, dataloader_fun = models_utils.get_dataloader(mv)
        test_data = features_fun(test_dict, tp, tf, bin_size, freq)
        batch_size = 128
        dataloader_test = train.create_dataloader(dataloader_fun, test_data, tp, bin_size, batch_size, shuffle=False)

        # create model
        num_sequences = tp // bin_size
        bin_size = 15
        eegnet_params = {
            'num_electrodes': 5,  # (EDA, ACC_X, ACC_Y, ACC_Z, BVP) ...
            'chunk_size': tp * freq // num_sequences  # muestras en cada ventana
        }
        lstm_hidden_dim = 64
        num_classes = 1

        model_path = f'./models/mv{mv}_f{feat_code}_tf{tf}_tp{tp}_fold{fold_idx}_model.pth'
        print(f"Evaluando modelo {model_path}")
        model = models_utils.load_model(mv, device, eegnet_params, lstm_hidden_dim, model_path, feat_code)
        all_probs, all_labels = models_utils.test_model(model, dataloader_test, device)
        auc_score, fpr, tpr, tnr, fnr, best_f1, best_threshold_f1, thresholds, f1_scores, best_threshold_roc, best_fpr_at_auc, best_tpr_at_auc = evaluate_all_results(
            all_labels, all_probs)

        print(f"Fold {fold_idx} - AUC: {auc_score:.4f}, Best F1: {best_f1:.4f}")

        #plot_combined_metrics(all_labels, all_probs, fpr, tpr, tnr, fnr, thresholds, f1_scores, auc_score, best_f1,
        #                      best_threshold_f1)

        folds_metrics.append({
            'fpr': fpr,
            'tpr': tpr,
            'fnr': fnr,
            'thresholds': thresholds,
            'auc_score': auc_score,
            'fold_idx': fold_idx + 1,
            'best_threshold_f1': best_threshold_f1,
            'best_f1': best_f1,
            'f1_scores': f1_scores,
            'all_probs': all_probs,
            'all_labels': all_labels
        })

    '''
    #plot_five_folds_metrics(folds_metrics)
    plot_five_folds_rates(folds_metrics)
    plot_f1_scores(folds_metrics)
    '''
    path_analysis_results = './results_analysis/'
    path_results = f"{path_analysis_results}PM_mv{mv}_f{feat_code}_tf{tf}_tp{tp}_probability_distribution.png"
    # Análisis para todos los folds
    plot_distribution_all_folds(folds_metrics, path_results)
    path_results = f"{path_analysis_results}PM_mv{mv}_f{feat_code}_tf{tf}_tp{tp}_tpr_vs_fpr.png"
    plot_tpr_vs_fpr_all_folds(folds_metrics, path_results)
    path_results = f"{path_analysis_results}PM_mv{mv}_f{feat_code}_tf{tf}_tp{tp}_combined_metrics.png"
    plot_combined_metrics(folds_metrics, path_results)