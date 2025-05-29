# model_path_example = mv1_f0_tf180_tp180_fold0_model.pth
import torch
import data_utils
import models_utils
import train
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, confusion_matrix
import seaborn as sns

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



def plot_f1_scores(folds_metrics, path_results):
    fig, axes = plt.subplots(1, 5, figsize=(25, 5), sharey=True)
    for metrics, ax in zip(folds_metrics, axes):
        ax.plot(metrics['thresholds'], metrics['f1_scores'], label='F1-Score', color='purple', linestyle='-')
        ax.set_title(f"Fold {metrics['fold_idx']}\nAUC: {metrics['auc_score']:.4f}")
        ax.set_xlabel('Threshold')
        ax.set_ylabel('F1-Score')
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path_results)
    plt.show()


def plot_mean_f1_scores(folds_metrics, path_results):
    # Obtener los thresholds comunes exactos en los 5 folds
    rounded_thresholds_list = [np.round(fm['thresholds'], 2) for fm in folds_metrics]
    common_thresholds = set(rounded_thresholds_list[0])
    for fm in folds_metrics[1:]:
        common_thresholds &= set(fm['thresholds'])
    common_thresholds = sorted(list(common_thresholds))

    if len(common_thresholds) == 0:
        raise ValueError("No hay thresholds comunes entre los folds.")

    # Extraer F1-scores correspondientes a los thresholds comunes
    f1_matrix = []
    for fm in folds_metrics:
        th_to_f1 = dict(zip(fm['thresholds'], fm['f1_scores']))
        f1_matrix.append([th_to_f1[th] for th in common_thresholds])

    f1_matrix = np.array(f1_matrix)
    mean_f1 = f1_matrix.mean(axis=0)

    # Encontrar el máximo F1 y el valor en threshold 0.5
    best_idx = np.argmax(mean_f1)
    best_thresh = common_thresholds[best_idx]
    best_f1 = mean_f1[best_idx]

    if 0.5 in common_thresholds:
        idx_05 = common_thresholds.index(0.5)
        f1_05 = mean_f1[idx_05]
    else:
        idx_05 = np.argmin(np.abs(np.array(common_thresholds) - 0.5))
        f1_05 = mean_f1[idx_05]

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(common_thresholds, mean_f1, color='seagreen', linewidth=2)
    plt.scatter([best_thresh], [best_f1], marker='x', color='darkgreen', s=80, label=f"Best F1 = {best_f1:.2f}")
    plt.scatter([common_thresholds[idx_05]], [f1_05], marker='x', color='olive', s=80, label=f"F1@0.5 = {f1_05:.2f}")
    plt.xlabel("Threshold")
    plt.ylabel("F1-Score")
    plt.title(f"Mean F1-Score Curve Across Folds\nBest F1: {best_f1:.2f} at threshold {best_thresh:.2f}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path_results)
    plt.show()




def plot_mean_combined_metrics(folds_metrics, path_results):
    # Extraer los thresholds comunes exactos entre los folds
    threshold_sets = [set(np.round(fm['thresholds'], 2)) for fm in folds_metrics]
    common_thresholds = sorted(list(set.intersection(*threshold_sets)))

    if not common_thresholds:
        print("No hay thresholds comunes entre los folds.")
        return

    # Inicializar matrices
    tpr_values, fnr_values, fpr_values = [], [], []
    auc_scores = []
    for fm in folds_metrics:
        th_to_tpr = dict(zip(np.round(fm['thresholds'], 2), fm['tpr']))
        th_to_fnr = dict(zip(np.round(fm['thresholds'], 2), fm['fnr']))
        th_to_fpr = dict(zip(np.round(fm['thresholds'], 2), fm['fpr']))
        auc_scores.append(fm['auc_score'])
        tpr_values.append([th_to_tpr[th] for th in common_thresholds])
        fnr_values.append([th_to_fnr[th] for th in common_thresholds])
        fpr_values.append([th_to_fpr[th] for th in common_thresholds])

    # Calcular medias y desviaciones estándar
    tpr_values = np.array(tpr_values)
    fnr_values = np.array(fnr_values)
    fpr_values = np.array(fpr_values)

    tpr_mean, tpr_std = tpr_values.mean(axis=0), tpr_values.std(axis=0)
    fnr_mean, fnr_std = fnr_values.mean(axis=0), fnr_values.std(axis=0)
    fpr_mean, fpr_std = fpr_values.mean(axis=0), fpr_values.std(axis=0)

    # Graficar
    plt.figure(figsize=(8, 5))
    plt.plot(common_thresholds, tpr_mean, color='#7eb08b', label='TPR (Episodes Detected)', linewidth=2)
    plt.fill_between(common_thresholds, tpr_mean - tpr_std, tpr_mean + tpr_std, alpha=0.2, color='#7eb08b')

    plt.plot(common_thresholds, fnr_mean, color='#1b5184', linestyle='--', label='FNR (Episodes Missed)', linewidth=2)
    plt.fill_between(common_thresholds, fnr_mean - fnr_std, fnr_mean + fnr_std, alpha=0.2, color='#1b5184')

    plt.plot(common_thresholds, fpr_mean, color='#c899c0', linestyle='-.', label='FPR (False Alarms)', linewidth=2)
    plt.fill_between(common_thresholds, fpr_mean - fpr_std, fpr_mean + fpr_std, alpha=0.2, color='#c899c0')

    plt.xlabel("Threshold")
    plt.ylabel("Rate")

    plt.title(f"AUC = {np.mean(auc_scores):.2f} ± {np.std(auc_scores):.2f}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path_results)
    plt.show()



def plot_distribution_all_folds(folds_metrics, path_to_save):
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


def plot_distribution_all_folds_combined(folds_metrics, path_to_save):
    all_labels = []
    all_probs = []

    for metrics in folds_metrics:
        all_labels.extend(metrics['all_labels'])
        all_probs.extend(metrics['all_probs'])

    # Separar probabilidades por clase
    positive_probs = [p for p, label in zip(all_probs, all_labels) if label == 1]
    negative_probs = [p for p, label in zip(all_probs, all_labels) if label == 0]

    # Crear histograma combinado
    plt.figure(figsize=(10, 6))
    plt.hist(positive_probs, bins=20, alpha=0.6, color='#7eb08b', label='Positive (Label=1)')
    plt.hist(negative_probs, bins=20, alpha=0.6, color='#c899c0', label='Negative (Label=0)')
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.title("Distribution of Predicted Probabilities Across All Folds")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path_to_save)
    plt.show()


def plot_tpr_vs_fpr_all_folds(folds_metrics, path_to_save):
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


def evaluate_anticipation_only(probs, labels, aggObs):
    labels = np.array(labels)
    probs = np.array(probs)
    aggObs = np.array(aggObs)
    print("Positives:", np.sum(labels == np.unique(labels)[1]))
    print("AggObs == 0:", np.sum(aggObs == np.unique(aggObs)[1]))
    print("Positives with AggObs == 0:", np.sum((labels == np.unique(labels)[1]) & (aggObs == np.unique(aggObs)[0])))
    # obtener muestras con label = 1 y agg = 0 o label = 0 , es decir, si en tp hay contexto agresivo, no evaluamos
    # (probar si termina en 1 solo en lugar del max)
    mask = ((labels == np.unique(labels)[1]) & (aggObs == np.unique(aggObs)[0]) | (labels == np.unique(labels)[0]))
    filtered_labels = labels[mask]
    filtered_probs = probs[mask]
    return filtered_labels, filtered_probs


def test_model(path_models, model_code, feats_code, tf, tp, bin_size, split_code, cw_type, onset_only=False):
    num_folds = 5
    folds = np.arange(num_folds)
    freq = 32
    data_path_resampled = './dataset_resampled/'
    ds_path = data_path_resampled + "dataset_" + str(freq) + "Hz.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eegnet_params = {"chunk_size": 151, "num_electrodes": 60, "F1": 8, "F2": 16, "D": 2, "num_classes": 2,
                     "kernel_1": 64, "kernel_2": 16, "dropout": 0.25}
    lstm_hidden_dim = 64
    folds_metrics = []
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
        PDM = False
        _, _, test_dict, _ = train.get_partitions_from_fold(data_dict, train_uids, test_uids, split_code, PDM, seed=1)  # same seed as in train
        features_fun, dataloader_fun = models_utils.get_dataloader(model_code)
        test_data = features_fun(test_dict, bin_size, freq)
        batch_size = 128
        dataloader_test = train.create_dataloader(dataloader_fun, test_data, tp, tf, bin_size, batch_size, shuffle=False)

        # create model
        num_sequences = tp // bin_size
        bin_size = 15
        eegnet_params = {
            'num_electrodes': 5,  # (EDA, ACC_X, ACC_Y, ACC_Z, BVP) ...
            'chunk_size': tp * freq // num_sequences  # muestras en cada ventana
        }
        lstm_hidden_dim = 64

        if cw_type != 0:
            model_path = f"{path_models}mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_sc{split_code}_cw{cw_type}_fold{fold_idx}_model.pth"
        else:
            model_path = f"{path_models}mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_sc{split_code}_fold{fold_idx}_model.pth"

        print(f"Evaluando modelo {model_path}")
        model = models_utils.load_model(model_code, device, eegnet_params, lstm_hidden_dim, model_path)
        all_probs, all_labels, all_aggObs = models_utils.test_model(model, dataloader_test, device, onset_only=onset_only)
        print(f"Positives: {np.sum(all_labels == np.unique(all_labels)[1])}, unique labels = {np.unique(all_labels)}")
        print("Positives sum:", np.sum(all_labels))
        print("Probs sum:", np.sum(all_probs))
        print(f"AggObs: {np.sum(all_aggObs == np.unique(all_labels)[1])}, unique all_aggObs = {np.unique(all_aggObs)}")
        print("AggObs sum:", np.sum(all_aggObs))
        if onset_only:
            aggObs_max = np.array(all_aggObs).max(axis=1) # cambiar por last only para ver si en t esta en calma en lugar de en tp
            all_labels, all_probs = evaluate_anticipation_only(all_probs, all_labels, list(aggObs_max))
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
    plot_five_folds_metrics(folds_metrics)
    plot_five_folds_rates(folds_metrics)
    plot_f1_scores(folds_metrics)
    '''


    path_analysis_results = './results_analysis/'
    path_results = f"{path_analysis_results}PM_mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_sc{split_code}_cw{cw_type}_onsetonly{onset_only}_probability_distribution.png"
    plot_distribution_all_folds(folds_metrics, path_results)
    '''
    path_results = f"{path_analysis_results}PM_mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_sc{split_code}_tpr_vs_fpr.png"
    plot_tpr_vs_fpr_all_folds(folds_metrics, path_results)
    path_results = f"{path_analysis_results}PM_mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_sc{split_code}_f1_scores.png"
    plot_f1_scores(folds_metrics, path_results)
    '''
    path_results = f"{path_analysis_results}PM_mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_sc{split_code}_cw{cw_type}_onsetonly{onset_only}_combined_metrics.png"
    plot_mean_combined_metrics(folds_metrics, path_results)
    path_results = f"{path_analysis_results}PM_mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_sc{split_code}_cw{cw_type}_onsetonly{onset_only}_probability_distribution.png"
    plot_distribution_all_folds_combined(folds_metrics, path_results)



def test_model_PDM(path_models, model_code, feats_code, tf, tp, bin_size, split_code, cw_type, onset_only=False):
    freq = 32
    stride = 15  # igual que bin_size
    data_path_resampled = './dataset_resampled/'
    ds_path = data_path_resampled + f"dataset_{freq}Hz.csv"
    data_dict = data_utils.load_data_to_dict(ds_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eegnet_params = {"chunk_size": 151, "num_electrodes": 60, "F1": 8, "F2": 16, "D": 2, "num_classes": 2,
                     "kernel_1": 64, "kernel_2": 16, "dropout": 0.25}
    lstm_hidden_dim = 64

    subjects_metrics = []
    features_fun, dataloader_fun = models_utils.get_dataloader(model_code)
    batch_size = 16

    for subject_id in sorted(data_dict.keys()):
        print(f"\nEvaluando sujeto {subject_id}")
        subject_data = {subject_id: data_dict[subject_id]}
        test_data = features_fun(subject_data, bin_size, freq)
        dataloader_test = train.create_dataloader(dataloader_fun, test_data, tp, tf, bin_size, batch_size, shuffle=False)

        if cw_type != 0:
            model_path = f"{path_models}mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_sc{split_code}_cw{cw_type}_subj{subject_id}_hybrid_model.pth"
        else:
            model_path = f"{path_models}mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_sc{split_code}_subj{subject_id}_hybrid_model.pth"
        print(f"Modelo cargado desde {model_path}")
        model = models_utils.load_model(model_code, device, eegnet_params, lstm_hidden_dim, model_path)

        all_probs, all_labels, all_aggObs = models_utils.test_model(model, dataloader_test, device, onset_only=onset_only)

        print(f"Positives: {np.sum(all_labels == 1)}, AggObs sum: {np.sum(all_aggObs)}")

        if onset_only:
            aggObs_max = np.array(all_aggObs).max(axis=1)
            all_labels, all_probs = evaluate_anticipation_only(all_probs, all_labels, list(aggObs_max))

        auc_score, fpr, tpr, tnr, fnr, best_f1, best_threshold_f1, thresholds, f1_scores, best_threshold_roc, best_fpr_at_auc, best_tpr_at_auc = evaluate_all_results(
            all_labels, all_probs)

        print(f"Sujeto {subject_id} - AUC: {auc_score:.4f}, Best F1: {best_f1:.4f}")

        subjects_metrics.append({
            'subject_id': subject_id,
            'fpr': fpr,
            'tpr': tpr,
            'fnr': fnr,
            'thresholds': thresholds,
            'auc_score': auc_score,
            'best_threshold_f1': best_threshold_f1,
            'best_f1': best_f1,
            'f1_scores': f1_scores,
            'all_probs': all_probs,
            'all_labels': all_labels
        })

    # Guardar y plotear resultados agregados
    path_analysis_results = './results_analysis/'
    plot_distribution_all_folds_combined(subjects_metrics,
        f"{path_analysis_results}PDM_mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_sc{split_code}_onsetonly{onset_only}_cw{cw_type}_probability_distribution_all.png")

    plot_mean_combined_metrics(subjects_metrics,
        f"{path_analysis_results}PDM_mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_sc{split_code}_onsetonly{onset_only}_cw{cw_type}_combined_metrics.png")

    return subjects_metrics



def test_all_models():
    model_exp = 0 # PM
    model_versions = [2]
    feat_code = 0
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
        split_code = 0 # configurar todo en args
        print(f'tp: {tp}, tf: {tf}, stride: {stride}, bin_size: {bin_size}.')
        for fold_idx, (train_uids, test_uids) in enumerate(folds):
            print(f"Fold {fold_idx + 1}:")
            print(f"  Train UIDs: {train_uids}")
            print(f"  Test UIDs: {test_uids}")
            PDM = False
            # Obtener particiones de test en funcion de los ids del fold
            _, _, test_dict, _ = train.get_partitions_from_fold(data_dict, train_uids, split_code, PDM, test_uids, seed=1)  # same seed as in train
            features_fun, dataloader_fun = models_utils.get_dataloader(mv)
            test_data = features_fun(test_dict, bin_size, freq)
            batch_size = 128
            dataloader_test = train.create_dataloader(dataloader_fun, test_data, tp, tf, bin_size, batch_size, shuffle=False)

            # create model
            num_sequences = tp // bin_size
            bin_size = 15
            eegnet_params = {
                'num_electrodes': 5,
                'chunk_size': tp * freq // num_sequences
            }
            lstm_hidden_dim = 64
            num_classes = 1

            model_path = f'./models/mv{mv}_f{feat_code}_tf{tf}_tp{tp}_bs{bin_size}_fold{fold_idx}_model.pth'
            # path_model = f"{path_models}mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_sc{split_code}_fold{fold_idx}_model.pth"
            print(f"Evaluando modelo {model_path}")
            model = models_utils.load_model(mv, device, eegnet_params, lstm_hidden_dim, model_path)
            all_probs, all_labels, _ = models_utils.test_model(model, dataloader_test, device, onset_only=False)
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
        plot_five_folds_metrics(folds_metrics)
        plot_five_folds_rates(folds_metrics)
        plot_f1_scores(folds_metrics)
        '''
        path_analysis_results = './results_analysis/'
        path_results = f"{path_analysis_results}PM_mv{mv}_f{feat_code}_tf{tf}_tp{tp}_probability_distribution.png"

        plot_distribution_all_folds(folds_metrics, path_results)
        path_results = f"{path_analysis_results}PM_mv{mv}_f{feat_code}_tf{tf}_tp{tp}_tpr_vs_fpr.png"
        plot_tpr_vs_fpr_all_folds(folds_metrics, path_results)
        path_results = f"{path_analysis_results}PM_mv{mv}_f{feat_code}_tf{tf}_tp{tp}_combined_metrics.png"
        plot_mean_combined_metrics(folds_metrics, path_results)


def plot_rates_with_best_scores(folds_metrics, path_results):
    num_folds = len(folds_metrics)
    fig, axes = plt.subplots(1, num_folds, figsize=(5 * num_folds, 5), sharey=True)

    if num_folds == 1:
        axes = [axes]

    for metrics, ax in zip(folds_metrics, axes):
        thresholds = metrics['thresholds']
        tpr = metrics['tpr']
        fnr = metrics['fnr']
        fpr = metrics['fpr']

        ax.plot(thresholds, tpr, label='TPR (Episodes Detected)', linestyle='-', color='green')
        ax.plot(thresholds, fnr, label='FNR (Episodes Missed)', linestyle='--', color='red')
        ax.plot(thresholds, fpr, label='FPR (False Alarms)', linestyle='-.', color='blue')

        best_auc_th = metrics['Best Threshold (AUC)']
        best_f1_th = metrics['Best Threshold (F1)']
        auc_score = metrics['auc_score']
        best_f1_score = metrics['Best F1-Score']

        #ax.scatter(best_auc_th, auc_score, color='navy', label=f"AUC = {auc_score:.4f}",
        #           marker='x', s=100, linewidths=2)
        #ax.scatter(best_f1_th, best_f1_score, edgecolors='darkorange', facecolors='none',
        #           label=f"F1 Score = {best_f1_score:.4f}", marker='^', s=100, linewidths=2)

        ax.set_title(f"Fold {metrics['fold_idx']}\nAUC: {metrics['auc_score']:.4f}")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Rate")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path_results)
    plt.show()


def plot_mean_roc_curve(folds_metrics, path_to_save):
    all_fprs = [metrics['fpr'] for metrics in folds_metrics]
    all_tprs = [metrics['tpr'] for metrics in folds_metrics]
    all_aucs = [metrics['auc_score'] for metrics in folds_metrics]

    min_len = min(map(len, all_fprs))
    fpr_common = np.mean([fpr[:min_len] for fpr in all_fprs], axis=0)
    mean_tpr = np.mean([tpr[:min_len] for tpr in all_tprs], axis=0)
    std_tpr = np.std([tpr[:min_len] for tpr in all_tprs], axis=0)

    plt.figure(figsize=(7, 7))
    plt.plot(fpr_common, mean_tpr, color='blue', label=f"Mean ROC (AUC = {np.mean(all_aucs):.4f})")
    plt.fill_between(fpr_common, mean_tpr - std_tpr, mean_tpr + std_tpr, color='blue', alpha=0.2, label="±1 Std Dev")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random Classifier")

    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Mean ROC Curve Across Folds")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.savefig(path_to_save)
    plt.show()


def plot_tpr_vs_fpr_mean_std(folds_metrics, path_to_save):
    common_fpr = np.linspace(0, 1, 100)
    tpr_interpolated = []
    for metrics in folds_metrics:
        fpr = np.array(metrics['fpr'])
        tpr = np.array(metrics['tpr'])
        interp_tpr = np.interp(common_fpr, fpr, tpr)  # Interpolar TPR a escala común
        tpr_interpolated.append(interp_tpr)
    tpr_interpolated = np.array(tpr_interpolated)

    mean_tpr = np.mean(tpr_interpolated, axis=0)
    std_tpr = np.std(tpr_interpolated, axis=0)
    auc_scores = [metrics['auc_score'] for metrics in folds_metrics]
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(common_fpr, mean_tpr, color='blue', label=f"Mean AUC = {mean_auc:.4f} ± {std_auc:.4f}", linestyle='-')
    plt.fill_between(common_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color='blue', alpha=0.2, label="±1 Std Dev")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random Classifier")

    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Mean TPR vs FPR Across All Folds")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    plt.savefig(path_to_save)
    plt.show()


def evaluate_ensemble_results_multi(all_labels, probs_sib, probs_agg, probs_ed, fold_idx):
    # soft voting
    final_probs = np.maximum.reduce([probs_sib, probs_agg, probs_ed])

    auc_score = roc_auc_score(all_labels, final_probs)
    fpr, tpr, thresholds = roc_curve(all_labels, final_probs)
    best_threshold_auc_idx = np.argmax(tpr - fpr)
    best_threshold_auc = thresholds[best_threshold_auc_idx]

    best_f1, best_threshold_f1 = 0, 0
    f1_scores = []
    for threshold in thresholds:
        predictions = (final_probs >= threshold).astype(int)
        f1 = f1_score(all_labels, predictions)
        f1_scores.append(f1)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold_f1 = threshold

    _, fpr, tpr, tnr, fnr, _, _, _, _, _, _, _ = evaluate_all_results(all_labels, final_probs)
    bin_predictions = (final_probs >= 0.5).astype(int)
    f1 = f1_score(all_labels, bin_predictions, zero_division=0)
    return {
        "AUC-ROC": auc_score,
        "Best F1-Score": best_f1,
        "Best Threshold (AUC)": best_threshold_auc,
        "Best Threshold (F1)": best_threshold_f1,
        'f1_scores': f1_scores,
        'all_probs': final_probs,
        'all_labels': all_labels,
        'fpr': fpr,
        'tpr': tpr,
        'fnr': fnr,
        'thresholds': thresholds,
        # se repiten para integrar rapido con funciones anteriores... unificar en el futuro
        'auc_score': auc_score,
        'best_threshold_f1': best_threshold_f1,
        'best_f1': best_f1,
        'f1:': f1,
        'fold_idx': fold_idx + 1
    }


def test_model_multi(path_models, model_code, feats_code, tf, tp, bin_size, split_code, cw_type):
    freq = 32
    data_path_resampled = './dataset_resampled/'
    ds_path = data_path_resampled + "dataset_" + str(freq) + "Hz.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds_metrics = []

    data_dict = data_utils.load_data_to_dict_multi (ds_path)
    model_fun, features_fun, dataloader_fun, train_fun, retrain_fun = train.set_model_multi(model_code)
    load_data_fun, model_fun, EEG_channels = train.set_features_multi(feats_code, model_fun)
    PDM = False
    num_folds = 5
    folds = train.generate_user_kfolds(data_dict, k=num_folds)
    labels_list = ['SIB', 'AGG', 'ED']
    all_labels_models = []
    all_probs_models = {"SIB": [], "AGG": [], "ED": []}
    print(f'model_code: {model_code}, tp: {tp}, tf: {tf}, feats_code: {feats_code}, split_code: {split_code} bin_size: {bin_size}, cw_type: {cw_type}.')
    for fold_idx, (train_uids, test_uids) in enumerate(folds):
        print(f"Fold {fold_idx + 1}:")
        print(f"  Train UIDs: {train_uids}")
        print(f"  Test UIDs: {test_uids}")
        _, _, test_dict, _ = train.get_partitions_from_fold(data_dict, train_uids, test_uids, split_code, PDM, seed=1)
        test_data = features_fun(test_dict, bin_size, freq)
        batch_size = 128

        # create model
        num_sequences = tp // bin_size
        bin_size = 15
        eegnet_params = {
            'num_electrodes': EEG_channels,
            'chunk_size': tp * freq // num_sequences
        }
        lstm_hidden_dim = 64
        num_classes = 1
        fold_labels = []
        for label_type in labels_list:
            if cw_type != 0:
                path_model = f"{path_models}mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_sc{split_code}_fold{fold_idx}_label{label_type}__cw{cw_type}_model.pth"
            else:
                path_model = f"{path_models}mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_sc{split_code}_fold{fold_idx}_label{label_type}_model.pth"
            model = models_utils.load_model(model_code, device, eegnet_params, lstm_hidden_dim, path_model)
            print(f"Evaluando modelo {path_model}")
            dataloader_test = train.create_dataloader_multi(dataloader_fun, test_data, label_type, tp, tf, bin_size,
                                                            batch_size,
                                                            shuffle=False)
            all_probs, all_labels, _ = models_utils.test_model(model, dataloader_test, device, onset_only=False)
            all_probs_models[label_type].append(all_probs)
            fold_labels.append(np.array(all_labels))
        all_labels_models_final = np.maximum.reduce(fold_labels) # etiqueta pos si al menos 1 de los modelos tiene etiqueta pos (analogo a 'Condition')
        results_dict = evaluate_ensemble_results_multi(all_labels_models_final, all_probs_models[labels_list[0]][fold_idx], all_probs_models[labels_list[1]][fold_idx],
                                        all_probs_models[labels_list[2]][fold_idx], fold_idx)

        folds_metrics.append(results_dict)

    '''
    plot_five_folds_metrics(folds_metrics)
    plot_five_folds_rates(folds_metrics)
    plot_f1_scores(folds_metrics)
    '''
    path_analysis_results = './results_analysis/'
    path_results = f"{path_analysis_results}PM_multi_mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_probability_distribution.png"
    plot_distribution_all_folds(folds_metrics, path_results)
    path_results = f"{path_analysis_results}PM_multi_mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_tpr_vs_fpr.png"
    plot_tpr_vs_fpr_all_folds(folds_metrics, path_results)
    path_results = f"{path_analysis_results}PM_multi_mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_combined_metrics.png"
    plot_mean_combined_metrics(folds_metrics, path_results)
    path_results = f"{path_analysis_results}PM_multi_mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_combined_metrics_v2.png"
    plot_rates_with_best_scores(folds_metrics, path_results)
    path_results = f"{path_analysis_results}PM_multi_mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_mean_roc_curve.png"
    plot_mean_roc_curve(folds_metrics, path_results)

    results_df = pd.DataFrame(folds_metrics)
    results_df = results_df.drop(columns=["all_labels", "all_probs", "f1_scores", "fpr", "fnr", "tpr", "thresholds"])
    results_df.set_index("fold_idx", inplace=True)
    avg_metrics = results_df.mean(numeric_only=True)
    std_metrics = results_df.std(numeric_only=True)
    summary_df = pd.DataFrame([avg_metrics, std_metrics], index=["Avg.", "Std."])
    summary_df.reset_index(inplace=True)
    final_results_df = pd.concat([results_df, summary_df], ignore_index=True)
    path_to_save_results = f"./results_analysis/PM_ensemble_mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_sc{split_code}_cw{cw_type}_ensemble_results_5cv.csv"
    final_results_df.to_csv(path_to_save_results, index=False)


def analyse_PDM_results(path_results, model_code, feats_code, tf, tp, bin_size, split_code, hm=False):
    if hm:
        strategy = 'HM'
    else: strategy = 'PDM'
    results_path = f"{path_results}/{strategy}_mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_sc{split_code}_all_experiments_results.csv"
    results_df = pd.read_csv(results_path, dtype={'Fold': str})
    mean_row = results_df.iloc[-2]
    std_row = results_df.iloc[-1]
    mean_auc_roc = mean_row['AUC-ROC']
    std_auc_roc = std_row['AUC-ROC']
    results_df = results_df.iloc[:-2]

    freq = 32
    data_path_resampled = './dataset_resampled/'
    ds_path = data_path_resampled + f"dataset_{freq}Hz.csv"
    data_dict = data_utils.load_data_to_dict(ds_path)
    subject_ids = results_df['Fold'].astype(str)
    auc_values = np.round(results_df['AUC-ROC'], 2)

    total_episodes = []
    total_sessions = []
    mean_episodes_per_session = []
    mean_episode_duration = []
    total_duration_episodes = []
    total_duration_sessions = []

    def compute_episode_count_and_durations(condition_series, time_series):
        condition_series = condition_series.reset_index(drop=True)
        if isinstance(time_series, pd.DatetimeIndex):
            time_series = time_series.to_series()
        time_series = pd.Series(time_series)
        time_series = pd.to_datetime(time_series)
        shifted = condition_series.shift(fill_value=0)
        episode_starts = (condition_series == 1) & (shifted == 0)
        episode_durations = []
        for start_idx in episode_starts[episode_starts].index:
            end_idx = start_idx
            while end_idx < len(condition_series) and condition_series.iloc[end_idx] == 1:
                end_idx += 1
            end_idx = min(end_idx, len(time_series) - 1)
            duration = (time_series.iloc[end_idx] - time_series.iloc[start_idx]).total_seconds()
            episode_durations.append(duration)
        return len(episode_durations), episode_durations

    for subject_id in subject_ids:
        subject_data = data_dict.get(subject_id, {})
        num_episodes = 0
        num_sessions = len(subject_data)
        episodes_per_session = []
        all_durations = []
        total_duration = 0
        session_durations = []
        for session_id, session_data in subject_data.items():
            session_data = session_data.sort_index()
            num_episodes_in_session, durations_in_session = compute_episode_count_and_durations(
                session_data['Condition'], session_data.index
            )
            num_episodes += num_episodes_in_session
            episodes_per_session.append(num_episodes_in_session)
            all_durations.extend(durations_in_session)
            total_duration += sum(durations_in_session)
            if not session_data.empty:
                session_duration = (session_data.index[-1] - session_data.index[0]).total_seconds()
                session_durations.append(session_duration / 60)

        total_episodes.append(num_episodes)
        total_sessions.append(num_sessions)
        mean_episodes_per_session.append(num_episodes / num_sessions if num_sessions > 0 else 0)
        mean_episode_duration.append(np.mean(all_durations))
        total_duration_episodes.append(total_duration / 60)
        total_duration_sessions.append(np.sum(session_durations) if session_durations else 0)

    analysis_df = pd.DataFrame({
        'Subject': subject_ids,
        'AUC-ROC': auc_values,
        'Total Episodes': total_episodes,
        'Total Sessions': total_sessions,
        'Mean Episodes per Session': np.round(mean_episodes_per_session, 2),
        'Mean Episode Duration (s.)': np.round(mean_episode_duration, 2),
        'Total Duration Episodes (min.)': np.round(total_duration_episodes, 2),
        'Total Duration Sessions (min.)': np.round(total_duration_sessions, 2)
    })

    output_path_csv = f"./results_analysis/{strategy}_mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_sc{split_code}_data_subjects_analysis.csv"
    analysis_df.to_csv(output_path_csv, index=False)
    print(f"DataFrame guardado en: {output_path_csv}")

    unique_subjects = analysis_df['Subject'].unique()
    color_map = plt.get_cmap('tab10')
    subject_colors = {subject: color_map(i % 10) for i, subject in enumerate(unique_subjects)}

    fig, axs = plt.subplots(3, 2, figsize=(18, 16))
    fig.suptitle(f'Performance Analysis - Mean AUC-ROC: {mean_auc_roc:.3f}, Std.: {std_auc_roc:.3f}',
                 fontsize=16, fontweight='bold')

    plot_data = [
        ("AUC vs Total Sessions", 'Total Sessions', axs[0, 0]),
        ("AUC vs Total Duration Sessions", 'Total Duration Sessions (min.)', axs[0, 1]),
        ("AUC vs Total Episodes", 'Total Episodes', axs[1, 0]),
        ("AUC vs Total Duration Episodes", 'Total Duration Episodes (min.)', axs[1, 1]),
        ("AUC vs Mean Episodes per Session", 'Mean Episodes per Session', axs[2, 0]),
        ("AUC vs Mean Episode Duration", 'Mean Episode Duration (s.)', axs[2, 1])
    ]

    for i, (title, column, ax) in enumerate(plot_data):
        for subject_id in unique_subjects:
            subject_data = analysis_df[analysis_df['Subject'] == subject_id]
            ax.scatter(subject_data[column], subject_data['AUC-ROC'],
                       color=subject_colors[subject_id], marker='+', s=100, linewidth=1.5)

        ax.set_xlabel(column, fontsize=14)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

        if i % 2 == 0:
            ax.set_ylabel("AUC-ROC", fontsize=14)
        else:
            ax.set_ylabel("")

        ax.axhline(0.5, color='black', linestyle='dashed', linewidth=1.2)
        ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path_to_save_results = f"./results_analysis/{strategy}_mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_sc{split_code}_data_vs_results_5cv.png"
    plt.savefig(path_to_save_results)
    plt.show()


'''
# Usage example
tp, tf, freq = 180, 60, 32
model_code, feats_code, split_code, bin_size = 2, 0, 0, 15
cw = 0
data_path_resampled = './dataset_resampled/'
results_path = './results/'
models_path = './models/'
#test_model_multi(models_path, model_code, feats_code, tf, tp, bin_size, split_code, cw)
tf=120
cw = 1
test_model_multi(models_path, model_code, feats_code, tf, tp, bin_size, split_code, cw)

#test_model(models_path, model_code, feats_code, tf, tp, bin_size, split_code, cw, True)

#test_model_PDM(models_path, model_code, feats_code, tf, tp, bin_size, split_code) # to-do
#analyse_PDM_results(results_path, model_code, feats_code, tf, tp, bin_size, split_code)
#analyse_PDM_results(results_path, model_code, feats_code, tf, tp, bin_size, split_code, True)
'''
