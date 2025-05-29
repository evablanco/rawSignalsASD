import train, data_utils
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from evaluation_utils import model_evaluation
from sklearn.metrics import precision_recall_fscore_support
import re
from sklearn.metrics import roc_curve, auc
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import glob


def extract_hyperparams_from_filename(filename):
    pattern = r"PM_mv(\d+)_f(\d+)_tf(\d+)_tp(\d+)_bs(\d+)_sc(\d+)_cw(\d+)"
    match = re.search(pattern, filename)
    if match:
        mv, f, tf, tp, bs, sc, cw = match.groups()
        return {
            "mv": int(mv),
            "f": int(f),
            "tf": int(tf),
            "tp": int(tp),
            "bs": int(bs),
            "sc": int(sc),
            "cw": int(cw)
        }
    else:
        return {}


def find_best_models_by_metric(results_dir):
    metric_names = [
        "F1_macro", "F1_weighted", "AUC_macro", "Accuracy",
        "F1_Calm", "F1_Pre-episode", "F1_Aggression"
    ]
    all_means = []
    for filename in os.listdir(results_dir):
        if filename.endswith(".csv") and "all_experiments_results" in filename:
            file_path = os.path.join(results_dir, filename)
            match = re.search(r"PM_mv(\d+)_f(\d+)_tf(\d+)_tp(\d+)_bs(\d+)_sc(\d+)_cw(\d+)", filename)
            if not match:
                continue
            try:
                df = pd.read_csv(file_path)
                avg_row = df[df["Fold"] == "Avg."].copy()
                if avg_row.empty:
                    continue
                mv, f, tf, tp, bs, sc, cw = match.groups()
                avg_row["mv"] = int(mv)
                avg_row["f"] = int(f)
                avg_row["tf"] = int(tf)
                avg_row["tp"] = int(tp)
                avg_row["bs"] = int(bs)
                avg_row["sc"] = int(sc)
                avg_row["cw"] = int(cw)
                avg_row["source_file"] = filename
                avg_row["model_path"] = df.loc[0, "Model_path"]  # Path del primer fold
                all_means.append(avg_row)

            except Exception as e:
                print(f"Error en {filename}: {e}")
    all_df = pd.concat(all_means, ignore_index=True)

    best_records = []
    for metric in metric_names:
        if metric in all_df.columns:
            idx = all_df[metric].astype(float).idxmax()
            row = all_df.loc[idx].copy()
            row["metric"] = metric
            row["metric_value"] = row[metric]
            best_records.append(row[["metric", "metric_value", "mv", "f", "tf", "tp", "bs", "sc", "cw", "model_path"]])
    final_df = pd.DataFrame(best_records)
    return final_df


def find_best_model_per_tf_and_metric(results_dir):
    metric_names = [
        "F1_macro", "F1_weighted", "AUC_macro", "Accuracy",
        "F1_Calm", "F1_Pre-episode", "F1_Aggression"
    ]
    records = []
    for filename in os.listdir(results_dir):
        if filename.endswith(".csv") and "all_experiments_results" in filename:
            path = os.path.join(results_dir, filename)
            hyperparams = extract_hyperparams_from_filename(filename)
            if not hyperparams:
                continue
            try:
                df = pd.read_csv(path)
                avg_row = df[df["Fold"] == "Avg."]
                if avg_row.empty:
                    continue
                for metric in metric_names:
                    if metric in avg_row.columns:
                        record = {
                            "metric": metric,
                            "metric_value": float(avg_row[metric].values[0]),
                            **hyperparams,
                            "model_path": df.loc[0, "Model_path"] if "Model_path" in df.columns else None
                        }
                        records.append(record)
            except Exception as e:
                print(f"Error leyendo {filename}: {e}")
    df_all = pd.DataFrame(records)
    best_per_tf = df_all.loc[df_all.groupby(["metric", "tf"])["metric_value"].idxmax()]
    return best_per_tf.reset_index(drop=True)


def test_model(path_models, model_version, feats_code, tf, tp, bin_size, split_code, cw_type, seed=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    freq = 32
    batch_size = 128
    ds_path = f'./dataset_resampled/dataset_{freq}Hz.csv'
    model_fun, features_fun, dataloader_fun, _, _ = train.set_model(model_version)
    selected_features, EEG_channels = train.set_features(feats_code)
    data_dict = data_utils.load_data_to_dict(ds_path, selected_features)
    folds = train.generate_subject_kfolds(data_dict, k=5)
    print('Testing folds...')
    all_results = []
    for fold_idx, (train_uids, test_uids) in enumerate(folds):
        print(f"\nEvaluando Fold {fold_idx + 1} -------------------------")
        _, _, test_dict, _ = train.get_partitions_from_fold(data_dict, train_uids, test_uids, split_code, seed)

        test_data = features_fun(test_dict, bin_size, freq)
        test_loader = train.create_dataloader(dataloader_fun, test_data, tp, tf, bin_size,
                                             batch_size=batch_size, shuffle=False)

        lstm_hidden_dim, lstm_num_layers, num_classes = train.set_lstm()
        num_sequences = tp // bin_size
        eegnet_params = {'num_electrodes': EEG_channels, 'chunk_size': tp * freq // num_sequences}
        model = model_fun(eegnet_params, lstm_hidden_dim, lstm_num_layers, num_classes).to(device)

        model_path = f"{path_models}PM_sc{split_code}_mv{model_version}_f{feats_code}_tf{tf}" + \
                     f"_tp{tp}_bs{bin_size}_cw{cw_type}_fold{fold_idx}_model.pth"
        if not os.path.exists(model_path):
            print(f"Warning!: falta modelo {model_path}. Se omite fold {fold_idx}.")
            continue
        model.load_state_dict(torch.load(model_path, map_location=device))

        f1_macro, f1_weighted, auc_macro, acc, f1_per_class = model_evaluation(model, test_loader, device)
        result = {
            'Fold': fold_idx + 1,
            'F1_macro': f1_macro,
            'F1_weighted': f1_weighted,
            'AUC_macro': auc_macro,
            'Accuracy': acc,
            **{f"F1_{k}": v for k, v in f1_per_class.items()}
        }
        all_results.append(result)
        print(f"Fold {fold_idx + 1} ➤ F1_macro: {f1_macro:.4f}, Accuracy: {acc:.4f}, AUC_macro: {auc_macro:.4f}")

    if all_results:
        df = pd.DataFrame(all_results)
        print("\nAvg:")
        print(df.drop(columns=["Fold"]).mean().round(4))
        print("\nStd:")
        print(df.drop(columns=["Fold"]).std().round(4))
        df.to_csv("evaluation_results_folds.csv", index=False)
        print("\nResultados guardados en 'evaluation_results_folds.csv'")
    else:
        print("No se ha evaluado ningún fold...")



def run_full_evaluation_analysis(path_models, model_version, feats_code, tf, tp, bin_size, split_code,
                                 cw_type, seed=1):
    output_path = "./results_analysis"  # TO-DO: args
    os.makedirs(output_path, exist_ok=True)
    config_name = f"PM_sc{split_code}_mv{model_version}_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}"
    print(f"evaluación completa para {config_name}")
    all_true, all_pred, all_probs, all_times = test_model_collect_predictions(
        path_models, model_version, feats_code, tf, tp, bin_size, split_code, cw_type, seed
    )

    thresholds = [
        np.array([0.5, 0.5, 0.5]),  # base
        np.array([0.1, 0.3, 0.35]) # TO-DO... arreglar lógica, actualmente male...
    ]
    plot_confusion_matrices_thresholds(all_true, all_probs, thresholds, output_path, config_name)
    plot_roc_pr_curves(all_true, all_probs, output_path, config_name)
    analyze_pre_attack_timing(all_true, all_pred, all_times, output_path, config_name)

    y_true_all = np.concatenate(all_true)
    y_probs_all = np.concatenate(all_probs)
    plot_metric_vs_threshold(y_true_all, y_probs_all, class_id=1, metric='f1',
                             save_path=f"{output_path}/f1_vs_threshold_class1_{config_name}.png")
    plot_all_class_metrics_vs_threshold(y_true_all, y_probs_all, metric='f1', save_path=output_path,
                                        config_name=config_name)
    print(f"resultados en: {output_path}")




def get_best_performing_models_prev(results_folder, f1_threshold_csv, output_csv="summary_best_models.csv"):
    if not os.path.exists(f1_threshold_csv):
        raise FileNotFoundError(f"Archivo no encontrado: {f1_threshold_csv}")
    print(f"Analizando resultados en: {results_folder}")

    csv_files = glob.glob(os.path.join(results_folder, "*.csv"))
    all_dfs = [pd.read_csv(f) for f in csv_files if "_experiments_results" in f]

    if not all_dfs:
        print("No se encontraron archivos...")
        return

    all_df = pd.concat(all_dfs, ignore_index=True)

    f1_df = pd.read_csv(f1_threshold_csv)
    summary_rows = []

    metrics = [
        ("F1_macro", "F1_macro", 0.5),
        ("F1_0", "F1_Calm_0.5", 0.5),
        ("F1_1", "F1_PreAggression_0.5", 0.5),
        ("F1_2", "F1_Aggression_0.5", 0.5),
        ("AUC_macro", "AUC_macro", 0.5),
    ]

    # Obtener modelos con mejores métricas para threshold 0.5
    for colname, label, threshold in metrics:
        if colname not in all_df.columns:
            print(f"Columna no encontrada: {colname}")
            continue
        best_row = all_df.sort_values(colname, ascending=False).iloc[0]
        summary_rows.append({
            "Metric": label,
            "Threshold": threshold,
            "Value": best_row[colname],
            "Config": best_row.get("Config", "unknown"),
            "Model_Path": best_row.get("Model_Path", "unknown"),
        })

    # Obtener mejores F1 por clase con threshold óptimo
    for class_id in range(3):
        class_label = ["Calm", "Pre-episode", "Aggression"][class_id]
        class_rows = f1_df[f1_df["Class_ID"] == class_id]
        if class_rows.empty:
            print(f"No hay resultados para clase {class_label}")
            continue
        best_idx = class_rows["Best_F1"].idxmax()
        best_f1_row = class_rows.loc[best_idx]
        summary_rows.append({
            "Metric": f"F1_{class_label}_Best",
            "Threshold": best_f1_row["Best_Threshold"],
            "Value": best_f1_row["Best_F1"],
            "Config": best_f1_row["Config"],
            "Model_Path": best_f1_row.get("Model_Path", "unknown")
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_csv, index=False)
    print(f"Resumen de mejores modelos guardado en: {output_csv}")


def get_best_performing_models(results_folder, f1_threshold_csv, output_csv="summary_best_models.csv"):
    result_files = [
        os.path.join(results_folder, f) for f in os.listdir(results_folder)
        if f.endswith("_experiments_results_5cv.csv")
    ]
    if not result_files:
        print("No se encontraron archivos de resultados en:", results_folder)
        return

    dfs = []
    for f in result_files:
        df = pd.read_csv(f)
        df['filename'] = os.path.basename(f)
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)
    f1_df = pd.read_csv(f1_threshold_csv)

    summary = []
    # F1_macro a threshold 0.5
    if "F1_macro" in all_df.columns:
        best_row = all_df.sort_values("F1_macro", ascending=False).iloc[0]
        summary.append({
            "Metric": "F1_macro",
            "Threshold": 0.5,
            "Value": best_row["F1_macro"],
            "Config": best_row.get("config_name", best_row.get("filename", "unknown")),
            "Model_Path": best_row.get("model_path", best_row.get("filename", "unknown"))
        })
    else:
        print("Columna no encontrada: F1_macro")

    # AUC_macro a threshold 0.5
    if "AUC_macro" in all_df.columns:
        best_row = all_df.sort_values("AUC_macro", ascending=False).iloc[0]
        summary.append({
            "Metric": "AUC_macro",
            "Threshold": 0.5,
            "Value": best_row["AUC_macro"],
            "Config": best_row.get("config_name", best_row.get("filename", "unknown")),
            "Model_Path": best_row.get("model_path", best_row.get("filename", "unknown"))
        })
    else:
        print("Columna no encontrada: AUC_macro")

    # F1 por clase en su mejor threshold
    for class_label, class_id in zip(["Calm", "Pre-episode", "Aggression"], [0, 1, 2]):
        row = f1_df[f1_df["Class"] == class_label]
        if not row.empty:
            r = row.iloc[0]
            summary.append({
                "Metric": f"F1_{class_label}_Best",
                "Threshold": r["Best_Threshold"],
                "Value": r["Best_F1"],
                "Config": r["Config"],
                "Model_Path": r.get("Model_Path", "unknown")
            })
        else:
            print(f"No se encontró F1_best para clase: {class_label}")

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(output_csv, index=False)
    print(f"Resumen de mejores modelos guardado en: {output_csv}")


###########################

def test_model_collect_predictions(path_models, model_version, feats_code, tf, tp, bin_size, split_code, cw_type, seed=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    freq = 32
    batch_size = 128
    ds_path = f'./dataset_resampled/dataset_{freq}Hz.csv'
    model_fun, features_fun, dataloader_fun, _, _ = train.set_model(model_version)
    selected_features, EEG_channels = train.set_features(feats_code)
    data_dict = data_utils.load_data_to_dict(ds_path, selected_features)
    folds = train.generate_subject_kfolds(data_dict, k=5)
    print('Testing folds...')
    all_true, all_pred, all_probs, all_times = [], [], [], []
    for fold_idx, (train_uids, test_uids) in enumerate(folds):
        print(f"\nEvaluando Fold {fold_idx + 1} -------------------------")
        _, _, test_dict, _ = train.get_partitions_from_fold(data_dict, train_uids, test_uids, split_code, seed)

        test_data = features_fun(test_dict, bin_size, freq)
        test_loader = train.create_dataloader(dataloader_fun, test_data, tp, tf, bin_size,
                                              batch_size=batch_size, shuffle=False)
        test_loader.dataset.return_onset = True  # Activa retorno del onset
        '''
        test_dataset = data_utils.AggressiveBehaviorDataset(
            test_data, tp=tp, tf=tf, bin_size=bin_size
        )
        test_dataset.return_onset = True
        test_loader = train.create_dataloader(dataloader_fun, test_dataset, tp, tf, bin_size,
                                              batch_size=batch_size, shuffle=False)
        '''

        lstm_hidden_dim, lstm_num_layers, num_classes = train.set_lstm()
        num_sequences = tp // bin_size
        eegnet_params = {'num_electrodes': EEG_channels, 'chunk_size': tp * freq // num_sequences}
        model = model_fun(eegnet_params, lstm_hidden_dim, lstm_num_layers, num_classes).to(device)

        model_path = f"{path_models}PM_sc{split_code}_mv{model_version}_f{feats_code}_tf{tf}" + \
                     f"_tp{tp}_bs{bin_size}_cw{cw_type}_fold{fold_idx}_model.pth"
        if not os.path.exists(model_path):
            print(f"Warning!: falta modelo {model_path}. Se omite fold {fold_idx}.")
            continue
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        fold_true, fold_pred, fold_probs, fold_onset = [], [], [], []
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                inputs, onsets, labels = batch
                if labels.ndim > 1:
                    labels = torch.argmax(labels, dim=1)

                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                fold_probs.extend(probs)
                fold_pred.extend(preds)
                fold_true.extend(labels.numpy())
                fold_onset.extend(onsets.numpy())

        all_true.append(np.array(fold_true))
        all_pred.append(np.array(fold_pred))
        all_probs.append(np.array(fold_probs))
        all_times.append(np.array(fold_onset))

    return all_true, all_pred, all_probs, all_times



def plot_confusion_matrices(all_true, all_pred, output_path, config_name):
    class_names = ["Calm", "Pre-episode", "Aggression"]
    num_folds = len(all_true)
    cm_list = [confusion_matrix(y_t, y_p, labels=[0, 1, 2]) for y_t, y_p in zip(all_true, all_pred)]
    cm_sum = np.sum(cm_list, axis=0)
    cm_avg = cm_sum.astype(float) / num_folds

    fig, axes = plt.subplots(1, num_folds + 2, figsize=(4 * (num_folds + 2), 4))
    titles = [f"Fold {i+1}" for i in range(num_folds)] + ["Sum", "Avg"]
    matrices = cm_list + [cm_sum, cm_avg]

    for ax, cm, title in zip(axes, matrices, titles):
        sns.heatmap(cm, annot=True, fmt='.0f' if title != "Avg" else '.2f',
                    xticklabels=class_names, yticklabels=class_names, cmap='Blues', ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    plt.tight_layout()
    fig.savefig(os.path.join(output_path, f"confusion_matrix_{config_name}.png"))
    plt.close(fig)


def plot_roc_pr_curves(all_true, all_probs, output_path, config_name):
    num_folds = len(all_true)
    class_names = ["Calm", "Pre-episode", "Aggression"]
    num_classes = len(class_names)

    fig, axes = plt.subplots(2, num_folds + 1, figsize=(5 * (num_folds + 1), 10))

    for fold_idx in range(num_folds + 1):
        if fold_idx < num_folds:
            y_true = np.array(all_true[fold_idx])
            y_probs = np.array(all_probs[fold_idx])
            title = f"Fold {fold_idx + 1}"
        else:
            y_true = np.concatenate(all_true)
            y_probs = np.concatenate(all_probs)
            title = "Average"

        for class_id in range(num_classes):
            y_true_bin = (y_true == class_id).astype(int)
            y_score = y_probs[:, class_id]
            fpr, tpr, _ = roc_curve(y_true_bin, y_score)
            prec, rec, _ = precision_recall_curve(y_true_bin, y_score)
            roc_auc = auc(fpr, tpr)
            ap = average_precision_score(y_true_bin, y_score)

            axes[0, fold_idx].plot(fpr, tpr, label=f"{class_names[class_id]} (AUC={roc_auc:.2f})")
            axes[1, fold_idx].plot(rec, prec, label=f"{class_names[class_id]} (AP={ap:.2f})")

        axes[0, fold_idx].set_title(f"ROC {title}")
        axes[1, fold_idx].set_title(f"PR {title}")
        axes[0, fold_idx].set_xlabel("FPR")
        axes[1, fold_idx].set_xlabel("Recall")
        if fold_idx == 0:
            axes[0, fold_idx].set_ylabel("TPR")
            axes[1, fold_idx].set_ylabel("Precision")
        axes[0, fold_idx].legend()
        axes[1, fold_idx].legend()

    plt.tight_layout()
    fig.savefig(os.path.join(output_path, f"roc_pr_curves_{config_name}.png"))
    plt.close(fig)


def analyze_pre_attack_timing(all_true, all_pred, all_times, output_path, config_name):
    pre_attack_times = []
    for y_t, y_p, t_on in zip(all_true, all_pred, all_times):
        for yt, yp, tt in zip(y_t, y_p, t_on):
            if yt == 1 and yp == 1 and tt != -1:
                pre_attack_times.append(tt)
    if not pre_attack_times:
        print("No correct Pre-episode predictions with valid onset found.")
        return

    pre_attack_times = np.array(pre_attack_times)
    mean_t = pre_attack_times.mean()
    median_t = np.median(pre_attack_times)
    std_t = pre_attack_times.std()
    print(f"Pre-episode onset analysis:")
    print(f" - N = {len(pre_attack_times)}")
    print(f" - Mean = {mean_t:.2f}s")
    print(f" - Median = {median_t:.2f}s")
    print(f" - Std = {std_t:.2f}s")

    plt.figure(figsize=(6, 4))
    plt.hist(pre_attack_times, bins=20, color='skyblue', edgecolor='black')
    plt.axvline(mean_t, color='red', linestyle='--', label=f"Mean = {mean_t:.1f}s")
    plt.title("Time to onset (correct Pre-episode predictions)")
    plt.xlabel("Seconds to aggressive episode onset")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"pre_attack_onset_hist_{config_name}.png"))
    plt.close()


def plot_confusion_matrices_thresholds(all_true, all_probs, thresholds, output_path, config_name):
    # Matrices de confusión para múltiples thresholds en predicciones multiclase.
    # Aplica umbrales por clase: sólo se consideran predicciones con probabilidad mayor o igual al umbral de su clase.
    # Si varias lo cumplen, se elige la de mayor probabilidad. Si ninguna, se elige la clase más probable. (TO-DO: descartar...)
    ## (TO-DO: descartar..., así no útil, solo con 0.5)
    class_names = ["Calm", "Pre-episode", "Aggression"]
    num_classes = len(class_names)
    num_folds = len(all_true)
    print('THRESHOLDS: ', thresholds)
    for th_idx, th_vec in enumerate(thresholds):
        cm_list = []
        for fold_idx in range(num_folds):
            y_true = np.array(all_true[fold_idx])
            y_probs = np.array(all_probs[fold_idx])
            y_pred = []
            for probs in y_probs:
                candidates = [i for i, p in enumerate(probs) if p >= th_vec[i]]
                if candidates:
                    pred = candidates[np.argmax([probs[i] for i in candidates])]
                else:
                    pred = np.argmax(probs)
                y_pred.append(pred)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
            cm_list.append(cm)
        cm_sum = np.sum(cm_list, axis=0)
        cm_avg = cm_sum.astype(float) / num_folds

        fig, axes = plt.subplots(1, num_folds + 2, figsize=(4 * (num_folds + 2), 4))
        titles = [f"Fold {i + 1}" for i in range(num_folds)] + ["Sum", "Avg"]
        matrices = cm_list + [cm_sum, cm_avg]
        for ax, cm, title in zip(axes, matrices, titles):
            sns.heatmap(cm, annot=True, fmt='.0f' if title != "Avg" else '.2f',
                        xticklabels=class_names, yticklabels=class_names, cmap='Blues', ax=ax)
            ax.set_title(title)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
        threshold_tag = "_".join([f"{t:.3f}" for t in th_vec])
        fig.tight_layout()
        fig.savefig(os.path.join(output_path, f"confusion_thresh_{threshold_tag}_{config_name}.png"))
        plt.close(fig)



def plot_metric_vs_threshold(y_true, y_probs, class_id=1, metric='f1', save_path=None):
    thresholds = np.linspace(0.01, 0.99, 50)
    metric_values = []

    for th in thresholds:
        y_pred = (y_probs[:, class_id] >= th).astype(int)
        y_true_bin = (y_true == class_id).astype(int)

        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true_bin, y_pred, average='binary', zero_division=0
        )

        if metric == 'precision':
            metric_values.append(prec)
        elif metric == 'recall':
            metric_values.append(rec)
        elif metric == 'f1':
            metric_values.append(f1)

    plt.figure(figsize=(6, 4))
    plt.plot(thresholds, metric_values, marker='o')
    plt.xlabel("Threshold")
    plt.ylabel(metric.capitalize())
    plt.title(f"{metric.capitalize()} vs Threshold (Class {class_id})")
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()



def plot_all_class_metrics_vs_threshold(y_true, y_probs, metric='f1', save_path=None, config_name="model"):
    thresholds = np.linspace(0.01, 0.99, 50)
    class_metrics = {0: [], 1: [], 2: []}
    colors = {0: 'blue', 1: 'green', 2: 'red'}
    labels = {0: 'Calm', 1: 'Pre-episode', 2: 'Aggression'}
    best_results = []
    for th in thresholds:
        for class_id in range(3):
            y_pred = (y_probs[:, class_id] >= th).astype(int)
            y_true_bin = (y_true == class_id).astype(int)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true_bin, y_pred, average='binary', zero_division=0
            )
            value = {'precision': prec, 'recall': rec, 'f1': f1}[metric]
            class_metrics[class_id].append(value)

    # Encontrar mejor threshold por clase
    for class_id in range(3):
        metric_values = class_metrics[class_id]
        best_idx = np.argmax(metric_values)
        best_results.append({
            "Class": labels[class_id],
            "Class_ID": class_id,
            "Best_Threshold": thresholds[best_idx],
            f"Best_{metric.upper()}": metric_values[best_idx],
            "Config": config_name
        })

    if save_path:
        df_summary = pd.DataFrame(best_results)
        csv_path = os.path.join(save_path, "f1_best_summary.csv")
        if os.path.exists(csv_path):
            df_summary.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df_summary.to_csv(csv_path, index=False)

        plt.figure(figsize=(8, 5))
        for class_id in range(3):
            plt.plot(thresholds, class_metrics[class_id], label=labels[class_id], color=colors[class_id])

        plt.xlabel("Threshold")
        plt.ylabel(metric.capitalize())
        plt.title(f"{metric.capitalize()} vs Threshold (All Classes)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{metric}_vs_threshold_{config_name}.png"))
        plt.close()
    else:
        plt.show()


def plot_binary_roc_curve(y_true_bin, y_pred_proba_bin, save_path):
    fpr, tpr, _ = roc_curve(y_true_bin, y_pred_proba_bin)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Binary (Calm vs Pre-episode + Aggression)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return roc_auc


def run_binary_calm_vs_aggressive_analysis(path_models, model_version, feats_code, tf, tp, bin_size, split_code,
                                           cw_type, seed=1):
    # Evaluación binaria: Calm (0) vs Agression (1: Pre-episode + Aggression).
    output_path = "./results_analysis"
    os.makedirs(output_path, exist_ok=True)
    config_name = f"PM_sc{split_code}_mv{model_version}_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}"
    all_true, _, all_probs, _ = test_model_collect_predictions(
        path_models, model_version, feats_code, tf, tp, bin_size, split_code, cw_type, seed
    )
    # Binarización: 0 → 0 (Calm), 1/2 → 1 (Aggression)
    y_true_all = np.concatenate(all_true)
    y_true_bin = (y_true_all > 0).astype(int)
    y_probs_all = np.concatenate(all_probs)
    # Suma de probs de clases 1 y 2 como clase Agression
    y_pred_proba_bin = y_probs_all[:, 1] + y_probs_all[:, 2]
    # ROC + AUC
    auc_score = plot_binary_roc_curve(
        y_true_bin, y_pred_proba_bin,
        save_path=f"{output_path}/roc_curve_binary_{config_name}.png"
    )
    print(f"Done! AUC: {auc_score:.4f}")


'''
results_dir = "./results/"
best_df = find_best_models_by_metric(results_dir)
best_df.to_csv("best_models_summary.csv", index=False)
'''
