import train, data_utils, models
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
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import glob
from captum.attr import IntegratedGradients


CALM, PRE_ATTACK, ATTACK = 0, 1, 2
results_analysis_path = './normalized_sigs/results_analysis'

def extract_hyperparams_from_filename(filename):
    pattern = r"PMTL_mv(\d+)_f(\d+)_tf(\d+)_tp(\d+)_bs(\d+)_sc(\d+)_cw(\d+)"
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
            match = re.search(r"PMTL_mv(\d+)_f(\d+)_tf(\d+)_tp(\d+)_bs(\d+)_sc(\d+)_cw(\d+)", filename)
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

        model_path = f"{path_models}PMTL_sc{split_code}_mv{model_version}_f{feats_code}_tf{tf}" + \
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
        df.to_csv(results_analysis_path + "evaluation_results_folds.csv", index=False)
        print("\nResultados guardados en 'evaluation_results_folds.csv'")
    else:
        print("No se ha evaluado ningún fold...")


def eval_features_and_gradients(preloaded_data, feats_code, bin_size, tp, tf, split_code, cw_type, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_features = preloaded_data[0]['selected_features']
    num_folds = len(preloaded_data)
    time_bins = np.arange(-tp, 0, bin_size)

    fig_grad, axs_grad = plt.subplots(1, num_folds + 1, figsize=(5 * (num_folds + 1), 4), sharey=True)
    all_fold_attrs = []
    for fold_data in preloaded_data:
        fold_idx = fold_data['fold_idx']
        model = fold_data['model']
        test_loader = fold_data['test_loader']
        ig = IntegratedGradients(model)
        fold_attrs = []
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            attrs = ig.attribute(inputs, target=1).cpu().detach().numpy()
            fold_attrs.append(attrs)
        fold_attrs = np.concatenate(fold_attrs, axis=0)
        mean_attributions = fold_attrs.mean(axis=0)
        all_fold_attrs.append(mean_attributions)
        for i, channel in enumerate(selected_features):
            channel_attr = mean_attributions[:, i, :].mean(axis=1)
            axs_grad[fold_idx].plot(time_bins, channel_attr, label=channel)
        axs_grad[fold_idx].set_title(f'Fold {fold_idx + 1}')
        axs_grad[fold_idx].set_xlabel('Time (sec)')
        axs_grad[fold_idx].grid(True)
    mean_all_folds = np.mean(all_fold_attrs, axis=0)
    for i, channel in enumerate(selected_features):
        channel_attr = mean_all_folds[:, i, :].mean(axis=1)
        axs_grad[-1].plot(time_bins, channel_attr, label=channel)
    axs_grad[-1].set_title('Mean Across All Folds')
    axs_grad[-1].set_xlabel('Time (sec)')
    axs_grad[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axs_grad[-1].grid(True)
    plt.tight_layout()
    grad_save_path = os.path.join(output_dir,
                                  f'PMTL_sc{split_code}_mv1_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}_model_grads_avg.png')
    plt.savefig(grad_save_path)
    plt.close(fig_grad)

    fig_imp, axs_imp = plt.subplots(1, 3, figsize=(18, 5))
    for target_class in range(3):
        class_attrs_all_folds = []
        for fold_data in preloaded_data:
            model = fold_data['model']
            test_loader = fold_data['test_loader']
            ig = IntegratedGradients(model)
            fold_class_attrs = []
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                attrs = ig.attribute(inputs, target=target_class).cpu().detach().numpy()
                fold_class_attrs.append(attrs)
            fold_class_attrs = np.concatenate(fold_class_attrs, axis=0)  # (N, chunks, features, time)
            mean_attr = fold_class_attrs.mean(axis=(0, 3))  # (chunks, features)
            mean_attr_per_feature = mean_attr.mean(axis=0)  # (features,)
            class_attrs_all_folds.append(mean_attr_per_feature)
        # Media sobre folds
        class_attrs_mean = np.mean(class_attrs_all_folds, axis=0)
        axs_imp[target_class].bar(selected_features, class_attrs_mean)
        axs_imp[target_class].set_xlabel('Features')
        axs_imp[target_class].set_ylabel('Mean Attribution')
        axs_imp[target_class].set_title(f'Feature Importance - Class {target_class}')
        axs_imp[target_class].grid(True)
        axs_imp[target_class].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    feat_imp_save_path = os.path.join(output_dir,
                                      f'PMTL_sc{split_code}_mv1_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}_feature_importance_all_classes.png')
    plt.savefig(feat_imp_save_path)
    plt.close(fig_imp)

    print(f"Saved in:\n{grad_save_path}\n{feat_imp_save_path}")


def eval_conv_layers(preloaded_data, feats_code, tp, bin_size, split_code, cw_type, output_dir, max_samples_per_class=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fs = 32
    selected_features = preloaded_data[0]['selected_features']
    target_classes = [0, 1, 2]

    # Creación del directorio de salida
    os.makedirs(output_dir, exist_ok=True)

    for fold_data in preloaded_data:
        fold_idx = fold_data['fold_idx']
        model = fold_data['model'].to(device)
        test_loader = fold_data['test_loader']
        EEG_channels = len(selected_features)
        model.eval()

        # Almacenar activaciones de la primera capa conv y segunda capa conv (F2)
        class_activations_conv1 = {cls: [] for cls in target_classes}
        class_activations_conv2 = {cls: [] for cls in target_classes}

        with torch.no_grad():
            for inputs, labels in test_loader:
                for input_single, label_single in zip(inputs, labels):
                    label_single = label_single.item()
                    if len(class_activations_conv1[label_single]) >= max_samples_per_class:
                        continue

                    input_single = input_single.unsqueeze(0).to(device)  # shape (1, C, Chunks, T)
                    if input_single.size(1) != 1:
                        input_single = input_single.permute(0, 2, 1, 3).reshape(1, 1, EEG_channels, -1)

                    # Conv1 activations
                    conv1_output = model.eegnet.block1[0](input_single)
                    class_activations_conv1[label_single].append(conv1_output.squeeze(0).cpu().numpy())

                    # Conv2 activations (F2)
                    conv2_input = conv1_output
                    conv2_output = model.eegnet.block2[0](conv2_input)
                    class_activations_conv2[label_single].append(conv2_output.squeeze(0).cpu().numpy())

                # Verifica si se alcanzó el límite de muestras por clase
                if all(len(class_activations_conv1[cls]) >= max_samples_per_class for cls in target_classes):
                    break

        # Guardar figuras por clase para la capa Conv1
        for cls in target_classes:
            samples = class_activations_conv1[cls]
            if not samples:
                continue
            num_filters = samples[0].shape[0]
            fig, axs = plt.subplots(len(samples), num_filters, figsize=(3*num_filters, 2.5*len(samples)), squeeze=False)
            for i, sample in enumerate(samples):
                for j in range(num_filters):
                    activation = sample[j]
                    xtick_positions = np.linspace(0, activation.shape[1] - 1, 5)
                    xtick_labels = np.round(xtick_positions / fs).astype(int)
                    axs[i, j].imshow(activation, aspect='auto', cmap='viridis')
                    axs[i, j].set_xticks(xtick_positions)
                    axs[i, j].set_xticklabels(xtick_labels)
                    axs[i, j].set_xlabel("Time (s)")
                    axs[i, j].set_yticks(np.arange(len(selected_features)))
                    axs[i, j].set_yticklabels(selected_features)
                    axs[i, j].set_ylabel("Channel")
                    if i == 0:
                        axs[i, j].set_title(f'Conv1 Filter {j + 1}')

            plt.suptitle(f'Conv1 Activations - Class {cls} - Fold {fold_idx}', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            conv1_save_path = os.path.join(output_dir, f'conv1_class{cls}_fold{fold_idx}.png')
            plt.savefig(conv1_save_path)
            plt.close(fig)

        # Guardar figuras por clase para la capa Conv2 (F2)
        for cls in target_classes:
            samples = class_activations_conv2[cls]
            if not samples:
                continue
            num_filters = samples[0].shape[0]
            fig, axs = plt.subplots(len(samples), num_filters, figsize=(3*num_filters, 2.5*len(samples)), squeeze=False)
            for i, sample in enumerate(samples):
                for j in range(num_filters):
                    activation = sample[j]
                    axs[i, j].imshow(activation, aspect='auto', cmap='viridis')
                    axs[i, j].axis('off')
                    if i == 0:
                        axs[i, j].set_title(f'Conv2 Filter {j + 1}')
            plt.suptitle(f'Conv2 Activations - Class {cls} - Fold {fold_idx}', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            conv2_save_path = os.path.join(output_dir, f'conv2_class{cls}_fold{fold_idx}.png') ### tp{}_tf{}_!!!
            plt.savefig(conv2_save_path)
            plt.close(fig)

    print("Conv visualizations saved.")



def preload_folds_and_models(path_models, ds_path, feats_code, bin_size, tp, tf, split_code, cw_type, num_folds=5):
    selected_features, EEG_channels = train.set_features(feats_code)
    data_dict = data_utils.load_data_to_dict(ds_path, selected_features)
    folds = train.generate_subject_kfolds(data_dict, k=num_folds)
    lstm_hidden_dim, lstm_num_layers, num_classes = train.set_lstm()
    num_sequences = tp // bin_size
    eegnet_params = {'num_electrodes': EEG_channels, 'chunk_size': tp * 32 // num_sequences}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preloaded = []
    for fold_idx, (train_uids, test_uids) in enumerate(folds):
        _, _, test_dict, _ = train.get_partitions_from_fold(data_dict, train_uids, test_uids, split_code, seed=42)
        test_data = data_utils.get_features_from_dict(test_dict, bin_size, 32)
        test_loader = train.create_dataloader(
            data_utils.AggressiveBehaviorDataset, test_data, tp, tf, bin_size, batch_size=32, shuffle=False
        )
        model_path = f"{path_models}PMTL_sc{split_code}_mv1_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}_fold{fold_idx}_model.pth"
        model = models.EEGNetLSTM(eegnet_params, lstm_hidden_dim, lstm_num_layers, num_classes).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
        model.train()
        preloaded.append({
            'fold_idx': fold_idx,
            'model': model,
            'test_loader': test_loader,
            'selected_features': selected_features
        })
    return preloaded


def evaluate_model_all():
    models_base_path = "./models/"
    ds_path = "./dataset_resampled/dataset_32Hz.csv"
    feats_code = 0
    bin_size = 15
    tp, tf = 300, 300
    split_code = 1
    cw_type = 1
    output_dir = './results_analysis/'

    # Precarga los datos y modelos (una sola vez)
    preloaded_data = preload_folds_and_models(
        models_base_path, ds_path, feats_code, bin_size, tp, tf, split_code, cw_type, num_folds=5
    )

    # Evalúa gradientes y características usando datos precargados
    eval_features_and_gradients(
        preloaded_data, feats_code, bin_size, tp, tf, split_code, cw_type, output_dir
    )

    # Capas convolucionales
    eval_conv_layers(
        preloaded_data, feats_code, tp, bin_size, split_code, tp, tf, cw_type, output_dir
    )


def run_full_evaluation_analysis(path_models, model_version, feats_code, tf, tp, bin_size, split_code,
                                 cw_type, seed=1):
    output_path = results_analysis_path # TO-DO: args
    os.makedirs(output_path, exist_ok=True)
    config_name = f"PMTL_sc{split_code}_mv{model_version}_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}"
    print(f"evaluación completa para {config_name}")
    all_true, all_pred, all_probs, all_times = test_model_collect_predictions(
        path_models, model_version, feats_code, tf, tp, bin_size, split_code, cw_type, seed
    )

    thresholds = [
        np.array([0.5, 0.5, 0.5]) #,  # base
        # np.array([0.5, 0.7, 0.7]) # TO-DO... arreglar lógica, actualmente male...
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
    ### eval features, covs and grads


    print(f"resultados en: {output_path}")




def get_best_performing_models_prev(results_folder, f1_threshold_csv, output_csv=results_analysis_path+"summary_best_models.csv"):
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


def get_best_performing_models(results_folder, f1_threshold_csv, output_csv=results_analysis_path+"summary_best_models.csv"):
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

        model_path = f"{path_models}PMTL_sc{split_code}_mv{model_version}_f{feats_code}_tf{tf}" + \
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
    config_name = f"PMTL_sc{split_code}_mv{model_version}_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}"
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


def custom_predict_with_threshold(logits, threshold=0.6):
    probs = torch.softmax(logits, dim=1)  # (batch_size, num_classes)
    predictions = []
    for sample_probs in probs:
        pre_attack_prob = sample_probs[PRE_ATTACK].item()
        if pre_attack_prob >= threshold:
            predictions.append(PRE_ATTACK)
        else:
            calm_attack_probs = [sample_probs[CALM].item(), sample_probs[ATTACK].item()]
            other_class = CALM if calm_attack_probs[0] > calm_attack_probs[1] else ATTACK
            predictions.append(other_class)
    return predictions


def evaluate_f1_score_model(model, dataloader, device, threshold_pre_attack=0.6):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_features, batch_labels in dataloader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(batch_features)
            probs = torch.softmax(logits, dim=1)  # shape: (batch_size, num_classes)
            for sample_probs in probs:
                pre_prob = sample_probs[PRE_ATTACK].item()
                if pre_prob >= threshold_pre_attack:
                    pred = PRE_ATTACK
                else:
                    calm_prob = sample_probs[CALM].item()
                    attack_prob = sample_probs[ATTACK].item()
                    pred = CALM if calm_prob > attack_prob else ATTACK
                all_preds.append(pred)
            all_labels.extend(batch_labels.cpu().numpy())
    return f1_score(all_labels, all_preds, average='macro')


def test_model_collect_predictions_with_threshold(
    path_models, model_version, feats_code, tf, tp, bin_size,
    split_code, cw_type, seed=1, threshold_pre_attack=0.6
):
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
        test_loader.dataset.return_onset = True
        lstm_hidden_dim, lstm_num_layers, num_classes = train.set_lstm()
        num_sequences = tp // bin_size
        eegnet_params = {'num_electrodes': EEG_channels, 'chunk_size': tp * freq // num_sequences}
        model = model_fun(eegnet_params, lstm_hidden_dim, lstm_num_layers, num_classes).to(device)
        model_path = f"{path_models}PMTL_sc{split_code}_mv{model_version}_f{feats_code}_tf{tf}" + \
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
                for prob_vec in probs:
                    pre_prob = prob_vec[PRE_ATTACK]
                    if pre_prob >= threshold_pre_attack:
                        pred = PRE_ATTACK
                    else:
                        pred = CALM if prob_vec[CALM] > prob_vec[ATTACK] else ATTACK
                    fold_pred.append(pred)
                    fold_probs.append(prob_vec)
                fold_true.extend(labels.numpy())
                fold_onset.extend(onsets.numpy())
        all_true.append(np.array(fold_true))
        all_pred.append(np.array(fold_pred))
        all_probs.append(np.array(fold_probs))
        all_times.append(np.array(fold_onset))
    return all_true, all_pred, all_probs, all_times




def evaluate_and_plot_confusion_matrices(
    path_models, model_version, feats_code, tf, tp, bin_size,
    split_code, cw_type, output_dir=results_analysis_path, threshold_active=0.5, seed=1
):

    # Constantes
    freq = 32
    batch_size = 128
    class_names = ["Calm", "Pre-attack", "Attack"]
    CALM, PRE_ATTACK, ATTACK = 0, 1, 2

    # Cargar datos y preparar entorno
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_path = f'./dataset_resampled/dataset_{freq}Hz.csv'
    model_fun, features_fun, dataloader_fun, _, _ = train.set_model(model_version)
    selected_features, EEG_channels = train.set_features(feats_code)
    data_dict = data_utils.load_data_to_dict(ds_path, selected_features)
    folds = train.generate_subject_kfolds(data_dict, k=5)

    all_true, all_pred = [], []
    print('Testing folds...')

    for fold_idx, (train_uids, test_uids) in enumerate(folds):
        print(f"\nEvaluando Fold {fold_idx + 1} -------------------------")
        _, _, test_dict, _ = train.get_partitions_from_fold(data_dict, train_uids, test_uids, split_code, seed)
        test_data = features_fun(test_dict, bin_size, freq)
        test_loader = train.create_dataloader(dataloader_fun, test_data, tp, tf, bin_size, batch_size=batch_size, shuffle=False)
        test_loader.dataset.return_onset = True

        lstm_hidden_dim, lstm_num_layers, num_classes = train.set_lstm()
        num_sequences = tp // bin_size
        eegnet_params = {'num_electrodes': EEG_channels, 'chunk_size': tp * freq // num_sequences}
        model = model_fun(eegnet_params, lstm_hidden_dim, lstm_num_layers, num_classes).to(device)

        model_path = f"{path_models}PMTL_sc{split_code}_mv{model_version}_f{feats_code}_tf{tf}" + \
                     f"_tp{tp}_bs{bin_size}_cw{cw_type}_fold{fold_idx}_model.pth"
        if not os.path.exists(model_path):
            print(f"Warning!: falta modelo {model_path}. Se omite fold {fold_idx}.")
            continue

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        fold_true, fold_pred = [], []
        with torch.no_grad():
            for batch in test_loader:
                inputs, onsets, labels = batch
                if labels.ndim > 1:
                    labels = torch.argmax(labels, dim=1)
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()

                for prob_vec in probs:
                    pre_prob = prob_vec[PRE_ATTACK]
                    att_prob = prob_vec[ATTACK]

                    if pre_prob >= threshold_active or att_prob >= threshold_active:
                        pred = np.argmax([prob_vec[CALM], pre_prob, att_prob])
                    else:
                        pred = CALM
                    fold_pred.append(pred)

                fold_true.extend(labels.numpy())

        all_true.append(np.array(fold_true))
        all_pred.append(np.array(fold_pred))

    # Calcular matrices de confusión por fold
    conf_matrices = [
        confusion_matrix(true, pred, labels=range(len(class_names)))
        for true, pred in zip(all_true, all_pred)
    ]

    os.makedirs(output_dir, exist_ok=True)
    fig1_path = f"{output_dir}PMTL_sc{split_code}_mv{model_version}_f{feats_code}_tf{tf}" + \
                 f"_tp{tp}_bs{bin_size}_cw{cw_type}_confM1.png"
    # === FIGURA 1: 5 subfiguras (una por fold) ===
    fig1, axs1 = plt.subplots(1, 5, figsize=(22, 4))
    for i, cm in enumerate(conf_matrices):
        sns.heatmap(cm, annot=True, fmt="d", ax=axs1[i], cmap="Blues",
                    cbar=False, xticklabels=class_names, yticklabels=class_names)
        axs1[i].set_title(f'Fold {i+1}')
        axs1[i].set_xlabel('Predicted')
        axs1[i].set_ylabel('True')
    fig1.tight_layout()
    fig1.savefig(fig1_path)
    plt.close(fig1)

    # === FIGURA 2: suma y media de las matrices ===
    cm_sum = np.sum(conf_matrices, axis=0)
    cm_mean = np.mean(conf_matrices, axis=0)

    fig2, axs2 = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm_sum, annot=True, fmt="d", ax=axs2[0], cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    axs2[0].set_title('Sum of Confusion Matrices')
    axs2[0].set_xlabel('Predicted')
    axs2[0].set_ylabel('True')

    sns.heatmap(cm_mean, annot=True, fmt=".1f", ax=axs2[1], cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    axs2[1].set_title('Mean of Confusion Matrices')
    axs2[1].set_xlabel('Predicted')
    axs2[1].set_ylabel('True')

    fig2.tight_layout()
    fig2_path = f"{output_dir}PMTL_sc{split_code}_mv{model_version}_f{feats_code}_tf{tf}" + \
                f"_tp{tp}_bs{bin_size}_cw{cw_type}_confM2.png"
    fig2.savefig(fig2_path)
    plt.close(fig2)

    print(f"Figuras guardadas en: {fig1_path} y {fig2_path}")



def plot_max_activation_regions(preloaded_data, feats_code, tp, bin_size, output_dir, max_samples_per_class=5):
    """
    Visualiza las regiones de activación máxima por clase y canal en la primera capa convolucional.
    - preloaded_data: dict con datos precargados, modelos y dataloaders por fold.
    - feats_code: índice de características usadas.
    - tp: tiempo de pasado en segundos.
    - bin_size: tamaño del bin para chunks temporales.
    - output_dir: directorio donde guardar las figuras.
    """
    os.makedirs(output_dir, exist_ok=True)
    fs = 32  # frecuencia de muestreo
    time_ticks = np.linspace(0, tp, tp // bin_size + 1)

    # Acumular activaciones por clase y canal
    class_channel_max_map = {}  # {clase: [list of (channel x time)]}
    for fold_idx, fold_data in enumerate(preloaded_data):
        model = fold_data['model']
        test_loader = fold_data['test_loader']
        selected_features = fold_data['selected_features']
        EEG_channels = len(selected_features)

        model.eval()
        device = next(model.parameters()).device

        activations_per_class = {0: [], 1: [], 2: []}

        with torch.no_grad():
            for inputs, labels in test_loader:
                label = labels.item()
                if label not in activations_per_class:
                    continue
                if len(activations_per_class[label]) >= max_samples_per_class:
                    continue

                inputs = inputs.to(device)
                if inputs.dim() == 4 and inputs.size(1) != 1:
                    inputs = inputs.permute(0, 2, 1, 3)
                    inputs = inputs.reshape(inputs.shape[0], 1, EEG_channels, -1)

                conv_output = model.eegnet.block1[0](inputs)
                act = conv_output.squeeze(0).cpu().numpy()  # (F1, C, T)
                activations_per_class[label].append(act)

        for cls, samples in activations_per_class.items():
            if cls not in class_channel_max_map:
                class_channel_max_map[cls] = []

            for sample in samples:
                max_act = sample.max(axis=0)  # (C, T): máximo entre filtros
                class_channel_max_map[cls].append(max_act)

    # Promediar y visualizar por clase
    for cls, maps in class_channel_max_map.items():
        maps = np.array(maps)  # (N, C, T)
        mean_map = maps.mean(axis=0)  # (C, T)
        plt.figure(figsize=(10, 6))
        sns.heatmap(mean_map, cmap="viridis", xticklabels=round(mean_map.shape[1] / 10), yticklabels=fold_data['selected_features'])
        plt.title(f"Zonas de Máxima Activación Promedio - Clase {cls}")
        plt.xlabel("Tiempo")
        plt.ylabel("Canales")
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"max_activation_regions_class{cls}.png")
        plt.savefig(save_path)
        plt.close()

    return True




'''
results_dir = "./results/"
best_df = find_best_models_by_metric(results_dir)
best_df.to_csv("best_models_summary.csv", index=False)
'''
