import train, data_utils, models
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from evaluation_utils import model_evaluation
from sklearn.metrics import precision_recall_fscore_support
import re
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from captum.attr import IntegratedGradients


CALM, PRE_ATTACK, ATTACK = 0, 1, 2
ROOT = './normalized_sigs/'
results_analysis_path = ROOT + 'results_analysis/'

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
        df.to_csv(results_analysis_path + "evaluation_results_folds.csv", index=False)
        print("\nResultados guardados en 'evaluation_results_folds.csv'")
    else:
        print("No se ha evaluado ningún fold...")


########################################################################################################################
###                                             SELECTED TEST FUNCTIONS                                              ###
########################################################################################################################

def eval_conv_layers_prev(preloaded_data, feats_code, bin_size, tp, tf, split_code, cw_type, output_dir, max_samples_per_class=3):
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
            conv1_save_path = os.path.join(output_dir, f'PM_sc{split_code}_mv1_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}_conv1_class{cls}_fold{fold_idx}.png')
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
            conv2_save_path = os.path.join(output_dir, f'PM_sc{split_code}_mv1_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}_conv2_class{cls}_fold{fold_idx}.png') ### tp{}_tf{}_!!!
            plt.savefig(conv2_save_path)
            plt.close(fig)

    print("Conv visualizations saved.")


def generate_full_visual_analysis_prev(preloaded_data, feats_code, bin_size, tp, tf, split_code, cw_type, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_features = preloaded_data[0]['selected_features']
    EEG_channels = len(selected_features)
    os.makedirs(output_dir, exist_ok=True)
    fs = 32  # Frecuencia de muestreo

    for cls in [0, 1, 2]:
        activations_cls = []
        for fold_data in preloaded_data:
            model, loader = fold_data['model'].to(device).eval(), fold_data['test_loader']
            with torch.no_grad():
                for inputs, labels in loader:
                    idxs_cls = (labels == cls).nonzero(as_tuple=True)[0]
                    if len(idxs_cls) == 0:
                        continue

                    # Reformatear inputs correctamente
                    inputs_cls = inputs[idxs_cls]  # [N, F, C, T]
                    inputs_cls = inputs_cls.permute(0, 2, 1, 3)  # [N, C, F, T]
                    inputs_cls = inputs_cls.reshape(inputs_cls.size(0), 1, EEG_channels, -1).to(
                        device)  # [N, 1, EEG_channels, T]

                    # Pasar por la primera capa convolucional
                    conv1_out = model.eegnet.block1[0](inputs_cls)  # Salida: [N, F1, EEG_channels, T']
                    mean_act = conv1_out.mean(dim=1)  # Media sobre filtros → [N, EEG_channels, T']
                    activations_cls.extend(mean_act.cpu().numpy())

        if not activations_cls:
            continue

        avg_activation = np.mean(activations_cls, axis=0)  # [EEG_channels, T']
        plt.figure(figsize=(10, 6))
        sns.heatmap(avg_activation, cmap="viridis", xticklabels=False, yticklabels=selected_features)
        plt.xlabel("Time")
        plt.ylabel("Channels")
        plt.title(f"Mean Conv1 Activation - Class {cls}")
        save_path = os.path.join(output_dir,
                                 f'PM_sc{split_code}_mv1_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}_avg_conv1_activation_region_class{cls}.png')
        plt.savefig(save_path)
        plt.close()

    print(f"Full visual analysis saved to: {output_dir}")


def eval_features_and_gradients(preloaded_data, feats_code, bin_size, tp, tf, split_code, cw_type, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_features = preloaded_data[0]['selected_features']
    num_folds = len(preloaded_data)
    time_bins = np.linspace(-tp, -bin_size, tp // bin_size)  # time_bins.shape = (tp // bin_size,)

    # --- GRADIENTS POR FOLD Y MEDIA ---
    fig_grad, axs_grad = plt.subplots(1, num_folds + 1, figsize=(5 * (num_folds + 1), 4), sharey=True)
    all_fold_attrs = []

    for fold_data in preloaded_data:
        fold_idx = fold_data['fold_idx']
        model = fold_data['model'].to(device)
        test_loader = fold_data['test_loader']
        model.train()  # Necesario para capturar gradientes en LSTM
        ig = IntegratedGradients(model)

        fold_attrs = []
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            attrs = ig.attribute(inputs, target=1).cpu().numpy()
            fold_attrs.append(attrs)

        fold_attrs = np.concatenate(fold_attrs, axis=0)  # (N, chunks, features, time)
        mean_attributions = fold_attrs.mean(axis=0)      # (chunks, features, time)
        all_fold_attrs.append(mean_attributions)

        for i, feature in enumerate(selected_features):
            channel_attr = mean_attributions[:, i, :].mean(axis=1)  # (chunks,), axis=0 (time,)
            axs_grad[fold_idx].plot(time_bins, channel_attr, label=feature)

        axs_grad[fold_idx].set_title(f'Fold {fold_idx + 1}')
        axs_grad[fold_idx].set_xlabel('Time (sec)')
        axs_grad[fold_idx].grid(True)

    # MEDIA DE TODOS LOS FOLDS
    mean_all_folds = np.mean(all_fold_attrs, axis=0)  # (chunks, features, time)
    for i, feature in enumerate(selected_features):
        channel_attr = mean_all_folds[:, i, :].mean(axis=1)
        axs_grad[-1].plot(time_bins, channel_attr, label=feature)

    axs_grad[-1].set_title('Mean Across All Folds')
    axs_grad[-1].set_xlabel('Time (sec)')
    axs_grad[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axs_grad[-1].grid(True)
    plt.tight_layout()

    grad_save_path = os.path.join(output_dir,
                                  f'PM_sc{split_code}_mv1_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}_model_grads_avg.png')
    plt.savefig(grad_save_path)
    plt.close(fig_grad)

    # --- IMPORTANCIA POR CLASE ---
    fig_imp, axs_imp = plt.subplots(1, 3, figsize=(18, 5))
    for target_class in range(3):
        class_attrs_all_folds = []
        for fold_data in preloaded_data:
            model = fold_data['model'].to(device)
            test_loader = fold_data['test_loader']
            model.train()
            ig = IntegratedGradients(model)

            fold_class_attrs = []
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                attrs = ig.attribute(inputs, target=target_class).cpu().numpy()
                fold_class_attrs.append(attrs)

            fold_class_attrs = np.concatenate(fold_class_attrs, axis=0)  # (N, chunks, features, time)
            mean_attr = fold_class_attrs.mean(axis=(0, 3))               # (chunks, features)
            mean_attr_per_feature = mean_attr.mean(axis=0)              # (features,)
            class_attrs_all_folds.append(mean_attr_per_feature)

        class_attrs_mean = np.mean(class_attrs_all_folds, axis=0)
        axs_imp[target_class].bar(selected_features, class_attrs_mean)
        axs_imp[target_class].set_xlabel('Features')
        axs_imp[target_class].set_ylabel('Mean Attribution')
        axs_imp[target_class].set_title(f'Feature Importance - Class {target_class}')
        axs_imp[target_class].grid(True)
        axs_imp[target_class].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    feat_imp_save_path = os.path.join(output_dir,
                                      f'PM_sc{split_code}_mv1_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}_feature_importance_all_classes.png')
    plt.savefig(feat_imp_save_path)
    plt.close(fig_imp)

    print(f"Saved in:\n{grad_save_path}\n{feat_imp_save_path}")


def evaluate_errors_model_signals(preloaded_data, feats_code, bin_size, split_code, tp, tf, cw_type, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_names = ["Calm", "Pre-attack", "Attack"]
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    time_bins = np.arange(-tp, 0, bin_size)

    for target_class in range(3):
        fold_attrs = []

        for fold_data in preloaded_data:
            model = fold_data['model'].to(device)
            test_loader = fold_data['test_loader']
            selected_features = fold_data['selected_features']
            model.train()

            ig = IntegratedGradients(model)
            attrs_fold = []

            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                preds = model(inputs).argmax(dim=1)
                incorrect = preds != labels

                if incorrect.any():
                    incorrect_inputs = inputs[incorrect]
                    attrs = ig.attribute(incorrect_inputs, target=target_class).cpu().detach().numpy()
                    attrs = attrs.mean(axis=3)  # Average over time
                    attrs_fold.append(attrs)

            if attrs_fold:
                mean_attr = np.concatenate(attrs_fold, axis=0).mean(axis=0)
                fold_attrs.append(mean_attr)

        if fold_attrs:
            mean_attr = np.mean(fold_attrs, axis=0)
            for i, channel in enumerate(selected_features):
                axs[target_class].plot(time_bins, mean_attr[:, i], label=channel)

        axs[target_class].set_title(f'Class {class_names[target_class]} - Errors')
        axs[target_class].set_xlabel('Time (sec)')
        axs[target_class].legend()
        axs[target_class].grid()

    fig.tight_layout()
    fig_path = os.path.join(
        output_dir,
        f"PM_sc{split_code}_mv1_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}_errors_attr.png"
    )
    plt.savefig(fig_path)
    plt.close(fig)
    print(f"Saved: {fig_path}")


def visualize_conv_layers(preloaded_data, feats_code, bin_size, split_code, tp, tf, cw_type, output_dir,
                                   max_samples_per_class=3, target_classes=[0, 1, 2]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fs = 32  # sampling frequency

    for fold_data in preloaded_data:
        model = fold_data['model'].to(device)
        test_loader = fold_data['test_loader']
        selected_features = fold_data['selected_features']
        fold_idx = fold_data['fold_idx']
        EEG_channels = len(selected_features)
        model.eval()

        activs_block1 = {cls: [] for cls in target_classes}
        activs_block2 = {cls: [] for cls in target_classes}

        for inputs, labels in test_loader:
            batch_size = inputs.size(0)
            for b in range(batch_size):
                label = labels[b].item()
                if label in target_classes and len(activs_block1[label]) < max_samples_per_class:
                    # Preparar muestra individual
                    sample = inputs[b].unsqueeze(0).to(device)  # [1, F, C, T]
                    sample = sample.permute(0, 2, 1, 3)          # [1, C, F, T]
                    sample = sample.reshape(1, 1, -1, sample.shape[-1])  # [1, 1, EEG_channels, T]

                    with torch.no_grad():
                        conv1_out = model.eegnet.block1[0](sample)       # solo conv temporal
                        full_block1 = model.eegnet.block1(sample)        # block1 completo
                        conv2_out = model.eegnet.block2[0](full_block1)  # solo depthwise separable

                    activs_block1[label].append(conv1_out.squeeze(0).cpu().numpy())  # (F1, H, W)
                    activs_block2[label].append(conv2_out.squeeze(0).cpu().numpy())  # (F2, H, W)

            if all(len(activs_block1[cls]) >= max_samples_per_class for cls in target_classes):
                break

        def plot_activations(layer_name, data_dict, show_yticks=True):
            for cls, samples in data_dict.items():
                num_samples = len(samples)
                num_filters = samples[0].shape[0]
                fig, axs = plt.subplots(num_samples, num_filters,
                                        figsize=(3 * num_filters, 2.5 * num_samples), squeeze=False)
                for i in range(num_samples):
                    for j in range(num_filters):
                        act = samples[i][j]
                        time_len = act.shape[1]
                        xtick_positions = np.linspace(0, time_len - 1, 5)
                        xtick_labels = np.round(xtick_positions / fs).astype(int)

                        axs[i, j].imshow(act, aspect='auto', cmap='viridis')
                        axs[i, j].set_xticks(xtick_positions)
                        axs[i, j].set_xticklabels(xtick_labels)
                        axs[i, j].set_xlabel("Time (s)")

                        if show_yticks:
                            axs[i, j].set_yticks(np.arange(len(selected_features)))
                            axs[i, j].set_yticklabels(selected_features)
                            axs[i, j].set_ylabel("Channel")
                        else:
                            axs[i, j].set_yticks([])

                        if i == 0:
                            axs[i, j].set_title(f'Filter {j + 1}')

                plt.suptitle(f'Fold {fold_idx} - {layer_name} - Class {cls}', fontsize=16)
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                os.makedirs(output_dir, exist_ok=True)
                fname = f"PM_sc{split_code}_mv1_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}_fold{fold_idx}_class{cls}_{layer_name}.png"
                plt.savefig(os.path.join(output_dir, fname))
                plt.close()

        # Plot conv1 and conv2
        plot_activations("Conv1", activs_block1, show_yticks=True)
        plot_activations("Conv2", activs_block2, show_yticks=False)


def visualize_single_sample_conv1(preloaded_data, feats_code, bin_size, tp, tf, split_code, cw_type, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_features = preloaded_data[0]['selected_features']
    EEG_channels = len(selected_features)
    fs = 32  # Hz
    os.makedirs(output_dir, exist_ok=True)

    activations = {}

    for cls in [0, 1, 2]:
        for fold_data in preloaded_data:
            model = fold_data['model'].to(device).eval()
            loader = fold_data['test_loader']

            with torch.no_grad():
                for inputs, labels in loader:
                    idxs_cls = (labels == cls).nonzero(as_tuple=True)[0]
                    if len(idxs_cls) == 0:
                        continue

                    sample = inputs[idxs_cls[0]].unsqueeze(0).permute(0, 2, 1, 3).to(device)
                    sample = sample.reshape(1, 1, EEG_channels, -1)
                    conv1_out = model.eegnet.block1[0](sample).squeeze(0).cpu().numpy()
                    activations[cls] = conv1_out
                    break  # solo una muestra
            if cls in activations:
                break

    F1 = list(activations.values())[0].shape[0]
    fig, axs = plt.subplots(len(activations), F1, figsize=(3 * F1, 3 * len(activations)), squeeze=False)

    for i, (cls, act_map) in enumerate(activations.items()):
        for j in range(F1):
            heat = act_map[j]
            heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
            im = axs[i, j].imshow(heat, aspect='auto', cmap='viridis')
            #im = axs[i, j].imshow(act_map[j], aspect='auto', cmap='viridis')
            axs[i, j].set_title(f"Class {cls} - F{j + 1}")
            axs[i, j].set_xlabel("Time (s)")

            xticks = np.linspace(0, act_map.shape[2] - 1, 5)
            xticklabels = np.round(xticks / fs).astype(int)
            axs[i, j].set_xticks(xticks)
            axs[i, j].set_xticklabels(xticklabels)

            if j == 0:
                axs[i, j].set_yticks(np.arange(len(selected_features)))
                axs[i, j].set_yticklabels(selected_features)
                axs[i, j].set_ylabel("Channels")
            else:
                axs[i, j].set_yticks([])

    plt.suptitle("Conv1 Activations - One Sample per Class", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(output_dir, f"PM_sc{split_code}_mv1_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}_conv1_sample_activations.png")
    plt.savefig(save_path)
    plt.close()

    print(f"Saved single-sample Conv1 activations to: {save_path}")


def get_sample_from_dataloader(dataloader, target_class=0):
    """
    Devuelve la primera muestra (dato y etiqueta) de una clase específica del DataLoader.

    Args:
        dataloader (DataLoader): DataLoader del conjunto de test.
        target_class (int): Clase objetivo (0, 1 o 2).

    Returns:
        Tuple[Tensor, int] o (None, None) si no se encuentra.
    """
    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            inputs, labels = batch[0], batch[-1]  # puede venir como (data, label) o (data, aggObs, label)
        else:
            continue  # no válido

        indices = (labels == target_class).nonzero(as_tuple=True)[0]
        if len(indices) > 0:
            idx = indices[0].item()
            return inputs[idx], labels[idx].item()

    return None, None


def gradcam_conv1_visualization(preloaded_data, feats_code, bin_size, split_code, tp, tf, cw_type, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_features = preloaded_data[0]['selected_features']
    EEG_channels = len(selected_features)
    os.makedirs(output_dir, exist_ok=True)
    fs = 32  # Hz

    for cls in [0, 1, 2]:  # calm, pre, attack
        for fold_data in preloaded_data:
            model = fold_data['model'].to(device)
            loader = fold_data['test_loader']

            input_sample = None
            for inputs, labels in loader:
                with torch.no_grad():
                    preds = model(inputs.to(device))
                    if isinstance(preds, tuple):
                        preds = preds[0]
                    preds = preds.argmax(dim=1).cpu()
                idxs = ((labels == cls) & (labels == preds)).nonzero(as_tuple=True)[0]
                if len(idxs) > 0:
                    input_sample = inputs[idxs[0]].unsqueeze(0).to(device)  # [1, seq, C, T]
                    break
            if input_sample is None:
                continue

            # Grad-CAM setup
            gradients = []
            activations = []

            def forward_hook(module, input, output):
                activations.append(output.detach())

            def backward_hook(module, grad_input, grad_output):
                gradients.append(grad_output[0].detach())

            handle_fwd = model.eegnet.block1.register_forward_hook(forward_hook)
            handle_bwd = model.eegnet.block1.register_full_backward_hook(backward_hook)

            # Reshape input like in training
            batch_size, num_seqs, C, chunk_len = input_sample.shape
            input_sample_reshaped = input_sample.view(-1, C, chunk_len).unsqueeze(1)  # [seq, 1, C, T]

            # Forward
            model.train()  # Needed for LSTM Grad-CAM
            x1 = model.eegnet.block1(input_sample_reshaped)   # [seq, F1*D, 1, T']
            x2 = model.eegnet.block2(x1)                      # [seq, F2, 1, T'']
            x2_flat = x2.flatten(start_dim=1)                # [seq, F2*T'']
            x_seq = x2_flat.view(batch_size, num_seqs, -1)   # [1, seq, feat]
            lstm_out, _ = model.lstm(x_seq)                  # [1, seq, H]
            output = model.fc(lstm_out[:, -1, :])            # [1, num_classes]

            pred_class = output.argmax(dim=1).item()
            model.zero_grad()
            output[0, pred_class].backward()
            model.eval()

            # Grad-CAM
            grad = gradients[0]                               # [seq, F1*D, 1, T']
            act = activations[0]                              # same
            weights = grad.mean(dim=[2, 3], keepdim=True)     # [seq, F1*D, 1, 1]
            cam = (weights * act).sum(dim=1)                  # [seq, 1, T']
            cam = torch.relu(cam).squeeze(1).cpu().numpy()    # [seq, T']
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)  # Normalize to [0, 1]

            # Input EEG (original signals)
            input_eeg = input_sample.squeeze(0).cpu().numpy()    # [seq, C, T]
            signal_len = input_eeg.shape[0] * input_eeg.shape[2]
            time_signal = np.linspace(-tp, 0, signal_len)

            fig, axs = plt.subplots(EEG_channels, 1, figsize=(12, 3 * EEG_channels), sharex=True)

            for ch in range(EEG_channels):
                eeg_ch = input_eeg[:, ch, :].reshape(-1)  # [seq*T]
                cam_ch = cam[:, :input_eeg.shape[2]].reshape(-1)  # [seq*T]

                axs[ch].plot(time_signal, eeg_ch, color='black', linewidth=0.8, label='EEG')
                pcm = axs[ch].imshow(cam_ch[np.newaxis, :],
                                     extent=[-tp, 0, eeg_ch.min(), eeg_ch.max()],
                                     aspect='auto', cmap='jet', alpha=0.6,
                                     vmin=0.0, vmax=1.0)

                axs[ch].set_ylabel(selected_features[ch])
                axs[ch].grid(True, linestyle='--', alpha=0.3)

            # Colorbar
            cbar = fig.colorbar(pcm, ax=axs, orientation='vertical', fraction=0.02, pad=0.01)
            cbar.set_label("Grad-CAM Activation", fontsize=10)

            axs[-1].set_xlabel("Time (s)")
            plt.suptitle(f"Grad-CAM - Class {cls} (Pred: {pred_class}) - Conv1", fontsize=14)
            plt.tight_layout(rect=[0, 0, 0.98, 0.96])

            fname = f"PM_sc{split_code}_mv1_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}_gradcam_conv1_class{cls}.png"
            plt.savefig(os.path.join(output_dir, fname))
            plt.close()

            handle_fwd.remove()
            handle_bwd.remove()
            break  # Solo una muestra por clase

    print("Grad-CAM visualizations saved.")


def gradcam_conv1_visualization_onlycorrect(preloaded_data, feats_code, bin_size, split_code, tp, tf, cw_type, output_dir):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_features = preloaded_data[0]['selected_features']
    EEG_channels = len(selected_features)
    os.makedirs(output_dir, exist_ok=True)
    fs = 32  # Hz

    for cls in [0, 1, 2]:  # calm, pre, attack
        found_sample = False  # para salir en cuanto encontremos una correcta por clase

        for fold_data in preloaded_data:
            model = fold_data['model'].to(device)
            loader = fold_data['test_loader']

            input_sample = None

            # Recorrer todo el loader hasta encontrar muestra correcta
            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                with torch.no_grad():
                    outputs = model(inputs)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    preds = outputs.argmax(dim=1)

                correct_idxs = ((labels == cls) & (preds == labels)).nonzero(as_tuple=True)
                if correct_idxs[0].numel() > 0:
                    idx = correct_idxs[0][0].item()
                    input_sample = inputs[idx].unsqueeze(0)  # [1, seq, C, T]
                    true_label = labels[idx].cpu()
                    true_pred = preds[idx].cpu()
                    print(
                        f"Fold {fold_data['fold_idx']}, Class {cls}: Selected correctly predicted sample at batch index {idx}")
                    found_sample = True
                    break
                else:
                    print(f"Fold {fold_data['fold_idx']}, Class {cls}: No correctly predicted samples in this batch.")

            if not found_sample:
                continue  # ve al siguiente fold

            # --- Grad-CAM setup ---
            gradients = []
            activations = []

            def forward_hook(module, input, output):
                activations.append(output.detach())

            def backward_hook(module, grad_input, grad_output):
                gradients.append(grad_output[0].detach())

            h1 = model.eegnet.block1.register_forward_hook(forward_hook)
            h2 = model.eegnet.block1.register_full_backward_hook(backward_hook)

            # Forward
            batch_size, num_seqs, C, chunk_len = input_sample.shape
            input_sample_reshaped = input_sample.view(-1, C, chunk_len).unsqueeze(1)  # [seq, 1, C, T]

            model.train()
            x1 = model.eegnet.block1(input_sample_reshaped)
            x2 = model.eegnet.block2(x1)
            x2_flat = x2.flatten(start_dim=1)
            x_seq = x2_flat.view(batch_size, num_seqs, -1)
            lstm_out, _ = model.lstm(x_seq)
            output = model.fc(lstm_out[:, -1, :])

            pred_class = output.argmax(dim=1).item()
            model.zero_grad()
            output[0, pred_class].backward()
            model.eval()

            # Grad-CAM
            grad = gradients[0]
            act = activations[0]
            weights = grad.mean(dim=[2, 3], keepdim=True)
            cam = (weights * act).sum(dim=1)
            cam = torch.relu(cam).squeeze(1).cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)

            # Signal
            input_eeg = input_sample.squeeze(0).cpu().numpy()  # [seq, C, T]
            signal_len = input_eeg.shape[0] * input_eeg.shape[2]
            time_signal = np.linspace(-tp, 0, signal_len)

            fig, axs = plt.subplots(EEG_channels, 1, figsize=(12, 3 * EEG_channels), sharex=True)

            for ch in range(EEG_channels):
                eeg_ch = input_eeg[:, ch, :].reshape(-1)
                cam_ch = cam[:, :input_eeg.shape[2]].reshape(-1)

                axs[ch].plot(time_signal, eeg_ch, color='black', linewidth=0.8, label='EEG')
                pcm = axs[ch].imshow(cam_ch[np.newaxis, :],
                                     extent=[-tp, 0, eeg_ch.min(), eeg_ch.max()],
                                     aspect='auto', cmap='jet', alpha=0.6,
                                     vmin=0.0, vmax=1.0)

                axs[ch].set_ylabel(selected_features[ch])
                axs[ch].grid(True, linestyle='--', alpha=0.3)

            cbar = fig.colorbar(pcm, ax=axs, orientation='vertical', fraction=0.02, pad=0.01)
            cbar.set_label("Grad-CAM Activation", fontsize=10)

            axs[-1].set_xlabel("Time (s)")
            plt.suptitle(f"Grad-CAM - Class {true_label.item()} (Pred: {true_pred.item()}) - Conv1", fontsize=14)
            #plt.suptitle(f"Grad-CAM - Class {cls} (Pred: {pred_class}) - Conv1", fontsize=14)
            plt.tight_layout(rect=[0, 0, 0.98, 0.96])

            fname = f"PM_sc{split_code}_mv1_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}_gradcam_conv1_class{cls}.png"
            plt.savefig(os.path.join(output_dir, fname))
            plt.close()

            h1.remove()
            h2.remove()
            break  # una muestra por clase

    print("Grad-CAM visualizations saved.")


def test_model_collect_predictions_from_preloaded(preloaded):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_true, all_pred, all_probs, all_times = [], [], [], []
    for fold_data in preloaded:
        model = fold_data['model'].to(device)
        loader = fold_data['test_loader']
        model.eval()
        # Activamos retorno del tiempo de onset
        loader.dataset.return_onset = True
        fold_true, fold_pred, fold_probs, fold_onset = [], [], [], []
        with torch.no_grad():
            for batch in loader:
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


def analize_onsetpreloaded_data(preloaded, feats_code, bin_size, split_code, tp, tf, cw_type, output_dir):
    all_true, all_pred, all_probs, all_times = test_model_collect_predictions_from_preloaded(preloaded)
    config_name = f"PM_sc{split_code}_mv{1}_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}"
    analyze_pre_attack_timing(all_true, all_pred, all_times, output_dir, config_name)


def evaluate_and_plot_confusion_matrices_preloaded(preloaded_data, feats_code, bin_size, split_code, tp, tf, cw_type, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["Calm", "Pre-attack", "Attack"]
    CALM, PRE_ATTACK, ATTACK = 0, 1, 2
    threshold_active = 0.5
    all_true, all_pred = [], []
    print('Testing preloaded folds...')
    for fold_data in preloaded_data:
        fold_idx = fold_data['fold_idx']
        model = fold_data['model'].to(device)
        test_loader = fold_data['test_loader']
        test_loader.dataset.return_onset = False
        print(f"\nEvaluating Fold {fold_idx + 1} -------------------------")
        model.eval()
        fold_true, fold_pred = [], []
        with torch.no_grad():
            for batch in test_loader:
                inputs, labels = batch
                if labels.ndim > 1:
                    labels = torch.argmax(labels, dim=1)
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()

                for prob_vec in probs:
                    pred_argmax = np.argmax(prob_vec)
                    '''
                    if pred_argmax == CALM:
                        # Aplicar threshold solo si el modelo predijo Calm
                        pre_prob = prob_vec[PRE_ATTACK]
                        att_prob = prob_vec[ATTACK]
                        if pre_prob >= threshold_active or att_prob >= threshold_active:
                            pred = np.argmax([prob_vec[CALM], pre_prob, att_prob])
                        else:
                            pred = CALM
                    else:
                        # Si predijo Pre-attack o Attack, se respeta esa predicción
                        pred = pred_argmax
                    '''
                    fold_pred.append(pred_argmax) # pred for custom

                fold_true.extend(labels.numpy())

        all_true.append(np.array(fold_true))
        all_pred.append(np.array(fold_pred))

    # Confusion matrices por fold
    conf_matrices = [
        confusion_matrix(true, pred, labels=range(len(class_names)))
        for true, pred in zip(all_true, all_pred)
    ]

    os.makedirs(output_dir, exist_ok=True)
    fig1_path = f"{output_dir}PM_sc{split_code}_mv1_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}_confM1.png"

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
    fig2_path = f"{output_dir}PM_sc{split_code}_mv1_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}_confM2.png"
    fig2.savefig(fig2_path)
    plt.close(fig2)

    print(f"Figures saved to: {fig1_path} and {fig2_path}")


def run_binary_calm_vs_aggressive_analysis_preloaded(preloaded, feats_code, bin_size, split_code, tp, tf, cw_type, output_dir):
    # Evaluación binaria: Calm (0) vs Agression (1: Pre-episode + Aggression).
    config_name = f"PM_sc{split_code}_mv{1}_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}"
    all_true, _, all_probs, _ = test_model_collect_predictions_from_preloaded(preloaded)
    # Binarización: 0 → 0 (Calm), 1/2 → 1 (Aggression)
    y_true_all = np.concatenate(all_true)
    y_true_bin = (y_true_all > 0).astype(int)
    y_probs_all = np.concatenate(all_probs)
    # Suma de probs de clases 1 y 2 como clase Agression
    y_pred_proba_bin = y_probs_all[:, 1] + y_probs_all[:, 2]
    # ROC + AUC
    auc_score = plot_binary_roc_curve(
        y_true_bin, y_pred_proba_bin,
        save_path=f"{output_dir}/{config_name}_roc_curve_binary.png"
    )
    print(f"Done! AUC: {auc_score:.4f}")


def plot_roc_pr_curves_preloaded(preloaded, feats_code, bin_size, split_code, tp, tf, cw_type, output_dir):
    """
    generates two rows (roc/pr) and num_folds+1 columns (each fold + average)
    saves to: <output_path>/roc_pr_curves_<config_name>.png
    """
    num_folds = len(preloaded)
    class_names = ["Calm", "Pre-episode", "Aggression"]
    num_classes = len(class_names)

    # collect per-fold arrays
    all_true = []
    all_probs = []

    device = next(preloaded[0]['model'].parameters()).device

    for entry in preloaded:
        model = entry['model']
        loader = entry['test_loader']
        loader.dataset.return_onset = False
        model.eval()

        fold_true = []
        fold_probs = []
        with torch.no_grad():
            for batch in loader:
                inputs, labels = batch
                # if labels one-hot, take argmax
                if labels.ndim > 1:
                    labels = labels.argmax(dim=1)
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                fold_probs.extend(probs)
                fold_true.extend(labels.cpu().numpy())

        all_true.append(np.array(fold_true))
        all_probs.append(np.array(fold_probs))

    # now plotting exactly as before
    fig, axes = plt.subplots(2, num_folds + 1, figsize=(5 * (num_folds + 1), 10))

    for fold_idx in range(num_folds + 1):
        if fold_idx < num_folds:
            y_true = all_true[fold_idx]
            y_probs = all_probs[fold_idx]
            title = f"Fold {fold_idx + 1}"
        else:
            y_true = np.concatenate(all_true, axis=0)
            y_probs = np.concatenate(all_probs, axis=0)
            title = "Average"

        for class_id in range(num_classes):
            y_true_bin = (y_true == class_id).astype(int)
            y_score = y_probs[:, class_id]

            fpr, tpr, _ = roc_curve(y_true_bin, y_score)
            prec, rec, _ = precision_recall_curve(y_true_bin, y_score)
            roc_auc = auc(fpr, tpr)
            ap = average_precision_score(y_true_bin, y_score)

            axes[0, fold_idx].plot(fpr, tpr,
                label=f"{class_names[class_id]} (AUC={roc_auc:.2f})")
            axes[1, fold_idx].plot(rec, prec,
                label=f"{class_names[class_id]} (AP={ap:.2f})")

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
    os.makedirs(output_dir, exist_ok=True)
    config_name = f"PM_sc{split_code}_mv{1}_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}"
    fig.savefig(os.path.join(output_dir, f"{config_name}_roc_pr_curves.png"))
    plt.close(fig)


def preload_folds_and_models(path_models, ds_path, feats_code, bin_size, tp, tf, split_code, cw_type, num_folds=5):
    freq = 32
    selected_features, EEG_channels = train.set_features(feats_code)
    data_dict = data_utils.load_data_to_dict(ds_path, selected_features)
    folds = train.generate_subject_kfolds(data_dict, k=num_folds)
    lstm_hidden_dim, lstm_num_layers, num_classes = train.set_lstm()
    num_sequences = tp // bin_size
    eegnet_params = {'num_electrodes': EEG_channels, 'chunk_size': tp * freq // num_sequences}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preloaded = []
    for fold_idx, (train_uids, test_uids) in enumerate(folds):
        _, _, test_dict, all_train_dict = train.get_partitions_from_fold(data_dict, train_uids, test_uids, split_code, seed=42)
        mean, std = data_utils.compute_normalization_stats(all_train_dict, bin_size, freq)
        test_data = data_utils.get_features_from_dict(test_dict, bin_size, freq, mean, std)
        test_loader = train.create_dataloader(
            data_utils.AggressiveBehaviorDataset, test_data, tp, tf, bin_size, batch_size=32, shuffle=False
        )
        model_path = f"{path_models}PM_sc{split_code}_mv1_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}_fold{fold_idx}_model.pth"
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


### final test fun
def evaluate_model_all():
    models_base_path = ROOT + "models/"
    ds_path = "./dataset_resampled/dataset_32Hz.csv"
    feats_code = 6
    bin_size = 15
    tp, tf = 300, 300
    split_code = 1
    cw_type = 1
    output_dir = ROOT + 'results_analysis/'

    # Precarga los datos y modelos (una sola vez)
    preloaded_data = preload_folds_and_models(models_base_path, ds_path, feats_code, bin_size, tp, tf, split_code,
                                              cw_type, num_folds=5)

    # Evalúa gradientes y características usando datos precargados: OK
    eval_features_and_gradients(preloaded_data, feats_code, bin_size, tp, tf, split_code, cw_type, output_dir)
    evaluate_errors_model_signals(preloaded_data, feats_code, bin_size, tp, tf, split_code, cw_type, output_dir)  # error con tf 60 pero no 300....

    # Capas convolucionales y activaciones, media y una muestra, por clase. OK
    visualize_single_sample_conv1(preloaded_data, feats_code, bin_size, split_code, tp, tf, cw_type, output_dir)

    gradcam_conv1_visualization_onlycorrect(preloaded_data, feats_code, bin_size, split_code, tp, tf, cw_type, output_dir)

    analize_onsetpreloaded_data(preloaded_data, feats_code, bin_size, split_code, tp, tf, cw_type, output_dir)

    run_binary_calm_vs_aggressive_analysis_preloaded(preloaded_data, feats_code, bin_size, split_code, tp, tf, cw_type, output_dir)

    evaluate_and_plot_confusion_matrices_preloaded(preloaded_data, feats_code, bin_size, split_code, tp, tf, cw_type, output_dir)

    plot_roc_pr_curves_preloaded(preloaded_data, feats_code, bin_size, split_code, tp, tf, cw_type, output_dir)


def run_full_evaluation_analysis(path_models, model_version, feats_code, tf, tp, bin_size, split_code,
                                 cw_type, seed=1):
    output_path = results_analysis_path # TO-DO: args
    os.makedirs(output_path, exist_ok=True)
    config_name = f"PM_sc{split_code}_mv{model_version}_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_cw{cw_type}"
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

        model_path = f"{path_models}PM_sc{split_code}_mv{model_version}_f{feats_code}_tf{tf}" + \
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
    fig1_path = f"{output_dir}PM_sc{split_code}_mv{model_version}_f{feats_code}_tf{tf}" + \
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
    fig2_path = f"{output_dir}PM_sc{split_code}_mv{model_version}_f{feats_code}_tf{tf}" + \
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


def summarize_results(dir_path: str, label: str):
    """
    scan all csv files in dir_path starting with label (eg pm_, pdm_, pmtl_)
    pick the best avg on last-2 row for each metric, save summary csv
    """
    METRICS = [
        "F1_macro","F1_weighted","AUC_macro","Accuracy",
        "F1_Calm","F1_Pre-attack","F1_Attack"
    ]
    best = {m: None for m in METRICS}

    for fname in os.listdir(dir_path):
        if not fname.endswith(".csv") or not fname.startswith(label + "_"):
            continue
        path = os.path.join(dir_path, fname)
        df = pd.read_csv(path)
        if df.shape[0] < 2:
            continue
        avg_row = df.iloc[-2]
        std_row = df.iloc[-1]

        # extract params from filename
        param_vals = {}
        for key in ("mv","f","tf","tp","bs","sc","cw"):
            m = re.search(rf"_{key}(\d+)", fname)
            param_vals[key] = m.group(1) if m else ""

        for met in METRICS:
            if met not in df.columns:
                continue
            try:
                mean_val = float(avg_row[met])
                std_val  = float(std_row[met])
            except:
                continue
            rec = {
                "Mean":      mean_val,
                "Std":       std_val,
                **param_vals,
                "File_name": fname
            }
            if best[met] is None or mean_val > best[met]["Mean"]:
                best[met] = rec

    # build output rows
    rows = []
    for met, rec in best.items():
        if rec is None:
            continue
        row = {"metric": met}
        row.update(rec)
        rows.append(row)

    df_out = pd.DataFrame(rows, columns=[
        "metric","Mean","Std",
        "sc","mv","f","tf","tp","bs","cw",
        "File_name"
    ])
    out_fn = os.path.join(dir_path, f"{label}_summary_results.csv")
    df_out.to_csv(out_fn, index=False)
    print(f"[+] saved summary csv: {out_fn}")




'''
#evaluate_model_all()  ## TO-DO: integrar desde main con params específicos en lugar de definirlos en la funcion
summarize_results("./normalized_sigs/results", "PM")
summarize_results("./normalized_sigs/results", "PDM")
summarize_results("./normalized_sigs/results", "PMTL")
summarize_results("./not_normalized/results", "PM")
summarize_results("./not_normalized/results", "PDM")
summarize_results("./not_normalized/results", "PMTL")
'''
