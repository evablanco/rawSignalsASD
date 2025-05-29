from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, roc_curve, auc, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
import torch
import train


def prepare_data_from_dataloader(dataloader):
    """ Extrae las características (X) y etiquetas (y) desde un DataLoader. """
    X = []
    y = []
    # batch_features, batch_aGGObs, batch_labels
    for _, (batch_features, batch_labels) in enumerate(dataloader):
        for batch_idx in range(batch_features.shape[0]):
            X.append(batch_features[batch_idx].numpy().flatten())
            y.append(batch_labels[batch_idx].item())
    return np.array(X), np.array(y)

def prepare_data_from_dataloader_AGGOBs(dataloader):
    """ Extrae las características (X) y etiquetas (y) desde un DataLoader. """
    X = []
    y = []
    # batch_features, batch_aGGObs, batch_labels
    for _, (batch_features, batch_aGGObs, batch_labels) in enumerate(dataloader):
        for batch_idx in range(batch_aGGObs.shape[0]):
            X.append(batch_aGGObs[batch_idx].numpy().flatten())
            y.append(batch_labels[batch_idx].item())
    return np.array(X), np.array(y)


def train_evaluate_dummy(dataloader_train, dataloader_test, model_code, strategy):
    """
    Entrena y evalúa un DummyClassifier con la estrategia especificada.
    Devuelve un diccionario con las métricas obtenidas.
    """
    if model_code == 1:
        X_train, y_train = prepare_data_from_dataloader(dataloader_train)
        X_test, y_test = prepare_data_from_dataloader(dataloader_test)
    elif model_code == 2:
        X_train, y_train = prepare_data_from_dataloader_AGGOBs(dataloader_train)
        X_test, y_test = prepare_data_from_dataloader_AGGOBs(dataloader_test)
    else:
        print('*** model_code error, Not supported yet: ', model_code)
        return

    dummy_clf = DummyClassifier(strategy=strategy)
    dummy_clf.fit(X_train, y_train)
    y_pred = dummy_clf.predict(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    accuracy = np.mean(y_pred == y_test)

    return {
        'F1-Score': f1,
        'Precision': precision,
        'Recall': recall,
        'AUC-ROC': roc_auc,
        'Accuracy': accuracy
    }


def start_exps_PM_dummy(tp, tf, freq, data_path_resampled, path_results, model_code, feats_code, split_code, b_size, seed=1):
    """
    Entrena y evalúa Dummy Classifiers (most_frequent y stratified) en cada fold, generando estadísticas
    en archivos CSV para cada estrategia.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    _, features_fun, dataloader_fun, _, _ = train.set_model(model_code)

    # Cargar datos
    ds_path = f"{data_path_resampled}dataset_{freq}Hz.csv"
    load_data_fun, _, _ = train.set_features(feats_code, None)
    data_dict = load_data_fun(ds_path)

    # Generar folds
    num_folds = 5
    folds = train.generate_user_kfolds(data_dict, k=num_folds)
    print(f"tp: {tp}, tf: {tf}, bin_size: {b_size}.")

    final_results_stratified = []
    final_results_mostfreq = []

    for fold_idx, (train_uids, test_uids) in enumerate(folds):
        print(f"Fold {fold_idx + 1}:")
        print(f"  Train UIDs: {train_uids}")
        print(f"  Test UIDs: {test_uids}")

        # Obtener particiones de train, val y test en función de los ids del fold
        _, _, test_dict, train_dict = train.get_partitions_from_fold(data_dict, train_uids, test_uids, split_code, False, seed)

        # Extraer características y etiquetas de los datos
        train_data = features_fun(train_dict, tp, tf, b_size, freq)
        test_data = features_fun(test_dict, tp, tf, b_size, freq)

        # Crear DataLoaders
        batch_size = 128
        dataloader_train = train.create_dataloader(dataloader_fun, train_data, tp, tf, b_size, batch_size, shuffle=True)
        dataloader_test = train.create_dataloader(dataloader_fun, test_data, tp, tf, b_size, batch_size, shuffle=False)

        # Entrenar y evaluar Dummy Classifiers
        results_strat = train_evaluate_dummy(dataloader_train, dataloader_test, model_code, strategy="stratified")
        results_strat["Fold"] = fold_idx
        final_results_stratified.append(results_strat)

        results_mostfreq = train_evaluate_dummy(dataloader_train, dataloader_test, model_code, strategy="most_frequent")
        results_mostfreq["Fold"] = fold_idx
        final_results_mostfreq.append(results_mostfreq)

    # Convertir resultados en DataFrame
    results_df_strat = pd.DataFrame(final_results_stratified)
    results_df_mostfreq = pd.DataFrame(final_results_mostfreq)

    # Calcular métricas de promedio y desviación estándar
    avg_metrics_strat = results_df_strat.mean()
    std_metrics_strat = results_df_strat.std()

    avg_metrics_mostfreq = results_df_mostfreq.mean()
    std_metrics_mostfreq = results_df_mostfreq.std()

    # Crear DataFrame resumen
    summary_df_strat = pd.DataFrame({
        "Fold": ["Avg.", "Std."],
        'F1-Score': [avg_metrics_strat['F1-Score'], std_metrics_strat['F1-Score']],
        'Precision': [avg_metrics_strat['Precision'], std_metrics_strat['Precision']],
        'Recall': [avg_metrics_strat['Recall'], std_metrics_strat['Recall']],
        'AUC-ROC': [avg_metrics_strat['AUC-ROC'], std_metrics_strat['AUC-ROC']],
        'Accuracy': [avg_metrics_strat['Accuracy'], std_metrics_strat['Accuracy']]
    })

    summary_df_mostfreq = pd.DataFrame({
        "Fold": ["Avg.", "Std."],
        'F1-Score': [avg_metrics_mostfreq['F1-Score'], std_metrics_mostfreq['F1-Score']],
        'Precision': [avg_metrics_mostfreq['Precision'], std_metrics_mostfreq['Precision']],
        'Recall': [avg_metrics_mostfreq['Recall'], std_metrics_mostfreq['Recall']],
        'AUC-ROC': [avg_metrics_mostfreq['AUC-ROC'], std_metrics_mostfreq['AUC-ROC']],
        'Accuracy': [avg_metrics_mostfreq['Accuracy'], std_metrics_mostfreq['Accuracy']]
    })

    # Guardar CSVs con los resultados
    final_results_df_stratified = pd.concat([results_df_strat, summary_df_strat], ignore_index=True)
    path_to_save_results = f'{path_results}dummy_stratified_PM_SS_mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_bs{b_size}_sc{split_code}_all_experiments_results_5cv.csv'
    final_results_df_stratified.to_csv(path_to_save_results, index=False)

    final_results_df_mostfreq = pd.concat([results_df_mostfreq, summary_df_mostfreq], ignore_index=True)
    path_to_save_results = f'{path_results}dummy_mostfreq_PM_SS_mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_bs{b_size}_sc{split_code}_all_experiments_results_5cv.csv'
    final_results_df_mostfreq.to_csv(path_to_save_results, index=False)

    print("Dummy model results saved successfully.")


def start_exps_PDM_dummy(tp, tf, freq, data_path_resampled, path_results, model_code, feats_code, split_code, b_size, seed=1):
    """
    Entrena y evalúa Dummy Classifiers (most_frequent y stratified) en cada fold, generando estadísticas
    en archivos CSV para cada estrategia.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    _, features_fun, dataloader_fun, _, _ = train.set_model(model_code)

    # Cargar datos
    ds_path = f"{data_path_resampled}dataset_{freq}Hz.csv"
    load_data_fun, _, _ = train.set_features(feats_code, None)
    data_dict = load_data_fun(ds_path)

    # Generar folds
    num_folds = 5
    folds = train.generate_user_kfolds(data_dict, k=num_folds)
    print(f"tp: {tp}, tf: {tf}, bin_size: {b_size}.")

    final_results_stratified = []
    final_results_mostfreq = []

    split_fun = train.get_split_fun(split_code, True)
    tp, tf, stride, bin_size = tp, tf, 15, 15
    print(f'tp: {tp}, tf: {tf}, stride: {stride}, bin_size: {bin_size}, sc:{split_code}.')

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
        dataloader = train.create_dataloader(dataloader_fun, train_dict, tp, tf, bin_size, batch_size, shuffle=True)
        dataloader_val = train.create_dataloader(dataloader_fun, val_dict, tp, tf, bin_size, batch_size, shuffle=False)
        dataloader_test = train.create_dataloader(dataloader_fun, test_dict, tp, tf, bin_size, batch_size, shuffle=False)
        dataloader_train = train.create_dataloader(dataloader_fun, train_dict_, tp, tf, bin_size, batch_size, shuffle=True)

        if train.invalid_data_PDM(dataloader) or train.invalid_data_PDM(dataloader_test) or train.invalid_data_PDM(dataloader_val) or train.invalid_data_PDM(dataloader_train):
            invalid_users.append(key_subject)
        else:
            # Entrenar y evaluar Dummy Classifiers
            results_strat = train_evaluate_dummy(dataloader_train, dataloader_test, model_code, strategy="stratified")
            results_strat["SubjectID"] = key_subject
            final_results_stratified.append(results_strat)

            results_mostfreq = train_evaluate_dummy(dataloader_train, dataloader_test, model_code,
                                                    strategy="most_frequent")
            results_mostfreq["SubjectID"] = key_subject
            final_results_mostfreq.append(results_mostfreq)
        data_dict.pop(next(iter(first_item)))


    # Convertir resultados en DataFrame
    results_df_strat = pd.DataFrame(final_results_stratified)
    results_df_mostfreq = pd.DataFrame(final_results_mostfreq)

    # Calcular métricas de promedio y desviación estándar
    avg_metrics_strat = results_df_strat.mean(numeric_only=True)
    std_metrics_strat = results_df_strat.std(numeric_only=True)

    avg_metrics_mostfreq = results_df_mostfreq.mean(numeric_only=True)
    std_metrics_mostfreq = results_df_mostfreq.std(numeric_only=True)

    # Crear DataFrame resumen
    summary_df_strat = pd.DataFrame({
        "Fold": ["Avg.", "Std."],
        'F1-Score': [avg_metrics_strat['F1-Score'], std_metrics_strat['F1-Score']],
        'Precision': [avg_metrics_strat['Precision'], std_metrics_strat['Precision']],
        'Recall': [avg_metrics_strat['Recall'], std_metrics_strat['Recall']],
        'AUC-ROC': [avg_metrics_strat['AUC-ROC'], std_metrics_strat['AUC-ROC']],
        'Accuracy': [avg_metrics_strat['Accuracy'], std_metrics_strat['Accuracy']]
    })

    summary_df_mostfreq = pd.DataFrame({
        "Fold": ["Avg.", "Std."],
        'F1-Score': [avg_metrics_mostfreq['F1-Score'], std_metrics_mostfreq['F1-Score']],
        'Precision': [avg_metrics_mostfreq['Precision'], std_metrics_mostfreq['Precision']],
        'Recall': [avg_metrics_mostfreq['Recall'], std_metrics_mostfreq['Recall']],
        'AUC-ROC': [avg_metrics_mostfreq['AUC-ROC'], std_metrics_mostfreq['AUC-ROC']],
        'Accuracy': [avg_metrics_mostfreq['Accuracy'], std_metrics_mostfreq['Accuracy']]
    })

    # Guardar CSVs con los resultados
    final_results_df_stratified = pd.concat([results_df_strat, summary_df_strat], ignore_index=True)
    path_to_save_results = f'{path_results}dummy_stratified_PDM_SS_mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_bs{b_size}_sc{split_code}_all_experiments_results_5cv.csv'
    final_results_df_stratified.to_csv(path_to_save_results, index=False)

    final_results_df_mostfreq = pd.concat([results_df_mostfreq, summary_df_mostfreq], ignore_index=True)
    path_to_save_results = f'{path_results}dummy_mostfreq_PDM_SS_mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_bs{b_size}_sc{split_code}_all_experiments_results_5cv.csv'
    final_results_df_mostfreq.to_csv(path_to_save_results, index=False)

    print("Dummy model results saved successfully.")


'''
# Usage example
tfs = [180, 120, 60]
tps = [60, 120, 180]

for tf in tfs:
    for tp in tps:
        tp, tf, freq = tp, tf, 32
        model_code, feats_code, split_code, b_size = 2, 0, 0, 15
        data_path_resampled = './dataset_resampled/'
        results_path = './results/'
        #start_exps_PM_dummy(tp, tf, freq, data_path_resampled, results_path, model_code, feats_code, split_code, b_size, seed=1)
        start_exps_PDM_dummy(tp, tf, freq, data_path_resampled, results_path, model_code, feats_code, split_code,
                             b_size, seed=1)
'''