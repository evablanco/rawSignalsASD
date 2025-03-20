import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import data_utils
import numpy as np


def plot_resampled_comparison(original_data, resampled_data, subject_id, session_id, start_seconds, interval_seconds):
    eda_data = original_data[subject_id][session_id]['EDA']
    bvp_data = original_data[subject_id][session_id]['BVP']
    acc_data = original_data[subject_id][session_id]['ACC']
    start_time = eda_data['Timestamp'].iloc[0] + (start_seconds * 1000)
    end_time = start_time + (interval_seconds * 1000)
    print(f"Intervalo de tiempo seleccionado: {start_time} - {end_time}")

    eda_filtered = eda_data[(eda_data['Timestamp'] >= start_time) & (eda_data['Timestamp'] < end_time)]
    bvp_filtered = bvp_data[(bvp_data['Timestamp'] >= start_time) & (bvp_data['Timestamp'] < end_time)]
    acc_filtered = acc_data[(acc_data['Timestamp'] >= start_time) & (acc_data['Timestamp'] < end_time)]
    condition_filtered = acc_data[(acc_data['Timestamp'] >= start_time) & (acc_data['Timestamp'] < end_time)][['Timestamp', 'Condition']]
    resampled_filtered = resampled_data[
        (resampled_data['SubjectID'] == subject_id) &
        (resampled_data['SessionID'] == session_id) &
        (resampled_data['Timestamp'] >= start_time) &
        (resampled_data['Timestamp'] < end_time)
        ]
    if resampled_filtered.empty:
        print("No hay datos en el intervalo especificado para las señales remuestreadas.")
        print(
            f"Rango de timestamps en `resampled_data`: {resampled_data['Timestamp'].min()} - {resampled_data['Timestamp'].max()}")
        return

    fig, axs = plt.subplots(4, 2, figsize=(15, 16), sharex=True)
    fig.suptitle(f'Subject {subject_id}, Session {session_id} - Interval: {interval_seconds} seconds', fontsize=16)
    styles = {
        'EDA': {'color': 'blue', 'marker': 'o', 'label': 'EDA'},
        'BVP': {'color': 'green', 'marker': 's', 'label': 'BVP'},
        'ACC_X': {'color': 'purple', 'marker': '^', 'label': 'ACC_X'},
        'ACC_Y': {'color': 'orange', 'marker': 'v', 'label': 'ACC_Y'},
        'ACC_Z': {'color': 'cyan', 'marker': 'D', 'label': 'ACC_Z'},
        'Condition': {'color': 'red', 'marker': 'x', 'label': 'Condition'}
    }
    point_size = 1
    line_alpha = 0.5
    line_style = '--'

    # EDA
    axs[0, 0].scatter(eda_filtered['Timestamp'], eda_filtered['EDA'], s=point_size, **styles['EDA'])
    axs[0, 0].plot(eda_filtered['Timestamp'], eda_filtered['EDA'], color=styles['EDA']['color'], linestyle=line_style,
                   alpha=line_alpha)
    axs[0, 1].scatter(resampled_filtered['Timestamp'], resampled_filtered['EDA'], s=point_size, **styles['EDA'])
    axs[0, 1].plot(resampled_filtered['Timestamp'], resampled_filtered['EDA'], color=styles['EDA']['color'],
                   linestyle=line_style, alpha=line_alpha)
    axs[0, 0].set_ylabel('EDA')
    axs[0, 0].legend([styles['EDA']['label']])
    axs[0, 1].legend([f'{styles["EDA"]["label"]}'])

    # BVP
    axs[1, 0].scatter(bvp_filtered['Timestamp'], bvp_filtered['BVP'], s=point_size, **styles['BVP'])
    axs[1, 0].plot(bvp_filtered['Timestamp'], bvp_filtered['BVP'], color=styles['BVP']['color'], linestyle=line_style,
                   alpha=line_alpha)
    axs[1, 1].scatter(resampled_filtered['Timestamp'], resampled_filtered['BVP'], s=point_size, **styles['BVP'])
    axs[1, 1].plot(resampled_filtered['Timestamp'], resampled_filtered['BVP'], color=styles['BVP']['color'],
                   linestyle=line_style, alpha=line_alpha)
    axs[1, 0].set_ylabel('BVP')
    axs[1, 0].legend([styles['BVP']['label']])
    axs[1, 1].legend([f'{styles["BVP"]["label"]}'])

    # ACC_X, ACC_Y y ACC_Z
    for component in ['ACC_X', 'ACC_Y', 'ACC_Z']:
        axs[2, 0].scatter(acc_filtered['Timestamp'], acc_filtered[component], s=point_size,
                          color=styles[component]['color'], marker=styles[component]['marker'],
                          label=styles[component]['label'])
        axs[2, 0].plot(acc_filtered['Timestamp'], acc_filtered[component], color=styles[component]['color'],
                       linestyle=line_style, alpha=line_alpha)

        axs[2, 1].scatter(resampled_filtered['Timestamp'], resampled_filtered[component], s=point_size,
                          color=styles[component]['color'], marker=styles[component]['marker'],
                          label=f'{styles[component]["label"]}')
        axs[2, 1].plot(resampled_filtered['Timestamp'], resampled_filtered[component], color=styles[component]['color'],
                       linestyle=line_style, alpha=line_alpha)

    axs[2, 0].set_ylabel('ACC')
    axs[2, 0].legend()
    axs[2, 1].legend()

    axs[3, 0].scatter(condition_filtered['Timestamp'], condition_filtered['Condition'], s=point_size,
                      **styles['Condition'])
    axs[3, 0].plot(condition_filtered['Timestamp'], condition_filtered['Condition'], color=styles['Condition']['color'],
                   linestyle=line_style, alpha=line_alpha)
    axs[3, 1].scatter(resampled_filtered['Timestamp'], resampled_filtered['Condition'], s=point_size,
                      **styles['Condition'])
    axs[3, 1].plot(resampled_filtered['Timestamp'], resampled_filtered['Condition'], color=styles['Condition']['color'],
                   linestyle=line_style, alpha=line_alpha)
    axs[3, 0].set_ylabel('Condition')
    axs[3, 0].legend([styles['Condition']['label']])
    axs[3, 1].legend([styles['Condition']['label']])

    for ax in axs[-1, :]:
        ax.set_xlabel('Timestamp (ms)')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()


def plot_subject_data(resampled_data, subject_id, session_id, start_seconds, interval_seconds):
    subject_session_data = resampled_data[
        (resampled_data['SubjectID'] == subject_id) &
        (resampled_data['SessionID'] == session_id)
        ]
    min_timestamp = subject_session_data['Timestamp'].min()
    max_timestamp = subject_session_data['Timestamp'].max()
    start_time = min_timestamp + (start_seconds * 1000)
    end_time = start_time + (interval_seconds * 1000)
    print(f"Intervalo de tiempo seleccionado: {start_time / 1000} - {end_time / 1000} segundos")

    filtered_data = subject_session_data[
        (subject_session_data['Timestamp'] >= start_time) &
        (subject_session_data['Timestamp'] < end_time)
        ]

    if filtered_data.empty:
        print(
            f"No hay datos en el intervalo especificado. Rango disponible para el sujeto {subject_id}, sesión {session_id}: {min_timestamp / 1000:.2f} - {max_timestamp / 1000:.2f} segundos")
        return

    fig, axs = plt.subplots(4, 1, figsize=(10, 16), sharex=True)
    fig.suptitle(f'Subject {subject_id}, Session {session_id} - Interval: {interval_seconds} seconds', fontsize=16)
    styles = {
        'EDA': {'color': 'blue', 'marker': 'o', 'label': 'EDA'},
        'BVP': {'color': 'green', 'marker': 's', 'label': 'BVP'},
        'ACC_X': {'color': 'purple', 'marker': '^', 'label': 'ACC_X'},
        'ACC_Y': {'color': 'orange', 'marker': 'v', 'label': 'ACC_Y'},
        'ACC_Z': {'color': 'cyan', 'marker': 'D', 'label': 'ACC_Z'},
        'Condition': {'color': 'red', 'marker': 'x', 'label': 'Condition'}
    }
    point_size = 1
    line_alpha = 0.5
    line_style = '--'

    axs[0].scatter(filtered_data['Timestamp'], filtered_data['EDA'], s=point_size, **styles['EDA'])
    axs[0].plot(filtered_data['Timestamp'], filtered_data['EDA'], color=styles['EDA']['color'], linestyle=line_style,
                alpha=line_alpha)
    axs[0].set_ylabel('EDA')
    axs[0].legend([styles['EDA']['label']])

    axs[1].scatter(filtered_data['Timestamp'], filtered_data['BVP'], s=point_size, **styles['BVP'])
    axs[1].plot(filtered_data['Timestamp'], filtered_data['BVP'], color=styles['BVP']['color'], linestyle=line_style,
                alpha=line_alpha)
    axs[1].set_ylabel('BVP')
    axs[1].legend([styles['BVP']['label']])

    for component in ['ACC_X', 'ACC_Y', 'ACC_Z']:
        axs[2].scatter(filtered_data['Timestamp'], filtered_data[component], s=point_size,
                       color=styles[component]['color'], marker=styles[component]['marker'],
                       label=styles[component]['label'])
        axs[2].plot(filtered_data['Timestamp'], filtered_data[component], color=styles[component]['color'],
                    linestyle=line_style, alpha=line_alpha)
    axs[2].set_ylabel('ACC')
    axs[2].legend()

    axs[3].scatter(filtered_data['Timestamp'], filtered_data['Condition'], s=point_size, **styles['Condition'])
    axs[3].plot(filtered_data['Timestamp'], filtered_data['Condition'], color=styles['Condition']['color'],
                linestyle=line_style, alpha=line_alpha)
    axs[3].set_ylabel('Condition')
    axs[3].legend([styles['Condition']['label']])
    axs[3].set_xlabel('Timestamp (s)')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()


def plot_all_subject_data_session(resampled_data, subject_id, session_id, save_path, show_plot=True):
    subject_session_data = resampled_data[
        (resampled_data['SubjectID'] == subject_id) &
        (resampled_data['SessionID'] == session_id)
        ]
    min_timestamp = subject_session_data['Timestamp'].min()
    max_timestamp = subject_session_data['Timestamp'].max()
    min_minutes = 0
    max_minutes = (max_timestamp - min_timestamp) / (60 * 1000)
    print(f"Duración de la sesión: {min_minutes} - {max_minutes:.2f} minutos")

    fig, axs = plt.subplots(4, 1, figsize=(12, 18), sharex=True)
    fig.suptitle(f'Subject {subject_id}, Session {session_id} - Full Session', fontsize=16)
    styles = {
        'EDA': {'color': 'blue', 'marker': 'o', 'label': 'EDA'},
        'BVP': {'color': 'green', 'marker': 's', 'label': 'BVP'},
        'ACC_X': {'color': 'purple', 'marker': '^', 'label': 'ACC_X'},
        'ACC_Y': {'color': 'orange', 'marker': 'v', 'label': 'ACC_Y'},
        'ACC_Z': {'color': 'cyan', 'marker': 'D', 'label': 'ACC_Z'},
        'Condition': {'color': 'red', 'marker': 'x', 'label': 'Condition'}
    }
    point_size = 1
    line_alpha = 0.5
    line_style = '--'
    time_in_minutes = (subject_session_data['Timestamp'] - min_timestamp) / (60 * 1000)

    axs[0].scatter(time_in_minutes, subject_session_data['EDA'], s=point_size, **styles['EDA'])
    axs[0].plot(time_in_minutes, subject_session_data['EDA'], color=styles['EDA']['color'], linestyle=line_style,
                alpha=line_alpha)
    axs[0].set_ylabel('EDA')
    axs[0].legend([styles['EDA']['label']])

    axs[1].scatter(time_in_minutes, subject_session_data['BVP'], s=point_size, **styles['BVP'])
    axs[1].plot(time_in_minutes, subject_session_data['BVP'], color=styles['BVP']['color'], linestyle=line_style,
                alpha=line_alpha)
    axs[1].set_ylabel('BVP')
    axs[1].legend([styles['BVP']['label']])

    for component in ['ACC_X', 'ACC_Y', 'ACC_Z']:
        axs[2].scatter(time_in_minutes, subject_session_data[component], s=point_size, color=styles[component]['color'],
                       marker=styles[component]['marker'], label=styles[component]['label'])
        axs[2].plot(time_in_minutes, subject_session_data[component], color=styles[component]['color'],
                    linestyle=line_style, alpha=line_alpha)
    axs[2].set_ylabel('ACC')
    axs[2].legend()

    axs[3].scatter(time_in_minutes, subject_session_data['Condition'], s=point_size, **styles['Condition'])
    axs[3].plot(time_in_minutes, subject_session_data['Condition'], color=styles['Condition']['color'],
                linestyle=line_style, alpha=line_alpha)
    axs[3].set_ylabel('Condition')
    axs[3].legend([styles['Condition']['label']])

    major_tick_interval = 5
    minor_tick_interval = 0.25

    axs[3].set_xticks(np.arange(min_minutes, max_minutes + 1, minor_tick_interval), minor=True)
    axs[3].set_xticks(np.arange(min_minutes, max_minutes + 1, major_tick_interval), minor=False)
    axs[3].set_xlabel('Tiempo (minutos)')
    axs[3].grid(True, which='both', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    # Guardar la figura
    os.makedirs(save_path, exist_ok=True)
    save_filename = f"{save_path}/subject_{subject_id}_session_{session_id}.png"
    plt.savefig(save_filename)
    if show_plot:
        plt.show()
    else:
        plt.close()


def format_timestamp(timestamp_ms):
    timestamp_s = timestamp_ms / 1000.0  # Convertir ms a segundos
    return datetime.fromtimestamp(timestamp_s).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]


### FUNCION CHECK OK
def resample_data_to_dataframe_EDA(subject_data, interp_method='linear'):
    combined_data_list = []
    for subject_id, sessions in subject_data.items():
        for session_id, signals in sessions.items():
            eda_data = signals['EDA'].copy()
            combined_data = eda_data[
                ['Timestamp', 'EDA', 'Note', 'AGG', 'ED', 'SIB', 'Condition']].copy()
            for signal_type, data in signals.items():
                if signal_type != 'EDA':
                    data = data.copy()
                    resampled_signal = data.set_index('Timestamp').reindex(eda_data['Timestamp']).interpolate(method=interp_method, limit_direction='both')
                    if signal_type == 'BVP':
                        combined_data = combined_data.join(resampled_signal[['BVP']], on='Timestamp', how='left')
                    elif signal_type == 'ACC':
                        combined_data = combined_data.join(resampled_signal[['ACC_X', 'ACC_Y', 'ACC_Z']],
                                                           on='Timestamp', how='left')
            combined_data.reset_index(drop=True, inplace=True)
            combined_data['SubjectID'] = subject_id
            combined_data['SessionID'] = session_id
            combined_data_list.append(combined_data)

    final_df = pd.concat(combined_data_list, ignore_index=True)
    return final_df


def resample_data_to_dataframe_ACC(subject_data, interp_method='linear'):
    combined_data_list = []
    for subject_id, sessions in subject_data.items():
        for session_id, signals in sessions.items():

            acc_data = signals['ACC'].copy()  # Usamos copy para evitar modificar el original
            #acc_data['TimestampAux'] = acc_data['Timestamp'].apply(format_timestamp)
            combined_data = acc_data[
                ['Timestamp', 'ACC_X', 'ACC_Y', 'ACC_Z', 'Note', 'AGG', 'ED', 'SIB', 'Condition']
            ].copy()

            if 'EDA' in signals:
                eda_data = signals['EDA'].copy()
                #eda_data['TimestampAux'] = eda_data['Timestamp'].apply(format_timestamp)
                resampled_eda = eda_data.set_index('Timestamp').reindex(acc_data['Timestamp']).interpolate(
                    method=interp_method, limit_direction='both')
                combined_data = combined_data.join(resampled_eda[['EDA']], on='Timestamp', how='left')

            if 'BVP' in signals:
                bvp_data = signals['BVP'].copy()
                #bvp_data['TimestampAux'] = bvp_data['Timestamp'].apply(format_timestamp)
                resampled_bvp = bvp_data.set_index('Timestamp').reindex(acc_data['Timestamp']).interpolate(
                    method=interp_method, limit_direction='both')
                combined_data = combined_data.join(resampled_bvp['BVP'], on='Timestamp', how='left')

            combined_data.reset_index(drop=True, inplace=True)
            combined_data['SubjectID'] = subject_id
            combined_data['SessionID'] = session_id

            combined_data_list.append(combined_data)

    final_df = pd.concat(combined_data_list, ignore_index=True)
    print("Filas con valores NaN después del procesamiento:")
    print(final_df[final_df.isna().any(axis=1)])
    return final_df


def resample_data_to_dataframe_BVP(subject_data, interp_method='linear'):
    combined_data_list = []
    for subject_id, sessions in subject_data.items():
        for session_id, signals in sessions.items():
            bvp_data = signals['BVP'].copy()
            #bvp_data['TimestampAux'] = bvp_data['Timestamp'].apply(format_timestamp)
            combined_data = bvp_data[
                ['Timestamp', 'BVP', 'Note', 'AGG', 'ED', 'SIB', 'Condition']
            ].copy()

            if 'EDA' in signals:
                eda_data = signals['EDA'].copy()
                #eda_data['TimestampAux'] = eda_data['Timestamp'].apply(format_timestamp)
                resampled_eda = eda_data.set_index('Timestamp').reindex(bvp_data['Timestamp']).interpolate(
                    method=interp_method, limit_direction='both')
                combined_data = combined_data.join(resampled_eda[['EDA']], on='Timestamp', how='left')
            if 'ACC' in signals:
                acc_data = signals['ACC'].copy()
                #acc_data['TimestampAux'] = acc_data['Timestamp'].apply(format_timestamp)
                resampled_acc = acc_data.set_index('Timestamp').reindex(bvp_data['Timestamp']).interpolate(
                    method=interp_method, limit_direction='both')
                combined_data = combined_data.join(resampled_acc[['ACC_X', 'ACC_Y', 'ACC_Z']], on='Timestamp',
                                                   how='left')
            combined_data.reset_index(drop=True, inplace=True)
            combined_data['SubjectID'] = subject_id
            combined_data['SessionID'] = session_id
            combined_data_list.append(combined_data)

    final_df = pd.concat(combined_data_list, ignore_index=True)
    num_nans = final_df.isna().sum()
    print('num_nans in final_df: ', num_nans)
    return final_df


def resample(orig_data, signal, path_to_save, interp_method='linear'):
    if signal == 'EDA':
        resampled_data = resample_data_to_dataframe_EDA(orig_data)
    elif signal == 'ACC':
        resampled_data = resample_data_to_dataframe_ACC(orig_data, interp_method=interp_method)
    elif signal == 'BVP':
        resampled_data = resample_data_to_dataframe_BVP(orig_data, interp_method=interp_method)
    else:
        print('Not supported signal...')
        resampled_data = pd.DataFrame([])
    resampled_data.to_csv(path_to_save)
    return resampled_data


def test_resampled(orig_data_path='./dataset', resample_data_path='./dataset_resampled/', hz_str='32Hz'):
    orig_data = data_utils.load_orig_data_dict(orig_data_path)
    resample_data = pd.read_csv(resample_data_path + 'dataset_' + hz_str + '.csv', dtype={'SubjectID': str, 'SessionID': str})
    plot_resampled_comparison(orig_data, resample_data, "1234.01", "12", start_seconds=180, interval_seconds=15)
    plot_resampled_comparison(orig_data, resample_data, "5056.01", "11", start_seconds=180, interval_seconds=15)
    plot_resampled_comparison(orig_data, resample_data, "3007.01", "03", start_seconds=180, interval_seconds=15)


def resample_dataset(dataset_path='./dataset/', output_path='./dataset_resampled/'):
    subject_data = data_utils.load_orig_data_dict(dataset_path)
    signals = ['EDA', 'ACC', 'BVP']
    frequencies = [4, 32, 64]
    frequencies_str = ['4Hz', '32Hz', '64Hz']
    signals = ['ACC']
    frequencies = [32]
    frequencies_str = ['32Hz']
    resample_dataset_path = './dataset_resampled/'
    #resample_dataset_path = './toy_dataset_resampled/' #test
    for signal, hz, hz_str in zip(signals, frequencies, frequencies_str):
        path_to_save= resample_dataset_path + 'dataset_' + hz_str + '.csv'
        resampled_data = resample(orig_data=subject_data, signal=signal, path_to_save=path_to_save)
        resampled_data.to_csv(output_path+'dataset'+hz_str+'.csv')
    test_resampled()


def generate_plots_for_all_subjects(resampled_data, save_path='./data_analysis'):
    subjects = resampled_data['SubjectID'].unique()
    for sub in subjects:
        sub_sessions = resampled_data[resampled_data['SubjectID'] == sub]['SessionID'].unique()
        for session in sub_sessions:
            plot_all_subject_data_session(resampled_data, sub, session, save_path, show_plot=False)


def save_subject_session_data(resampled_data, subject_id, session_id, save_path):
    subject_session_data = resampled_data[
        (resampled_data['SubjectID'] == subject_id) &
        (resampled_data['SessionID'] == session_id)
        ]

    if subject_session_data.empty:
        print(f"No hay datos para el sujeto {subject_id}, sesión {session_id}.")
        return

    os.makedirs(save_path, exist_ok=True)
    save_filename = f"{save_path}/subject_{subject_id}_session_{session_id}_data_from_resampled.csv"
    subject_session_data.to_csv(save_filename, index=False)
    print(f"Datos guardados en: {save_filename}")


def save_subject_session_data_from_dict(data_dict, subject_id, session_id, save_path):
    if subject_id not in data_dict or session_id not in data_dict[subject_id]:
        print(f"No hay datos disponibles para SubjectID {subject_id}, SessionID {session_id}.")
        return
    session_data = data_dict[subject_id][session_id]
    acc_data = session_data['ACC'][['Timestamp', 'ACC_X', 'ACC_Y', 'ACC_Z', 'Condition']] if 'ACC' in session_data else None
    bvp_data = session_data['BVP'][['Timestamp', 'BVP']] if 'BVP' in session_data else None
    eda_data = session_data['EDA'][['Timestamp', 'EDA']] if 'EDA' in session_data else None
    print('sum conditon: ', np.sum(acc_data['Condition']))
    merged_data = None
    if eda_data is not None:
        merged_data = eda_data
    if bvp_data is not None:
        merged_data = merged_data.merge(bvp_data, on='Timestamp', how='outer') if merged_data is not None else bvp_data
    if acc_data is not None:
        merged_data = merged_data.merge(acc_data, on='Timestamp', how='outer') if merged_data is not None else acc_data
    if merged_data is None or merged_data.empty:
        print(f"Error: No se pudo fusionar los datos de SubjectID {subject_id}, SessionID {session_id}.")
        return

    merged_data['SubjectID'] = subject_id
    merged_data['SessionID'] = session_id
    print('sum conditon merged: ', np.sum(merged_data['Condition']))
    columns_order = ['Timestamp', 'ACC_X', 'ACC_Y', 'ACC_Z', 'Condition', 'EDA', 'BVP', 'SubjectID', 'SessionID']
    available_columns = [col for col in columns_order if col in merged_data.columns]
    merged_data = merged_data[available_columns]

    os.makedirs(save_path, exist_ok=True)
    save_filename = os.path.join(save_path, f"subject_{subject_id}_session_{session_id}_data_from_dict.csv")
    merged_data.to_csv(save_filename, index=False)
    print(f"Datos guardados en: {save_filename}")


def analyse_subjects_data():
    freq = 32
    data_path_resampled = './dataset_resampled/'
    ds_path = data_path_resampled + f"dataset_{freq}Hz.csv"
    data_dict = data_utils.load_data_to_dict(ds_path)
    subject_ids = list(data_dict.keys())

    total_episodes = []
    total_sessions = []
    mean_episodes_per_session = []
    mean_episode_duration = []
    total_duration_episodes = []

    def compute_episode_count_and_duration(condition_series, time_series):
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
        return len(episode_durations), np.sum(episode_durations) if episode_durations else 0

    for subject_id in subject_ids:
        subject_data = data_dict.get(subject_id, {})
        num_episodes = 0
        num_sessions = len(subject_data)
        episodes_per_session = []
        total_duration = 0

        for session_id, session_data in subject_data.items():
            session_data = session_data.sort_index()
            num_episodes_in_session, total_duration_in_session = compute_episode_count_and_duration(
                session_data['Condition'], session_data.index
            )
            num_episodes += num_episodes_in_session
            episodes_per_session.append(num_episodes_in_session)
            total_duration += total_duration_in_session

        total_episodes.append(num_episodes)
        total_sessions.append(num_sessions)
        mean_episodes_per_session.append(num_episodes / num_sessions if num_sessions > 0 else 0)
        mean_episode_duration.append((total_duration / num_episodes) / 60 if num_episodes > 0 else 0)
        total_duration_episodes.append(total_duration / 60)

    analysis_df = pd.DataFrame({
        'Subject': subject_ids,
        'Total Episodes': total_episodes,
        'Total Sessions': total_sessions,
        'Mean Episodes per Session': np.round(mean_episodes_per_session, 2),
        'Mean Episode Duration (min.)': np.round(mean_episode_duration, 2),
        'Total Duration Episodes (min.)': np.round(total_duration_episodes, 2)
    })

    mean_values = analysis_df.iloc[:, 1:].mean().round(2)
    std_values = analysis_df.iloc[:, 1:].std().round(2)
    summary_df = pd.DataFrame([mean_values, std_values])
    summary_df.insert(0, 'Subject', ['Mean', 'STD'])
    analysis_df = pd.concat([analysis_df, summary_df], ignore_index=True)

    output_dir = "./results_analysis/"
    os.makedirs(output_dir, exist_ok=True)
    output_path_csv = os.path.join(output_dir, "all_subjects_data_analysis.csv")
    analysis_df.to_csv(output_path_csv, index=False)

    print(f"DataFrame saved at: {output_path_csv}")



'''
### TEST ###

dataset_path = './toy_data'
orig_data = data_utils.load_orig_data_dict(dataset_path)

new_data = resample_data_to_dataframe_EDA(orig_data)
save_data_path = './toy_data_resampled_EDA.csv'
new_data.to_csv(save_data_path)
plot_resampled_comparison(orig_data, new_data, "1223.01", "01", start_seconds=0, interval_seconds=3)

new_data = resample_data_to_dataframe_ACC(orig_data, interp_method='cubic')
save_data_path = './toy_data_resampled_ACC.csv'
new_data.to_csv(save_data_path)
plot_resampled_comparison(orig_data, new_data, "1223.01", "01", start_seconds=0, interval_seconds=3)

new_data = resample_data_to_dataframe_BVP(orig_data, interp_method='cubic')
save_data_path = './toy_data_resampled_BVP.csv'
new_data.to_csv(save_data_path)
plot_resampled_comparison(orig_data, new_data, "1223.01", "01", start_seconds=0, interval_seconds=3)
'''


'''
### Testing resampled data example
orig_data = data_utils.load_orig_data_dict('./dataset/')
resampled_data = pd.read_csv('./dataset_resampled/dataset_32Hz.csv', dtype={'SubjectID': str, 'SessionID': str})
##plot_resampled_comparison(orig_data, resampled_data, "1224.01", "01", start_seconds=0, interval_seconds=3)
plot_resampled_comparison(orig_data, resampled_data, "4107.01", "05", start_seconds=0, interval_seconds=int(60*45))
plot_subject_data(resampled_data, "4107.01", "05", start_seconds=0, interval_seconds=1200)
save_path = './data_analysis/'
plot_all_subject_data_session(resampled_data, "4107.01", "05", save_path, show_plot=True)
#generate_plots_for_all_subjects(resampled_data, save_path)
save_subject_session_data(resampled_data, "4107.01", "05", save_path)
save_subject_session_data_from_dict(orig_data, "4107.01", "05", save_path)
'''


'''
# Usage examples for paper
resampled_data = pd.read_csv('./dataset_resampled/dataset_32Hz.csv', dtype={'SubjectID': str, 'SessionID': str})
plot_subject_data(resampled_data, "4356.01", "01", start_seconds=0, interval_seconds=900)
analyse_subjects_data()
'''

