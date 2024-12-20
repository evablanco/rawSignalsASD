import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import data_utils

def plot_resampled_comparison(original_data, resampled_data, subject_id, session_id, start_seconds, interval_seconds):

    eda_data = original_data[subject_id][session_id]['EDA']
    bvp_data = original_data[subject_id][session_id]['BVP']
    acc_data = original_data[subject_id][session_id]['ACC']

    start_time = eda_data['Timestamp'].iloc[0] + (start_seconds * 1000)  # Convertimos segundos a ms
    end_time = start_time + (interval_seconds * 1000)
    print(f"Intervalo de tiempo seleccionado: {start_time} - {end_time}")

    eda_filtered = eda_data[(eda_data['Timestamp'] >= start_time) & (eda_data['Timestamp'] < end_time)]
    bvp_filtered = bvp_data[(bvp_data['Timestamp'] >= start_time) & (bvp_data['Timestamp'] < end_time)]
    acc_filtered = acc_data[(acc_data['Timestamp'] >= start_time) & (acc_data['Timestamp'] < end_time)]

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

    fig, axs = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    fig.suptitle(f'Subject {subject_id}, Session {session_id} - Interval: {interval_seconds} seconds', fontsize=16)

    styles = {
        'EDA': {'color': 'blue', 'marker': 'o', 'label': 'EDA'},
        'BVP': {'color': 'green', 'marker': 's', 'label': 'BVP'},
        'ACC_X': {'color': 'purple', 'marker': '^', 'label': 'ACC_X'},
        'ACC_Y': {'color': 'orange', 'marker': 'v', 'label': 'ACC_Y'},
        'ACC_Z': {'color': 'cyan', 'marker': 'D', 'label': 'ACC_Z'}
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

    # ACC_X, ACC_Y y ACC_Z juntas en una sola gráfica
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

    for ax in axs[-1, :]:
        ax.set_xlabel('Timestamp (ms)')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()


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

    # Concatenar todos los datos en un solo DataFrame final sin índices adicionales
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


def test_resample(orig_data_path='./dataset', resample_data_path='./dataset_resampled/', hz_str='32Hz'):
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
    test_resample()

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

