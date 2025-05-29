import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import data_utils

def analyse_PDM_results_v4(path_results, model_code, feats_code, tf, tp, bin_size, split_code):
    results_path = f"{path_results}/PDM_mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_sc{split_code}_all_experiments_results.csv"
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

    output_path_csv = f"./results_analysis/PDM_mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_sc{split_code}_data_subjects_analysis.csv"
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
    path_to_save_results = f"./results_analysis/PDM_mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_sc{split_code}_data_vs_results_5cv.png"
    plt.savefig(path_to_save_results)
    plt.show()

'''
# Usage example
path_results = './results/'
model_code = 2
feats_code = 0
tf, tp = 180, 180
bin_size = 15
split_code = 0
analyse_PDM_results_v4(path_results, model_code, feats_code, tf, tp, bin_size, split_code)
'''

