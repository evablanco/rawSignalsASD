import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import data_utils

def analyse_PDM_results_v4(path_results, model_code, feats_code, tf, tp, bin_size, split_code):
    results_path = f"{path_results}/PDM_mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_sc{split_code}_all_experiments_results.csv"
    results_df = pd.read_csv(results_path, dtype={'Fold': str})
    # last two rows: mean and standard deviation
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
    # counts the number of aggressive episodes and their average duration based on changes
    # from 0 to 1 in the Condition column. An episode is also counted if the series starts with 1.
    def compute_episode_count_and_duration(condition_series, time_series):
        condition_series = condition_series.reset_index(drop=True)
        time_series = pd.Series(time_series).reset_index(drop=True)
        shifted = condition_series.shift(fill_value=0)
        episode_starts = (condition_series == 1) & (shifted == 0)
        episode_durations = []
        for start_idx in episode_starts[episode_starts].index:
            end_idx = start_idx
            while end_idx < len(condition_series) and condition_series[end_idx] == 1:
                end_idx += 1
            duration = (time_series[end_idx - 1] - time_series[start_idx]).total_seconds()
            episode_durations.append(duration)
        return len(episode_durations), np.sum(episode_durations) if episode_durations else 0

    for subject_id in subject_ids:
        subject_data = data_dict.get(subject_id, {})
        num_episodes = 0
        num_sessions = len(subject_data)
        episodes_per_session = []
        durations = []
        total_duration = 0
        total_session_duration = 0
        for session_id, session_data in subject_data.items():
            session_data = session_data.sort_index()
            num_episodes_in_session, total_duration_in_session = compute_episode_count_and_duration(
                session_data['Condition'], session_data.index
            )
            num_episodes += num_episodes_in_session
            episodes_per_session.append(num_episodes_in_session)
            durations.append(total_duration_in_session)
            total_duration += total_duration_in_session
            session_duration = (session_data.index[-1] - session_data.index[0]).total_seconds()
            total_session_duration += session_duration
        total_episodes.append(num_episodes)
        total_sessions.append(num_sessions)
        mean_episodes_per_session.append(np.mean(episodes_per_session) if episodes_per_session else 0)
        mean_episode_duration.append(np.mean(durations) if durations else 0)
        total_duration_episodes.append(total_duration)
        total_duration_sessions.append(total_session_duration)

    analysis_df = pd.DataFrame({
        'Subject': subject_ids,
        'AUC-ROC': auc_values,
        'Total Episodes': total_episodes,
        'Total Sessions': total_sessions,
        'Mean Episodes per Session': np.round(mean_episodes_per_session, 2),
        'Mean Episode Duration (min.)': np.round(np.array(mean_episode_duration) / 60, 2),
        'Total Duration Episodes (min.)': np.round(np.array(total_duration_episodes) / 60, 2),
        'Total Duration Sessions (min.)': np.round(np.array(total_duration_sessions) / 60, 2)
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
        ("AUC vs Mean Episode Duration", 'Mean Episode Duration (min.)', axs[2, 1])
    ]

    for i, (title, column, ax) in enumerate(plot_data):
        for subject_id in unique_subjects:
            subject_data = analysis_df[analysis_df['Subject'] == subject_id]
            ax.scatter(subject_data[column], subject_data['AUC-ROC'],
                       color=subject_colors[subject_id], marker='+',  s=100, linewidth=1.5)

        ax.set_xlabel(column, fontsize=14)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

        if i % 2 == 0:
            ax.set_ylabel("AUC-ROC", fontsize=14)
        else:
            ax.set_ylabel("")

        ax.axhline(0.5, color='black', linestyle='dashed', linewidth=1.2)  # AUC=0.5 reference line
        ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path_to_save_results = f"./results_analysis/PDM_mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_sc{split_code}_data_vs_results_5cv.png"
    plt.savefig(path_to_save_results)
    plt.show()


def analyse_PDM_results_v3(path_results, model_code, feats_code, tf, tp, bin_size, split_code):
    results_path = f"{path_results}/PDM_mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_sc{split_code}_all_experiments_results.csv"
    results_df = pd.read_csv(results_path, dtype={'Fold': str})
    # last two rows: mean and standard deviation
    mean_row = results_df.iloc[-2]
    std_row = results_df.iloc[-1]
    mean_auc_roc = mean_row['AUC-ROC']
    std_auc_roc = std_row['AUC-ROC']
    mean_f1_score = mean_row['F1-Score']
    results_df = results_df.iloc[:-2]

    freq = 32
    data_path_resampled = './dataset_resampled/'
    ds_path = data_path_resampled + "dataset_" + str(freq) + "Hz.csv"
    data_dict = data_utils.load_data_to_dict(ds_path)

    subject_ids = results_df['Fold'].astype(str)
    auc_values = results_df['AUC-ROC']
    f1_values = results_df['Best_F1-score']

    total_episodes = []
    total_sessions = []
    mean_episodes_per_session = []
    mean_episode_duration = []
    total_duration_episodes = []
    total_duration_sessions = []

    def compute_episode_count_and_duration(condition_series, time_series):
        condition_series = condition_series.reset_index(drop=True)
        time_series = pd.Series(time_series).reset_index(drop=True)
        shifted = condition_series.shift(fill_value=0)
        episode_starts = (condition_series == 1) & (shifted == 0)
        episode_durations = []
        for start_idx in episode_starts[episode_starts].index:
            end_idx = start_idx
            while end_idx < len(condition_series) and condition_series[end_idx] == 1:
                end_idx += 1
            duration = (time_series[end_idx - 1] - time_series[start_idx]).total_seconds()
            episode_durations.append(duration)
        return len(episode_durations), np.sum(episode_durations) if episode_durations else 0

    for subject_id in subject_ids:
        subject_data = data_dict.get(subject_id, {})
        num_episodes = 0
        num_sessions = len(subject_data)
        episodes_per_session = []
        durations = []
        total_duration = 0
        total_session_duration = 0
        for session_id, session_data in subject_data.items():
            session_data = session_data.sort_index()
            num_episodes_in_session, total_duration_in_session = compute_episode_count_and_duration(
                session_data['Condition'], session_data.index
            )
            num_episodes += num_episodes_in_session
            episodes_per_session.append(num_episodes_in_session)
            durations.append(total_duration_in_session)
            total_duration += total_duration_in_session
            session_duration = (session_data.index[-1] - session_data.index[0]).total_seconds()
            total_session_duration += session_duration

        total_episodes.append(num_episodes)
        total_sessions.append(num_sessions)
        mean_episodes_per_session.append(np.mean(episodes_per_session) if episodes_per_session else 0)
        mean_episode_duration.append(np.mean(durations) if durations else 0)
        total_duration_episodes.append(total_duration)
        total_duration_sessions.append(total_session_duration)

    analysis_df = pd.DataFrame({
        'Subject': subject_ids,
        'AUC-ROC': np.round(auc_values, 2),
        'F1-Score': np.round(f1_values, 2),
        'Total Episodes': total_episodes,
        'Total Sessions': total_sessions,
        'Mean Episodes per Session': np.round(mean_episodes_per_session, 2),
        'Mean Episode Duration (min.)': np.round(np.array(mean_episode_duration) / 60, 2),
        'Total Duration Episodes (min.)': np.round(np.array(total_duration_episodes) / 60, 2),
        'Total Duration Sessions (min.)': np.round(np.array(total_duration_sessions) / 60, 2)
    })

    mean_values = analysis_df.iloc[:, 1:].mean().round(2)
    std_values = analysis_df.iloc[:, 1:].std().round(2)
    summary_df = pd.DataFrame([mean_values, std_values])
    summary_df.insert(0, 'Subject', ['Mean', 'STD'])
    analysis_df = pd.concat([analysis_df, summary_df], ignore_index=True)
    output_path_csv = f"./results_analysis/PDM_mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_sc{split_code}_data_subjects_analysis.csv"
    analysis_df.to_csv(output_path_csv, index=False)
    print(f"DataFrame guardado en: {output_path_csv}")

    corr_vars = ['Total Sessions', 'Total Duration Sessions (min.)', 'Total Episodes',
                 'Total Duration Episodes (min.)', 'Mean Episode Duration (min.)', 'AUC-ROC']
    correlation_matrix = analysis_df[corr_vars].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="Blues", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix")

    #### v2:
    sns.heatmap(correlation_matrix, annot=True, cmap="Blues", fmt=".2f",
                linewidths=1, linecolor='white', square=True, cbar_kws={'shrink': 0.8})
    plt.title("Correlation Matrix", fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    path_to_save_corr = f"./results_analysis/PDM_mv{model_code}_f{feats_code}_tf{tf}_tp{tp}_bs{bin_size}_sc{split_code}_correlation_data_vs_results_5cv.png"
    plt.savefig(path_to_save_corr)
    print(f"Matriz guardada en: {path_to_save_corr}")
    plt.show()

    unique_subjects = analysis_df['Subject'].unique()
    color_map = plt.get_cmap('tab10')
    subject_colors = {subject: color_map(i % 10) for i, subject in enumerate(unique_subjects)}

    fig, axs = plt.subplots(3, 3, figsize=(18, 18))
    fig.suptitle(f'Performance Analysis - Mean AUC-ROC: {mean_auc_roc:.3f}, Std.: {std_auc_roc:.3f}.', fontsize=14, fontweight='bold')
    plot_data = [

        ("AUC vs Total Sessions", 'Total Sessions', axs[0, 0]),
        ("AUC vs Total Duration Sessions", 'Total Duration Sessions (min.)', axs[0, 1]),
        ("AUC vs Mean Episodes per Session", 'Mean Episodes per Session', axs[0, 2]),
        ("AUC vs Total Episodes", 'Total Episodes', axs[1, 0]),
        ("AUC vs Total Duration Episodes", 'Total Duration Episodes (min.)', axs[1, 1]),
        ("AUC vs Mean Episode Duration", 'Mean Episode Duration (min.)', axs[1, 2])
        #("AUC vs F1-Score", 'F1-Score', axs[1, 2])
    ]
    for title, column, ax in plot_data:
        for subject_id in unique_subjects:
            subject_data = analysis_df[analysis_df['Subject'] == subject_id]
            ax.scatter(subject_data[column], subject_data['AUC-ROC'], color=subject_colors[subject_id], marker='+', linewidth=1)
        ax.set_title(title)
        ax.set_xlabel(column)
        ax.set_ylabel("AUC-ROC")
        ax.axhline(0.5, color='black', linestyle='dashed', linewidth=1.2)
        ax.grid(True)

    for i in range(3):
        fig.delaxes(axs[2, i])

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
