import pandas as pd
import numpy as np
import data_utils
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import math


def compute_metrics(data_dict):
    results = []

    for subject_id, sessions in data_dict.items():
        total_sessions = len(sessions)
        total_episodes = 0
        total_episode_duration = 0
        total_session_duration = 0
        episodes_per_session = []
        durations_per_session = []

        for session_id, session_data in sessions.items():
            session_data = session_data.sort_index()
            # Compute session duration
            session_duration = (session_data.index[-1] - session_data.index[0]).total_seconds() / 60
            total_session_duration += session_duration
            # Identify episodes based on changes from 0 to 1 in Condition column
            condition = session_data['Condition']
            shifted = condition.shift(fill_value=0) # shift values to detect changes
            episode_starts = (condition == 1) & (shifted == 0) # identify episode start points
            # Store individual episode durations
            episode_durations = []
            for start_time in episode_starts[episode_starts].index:
                # Filter the DataFrame from episode onset
                episode_data = session_data.loc[start_time:]
                # Find the first instance where Condition returns to 0 (end of episode)
                end_times = episode_data[episode_data['Condition'] == 0].index
                # Set episode end time to first 0 found or last timestamp if no 0 is found
                if len(end_times) > 0:
                    end_time = end_times[0]
                else:
                    end_time = episode_data.index[-1]
                # Determine end time of the episode
                episode_duration = (end_time - start_time).total_seconds() / 60
                if episode_duration > 0:
                    episode_durations.append(episode_duration)

            num_episodes = len(episode_durations)
            total_episodes += num_episodes
            total_episode_duration += sum(episode_durations)
            episodes_per_session.append(num_episodes)
            durations_per_session.append(sum(episode_durations))

        mean_episodes_per_session = np.mean(episodes_per_session) if episodes_per_session else 0
        mean_episode_duration = total_episode_duration / total_episodes if total_episodes > 0 else 0

        results.append({
            'Subject': subject_id,
            'Total_Sessions': total_sessions,
            'Total_Episodes': total_episodes,
            'Mean Episodes/Session': mean_episodes_per_session,
            'Mean Duration Episodes': mean_episode_duration,
            'Total Duration Episodes': total_episode_duration,
            'Total Duration Sessions': total_session_duration
        })

    return pd.DataFrame(results)


def merge_with_pdm_results(metrics_df, pdm_results_path):
    pdm_results = pd.read_csv(pdm_results_path)
    # Remove the last two rows (mean and std rows) if present
    pdm_results = pdm_results.iloc[:-2]
    pdm_results['Subject'] = pdm_results['Subject'].astype(str).str.strip()
    metrics_df['Subject'] = metrics_df['Subject'].astype(str).str.strip()
    # Perform the merge, keeping only the subjects that exist in both datasets
    merged_df = pd.merge(metrics_df, pdm_results, left_on='Subject', right_on='Subject', how='inner')
    merged_df.reset_index(drop=True, inplace=True)
    return merged_df


def plot_episodes_vs_auc_scatter(results_pdf_path):
    df = pd.read_csv(results_pdf_path)
    fig = px.scatter(
        df,
        x= 'Mean Duration Episodes (minutes)', #'Total_Episodes',
        y='AUC-ROC',
        text='Subject',
        title='Distribution of Aggressive Episodes vs AUC-ROC',
        labels={'Total_Episodes': 'Total Episodes', 'AUC-ROC': 'AUC-ROC'},
        hover_data=['Total_Sessions', 'Mean Episodes/Session', 'Total_Episodes'] #, Mean Duration Episodes (minutes)
    )
    fig.update_traces(marker=dict(size=10, opacity=0.7))
    fig.update_layout(showlegend=False)

    fig.show()


def plot_correlation_matrix(df, included_vars):
    df_filtered = df[included_vars]
    df_filtered = df_filtered.rename(columns={"Best_F1-score": "F1-Score"})
    correlation_matrix = df_filtered.corr()
    plt.figure(figsize=(10, 7))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap=sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True), linewidths=0.5)
    plt.title("Correlation Matrix")
    plt.show()
    return correlation_matrix["AUC-ROC"].sort_values(ascending=False)


def plot_scatter_plots(df, variables):
    plt.figure(figsize=(15, 10))
    for i, var in enumerate(variables, 1):
        plt.subplot(2, 3, i)
        sns.scatterplot(x=df[var], y=df["AUC-ROC"])
        plt.xlabel(var)
        plt.ylabel("AUC-ROC")
        plt.title(f"AUC-ROC vs {var}")

    plt.tight_layout()
    plt.show()


def plot_scatter_plots_v2(df, variables):
    num_vars = len(variables)
    cols = 3
    rows = math.ceil(num_vars / cols)
    plt.figure(figsize=(5 * cols, 5 * rows))
    for i, var in enumerate(variables, 1):
        plt.subplot(rows, cols, i)
        sns.scatterplot(x=df[var], y=df["AUC-ROC"], marker='x', hue=df["Subject"], palette="tab10", legend=False)  # Cruces naranjas
        plt.xlabel(var)
        plt.ylabel("AUC-ROC")
        plt.title(f"AUC-ROC vs {var}")
    plt.tight_layout()
    plt.show()


def plot_histograms(df, variables):
    plt.figure(figsize=(15, 10))
    for i, var in enumerate(variables, 1):
        plt.subplot(2, 3, i)
        sns.histplot(df[var], kde=True, bins=20)
        plt.xlabel(var)
        plt.ylabel("Frecuencia")
        plt.title(f"Distribución de {var}")

    plt.tight_layout()
    plt.show()



def plot_episodes_vs_auc_hist(df):
    fig = px.histogram(
        df,
        x= 'Mean Duration Episodes (minutes)', #'Total_Episodes',
        y='AUC-ROC',
        color='Subject',
        nbins=20,
        marginal='box',
        title='Distribution of Aggressive Episodes vs AUC-ROC',
        labels={'Total_Episodes': 'Total Episodes', 'AUC-ROC': 'AUC-ROC'},
        hover_data=['Total_Sessions', 'Mean Episodes/Session', 'Total_Episodes'] #, Mean Duration Episodes (minutes)
    )
    fig.update_layout(title="Distribution in subjects by AUC reached.",
                      xaxis_title="Count",
                      yaxis_title="No of Reviews")
    #fig.update_traces(marker=dict(size=10, opacity=0.7))
    fig.update_layout(showlegend=False)

    fig.show()




'''

# Example usage

### get subject analysis
freq = 32
data_path_resampled = './dataset_resampled/'
ds_path = data_path_resampled + "dataset_" + str(freq) + "Hz.csv"
data_dict = data_utils.load_data_to_dict(ds_path)
metrics_df = compute_metrics(data_dict)
print(metrics_df)
metrics_df.to_csv('subjects_data_danalysis.csv')


### merge subject analysis data with PDM results
metrics_df = pd.read_csv('subjects_data_danalysis.csv')
# Load the CSV results file
results_path = './results/PDM_SS_model2_tf180_tp180_all_experiments_results.csv'
#results_df = pd.read_csv(results_path)
merged_df = merge_with_pdm_results(metrics_df, results_path)
print(merged_df)
merged_df.to_csv('PDM_data_vs_results.csv')


results_path = 'PDM_data_vs_results.csv'
df = pd.read_csv(results_path)
df = df.rename(columns={"Best_F1-score": "F1-score"})
#plot_episodes_vs_auc_hist(df)

variables_interes = ["Total_Sessions", "Total_Episodes", "Total Duration Episodes",
                     "Mean Duration Episodes", "Total Duration Sessions",  "F1-score", "AUC-ROC"]
correlation_auc = plot_correlation_matrix(df, variables_interes)
variables_interes = ["Total Duration Episodes", "Total_Sessions",
                     "Mean Duration Episodes", "Total Duration Sessions", "Total_Episodes", "F1-score"]
# Generar diagramas de dispersión y distribuciones
plot_scatter_plots_v2(df, variables_interes)
variables_interes = ["Total Duration Episodes", "Total_Sessions",
                     "Mean Duration Episodes", "Total Duration Sessions", "Total_Episodes", "AUC-ROC"]
plot_histograms(df, variables_interes)

'''
















