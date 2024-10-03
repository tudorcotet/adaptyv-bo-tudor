import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import os
import matplotlib.pyplot as plt
import tempfile
from utils.trackers.mlflow import MLflowTracker
from config.optimization import MLflowConfig

import numpy as np
import seaborn as sns


def fetch_csvs_from_mlflow(experiment_name):
    config = MLflowConfig()
    tracker = MLflowTracker(config)
    experiment_id = tracker.experiment_id
    client = MlflowClient()
    
    runs = client.search_runs(experiment_ids=[experiment_id])
    
    dfs = []
    for run in runs:
        artifacts = client.list_artifacts(run.info.run_id)
        csv_artifacts = [artifact for artifact in artifacts if artifact.path.endswith('.csv')]

        for csv_artifact in csv_artifacts:
            with tempfile.TemporaryDirectory() as temp_dir:
                local_path = client.download_artifacts(run.info.run_id, csv_artifact.path, temp_dir)
                df = pd.read_csv(local_path)
                df['run_id'] = run.info.run_id
                dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)

def process_csvs(experiment_name):
    combined_df = fetch_csvs_from_mlflow(experiment_name)
    
    # Save the combined dataframe to a CSV file
    output_dir = './output_benchmark_csv'
    os.makedirs(output_dir, exist_ok=True)
    combined_df.to_csv(os.path.join(output_dir, 'combined_metrics.csv'), index=False)
    print(f"Combined metrics saved to {os.path.join(output_dir, 'combined_metrics.csv')}")
    
    return combined_df

def get_cumulative_max_per_round(df_per_combination):
    all_seeds = np.unique(df_per_combination['Seed'].tolist())
    per_seed = []
    
    for seed in all_seeds:
        current_df = df_per_combination[df_per_combination['Seed'] == seed]
        rounds = np.arange(current_df['Round'].max() + 1)
        max_fitness = -np.inf
        max_fitness_per_round = []

        for round_no in rounds:
            round_fitness = current_df[current_df['Round'] == round_no]['Fitness'].max()
            max_fitness = max(max_fitness, round_fitness)
            max_fitness_per_round.append(max_fitness)
        
        df = pd.DataFrame({'round': rounds, 'fitness': max_fitness_per_round, 'combination': [df_per_combination['model_combination'].iloc[0]] * len(rounds), 'seed': [seed] * len(rounds)})
        
        per_seed.append(df)

    if len(per_seed) > 0:
        df_final = pd.concat(per_seed)
        return df_final
    else:
        return None

def run_artifact_processing_pipeline(experiment_name):
    combined_df = process_csvs(experiment_name)
    combined_df = pd.read_csv('./output_benchmark_csv/combined_metrics.csv', low_memory = False)
    combined_df = aggregate_by_model_combination(combined_df)
    unique_model_comb = list(set(combined_df['model_combination'].tolist()))
    dfs = [get_cumulative_max_per_round(combined_df[combined_df.model_combination == comb]) for comb in unique_model_comb]
    aggregated_df = pd.concat(dfs).reset_index()

    output_dir = './output_benchmark_csv'
    aggregated_df.to_csv(os.path.join(output_dir, 'aggregated_df.csv'), index = False)

    plot_optimization_curves(aggregated_df, output_dir)

    efficiency_metrics = analyze_model_efficiency(aggregated_df)
    efficiency_metrics.to_csv(os.path.join(output_dir, 'model_efficiency_metrics.csv'), index=False)
    print(f"Model efficiency metrics saved to {os.path.join(output_dir, 'model_efficiency_metrics.csv')}")
    
    # Plot top 5 most efficient models
    top_5_models = efficiency_metrics.head(5).index.tolist()
    plot_optimization_curves(aggregated_df[aggregated_df['combination'].isin(top_5_models)], output_dir, filename='top_5_models_optimization_curves.png')
    #print(f"Top 5 models optimization curves plot saved to {os.path.join(output_dir, 'top_5_models_optimization_curves.png')}")
  
def aggregate_by_model_combination(df):

    df.loc[df['Loss_Function'].isna(), 'Loss_Function'] = 'None'
    df.loc[df['Kernel'].isna(), 'Loss_Function'] = 'None'

    df['model_combination'] = df['Surrogate'] + '_' + df['Acquisition'] + '_' + df['Loss_Function'] + '_' + df['Kernel']
    return df

def plot_optimization_curves(aggregated_df, output_dir, filename='optimization_curves.png'):
    plt.figure(figsize=(15, 10))
    sns.lineplot(aggregated_df, x = 'round', y = 'fitness', errorbar = 'sd', hue = 'combination')
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Optimization curves plot saved to {os.path.join(output_dir, filename)}")

def analyze_model_efficiency(aggregated_df):
    # Group by combination and get the last fitness value for each group
    final_performance = aggregated_df.groupby('combination')['fitness'].last()
    
    start_performance = aggregated_df.groupby('combination')['fitness'].first()

    # Calculate the round where fitness reaches 90% of its maximum for each combination
    convergence_speed = aggregated_df.groupby('combination').apply(
        lambda group: group[group['fitness'] >= .9 * group['fitness'].max()]['round'].min()
    )
    
    efficiency_metrics = pd.DataFrame({
        'final_fitness': final_performance,
        'rounds_to_90_percent': convergence_speed,
        'starting_fitness': start_performance
    })
    efficiency_metrics['efficiency_score'] = efficiency_metrics['final_fitness'] / efficiency_metrics['rounds_to_90_percent']
    efficiency_metrics['fold_change'] = efficiency_metrics['final_fitness'] / efficiency_metrics['starting_fitness']
    
    return efficiency_metrics.sort_values('fold_change', ascending=False)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process MLflow artifacts and generate analysis")
    args = parser.parse_args()
    experiment_name = 'https://mlflow.internal.adaptyvbio.com/2024/02_adaptyv_bo_tudor/001_benchmark_test'
    run_artifact_processing_pipeline('https://mlflow.internal.adaptyvbio.com/2024/02_adaptyv_bo_tudor/001_benchmark_test')
