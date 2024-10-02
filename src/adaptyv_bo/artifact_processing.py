import mlflow
import pandas as pd
import os
import matplotlib.pyplot as plt
from mlflow import MlflowClient

def process_csvs(output_dir = './output_benchmark_configs'):
    all_runs = os.listdir(output_dir)
    all_runs_paths = [os.path.join(output_dir, i) for i in all_runs]
    all_runs_paths = [path for path in all_runs_paths if not path.endswith('.DS_Store')]

    dfs = []
    for path in all_runs_paths:
        csv_path = os.path.join(path, 'combined', 'csv')
        if os.path.exists(csv_path):
            csv_path = os.path.join(csv_path, 'combined_results.csv')
        else:
            next
        dfs.append(pd.read_csv(csv_path))

    # Concatenate all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)

    # Save the combined dataframe to a CSV file
    combined_df.to_csv(os.path.join(output_dir, 'combined_metrics.csv'), index=False)

    print(f"Combined metrics saved to {os.path.join(output_dir, 'combined_metrics.csv')}")

def plot_fitness_over_iterations(output_dir = './output_benchmark_configs'):
    # Load the combined metrics dataframe
    df = pd.read_csv(os.path.join(output_dir, 'combined_metrics.csv'))

    # Plot the fitness over iterations
    plt.figure(figsize=(12, 6))
    metric = 'fitness'
    plt.plot(df['iteration'] if 'iteration' in df.columns else df.index, df[metric], label=metric)
  
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.title('Fitness over Iterations')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'fitness_over_iterations.png'))
    plt.close()  # Close the plot to free up memory

    print(f"Fitness plot saved to {os.path.join(output_dir, 'fitness_over_iterations.png')}")

if __name__ == "__main__":
    process_artifacts("2024/02_adaptyv_bo_tudor", "artifacts")  


