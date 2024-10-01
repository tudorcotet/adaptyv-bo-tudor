import mlflow
from mlflow.tracking import MlflowClient
from config.optimization import MLflowConfig
from typing import Dict, Any
from contextlib import contextmanager
import pandas as pd

class MLflowTracker:
    def __init__(self, config: MLflowConfig):
        self.mlflow_config = config
        mlflow.set_tracking_uri(self.mlflow_config.tracking_uri)
        mlflow.set_experiment(self.mlflow_config.experiment_name)
        self.client = MlflowClient()
        experiment_name = self.mlflow_config.experiment_name
        current_experiment=dict(mlflow.get_experiment_by_name(experiment_name))
        self.experiment_id=current_experiment['experiment_id']
        self.parent_run_id = None

    @contextmanager
    def start_parent_run(self, run_name: str, description: str):
        with mlflow.start_run(run_name=run_name, experiment_id=self.experiment_id, description=description) as parent_run:
            self.parent_run_id = parent_run.info.run_id
            yield parent_run

    @contextmanager
    def start_child_run(self, child_run_name: str, parent_run_id: str):
        with mlflow.start_run(run_name=child_run_name, nested=True, parent_run_id=parent_run_id, experiment_id=self.experiment_id) as child_run:
            yield child_run

    def log_params(self, params: Dict[str, Any], run_id: str):
        mlflow.log_params(params, run_id=run_id)

    def log_metric(self, key: str, value: float, run_id: str, step: int = None):
        mlflow.log_metric(key, value, run_id=run_id, step=step)

    def log_artifact(self, local_path: str, run_id: str):
        mlflow.log_artifact(local_path, run_id=run_id)

    def log_figure(self, figure, filename: str, run_id: str):
        mlflow.log_figure(figure, filename, run_id=run_id)

    def log_model_summary(self, model_name: str, model_summary: str, run_id: str):
        mlflow.log_text(model_summary, f"models/{model_name}_summary.txt", run_id=run_id)

    def log_model_graph(self, model_name: str, model_graph: str, run_id: str):
        mlflow.log_text(model_graph, f"models/{model_name}_graph.txt", run_id=run_id)

    def log_model_weights(self, model_name: str, model_weights: Dict[str, Any], run_id: str):
        mlflow.log_dict(model_weights, f"models/{model_name}_weights.json", run_id=run_id)
    
    def end_run(self):
        mlflow.end_run()
    
    def log_dataset(self, dataset: pd.DataFrame, context: str):
        mlflow.log_input(dataset, context = context)
    
    def set_tags(self, tags: Dict[str, Any]):
        mlflow.set_tags(tags)
    