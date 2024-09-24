import mlflow
from typing import Dict, Any
from utils.trackers.base import BaseTracker
from surrogates.base import BaseSurrogate
from config.optimization import MLflowConfig

class MLflowTracker(BaseTracker):
    def __init__(self, config: MLflowConfig):
        self.mlflow_config = config
        mlflow.set_tracking_uri(self.mlflow_config.tracking_uri)
        mlflow.set_experiment(self.mlflow_config.experiment_name)
        self.run = None

    def start_run(self, run_name: str):
        self.run = mlflow.start_run(run_name=run_name)

    def end_run(self):
        if self.run:
            mlflow.end_run()
            self.run = None

    def __enter__(self):
        self.start_run(f"seed_{self.mlflow_config.seed}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_run()

    def log_params(self, params: Dict[str, Any]):
        mlflow.log_params(params)

    def log_metric(self, key: str, value: float, step: int = None):
        mlflow.log_metric(key, value, step=step)

    def log_artifact(self, local_path: str):
        mlflow.log_artifact(local_path)

    def log_figure(self, figure, filename: str):
        mlflow.log_figure(figure, filename)

    def log_model_summary(self, model_name: str, model_summary: str):
        mlflow.log_text(model_summary, f"models/{model_name}_summary.txt")

    def log_model_graph(self, model_name: str, model_graph: str):
        mlflow.log_text(model_graph, f"models/{model_name}_graph.txt")

    def log_model_weights(self, model_name: str, model_weights: Dict[str, Any]):
        mlflow.log_dict(model_weights, f"models/{model_name}_weights.json")

    def log_training_loss(self, loss: float, step: int):
        self.log_metric("training_loss", loss, step=step)

    def log_validation_loss(self, loss: float, step: int):
        self.log_metric("validation_loss", loss, step=step)

    def log_training_metrics(self, metrics: Dict[str, float], step: int):
        for key, value in metrics.items():
            self.log_metric(f"training_{key}", value, step=step)

    def log_validation_metrics(self, metrics: Dict[str, float], step: int):
        for key, value in metrics.items():
            self.log_metric(f"validation_{key}", value, step=step)

    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        self.log_params(hyperparameters)

    def log_dataset(self, dataset: Dict[str, Any]):
        mlflow.log_dict(dataset, "dataset.json")

   


