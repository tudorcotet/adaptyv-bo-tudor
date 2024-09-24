import mlflow
from typing import Dict, Any
from utils.trackers.base import BaseTracker
from models.base import BaseSurrogate
from config.mlflow import MLflowConfig

class MLflowTracker(BaseTracker):
    def __init__(self, loop: BayesianOptimizationLoop, model: BaseSurrogate, config: MLflowConfig):
        self.model = model
        self.loop = loop
        self.mlflow_config = config
        mlflow.set_tracking_uri(self.mlflow_config.tracking_uri)
        mlflow.set_experiment(self.mlflow_config.experiment_name)
        self.run = None

    def start_run(self, run_name: str):
        self.run = mlflow.start_run(run_name=run_name)
        self.log(self.model, self.mlflow_config)

    def end_run(self):
        if self.run:
            mlflow.end_run()
            self.run = None
    
    def __enter__(self):
        self.start_run(f"seed_{self.config.seed}")
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

    def log(self, model: BaseSurrogate, config: OptimizationConfig):
        if self.mlflow_config.log_params:
            self.log_params(config.to_dict())
        if self.mlflow_config.log_model_summary:
            self.log_model_summary(model.__class__.__name__, str(model))
        if self.mlflow_config.log_model_graph:
            self.log_model_graph(model.__class__.__name__, str(model))
        if self.mlflow_config.log_model_weights:
            self.log_model_weights(model.__class__.__name__, model.state_dict())
        if self.mlflow_config.log_training_loss:
            self.log_training_loss(model.training_loss, step=0)
        if self.mlflow_config.log_validation_loss:
            self.log_validation_loss(model.validation_loss, step=0)
        if self.mlflow_config.log_training_metrics:
            self.log_training_metrics(model.training_metrics, step=0)
        if self.mlflow_config.log_validation_metrics:
            self.log_validation_metrics(model.validation_metrics, step=0)
        if self.mlflow_config.log_figures:
            self.log_figure(self.loop.plotter.plot_embeddings(), "embeddings.png")
            self.log_figure(self.loop.plotter.plot_max_fitness(), "max_fitness.png")
            self.log_figure(self.loop.plotter.plot_training_loss(), "training_loss.png")
            self.log_figure(self.loop.plotter.plot_validation_loss(), "validation_loss.png")
            self.log_figure(self.loop.plotter.plot_training_metrics(), "training_metrics.png")
            self.log_figure(self.loop.plotter.plot_validation_metrics(), "validation_metrics.png")




