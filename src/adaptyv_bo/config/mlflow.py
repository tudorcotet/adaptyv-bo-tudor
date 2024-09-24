from dataclasses import dataclass

@dataclass
class MLflowConfig:
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "testing"
    log_params: bool = True
    log_model_summary: bool = True
    log_model_graph: bool = True
    log_model_weights: bool = True
    log_training_loss: bool = True
    log_validation_loss: bool = True
    log_training_metrics: bool = True
    log_validation_metrics: bool = True 
    log_figures: bool = True
    log_dataset: bool = True
