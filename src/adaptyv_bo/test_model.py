# test_model.py
from cnn_code import BLIModel
import torch

def test_model_initialization():
    num_scalar_features = 16  # Replace with actual number
    num_classes = 4  # Replace with actual number
    class_weights = torch.tensor([0.5, 1.0, 1.5, 2.0])  # Example weights
    try:
        model = BLIModel(num_scalar_features, num_classes, class_weights=class_weights)
        print("Model initialized successfully.")
    except Exception as e:
        print(f"Model initialization failed: {e}")

if __name__ == "__main__":
    test_model_initialization()
