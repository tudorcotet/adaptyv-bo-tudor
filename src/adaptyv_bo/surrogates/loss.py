import torch
import torch.nn.functional as F

            
def generate_comparisons(y, noise = .0):
    y_t = y.t()
    y_t = y_t + noise * torch.randn_like(y_t)

    comparison_matrix = (y > y_t).int().float()  # Shape (batch, batch)

    return comparison_matrix

def bt_model(y_hat, beta = 1.):
    """
    Computes the comparison matrix using the Bradley-Terry model.

    Args:
        y_hat (torch.Tensor): The predicted values for each sample.
        beta (float, optional): The scaling factor for the difference between predicted values. Defaults to 1.

    Returns:
        torch.Tensor: The comparison matrix computed using the Bradley-Terry model.
    """
    comparison_matrix = F.sigmoid(beta * (y_hat - y_hat.t())) #BT model = sigmoid(beta(y_i - y_j))
    return comparison_matrix

def bt_loss(y_hat, y, beta=1., noise=0.):
    """
    Calculates the Bradley-Terry loss given predicted scores and true labels.

    Args:
        y_hat (torch.Tensor): Predicted scores of shape (batch_size, num_classes).
        y (torch.Tensor): True labels of shape (batch_size).
        beta (float, optional): Scaling factor for the Bradley-Terry model. Defaults to 1.
        noise (float, optional): Amount of noise to add to the true labels. Defaults to 0.

    Returns:
        torch.Tensor: Calculated loss value.

    """
    y = generate_comparisons(y, noise=noise)  # Generate comparison given a set of scores of shape (batch)
    y_hat = bt_model(y_hat, beta=beta)  # BT model = sigmoid(beta(y_i - y_j))

    loss = F.cross_entropy(y_hat, y)
    return loss

class BaseLoss(torch.nn.Module):
    def __init__(self, args):
        super(BaseLoss, self).__init__()
        self.args = args
        self.configure_loss()

    def forward(self, y_hat, y):
        return self.loss(y_hat, y)

    def configure_loss(self):
        if self.args.loss_fn == "mse":
            self.loss = torch.nn.MSELoss()
        elif self.args.loss_fn == "mae":
            self.loss = torch.nn.L1Loss()
        elif self.args.loss_fn == "bt":
            self.loss = bt_loss
        elif self.args.loss_fn == "cross_entropy":
            self.loss = F.cross_entropy
        else:
            raise ValueError(f"Loss function {self.args.loss_fn} not supported")