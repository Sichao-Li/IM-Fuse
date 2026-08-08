import torch


def regression_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict[str, float]:
    y_true = y_true.detach().float().reshape(-1)
    y_pred = y_pred.detach().float().reshape(-1)
    diff = y_pred - y_true
    mse = torch.mean(diff**2).item()
    rmse = float(mse**0.5)
    mae = torch.mean(torch.abs(diff)).item()
    total = torch.sum((y_true - torch.mean(y_true)) ** 2)
    residual = torch.sum(diff**2)
    r2 = (1.0 - residual / total).item() if total.item() > 0 else 0.0
    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}
