from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from battery_fusion.training.metrics import regression_metrics
from battery_fusion.training.target_transform import TargetTransform


BatchAdapter = Callable[[Any, torch.device], tuple[Any, torch.Tensor]]


def default_batch_adapter(batch: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    inputs, targets = batch
    return inputs.to(device), targets.to(device).float()


def train_regressor(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int,
    learning_rate: float,
    device: str | torch.device = "cpu",
    batch_adapter: BatchAdapter = default_batch_adapter,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
    restore_best: bool = True,
    target_transform: TargetTransform | None = None,
) -> list[dict[str, float]]:
    device_obj = torch.device(device)
    model.to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    history: list[dict[str, float]] = []
    best_val_mae = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            inputs, targets = batch_adapter(batch, device_obj)
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = loss_fn(predictions.reshape(-1), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        val_metrics = evaluate_regressor(
            model,
            val_loader,
            device_obj,
            batch_adapter,
            target_transform=target_transform,
        )
        improved = val_metrics["mae"] < best_val_mae - early_stopping_min_delta
        if improved:
            best_val_mae = val_metrics["mae"]
            best_epoch = epoch
            epochs_without_improvement = 0
            if restore_best:
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
        else:
            epochs_without_improvement += 1

        early_stopped = (
            early_stopping_patience is not None
            and epochs_without_improvement >= early_stopping_patience
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": sum(losses) / max(len(losses), 1),
                "val_mae": val_metrics["mae"],
                "val_mse": val_metrics["mse"],
                "val_r2": val_metrics["r2"],
                "best_epoch": best_epoch,
                "best_val_mae": best_val_mae,
                "epochs_without_improvement": epochs_without_improvement,
                "early_stopped": early_stopped,
            }
        )
        if early_stopped:
            break
    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
    return history


def evaluate_regressor(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str | torch.device = "cpu",
    batch_adapter: BatchAdapter = default_batch_adapter,
    target_transform: TargetTransform | None = None,
) -> dict[str, float]:
    device_obj = torch.device(device)
    model.to(device_obj)
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            inputs, target = batch_adapter(batch, device_obj)
            predictions.append(model(inputs).detach().cpu())
            targets.append(target.detach().cpu())
    y_true = torch.cat(targets)
    y_pred = torch.cat(predictions)
    if target_transform is not None:
        y_true = target_transform.inverse_tensor(y_true)
        y_pred = target_transform.inverse_tensor(y_pred)
    return regression_metrics(y_true, y_pred)
