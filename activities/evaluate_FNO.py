from temporalio import activity
from typing import Tuple
import os
import torch
from torch import nn
import matplotlib.pyplot as plt
from utils.visualization import save_fno_comparison_plots
from shared import EvaluateConfig
from utils.dataset import create_dataloaders
from models.FNO import FNO2d

@activity.defn
async def evaluate_FNO(
    evaluate_config: EvaluateConfig,
) -> Tuple[float, str]:
    """
    Activity to evaluate trained FNO on test dataset and generate plots.
    
    Args:
        evaluate_config: EvaluateConfig containing model_path and train_config
        
    Returns:
        Tuple of (test_mse, plot_dir)
    """

    activity.logger.info("Starting FNO evaluation...")

    # Extract configs
    model_path = evaluate_config.model_path
    train_config = evaluate_config.train_config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------
    # Load test dataloader
    # -----------------------
    _, _, test_loader = create_dataloaders(
        dataset_root=train_config.dataset_path,
        batch_size=train_config.batch_size,
        train_ratio=train_config.train_ratio,
        val_ratio=train_config.val_ratio,
        seed=train_config.seed,
        num_workers=train_config.num_workers,
    )

    # -----------------------
    # Load model
    # -----------------------
    model = FNO2d(
        in_channels=train_config.in_channels,
        out_channels=train_config.out_channels,
        width=train_config.fno_width,
        modes1=train_config.modes1,
        modes2=train_config.modes2,
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    activity.logger.info(f"Model loaded from {model_path}")

    # -----------------------
    # Evaluation
    # -----------------------
    mse_loss = nn.MSELoss(reduction="sum")
    total_mse = 0.0
    total_samples = 0

    plots_dir = os.path.join(train_config.log_dir, "evaluation_plots")
    os.makedirs(plots_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)

            total_mse += mse_loss(y_pred, y).item()
            total_samples += y.numel()

            # Plot only first batch
            if batch_idx == 0:
                save_fno_comparison_plots(
                    y[0].cpu(),
                    y_pred[0].cpu(),
                    plots_dir,
                )
                activity.logger.info(f"Comparison plots saved to {plots_dir}")

    test_mse = total_mse / total_samples

    activity.logger.info(f"Test MSE: {test_mse:.6e}")
    activity.logger.info(f"Evaluation complete. Plots directory: {plots_dir}")

    return test_mse, plots_dir
