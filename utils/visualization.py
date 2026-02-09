# import matplotlib.pyplot as plt
# import torch 
# import os

# def save_fno_comparison_plots(
#     y_true: torch.Tensor,
#     y_pred: torch.Tensor,
#     save_dir: str,
# ):
#     """
#     Save comparison plots between ground truth and predictions.
    
#     Args:
#         y_true: Ground truth tensor [C, H, W]
#         y_pred: Predicted tensor [C, H, W]
#         save_dir: Directory to save plots
#     """

#     channels = y_true.shape[0]
#     field_names = ["u", "v", "p", "omega"]  # adjust if needed

#     fig, axes = plt.subplots(
#         channels,
#         2,
#         figsize=(10, 3 * channels),
#         constrained_layout=True,
#     )

#     for c in range(channels):
#         vmin = min(y_true[c].min(), y_pred[c].min())
#         vmax = max(y_true[c].max(), y_pred[c].max())

#         im0 = axes[c, 0].imshow(
#             y_true[c],
#             cmap="jet",
#             vmin=vmin,
#             vmax=vmax,
#         )
#         axes[c, 0].set_title(f"GT - {field_names[c]}")
#         axes[c, 0].axis("off")

#         im1 = axes[c, 1].imshow(
#             y_pred[c],
#             cmap="jet",
#             vmin=vmin,
#             vmax=vmax,
#         )
#         axes[c, 1].set_title(f"Pred - {field_names[c]}")
#         axes[c, 1].axis("off")

#         fig.colorbar(im0, ax=axes[c, :], shrink=0.8)

#     save_path = os.path.join(save_dir, "comparison.png")
#     plt.savefig(save_path, dpi=200)
#     plt.close(fig)
    
#     print(f"Comparison plot saved to {save_path}")

import matplotlib.pyplot as plt
import torch 
import os
import numpy as np


def save_fno_comparison_plots(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    save_dir: str,
):
    """
    Save comparison plots between ground truth and predictions.
    Generates two types of visualizations:
    1. Field comparison plots (heatmaps for u, v, p, omega)
    2. Streamline comparison plot (for velocity field u, v)
    
    Args:
        y_true: Ground truth tensor [C, H, W] where C >= 2 (u, v, ...)
        y_pred: Predicted tensor [C, H, W]
        save_dir: Directory to save plots
    """

    channels = y_true.shape[0]
    field_names = ["u", "v", "p", "omega"]  # adjust if needed

    # ====================================
    # 1. Field Comparison Plots (Heatmaps)
    # ====================================
    fig, axes = plt.subplots(
        channels,
        2,
        figsize=(10, 3 * channels),
        constrained_layout=True,
    )

    for c in range(channels):
        vmin = min(y_true[c].min(), y_pred[c].min())
        vmax = max(y_true[c].max(), y_pred[c].max())

        im0 = axes[c, 0].imshow(
            y_true[c],
            cmap="jet",
            vmin=vmin,
            vmax=vmax,
        )
        axes[c, 0].set_title(f"GT - {field_names[c]}")
        axes[c, 0].axis("off")

        im1 = axes[c, 1].imshow(
            y_pred[c],
            cmap="jet",
            vmin=vmin,
            vmax=vmax,
        )
        axes[c, 1].set_title(f"Pred - {field_names[c]}")
        axes[c, 1].axis("off")

        fig.colorbar(im0, ax=axes[c, :], shrink=0.8)

    save_path = os.path.join(save_dir, "comparison.png")
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Field comparison plot saved to {save_path}")

    # ====================================
    # 2. Streamline Comparison Plot
    # ====================================
    if channels >= 2:  # Need at least u and v components
        _save_streamline_comparison(y_true, y_pred, save_dir)


def _save_streamline_comparison(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    save_dir: str,
    sample_idx: int = 20,
):
    """
    Create streamline plots comparing ground truth and predicted velocity fields.
    Matches the style of the reference image with proper formatting.
    
    Args:
        y_true: Ground truth tensor [C, H, W]
        y_pred: Predicted tensor [C, H, W]
        save_dir: Directory to save plots
        sample_idx: Sample number to display in title
    """
    # Extract velocity components (u, v are typically the first two channels)
    u_true = y_true[0].cpu().numpy()
    v_true = y_true[1].cpu().numpy()
    u_pred = y_pred[0].cpu().numpy()
    v_pred = y_pred[1].cpu().numpy()
    
    # Get grid dimensions
    H, W = u_true.shape
    
    # Create coordinate grids
    x = np.linspace(0, 1, W)
    y = np.linspace(0, 1, H)
    X, Y = np.meshgrid(x, y)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot true streamlines
    ax = axes[0]
    ax.streamplot(
        X, Y, u_true, v_true,
        color='#1f77b4',  # Blue color
        linewidth=0.8,
        density=2.5,
        arrowsize=1.2,
        arrowstyle='->',
    )
    ax.set_xlabel('X', fontsize=13)
    ax.set_ylabel('Y', fontsize=13)
    ax.set_title(f'True Streamlines (Sample {sample_idx})', fontsize=15, pad=15)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=11)
    # Add grid for better readability
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    # Plot predicted streamlines
    ax = axes[1]
    ax.streamplot(
        X, Y, u_pred, v_pred,
        color='#1f77b4',  # Blue color
        linewidth=0.8,
        density=2.5,
        arrowsize=1.2,
        arrowstyle='->',
    )
    ax.set_xlabel('X', fontsize=13)
    ax.set_ylabel('Y', fontsize=13)
    ax.set_title(f'Predicted Streamlines (Sample {sample_idx})', fontsize=15, pad=15)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=11)
    # Add grid for better readability
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    
    # Save streamline plot
    streamline_path = os.path.join(save_dir, "streamline_comparison.png")
    plt.savefig(streamline_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"Streamline comparison plot saved to {streamline_path}")
