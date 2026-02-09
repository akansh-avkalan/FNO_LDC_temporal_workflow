from temporalio import activity
import torch
import torch.nn as nn
from torch.optim import Adam
from shared import TrainConfig
from utils.dataset import create_dataloaders
from models.FNO import FNO2d
from utils.trainer import TrainFNO


@activity.defn
async def train_FNO(train_config: TrainConfig) -> str:
    """
    Activity to train FNO model.
    
    Args:
        train_config: Training configuration
        
    Returns:
        str: Path to the saved model checkpoint
    """
    activity.logger.info("Starting FNO training...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    activity.logger.info(f"Using device: {device}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_root=train_config.dataset_path,
        batch_size=train_config.batch_size,
        train_ratio=train_config.train_ratio,
        val_ratio=train_config.val_ratio,
        seed=train_config.seed,
        num_workers=train_config.num_workers,
    )
    
    activity.logger.info(f"Train samples: {len(train_loader.dataset)}")
    activity.logger.info(f"Val samples: {len(val_loader.dataset)}")
    activity.logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Initialize model
    model = FNO2d(
        in_channels=train_config.in_channels,
        out_channels=train_config.out_channels,
        width=train_config.fno_width,
        modes1=train_config.modes1,
        modes2=train_config.modes2,
    )
    
    activity.logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Initialize optimizer and loss
    optimizer = Adam(model.parameters(), lr=train_config.learning_rate)
    loss_fn = nn.MSELoss()
    
    # Initialize trainer
    trainer = TrainFNO(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=train_config.epochs,
        device=device,
        log_dir=train_config.log_dir,
        checkpoint_frequency=train_config.checkpoint_frequency,
    )
    
    # Train model
    activity.logger.info(f"Starting training for {train_config.epochs} epochs...")
    model_path = trainer.train()
    
    activity.logger.info(f"Training complete. Model saved at: {model_path}")
    
    return model_path
