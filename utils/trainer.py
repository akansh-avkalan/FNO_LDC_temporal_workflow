# utils/trainer.py
import os
from temporalio import activity
import torch
from models.FNO import FNO2d
from utils.metrics import L2_norm, LInf_norm


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        train_loader,
        val_loader,
        epochs,
        device,
        log_dir=None,
        checkpoint_frequency=1,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.device = device
        self.log_dir = log_dir
        self.checkpoint_frequency = checkpoint_frequency

        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            self.log_file = os.path.join(self.log_dir, "log.txt")
            self.ckpt_dir = os.path.join(self.log_dir, "checkpoints")
            os.makedirs(self.ckpt_dir, exist_ok=True)
        else:
            self.log_file = None
            self.ckpt_dir = None

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0

        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)

        return total_loss / len(self.train_loader.dataset)

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        total_l2 = 0.0
        total_linf = 0.0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)

                total_loss += loss.item() * inputs.size(0)
                total_l2 += L2_norm(outputs.cpu().numpy(), targets.cpu().numpy())
                total_linf += LInf_norm(outputs.cpu().numpy(), targets.cpu().numpy())

        n = len(self.val_loader.dataset)
        return total_loss / n, total_l2 / n, total_linf / n

    def save_checkpoint(self, epoch, train_loss, val_loss):
        if not self.ckpt_dir:
            return

        ckpt_path = os.path.join(self.ckpt_dir, f"epoch_{epoch}.pt")

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
            ckpt_path,
        )
    def train(self):
        last_ckpt_path = None

        for epoch in range(self.epochs):
            train_loss = self.train_epoch()
            val_loss, l2_loss, linf_loss = self.evaluate()

            log_line = (
                f"Epoch {epoch+1}/{self.epochs}, "
                f"Train Loss: {train_loss:.6f}, "
                f"Val Loss: {val_loss:.6f}, "
                f"L2 Loss: {l2_loss:.6f}, "
                f"LInf Loss: {linf_loss:.6f}"
            )
            print(log_line)

            if self.log_file:
                with open(self.log_file, "a") as f:
                    f.write(log_line + "\n")

            activity.heartbeat(
            {
                "epoch": epoch + 1,
            }
            )
            # PERIODIC CHECKPOINT
            if (
                self.ckpt_dir
                and (epoch + 1) % self.checkpoint_frequency == 0
            ):
                ckpt_path = os.path.join(
                    self.ckpt_dir, f"epoch_{epoch+1}.pt"
                )

                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                    },
                    ckpt_path,
                )

                last_ckpt_path = ckpt_path
                print(f"Checkpoint saved: {ckpt_path}")

        # SAVE FINAL MODEL
        final_model_path = None
        if self.log_dir:
            final_model_path = os.path.join(self.log_dir, "model_final.pt")

            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "epochs": self.epochs,
                },
                final_model_path,
            )

            print(f"Final model saved: {final_model_path}")

        print("Training complete.")
        return final_model_path


class TrainFNO(Trainer):
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        train_loader,
        val_loader,
        epochs,
        device,
        log_dir="experiments/fno/",
        checkpoint_frequency=1,
    ):
        if not isinstance(model, FNO2d):
            raise TypeError("Model must be an instance of FNO2d")

        super().__init__(
            model,
            optimizer,
            loss_fn,
            train_loader,
            val_loader,
            epochs,
            device,
            log_dir,
            checkpoint_frequency,
        )
