"""
RSNA Knee Abnormality Detection - Enterprise PyTorch Model Trainer.
Features Automatic Mixed Precision (FP16 AMP), Gradient Accumulation, Gradient Scaling,
Model EMA, Cosine Annealing with Warmup, and Competition Macro AUC-ROC Checkpointing.
"""

import os
import copy
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

from src.metrics.auc_metrics import compute_macro_auc, TARGET_COLUMNS
from src.training.losses import AsymmetricLoss


class ModelEMA:
    """
    Maintains Exponential Moving Average (EMA) of model parameters for smoother convergence.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.module = copy.deepcopy(model).eval()
        self.module.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_p, p in zip(self.module.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)


class KneeTrainer:
    """
    Production-Grade Trainer for 2.5D MIL Knee Abnormality Models with Gradient Accumulation.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        criterion: Optional[nn.Module] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_amp: bool = True,
        use_ema: bool = True,
        grad_accum_steps: int = 1,
        max_grad_norm: float = 2.0,
        checkpoint_dir: str = "checkpoints",
        experiment_name: str = "mil_baseline",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion if criterion is not None else AsymmetricLoss()
        self.device = device
        self.use_amp = use_amp and (device == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = checkpoint_dir
        self.experiment_name = experiment_name
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.ema = ModelEMA(model) if use_ema else None
        self.console = Console()
        self.best_macro_auc = -1.0
        self.best_epoch = 0

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:02d} [Train]", leave=False)
        for step, batch in enumerate(pbar):
            images = batch["images"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True)
            targets = batch["targets"].to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model(images, mask=mask)
                logits = outputs["logits"]
                loss = self.criterion(logits, targets)
                scaled_loss = loss / self.grad_accum_steps

            self.scaler.scale(scaled_loss).backward()

            if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == num_batches:
                if self.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                if self.ema is not None:
                    self.ema.update(self.model)

            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        return total_loss / max(1, num_batches)

    @torch.no_grad()
    def evaluate(self, use_ema_for_eval: bool = True) -> Dict[str, Any]:
        eval_model = self.ema.module if (use_ema_for_eval and self.ema is not None) else self.model
        eval_model.eval()

        all_preds = []
        all_targets = []
        all_study_uids = []
        total_val_loss = 0.0
        num_batches = len(self.val_loader)

        for batch in tqdm(self.val_loader, desc="[Validation]", leave=False):
            images = batch["images"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True)
            targets = batch["targets"].to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = eval_model(images, mask=mask)
                logits = outputs["logits"]
                loss = self.criterion(logits, targets)

            total_val_loss += loss.item()
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_preds.append(probs)
            all_targets.append(targets.detach().cpu().numpy())
            all_study_uids.extend(batch.get("study_uids", []))

        y_pred = np.vstack(all_preds)
        y_true = np.vstack(all_targets)

        auc_res = compute_macro_auc(y_true, y_pred)
        auc_res["val_loss"] = total_val_loss / max(1, num_batches)
        auc_res["y_pred"] = y_pred
        auc_res["y_true"] = y_true
        auc_res["study_uids"] = all_study_uids
        return auc_res

    def fit(self, num_epochs: int = 15, fold: int = 0) -> Dict[str, Any]:
        self.console.print(f"\n[bold green]=== Starting Training: {self.experiment_name} (Fold {fold}) ===[/bold green]")
        self.console.print(f"Device: [cyan]{self.device}[/cyan] | AMP FP16: [cyan]{self.use_amp}[/cyan] | Accum Steps: [cyan]{self.grad_accum_steps}[/cyan] | Epochs: [cyan]{num_epochs}[/cyan]")

        history = []

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_results = self.evaluate(use_ema_for_eval=True)

            if self.scheduler is not None:
                self.scheduler.step()

            macro_auc = val_results["macro_auc"]
            val_loss = val_results["val_loss"]

            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "macro_auc": macro_auc,
            })

            is_best = macro_auc > self.best_macro_auc
            if is_best:
                self.best_macro_auc = macro_auc
                self.best_epoch = epoch
                ckpt_path = os.path.join(self.checkpoint_dir, f"{self.experiment_name}_fold{fold}_best.pth")
                save_model = self.ema.module if self.ema is not None else self.model
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": save_model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "macro_auc": macro_auc,
                    "per_class_auc": val_results["per_class_auc"],
                }, ckpt_path)

            status_str = f"Epoch {epoch:02d}/{num_epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro AUC: [bold magenta]{macro_auc:.4f}[/bold magenta]"
            if is_best:
                status_str += " [bold green]⭐ (Best)[/bold green]"
            self.console.print(status_str)

        self.console.print(f"\n[bold green]✓ Finished Fold {fold}. Best Val Macro AUC: {self.best_macro_auc:.4f} at Epoch {self.best_epoch}[/bold green]")
        return {"best_macro_auc": self.best_macro_auc, "history": history}
