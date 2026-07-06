"""
Generic Trainer -- shared training loop for LoRA and HRM experiments.

Adapter-agnostic: receives model, optimizer, dataloaders, and config.
Adapter-specific logic (e.g., enabling state collection at a specific epoch)
is passed via an optional on_epoch_end(epoch, val_metrics) callback.

Features:
  - Masked cross-entropy loss (loss_mask applied per batch)
  - Gradient clipping
  - Per-step LR scheduling (cosine + warmup via make_scheduler)
  - Best-checkpoint saving (highest val_accuracy) + periodic saves
  - Epoch-level logging via src/utils/logging.Logger

Public API:
  Trainer(model, optimizer, scheduler, train_loader, val_loader, config, logger,
          on_epoch_end=None)
  trainer.train()                  -- full training loop
  trainer.eval() -> dict           -- single eval pass, returns {'loss', 'accuracy'}
  trainer.save_checkpoint(path)
  trainer.load_checkpoint(path)
"""

import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from src.training.metrics import copy_accuracy


# ------------------------------------------------------------------
# Scheduler factory (used by training scripts)
# ------------------------------------------------------------------

def make_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """Cosine decay with linear warmup, stepped every batch."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


# ------------------------------------------------------------------
# Trainer
# ------------------------------------------------------------------

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,                        # LambdaLR or None
        train_loader,
        val_loader,
        config: dict,
        logger,
        on_epoch_end=None,                # callable(epoch, val_metrics) | None
    ):
        self.model        = model
        self.optimizer    = optimizer
        self.scheduler    = scheduler
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.config       = config
        self.logger       = logger
        self.on_epoch_end = on_epoch_end

        self.device = next(model.parameters()).device
        self.best_val_acc  = 0.0
        self.best_val_loss = float("inf")
        self.global_step   = 0

        os.makedirs(config.get("save_dir", "checkpoints"), exist_ok=True)

    # ------------------------------------------------------------------

    def train(self) -> None:
        epochs     = self.config["epochs"]
        save_every = self.config.get("save_every", 5)
        save_dir   = self.config.get("save_dir", "checkpoints")

        for epoch in range(epochs):
            t0 = time.perf_counter()
            train_loss = self._train_epoch()
            val_metrics = self.eval()
            epoch_time = time.perf_counter() - t0

            self.logger.log(
                epoch,
                train_loss=train_loss,
                val_loss=val_metrics["loss"],
                val_acc=val_metrics["accuracy"],
                epoch_time_s=epoch_time,
            )

            if val_metrics["accuracy"] > self.best_val_acc:
                self.best_val_acc  = val_metrics["accuracy"]
                self.best_val_loss = val_metrics["loss"]
                self.save_checkpoint(os.path.join(save_dir, "best.pt"))

            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(os.path.join(save_dir, f"epoch_{epoch+1:03d}.pt"))

            if self.on_epoch_end is not None:
                self.on_epoch_end(epoch, val_metrics)

        print(f"\nTraining complete. Best val_acc={self.best_val_acc:.4f}")

    # ------------------------------------------------------------------

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        for input_ids, target_ids, loss_mask in self.train_loader:
            input_ids  = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            loss_mask  = loss_mask.to(self.device)   # (T,) or (B, T)

            logits = self.model(input_ids)            # (B, T, V)
            loss   = self._masked_ce(logits, target_ids, loss_mask)

            self.optimizer.zero_grad()
            loss.backward()

            grad_clip = self.config.get("grad_clip", 1.0)
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            total_loss   += loss.item()
            self.global_step += 1

        return total_loss / len(self.train_loader)

    # ------------------------------------------------------------------

    def eval(self) -> dict:
        self.model.eval()
        total_loss = 0.0
        total_acc  = 0.0
        n = 0

        with torch.no_grad():
            for input_ids, target_ids, loss_mask in self.val_loader:
                input_ids  = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                loss_mask  = loss_mask.to(self.device)

                logits = self.model(input_ids)
                loss   = self._masked_ce(logits, target_ids, loss_mask)
                acc    = copy_accuracy(logits, target_ids, loss_mask)

                total_loss += loss.item()
                total_acc  += acc
                n += 1

        return {"loss": total_loss / n, "accuracy": total_acc / n}

    # ------------------------------------------------------------------

    @staticmethod
    def _masked_ce(
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, T, V = logits.shape
        per_token = F.cross_entropy(
            logits.view(B * T, V),
            targets.view(B * T),
            reduction="none",
            label_smoothing=0.1,
        ).view(B, T)

        if loss_mask.dim() == 1:
            mask = loss_mask.unsqueeze(0).expand(B, T)
        else:
            mask = loss_mask

        return (per_token * mask).sum() / mask.sum()

    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "model_state_dict":     self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "global_step":          self.global_step,
                "best_val_acc":         self.best_val_acc,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.global_step  = ckpt.get("global_step", 0)
        self.best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"Loaded checkpoint: {path}  (step={self.global_step})")
