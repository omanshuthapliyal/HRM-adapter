"""
Task-specific evaluation metrics.

  copy_accuracy(logits, targets, loss_mask) -> float
    Token-level accuracy restricted to copy positions (loss_mask == True).

  recall_accuracy(logits, target_ids) -> float
    Exact-match accuracy at the single query position for Associative Recall.
    logits: (B, vocab_size) at query position; target_ids: (B,)

  parameter_count(model) -> dict
    Returns {'total': int, 'trainable': int, 'frozen': int}.
"""

import torch
import torch.nn as nn


def copy_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
) -> float:
    """
    logits   : (B, T, vocab_size)
    targets  : (B, T)
    loss_mask: (T,) or (B, T)  -- True at copy positions

    Returns token-level accuracy over masked positions as a Python float.
    """
    preds = logits.argmax(dim=-1)                    # (B, T)

    if loss_mask.dim() == 1:
        mask = loss_mask.unsqueeze(0).expand_as(preds)
    else:
        mask = loss_mask

    correct = (preds == targets) & mask
    return correct.sum().item() / mask.sum().item()


def recall_accuracy(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
) -> float:
    """
    logits    : (B, vocab_size)  -- logits at the single query position
    target_ids: (B,)

    Returns exact-match accuracy as a Python float.
    """
    preds = logits.argmax(dim=-1)
    return (preds == target_ids).float().mean().item()


def parameter_count(model: nn.Module) -> dict:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
