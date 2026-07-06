"""
Adapter insertion -- inject LoRA or HRM adapters into TinyGPT in-place.

inject_lora(model, config) -> model
  Replaces target nn.Linear modules (config['lora']['target_modules']) with LoRALinear.
  Freezes all non-LoRA parameters. Prints trainable parameter count.

inject_hrm(model, config) -> model
  Attaches one HRMAdapter per TransformerBlock, wired in parallel to the MLP.
  Freezes all non-HRM parameters. Prints trainable parameter count.

Both functions verify that only adapter parameters have requires_grad=True.
"""

import torch.nn as nn

from src.adapters.lora import LoRALinear


def _get_nested_attr(module: nn.Module, path: str) -> nn.Module:
    """Navigate a dotted path like 'attn.q_proj' from a parent module."""
    parts = path.split(".")
    obj = module
    for part in parts:
        obj = getattr(obj, part)
    return obj


def _set_nested_attr(module: nn.Module, path: str, value: nn.Module) -> None:
    parts = path.split(".")
    obj = module
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def inject_lora(model: nn.Module, config: dict) -> nn.Module:
    """
    Inject LoRA adapters into model.layers in-place.

    config must contain:
      config['lora']['rank']           int
      config['lora']['alpha']          float
      config['lora']['target_modules'] list[str]  e.g. ['attn.q_proj', 'attn.v_proj']
    """
    rank   = config["lora"]["rank"]
    alpha  = config["lora"]["alpha"]
    targets = config["lora"]["target_modules"]

    # Step 1: freeze everything
    for p in model.parameters():
        p.requires_grad_(False)

    # Step 2: replace target Linear modules with LoRALinear
    for block in model.layers:
        for path in targets:
            base_layer = _get_nested_attr(block, path)
            if not isinstance(base_layer, nn.Linear):
                raise TypeError(
                    f"Expected nn.Linear at '{path}', got {type(base_layer).__name__}"
                )
            lora_layer = LoRALinear(
                in_features=base_layer.in_features,
                out_features=base_layer.out_features,
                rank=rank,
                alpha=alpha,
                base_layer=base_layer,
            )
            _set_nested_attr(block, path, lora_layer)

    _print_param_summary(model, label="LoRA")
    return model


def inject_hrm(model: nn.Module, config: dict) -> nn.Module:
    """
    Inject HRM adapters alongside the MLP in each TransformerBlock.

    config must contain config['hrm'] matching hrm.yaml.
    Implemented in Step 3; placeholder here to keep insertion.py complete.
    """
    # Imported lazily to avoid circular deps before HRMAdapter exists
    from src.adapters.hrm_adapter import HRMAdapter
    from src.models.ssm import SSM

    state_dim  = config["hrm"]["state_dim"]
    input_dim  = config["hrm"]["input_dim"]
    output_dim = config["hrm"]["output_dim"]
    dt_init    = config["hrm"].get("dt_init", 0.01)
    dt_min     = config["hrm"].get("dt_min", 1e-4)
    dt_max     = config["hrm"].get("dt_max", 0.1)
    gate_init  = config["hrm"].get("gate_init", 0.0)

    # Freeze everything first
    for p in model.parameters():
        p.requires_grad_(False)

    for block in model.layers:
        ssm = SSM(input_dim, state_dim, output_dim, dt_init, dt_min, dt_max)
        adapter = HRMAdapter(ssm, input_dim, output_dim, gate_init=gate_init)
        block.hrm = adapter          # attach as named sub-module

        # Monkey-patch block.forward to add the HRM residual after MLP
        _wrap_block_with_hrm(block)

    _print_param_summary(model, label="HRM")
    return model


def _wrap_block_with_hrm(block: nn.Module) -> None:
    """Replace block.forward so it adds hrm(ln2(x)) residual alongside mlp."""
    original_forward = block.__class__.forward

    def new_forward(self, x):
        x = x + self.attn(self.ln1(x))
        normed = self.ln2(x)
        x = x + self.mlp(normed) + self.hrm(normed)
        return x

    import types
    block.forward = types.MethodType(new_forward, block)


def _print_param_summary(model: nn.Module, label: str) -> None:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable
    print(
        f"[{label}] trainable={trainable:,}  frozen={frozen:,}  total={total:,}"
    )
