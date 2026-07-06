"""
HRM adapter injection for HuggingFace pre-trained models.

Supports:
  - GPT-2 family  (gpt2, gpt2-medium, gpt2-large, gpt2-xl)
  - LLaMA-2 / Mistral (transformers >= 4.36)

Injection point: parallel residual after the MLP block in each decoder layer,
receiving the same layer-normalized input as the MLP:
    h_out = h_attn + MLP(LN(h_attn)) + alpha * SSM(LN(h_attn))

Parameter count at Mistral-7B scale (d_model=4096, n_layers=32):
  d=32  ->  (2 x 32 x 4096 + 32 + 32 + 1) x 32  ~=  8.39M  (Tier-2 iso-param with LoRA r=16)
  d=16  ->  (2 x 16 x 4096 + 16 + 16 + 1) x 32  ~=  4.20M  (Tier-1 iso-param with LoRA r=8)

Parameter count at GPT-2-medium scale (d_model=1024, n_layers=24):
  d=16  ->  (2 x 16 x 1024 + 16 + 16 + 1) x 24  ~=  0.79M  (Tier-1 iso-param with LoRA r=8)
"""

import inspect
import types

import torch
import torch.nn as nn

from src.adapters.hrm_adapter import HRMAdapter
from src.models.ssm import SSM


# ---------------------------------------------------------------------------
# Architecture detection
# ---------------------------------------------------------------------------

def _detect_arch(model: nn.Module) -> str:
    cls = type(model).__name__.lower()
    if "gpt2" in cls:
        return "gpt2"
    if "mistral" in cls or "llama" in cls:
        return "llama_mistral"
    raise ValueError(
        f"Unsupported model class '{type(model).__name__}'. "
        "Add support in hf_injection._detect_arch()."
    )


def _get_decoder_layers(model: nn.Module, arch: str) -> nn.ModuleList:
    if arch == "gpt2":
        return model.transformer.h
    if arch == "llama_mistral":
        return model.model.layers
    raise ValueError(arch)


def _get_d_model(model: nn.Module) -> int:
    """Return the hidden dimension from model.config."""
    cfg = model.config
    # GPT-2 uses n_embd; LLaMA/Mistral use hidden_size
    return getattr(cfg, "hidden_size", getattr(cfg, "n_embd", None))


def _gpt2_block_returns_tuple(model: nn.Module) -> bool:
    """
    Detect whether GPT2Model.forward expects block() to return a tuple (old API,
    transformers <=4.45) or a plain tensor (new API, some >=4.46 builds).

    Old API: `outputs = block(...)` then `hidden_states = outputs[0]`
    New API: `hidden_states = block(...)` directly
    """
    try:
        src = inspect.getsource(model.transformer.forward)
        return "hidden_states = outputs[0]" in src
    except Exception:
        return True  # fall back to old tuple API


# ---------------------------------------------------------------------------
# Block wrappers
# ---------------------------------------------------------------------------

def _wrap_gpt2_block(block: nn.Module, hrm: HRMAdapter, return_tuple: bool = True) -> None:
    """
    Patch a GPT2Block to run HRM parallel to the MLP.

    return_tuple=True  -> old transformers API: outer loop does outputs[0]
    return_tuple=False -> new transformers API: outer loop stores block() directly
    """
    block.hrm = hrm

    def new_forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
        **kwargs,
    ):
        # transformers >=4.46 GradientCheckpointingLayer packs args as a tuple
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )
        attn_out = attn_outputs[0]
        outputs = attn_outputs[1:]
        # accelerate device_map="auto" hooks may move attn_out to a different
        # device than residual; pin residual to match the computed output.
        hidden_states = attn_out + residual.to(attn_out.device)

        residual = hidden_states
        normed = self.ln_2(hidden_states)
        mlp_out = self.mlp(normed)
        hrm_out = self.hrm(normed.to(next(self.hrm.parameters()).device)).to(mlp_out.device)
        hidden_states = residual.to(mlp_out.device) + mlp_out + hrm_out

        if not return_tuple:
            # New transformers API: GPT2Model.forward stores block() directly
            # in hidden_states without unpacking outputs[0].
            return hidden_states
        if use_cache:
            return (hidden_states,) + outputs
        return (hidden_states,) + outputs[1:]

    block.forward = types.MethodType(new_forward, block)


def _wrap_llama_mistral_block(block: nn.Module, hrm: HRMAdapter) -> None:
    """
    Inject HRM parallel to the MLP via a forward hook on block.mlp.

    The hook intercepts mlp(normed) and returns mlp(normed) + hrm(normed),
    so the decoder layer computes:
        h_out = h_attn + MLP(LN(h_attn)) + HRM(LN(h_attn))

    This avoids monkey-patching block.forward entirely, making it compatible
    with all transformers versions and gradient checkpointing implementations.
    """
    block.hrm = hrm  # register as submodule so optimizer sees HRM parameters

    hrm_device = next(hrm.parameters()).device

    def mlp_hook(module, input, output):
        # input[0]: the layer-normed hidden states fed to the MLP
        normed = input[0].to(hrm_device)
        return output + block.hrm(normed).to(output.device)

    block.mlp.register_forward_hook(mlp_hook)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def inject_hrm_hf(
    model: nn.Module,
    state_dim: int,
    gate_init: float = 0.1,
    dt_init: float = 0.01,
    dt_min: float = 1e-4,
    dt_max: float = 0.1,
) -> nn.Module:
    """
    Freeze all backbone parameters and inject one HRMAdapter per decoder layer.

    The adapter is inserted in parallel with each MLP block, receiving the
    same layer-normalized input (zero-residual init preserves pretrained behavior).

    Args:
        model:      A loaded HuggingFace CausalLM model (GPT-2 or LLaMA/Mistral family).
        state_dim:  SSM state dimension d.  Use d=16 (Tier-1) or d=32 (Tier-2).
        gate_init:  Initial value of the scalar gate alpha.  0.1 gives a small but
                    non-zero gradient from step 0 (faster early convergence than 0.0).
        dt_init, dt_min, dt_max: ZOH step-size range passed to SSM.

    Returns:
        The same model object with HRM adapters injected and backbone frozen.
        Only HRM parameters (B, C, log_A, log_dt, gate x n_layers) require grad.
    """
    arch = _detect_arch(model)
    d_model = _get_d_model(model)
    if d_model is None:
        raise ValueError("Cannot infer d_model from model.config -- set manually.")

    # Freeze backbone
    for p in model.parameters():
        p.requires_grad_(False)

    layers = _get_decoder_layers(model, arch)
    gpt2_return_tuple = _gpt2_block_returns_tuple(model) if arch == "gpt2" else True
    for block in layers:
        # Infer device from the block's own parameters so HRM lands on the
        # same GPU shard as the block (handles device_map="auto" multi-GPU).
        block_device = next(block.parameters()).device
        ssm = SSM(d_model, state_dim, d_model, dt_init, dt_min, dt_max)
        hrm = HRMAdapter(ssm, d_model, d_model, gate_init=gate_init)
        hrm = hrm.to(block_device)
        if arch == "gpt2":
            _wrap_gpt2_block(block, hrm, return_tuple=gpt2_return_tuple)
        else:
            _wrap_llama_mistral_block(block, hrm)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    n_layers = len(layers)
    print(
        f"[HRM-HF] arch={arch}  d_model={d_model}  state_dim={state_dim}  "
        f"n_layers={n_layers}\n"
        f"         trainable={trainable:,}  /  total={total:,}  "
        f"({100 * trainable / total:.4f}%)"
    )
    return model
