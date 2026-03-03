# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn.functional as F


@torch.compile(dynamic=None)
def compiled_forward_kl_loss(
    prenorm_hidden_states_flat,
    target_p_flat,
    valid_idx,
    norm_weight,
    lm_head_weight,
    norm_eps,
):
    """torch.compile'd index_select + RMSNorm + lm_head + Forward KL loss.

    Takes full (B*T, ...) flat tensors and performs index_select inside the
    compiled graph so the compiler can fuse the gather with subsequent ops.

    Args:
        prenorm_hidden_states_flat: (B*T, H) — flattened draft hidden states
        target_p_flat: (B*T, V_out) — flattened target probs (detached)
        valid_idx: (N,) int64 — indices of non-masked positions
        norm_weight: (H,)
        lm_head_weight: (V_out, H) — draft lm_head weight
        norm_eps: float
    """
    hs = prenorm_hidden_states_flat.index_select(0, valid_idx)
    tp = target_p_flat.index_select(0, valid_idx)

    # RMSNorm
    hs_f32 = hs.float()
    variance = hs_f32.pow(2).mean(-1, keepdim=True)
    rstd = torch.rsqrt(variance + norm_eps)
    norm_hs = (hs_f32 * rstd).to(hs.dtype) * norm_weight

    logits = F.linear(norm_hs, lm_head_weight)  # (N, V_out)

    # Forward KL loss
    log_p = F.log_softmax(logits.float(), dim=-1)
    loss = -(tp * log_p).sum(-1).mean()

    # Accuracy
    acc = (logits.argmax(-1) == tp.argmax(-1)).float().mean()

    return loss, acc


@torch.compile(dynamic=None)
def compiled_lk_alpha_loss(
    prenorm_hidden_states_flat,
    target_p_flat,
    valid_idx,
    norm_weight,
    lm_head_weight,
    norm_eps,
):
    """LK^α loss: -log(acceptance_rate).mean().

    Directly optimizes the log acceptance rate α_i = Σ_x min(p_i(x), q_i(x)).
    """
    hs = prenorm_hidden_states_flat.index_select(0, valid_idx)
    tp = target_p_flat.index_select(0, valid_idx)

    # RMSNorm
    hs_f32 = hs.float()
    variance = hs_f32.pow(2).mean(-1, keepdim=True)
    rstd = torch.rsqrt(variance + norm_eps)
    norm_hs = (hs_f32 * rstd).to(hs.dtype) * norm_weight

    logits = F.linear(norm_hs, lm_head_weight)  # (N, V_out)
    q = F.softmax(logits.float(), dim=-1)

    # Acceptance rate per position
    alpha = torch.min(tp, q).sum(-1)  # (N,)
    loss = -torch.log(alpha.clamp(min=1e-8)).mean()

    acc = (logits.argmax(-1) == tp.argmax(-1)).float().mean()

    return loss, acc, alpha.mean()


@torch.compile(dynamic=None)
def compiled_lk_lambda_loss(
    prenorm_hidden_states_flat,
    target_p_flat,
    valid_idx,
    norm_weight,
    lm_head_weight,
    norm_eps,
    eta,
):
    """LK^λ loss: λ·KL(p‖q) + (1-λ)·TV(p,q) where λ = exp(-η·sg[α])."""
    hs = prenorm_hidden_states_flat.index_select(0, valid_idx)
    tp = target_p_flat.index_select(0, valid_idx)

    # RMSNorm
    hs_f32 = hs.float()
    variance = hs_f32.pow(2).mean(-1, keepdim=True)
    rstd = torch.rsqrt(variance + norm_eps)
    norm_hs = (hs_f32 * rstd).to(hs.dtype) * norm_weight

    logits = F.linear(norm_hs, lm_head_weight)  # (N, V_out)
    q = F.softmax(logits.float(), dim=-1)
    log_q = F.log_softmax(logits.float(), dim=-1)

    # Acceptance rate (stop-gradient for λ computation)
    alpha = torch.min(tp, q).sum(-1)  # (N,)
    lam = torch.exp(-eta * alpha.detach())  # (N,)

    # KL(p‖q) per position
    kl = F.kl_div(log_q, tp, reduction="none").sum(-1)  # (N,)

    # TV(p,q) per position
    tv = 0.5 * (tp - q).abs().sum(-1)  # (N,)

    # Combined loss
    loss = (lam * kl + (1.0 - lam) * tv).mean()

    acc = (logits.argmax(-1) == tp.argmax(-1)).float().mean()

    return loss, acc, alpha.mean()


@torch.compile(dynamic=None)
def compiled_forward_kl_loss_from_hs(
    prenorm_hidden_states_flat,
    target_hidden_states_flat,
    valid_idx,
    norm_weight,
    lm_head_weight,
    target_lm_head_weight,
    norm_eps,
):
    """torch.compile'd index_select + target softmax + RMSNorm + lm_head + Forward KL loss.

    Like compiled_forward_kl_loss but takes full (B*T, ...) flat tensors and
    performs index_select inside the compiled graph.  This lets the compiler
    fuse the gather with subsequent ops, avoiding a separate (N, V_full) copy
    outside the compiled region.

    Used for the non-pruning (LazyTarget) path where V_full is large.
    """
    hs = prenorm_hidden_states_flat.index_select(0, valid_idx)
    ths = target_hidden_states_flat.index_select(0, valid_idx)

    # Target probs (detached weights → no grad flows through target)
    tp = F.softmax(F.linear(ths, target_lm_head_weight).float(), dim=-1)

    # RMSNorm
    hs_f32 = hs.float()
    variance = hs_f32.pow(2).mean(-1, keepdim=True)
    rstd = torch.rsqrt(variance + norm_eps)
    norm_hs = (hs_f32 * rstd).to(hs.dtype) * norm_weight

    logits = F.linear(norm_hs, lm_head_weight)

    # Forward KL loss
    log_p = F.log_softmax(logits.float(), dim=-1)
    loss = -(tp * log_p).sum(-1).mean()

    # Accuracy
    acc = (logits.argmax(-1) == tp.argmax(-1)).float().mean()

    return loss, acc


@torch.compile(dynamic=None)
def compiled_lk_alpha_loss_from_hs(
    prenorm_hidden_states_flat,
    target_hidden_states_flat,
    valid_idx,
    norm_weight,
    lm_head_weight,
    target_lm_head_weight,
    norm_eps,
):
    """LK^α loss from hidden states (LazyTarget path)."""
    hs = prenorm_hidden_states_flat.index_select(0, valid_idx)
    ths = target_hidden_states_flat.index_select(0, valid_idx)

    # Target probs
    tp = F.softmax(F.linear(ths, target_lm_head_weight).float(), dim=-1)

    # RMSNorm
    hs_f32 = hs.float()
    variance = hs_f32.pow(2).mean(-1, keepdim=True)
    rstd = torch.rsqrt(variance + norm_eps)
    norm_hs = (hs_f32 * rstd).to(hs.dtype) * norm_weight

    logits = F.linear(norm_hs, lm_head_weight)
    q = F.softmax(logits.float(), dim=-1)

    alpha = torch.min(tp, q).sum(-1)
    loss = -torch.log(alpha.clamp(min=1e-8)).mean()

    acc = (logits.argmax(-1) == tp.argmax(-1)).float().mean()

    return loss, acc, alpha.mean()


@torch.compile(dynamic=None)
def compiled_lk_lambda_loss_from_hs(
    prenorm_hidden_states_flat,
    target_hidden_states_flat,
    valid_idx,
    norm_weight,
    lm_head_weight,
    target_lm_head_weight,
    norm_eps,
    eta,
):
    """LK^λ loss from hidden states (LazyTarget path)."""
    hs = prenorm_hidden_states_flat.index_select(0, valid_idx)
    ths = target_hidden_states_flat.index_select(0, valid_idx)

    # Target probs
    tp = F.softmax(F.linear(ths, target_lm_head_weight).float(), dim=-1)

    # RMSNorm
    hs_f32 = hs.float()
    variance = hs_f32.pow(2).mean(-1, keepdim=True)
    rstd = torch.rsqrt(variance + norm_eps)
    norm_hs = (hs_f32 * rstd).to(hs.dtype) * norm_weight

    logits = F.linear(norm_hs, lm_head_weight)
    q = F.softmax(logits.float(), dim=-1)
    log_q = F.log_softmax(logits.float(), dim=-1)

    alpha = torch.min(tp, q).sum(-1)
    lam = torch.exp(-eta * alpha.detach())

    kl = F.kl_div(log_q, tp, reduction="none").sum(-1)
    tv = 0.5 * (tp - q).abs().sum(-1)

    loss = (lam * kl + (1.0 - lam) * tv).mean()

    acc = (logits.argmax(-1) == tp.argmax(-1)).float().mean()

    return loss, acc, alpha.mean()
