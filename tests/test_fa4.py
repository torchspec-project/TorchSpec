import unittest

import torch
import torch._dynamo as dynamo
from transformers import LlamaConfig

from tests.utils import norm_tensor
from torchspec.models.draft.llama3_eagle import (
    LlamaFlashAttentionMasked,
    LlamaFlexAttention,
    _flash_attn_func,
    _has_cute_dsl,
)

dynamo.config.recompile_limit = 64
TTT_LENGTH = 7
torch.manual_seed(0)

_has_flash_attn = _flash_attn_func is not None


@unittest.skipUnless(
    _has_flash_attn and _has_cute_dsl,
    "flash_attn.cute or cutlass DSL not installed",
)
class TestFlashAttentionMasked(unittest.TestCase):
    """Compare LlamaFlashAttentionMasked against LlamaFlexAttention.

    LlamaFlashAttentionMasked concatenates all KV blocks and passes the full
    EAGLE3 attention pattern to a single flash_attn kernel via mask_mod,
    eliminating the nested logsumexp that causes q/k_proj gradient errors in
    LlamaFlashAttention, so the gradient tolerance here should be tight.
    """

    def setUp(self):
        torch.manual_seed(0)
        self.config_dict = {
            "hidden_size": 512,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-05,
            "vocab_size": 32000,
            "intermediate_size": 1376,
            "hidden_act": "silu",
            "num_hidden_layers": 1,
            "torch_dtype": "bfloat16",
        }
        self.config = LlamaConfig(**self.config_dict)
        self.seq_lengths = [128, 256, 512, 1024, 2048]
        self.dtype = torch.bfloat16
        # Masked version uses a single kernel so gradients should be tight.
        self.fwd_atol = 2e-2
        self.fwd_rtol = 2e-2
        self.bwd_atol = 2e-2
        self.bwd_rtol = 2e-2

    def _make_modules(self):
        flex_attn = LlamaFlexAttention(self.config).to("cuda").to(self.dtype)
        masked_attn = LlamaFlashAttentionMasked(self.config).to("cuda").to(self.dtype)
        with torch.no_grad():
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                getattr(masked_attn, proj).weight.copy_(getattr(flex_attn, proj).weight)
        return flex_attn, masked_attn

    def test_forward_pass_comparison(self):
        for seq_len in self.seq_lengths:
            with self.subTest(seq_len=seq_len):
                self._test_forward(seq_len)

    def test_forward_pass_non_aligned(self):
        """128-snapping pad/slice path: seq_len not a multiple of 128."""
        for seq_len in [120, 200, 400]:
            with self.subTest(seq_len=seq_len):
                self._test_forward(seq_len)

    def _test_forward(self, seq_len):
        flex_attn, masked_attn = self._make_modules()
        flex_attn.eval()
        masked_attn.eval()

        batch_size = 2
        hidden_size = self.config.hidden_size * 2
        position_ids = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1).to("cuda")
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool).to("cuda")

        flex_cache_keys, flex_cache_values = None, None
        masked_cache_keys, masked_cache_values = None, None

        for _ in range(TTT_LENGTH):
            hidden_states = norm_tensor(
                (batch_size, seq_len, hidden_size), device="cuda", dtype=self.dtype
            )
            with torch.no_grad():
                flex_out, flex_cache_keys, flex_cache_values = flex_attn(
                    hidden_states=hidden_states.clone(),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    cache_keys=flex_cache_keys,
                    cache_values=flex_cache_values,
                    use_cache=True,
                )
                masked_out, masked_cache_keys, masked_cache_values = masked_attn(
                    hidden_states=hidden_states.clone(),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    cache_keys=masked_cache_keys,
                    cache_values=masked_cache_values,
                    use_cache=True,
                )
            torch.testing.assert_close(flex_out, masked_out, atol=self.fwd_atol, rtol=self.fwd_rtol)

    def test_backward_pass_gradient_comparison(self):
        for seq_len in self.seq_lengths:
            with self.subTest(seq_len=seq_len):
                self._test_backward(seq_len)

    def test_backward_pass_non_aligned(self):
        """128-snapping pad/slice path: gradients correct for non-aligned seq_len."""
        for seq_len in [120, 200, 400]:
            with self.subTest(seq_len=seq_len):
                self._test_backward(seq_len)

    def _test_backward(self, seq_len):
        from torchspec.utils.tensor import padding

        flex_attn, masked_attn = self._make_modules()

        batch_size = 2
        hidden_size = self.config.hidden_size * 2
        position_ids = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1).to("cuda")
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool).to("cuda")

        flex_cache_keys, flex_cache_values = None, None
        masked_cache_keys, masked_cache_values = None, None
        loss_mask = torch.ones(batch_size, seq_len, dtype=self.dtype, device="cuda")

        flex_losses, masked_losses = [], []
        hidden_list = [
            norm_tensor((batch_size, seq_len, hidden_size), device="cuda", dtype=self.dtype)
            for _ in range(TTT_LENGTH)
        ]

        for idx in range(TTT_LENGTH):
            flex_out, flex_cache_keys, flex_cache_values = flex_attn(
                hidden_states=hidden_list[idx].clone(),
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_keys=flex_cache_keys,
                cache_values=flex_cache_values,
                use_cache=True,
            )
            masked_out, masked_cache_keys, masked_cache_values = masked_attn(
                hidden_states=hidden_list[idx].clone(),
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_keys=masked_cache_keys,
                cache_values=masked_cache_values,
                use_cache=True,
            )
            flex_losses.append((flex_out * loss_mask[..., None]).sum().mean())
            masked_losses.append((masked_out * loss_mask[..., None]).sum().mean())
            if idx < TTT_LENGTH - 1:
                loss_mask = padding(loss_mask, left=False)

        (sum(flex_losses) / TTT_LENGTH).backward()
        (sum(masked_losses) / TTT_LENGTH).backward()

        for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            torch.testing.assert_close(
                getattr(flex_attn, proj_name).weight.grad,
                getattr(masked_attn, proj_name).weight.grad,
                atol=self.bwd_atol,
                rtol=self.bwd_rtol,
                msg=f"{proj_name} grad mismatch at seq_len={seq_len}",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
