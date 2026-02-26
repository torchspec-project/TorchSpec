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

"""
Sgl Ray actor engine for distributed deployment across multiple nodes.

This wraps sgl.Engine (patched sglang) as a Ray actor with placement group support,
parallel to HFEngine. Uses sglang's spec_training mode with mooncake storage.

Accepts pre-tokenized input_ids and loss_mask from batch preprocessing.
"""

import os
import socket
from typing import Any

import ray
import sglang as sgl
import torch

from torchspec.inference.engine.base import InferenceEngine
from torchspec.ray.ray_actor import RayActor
from torchspec.utils.logging import logger
from torchspec.utils.misc import get_default_eagle3_aux_layer_ids, get_free_port

# Keys managed by TorchSpec that extra_args must not override.
_PROTECTED_ENGINE_KEYS = frozenset(
    {
        "model_path",
        "trust_remote_code",
        "disable_radix_cache",
        "enable_return_hidden_states",
        "enable_aux_hidden_states",
        "aux_hidden_state_layer_ids",
        "enable_spec_training_mooncake",
        "chunked_prefill_size",
        "disable_cuda_graph",
        "tp_size",
        "pp_size",
        "base_gpu_id",
        "gpu_id_step",
        "mem_fraction_static",
        "port",
        "nccl_port",
        "nnodes",
        "node_rank",
        "dist_init_addr",
        "dist_timeout",
    }
)


class SglEngine(InferenceEngine, RayActor):
    """Ray actor wrapper for sgl.Engine with distributed deployment support.

    Uses patched sglang's spec_training mode to generate training data and store
    it in mooncake. Returns mooncake keys instead of tensors for efficient
    distributed training.

    Accepts pre-tokenized input_ids and loss_mask instead of raw prompts.
    """

    def __init__(
        self,
        args,
        rank: int,
        base_gpu_id: int | None = None,
        num_gpus_per_engine: int = 1,
        node_rank: int = 0,
    ):
        """Store configuration but don't load model yet.

        Args:
            args: Configuration arguments.
            rank: Engine rank for distributed setup.
            base_gpu_id: Base GPU ID from placement group.
            num_gpus_per_engine: Number of GPUs this engine uses (for TP).
            node_rank: Node rank for multi-node TP (0 = head).
        """
        self.args = args
        self.rank = rank
        self.base_gpu_id = base_gpu_id
        self.num_gpus_per_engine = num_gpus_per_engine
        self.node_rank = node_rank
        self._engine = None
        self._mooncake_config = None
        self._mooncake_store = None
        self._hidden_size = None
        self.local_gpu_id = None

    def init(self, mooncake_config=None, dist_init_addr: str | None = None) -> None:
        """Initialize the sgl.Engine on the allocated GPU.

        This is called after the Ray actor is scheduled on a node.

        Args:
            mooncake_config: MooncakeConfig object for distributed storage.
            dist_init_addr: Address for sglang cross-node NCCL init (auto-negotiated by factory).
        """
        if self.base_gpu_id is not None:
            self.local_gpu_id = self.setup_gpu(self.base_gpu_id)
            logger.info(
                f"SglEngine rank {self.rank}: base_gpu_id={self.base_gpu_id}, "
                f"using local GPU {self.local_gpu_id}"
            )

        self._mooncake_config = mooncake_config
        if mooncake_config is not None:
            logger.info(f"SglEngine rank {self.rank}: received mooncake_config={mooncake_config}")

            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
                s.close()
            except Exception:
                local_ip = "localhost"
                logger.warning(
                    f"SglEngine rank {self.rank}: failed to get local IP, using localhost"
                )

            mooncake_config.local_hostname = local_ip
            mooncake_config.export_env()

            logger.info(
                f"SglEngine rank {self.rank}: mooncake env vars set - "
                f"local_hostname={local_ip}, "
                f"master_server={mooncake_config.master_server_address}, "
                f"metadata_server={mooncake_config.metadata_server}, "
                f"protocol={mooncake_config.protocol}, "
                f"device_name={mooncake_config.device_name}"
            )

            from torchspec.transfer.mooncake.utils import (
                check_mooncake_master_available,
            )

            check_mooncake_master_available(
                mooncake_config.master_server_address,
                mooncake_config.metadata_server,
            )

        # Get configuration
        mem_fraction = getattr(self.args, "sglang_mem_fraction_static", 0.8)
        pp_size = getattr(self.args, "sglang_pp_size", 1)
        if self.args.aux_hidden_states_layers is not None:
            self.aux_hidden_state_layer_ids = self.args.aux_hidden_states_layers
        else:
            self.aux_hidden_state_layer_ids = get_default_eagle3_aux_layer_ids(
                self.args.target_model_path
            )
            if self.rank == 0:
                logger.info(
                    f"Using default aux hidden state layer ids: {self.aux_hidden_state_layer_ids}"
                )

        assert pp_size == 1, f"pp_size must be 1, got {pp_size}"

        # tp_size: sglang tp_size is the TOTAL TP degree across all nodes
        # e.g. 3 nodes × 8 GPUs → tp_size=24
        nnodes = getattr(self.args, "sglang_nnodes", 1)
        tp_size = nnodes * self.num_gpus_per_engine
        configured_tp_size = getattr(self.args, "sglang_tp_size", None)
        if configured_tp_size is not None:
            assert configured_tp_size == tp_size, (
                f"sglang_tp_size ({configured_tp_size}) must equal "
                f"nnodes * num_gpus_per_engine ({nnodes} * {self.num_gpus_per_engine} "
                f"= {tp_size})"
            )

        logger.info(
            f"SglEngine rank {self.rank}: BEFORE init - "
            f"base_gpu_id={self.base_gpu_id}, num_gpus={self.num_gpus_per_engine}, "
            f"tp_size={tp_size}, pp_size={pp_size}, nnodes={nnodes}, node_rank={self.node_rank}, "
            f"aux_hidden_state_layer_ids={self.aux_hidden_state_layer_ids}"
        )

        # Build engine kwargs - base config for spec_training mode
        engine_kwargs = {
            "model_path": self.args.target_model_path,
            "disable_cuda_graph": True,  # Always disabled, prefill-only.
            "disable_radix_cache": True,  # IMPORTANT: radix cache interferes with hidden states capture
            "enable_return_hidden_states": True,
            "enable_aux_hidden_states": True,
            "aux_hidden_state_layer_ids": self.aux_hidden_state_layer_ids,
            "enable_spec_training_mooncake": True,  # Let sglang store to mooncake
            "tp_size": tp_size,
            "pp_size": pp_size,
            "base_gpu_id": self.local_gpu_id,
            "gpu_id_step": 1,  # Use consecutive GPUs for TP
            "mem_fraction_static": mem_fraction,
            "trust_remote_code": getattr(self.args, "trust_remote_code", True),
            "log_level": getattr(self.args, "sglang_log_level", "warning"),
            "chunked_prefill_size": -1,  # Disable chunked prefill
            "log_requests": getattr(self.args, "sglang_log_requests", False),
            "log_requests_level": getattr(self.args, "sglang_log_requests_level", 0),
        }

        # Optional sglang fields - only add if set (avoids passing None to sgl.Engine)
        _optional_sglang_fields = {
            "quantization": getattr(self.args, "sglang_quantization", None),
            "kv_cache_dtype": getattr(self.args, "sglang_kv_cache_dtype", None),
            "moe_runner_backend": getattr(self.args, "sglang_moe_runner_backend", None),
            "model_loader_extra_config": getattr(
                self.args, "sglang_model_loader_extra_config", None
            ),
            "attention_backend": getattr(self.args, "sglang_attention_backend", None),
            "context_length": getattr(self.args, "sglang_context_length", None),
        }
        for k, v in _optional_sglang_fields.items():
            if v is not None:
                engine_kwargs[k] = v

        _optional_sglang_bool_fields = {
            "disable_flashinfer_autotune": "sglang_disable_flashinfer_autotune",
            "enable_multimodal": "sglang_enable_multimodal",
        }
        for engine_key, args_key in _optional_sglang_bool_fields.items():
            val = getattr(self.args, args_key, False)
            if val:
                engine_kwargs[engine_key] = val

        # Power-user passthrough: extra_args are forwarded as-is to sgl.Engine,
        # except for keys that TorchSpec manages internally.
        extra_args = getattr(self.args, "sglang_extra_args", None)
        if extra_args:
            extra = dict(extra_args) if not isinstance(extra_args, dict) else extra_args
            blocked = extra.keys() & _PROTECTED_ENGINE_KEYS
            if blocked:
                logger.warning(
                    f"sglang extra_args contains protected keys that will be ignored: "
                    f"{sorted(blocked)}. These are managed internally by TorchSpec."
                )
                extra = {k: v for k, v in extra.items() if k not in _PROTECTED_ENGINE_KEYS}
            engine_kwargs.update(extra)

        # Avoid port collisions when multiple engines share the same node
        engine_kwargs["port"] = get_free_port()
        engine_kwargs["nccl_port"] = get_free_port()

        # Multi-node TP support — always set nnodes/node_rank
        engine_kwargs["nnodes"] = nnodes
        engine_kwargs["node_rank"] = self.node_rank
        if nnodes > 1:
            # dist_init_addr: prefer parameter (auto-negotiated by factory), fallback to config
            effective_addr = dist_init_addr or getattr(self.args, "sglang_dist_init_addr", None)
            if effective_addr:
                engine_kwargs["dist_init_addr"] = effective_addr
            sglang_dist_timeout = getattr(self.args, "sglang_dist_timeout", 60)
            engine_kwargs["dist_timeout"] = sglang_dist_timeout
            logger.info(
                f"SglEngine rank {self.rank}: multi-node TP enabled - "
                f"nnodes={nnodes}, node_rank={self.node_rank}, "
                f"dist_init_addr={effective_addr}, dist_timeout={sglang_dist_timeout}"
            )

        # Worker nodes (node_rank >= 1) block forever in _launch_subprocesses
        # unless this env var is set. See sglang's engine.py.
        if self.node_rank >= 1:
            os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"

        # Initialize sgl.Engine
        self._engine = sgl.Engine(**engine_kwargs)

        # Get hidden size from model config
        self._hidden_size = self._get_hidden_size_from_engine()

        if hasattr(self._engine, "model_config"):
            model_config = self._engine.model_config
            logger.info(
                f"SglEngine rank {self.rank}: model_config attributes: "
                f"{[attr for attr in dir(model_config) if not attr.startswith('_')]}"
            )
            logger.info(
                f"SglEngine rank {self.rank}: model_config.hidden_size = "
                f"{getattr(model_config, 'hidden_size', 'NOT_FOUND')}"
            )
            logger.info(
                f"SglEngine rank {self.rank}: model_config.num_hidden_layers = "
                f"{getattr(model_config, 'num_hidden_layers', 'NOT_FOUND')}"
            )

        logger.info(
            f"SglEngine rank {self.rank}: initialized from {self.args.target_model_path} "
            f"(tp_size={tp_size}, aux_layers={self.aux_hidden_state_layer_ids}, "
            f"hidden_size={self._hidden_size})"
        )

    @staticmethod
    def _extract_image_data(multimodal_inputs: list[dict] | None) -> list | None:
        """Extract image_data list from multimodal_inputs for sgl.Engine."""
        if not multimodal_inputs:
            return None
        image_data = []
        for mm_input in multimodal_inputs:
            if mm_input and mm_input.get("images"):
                image_data.append(mm_input["images"])
            else:
                image_data.append(None)
        return image_data

    def generate(
        self,
        data_id: str | list[str],
        input_ids_ref: ray.ObjectRef | list[torch.Tensor] | None = None,
        packed_loss_mask_list: list[str] | None = None,
        formatted_prompts: list[str] | None = None,
        return_last_hidden_states: bool = False,
        return_logits: bool = True,
        multimodal_inputs: list[dict] | None = None,
    ) -> list[dict[str, Any]]:
        """Generate training data using spec_training mode.

        Accepts either pre-tokenized input_ids or formatted prompt strings.
        Exactly one of input_ids_ref or formatted_prompts must be set.

        Args:
            data_id: Data ID(s) for the batch.
            input_ids_ref: Ray ObjectRef or list of input_ids tensors.
            packed_loss_mask_list: List of packed loss_mask strings (e.g. "2,3,2,2,1").
            formatted_prompts: List of already chat-template-formatted prompt strings.
            return_last_hidden_states: Whether to return last hidden states (ignored, always in mooncake).
            return_logits: Whether to return target logits (ignored, always in mooncake).

        Returns:
            List of dicts with mooncake_key and tensor metadata.
        """
        if self._engine is None:
            raise RuntimeError("SglEngine not initialized. Call init() first.")

        if (input_ids_ref is None) == (formatted_prompts is None):
            raise ValueError("Exactly one of input_ids_ref or formatted_prompts must be set")

        use_prompts = formatted_prompts is not None

        if use_prompts:
            batch_size = len(formatted_prompts)
        else:
            if isinstance(input_ids_ref, ray.ObjectRef):
                input_ids_list = ray.get(input_ids_ref)
            else:
                input_ids_list = input_ids_ref
            batch_size = len(input_ids_list)

        if isinstance(data_id, str):
            data_ids = [f"{data_id}_{i}" for i in range(batch_size)]
        elif len(data_id) == batch_size:
            data_ids = data_id
        else:
            raise ValueError(
                f"data_id length {len(data_id)} does not match batch size {batch_size}"
            )

        if use_prompts:
            logger.debug(
                f"SglEngine rank {self.rank}: prompt mode processing data_ids={data_ids}, "
                f"num_prompts={len(formatted_prompts)}"
            )
            # loss mask is computed during training time based on input_ids.
            engine_kwargs = {
                "prompt": formatted_prompts,
                "spec_training_data_id": data_ids,
                "sampling_params": {"max_new_tokens": 0},
                "return_hidden_states": True,
            }
        else:
            input_ids_list_of_lists = []
            for ids in input_ids_list:
                if ids.dim() == 2 and ids.shape[0] == 1:
                    ids = ids.squeeze(0)
                elif ids.dim() > 2:
                    raise ValueError(f"Unexpected input_ids shape: {ids.shape}")
                input_ids_list_of_lists.append(ids.tolist())

            logger.debug(
                f"SglEngine rank {self.rank}: processing data_ids={data_ids}, "
                f"seq_lengths={[len(ids) for ids in input_ids_list_of_lists]}"
            )
            engine_kwargs = {
                "input_ids": input_ids_list_of_lists,
                "spec_training_data_id": data_ids,
                "packed_loss_mask": packed_loss_mask_list,
                "sampling_params": {"max_new_tokens": 0},
                "return_hidden_states": True,
            }

        image_data = self._extract_image_data(multimodal_inputs)
        if image_data is not None:
            engine_kwargs["image_data"] = image_data

        results = self._engine.generate(**engine_kwargs)

        # Extract mooncake keys and construct shapes based on actual sequence length
        outputs = []
        for i, result in enumerate(results):
            store_keys = result["meta_info"].get("spec_training_mooncake_store_keys", [])
            if not store_keys:
                logger.error(
                    f"SglEngine rank {self.rank}: ERROR: No mooncake keys returned for "
                    f"data_id={data_ids[i]}. Training may be corrupted."
                )
                continue

            logger.debug(
                f"SglEngine rank {self.rank}: result meta_info keys: {list(result['meta_info'].keys())}"
            )

            for key in store_keys:
                seq_len = result["meta_info"].get(
                    "prompt_tokens",
                    len(formatted_prompts[i]) if use_prompts else len(input_ids_list_of_lists[i]),
                )

                tensor_shapes = self._get_tensor_shapes(seq_len)
                logger.debug(
                    f"SglEngine rank {self.rank}: mooncake_key={key}, seq_len={seq_len}, "
                    f"tensor_shapes={tensor_shapes}"
                )

                output = {
                    "mooncake_key": key,
                    "tensor_shapes": tensor_shapes,
                    "tensor_dtypes": self._get_tensor_dtypes(),
                }
                outputs.append(output)

        logger.debug(
            f"SglEngine rank {self.rank}: generated {len(outputs)} mooncake keys "
            f"for data_ids={data_ids}"
        )
        return outputs

    def health_check(self, timeout: float = 5.0) -> bool:
        """Check if the engine is healthy."""
        return self._engine is not None

    def shutdown(self) -> None:
        """Clean up resources."""
        if self._mooncake_store is not None:
            self._mooncake_store.close()
            self._mooncake_store = None
        if self._engine is not None:
            self._engine.shutdown()
            self._engine = None
        logger.info(f"SglEngine rank {self.rank}: shutdown complete")

    def get_status(self) -> dict:
        """Get engine status."""
        return {
            "rank": self.rank,
            "initialized": self._engine is not None,
            "base_gpu_id": self.base_gpu_id,
            "hidden_size": self._hidden_size,
        }

    def _get_hidden_size_from_engine(self) -> int:
        """Get hidden size from the model config using AutoConfig.

        Returns:
            Hidden size dimension.
        """
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            self.args.target_model_path,
            trust_remote_code=getattr(self.args, "trust_remote_code", True),
        )
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            text_config = getattr(config, "text_config", None)
            if text_config is not None:
                hidden_size = getattr(text_config, "hidden_size", None)
        return hidden_size

    def _get_tensor_shapes(self, seq_len: int) -> dict:
        """Get tensor shapes for mooncake metadata.

        Args:
            seq_len: Sequence length for this sample (input length, not generated length).

        Returns:
            Dict mapping tensor names to shapes.
        """
        aux_hidden_state_layer_ids = self.aux_hidden_state_layer_ids
        num_aux_layers = len(aux_hidden_state_layer_ids)
        if self._hidden_size is None:
            raise ValueError(
                f"SglEngine rank {self.rank}: hidden_size not initialized. Call init() first."
            )
        hidden_size = self._hidden_size

        # Concatenated hidden states from all aux layers
        # Sglang concatenates hidden states from all specified layers along the last dimension
        concat_hidden_size = num_aux_layers * hidden_size

        # IMPORTANT: Sglang stores tensors WITHOUT batch dimension in mooncake
        # We must request the SAME shapes that sglang stored, otherwise we get size mismatch
        # The collator will add batch dimension when needed
        return {
            "hidden_states": (seq_len, concat_hidden_size),  # 2D without batch dim
            "input_ids": (seq_len,),  # 1D without batch dim
            "last_hidden_states": (seq_len, hidden_size),  # 2D without batch dim
        }

    def _get_tensor_dtypes(self) -> dict:
        """Get tensor dtypes for mooncake metadata."""
        return {
            "hidden_states": torch.bfloat16,
            "input_ids": torch.long,
            "last_hidden_states": torch.bfloat16,
        }
