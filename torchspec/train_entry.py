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

"""Training entry point for Eagle3 speculative decoding."""

import argparse
import copy
import os
import sys

import ray
import yaml

from torchspec.controller import (
    auto_calculate_training_steps,
    build_mooncake_config,
    run_training_loop,
    setup_async_training_with_engines,
)
from torchspec.data.dataset import load_conversation_dataset
from torchspec.data.preprocessing import generate_vocab_mapping
from torchspec.inference import prepare_inference_engines
from torchspec.ray.placement_group import (
    allocate_train_group,
    create_placement_groups,
)
from torchspec.transfer.mooncake.utils import launch_mooncake_master
from torchspec.utils.logging import init_tracking, logger


def parse_nested_config():
    """Parse nested YAML config and convert to flat args.

    Supports configs with nested sections matching the Config dataclass:
    model, dataset, training, debug, inference, logging, mooncake, sglang.

    The nested config is flattened via config_to_flat_args(), with mooncake
    sections getting a name prefix (mooncake_*).
    """
    from torchspec.config.train_config import config_to_flat_args, load_config

    parser = argparse.ArgumentParser(description="Train with nested config")
    parser.add_argument(
        "--config", "-c", type=str, required=True, help="Path to nested YAML config"
    )
    parser.add_argument("--print-config", action="store_true", help="Print config and exit")

    args, unknown = parser.parse_known_args()

    config = load_config(config_path=args.config, cli_args=unknown if unknown else None)

    if args.print_config:
        from omegaconf import OmegaConf

        print(OmegaConf.to_yaml(config))
        sys.exit(0)

    flat_args = config_to_flat_args(config)

    flat_args.rank = 0
    flat_args.world_size = flat_args.training_num_nodes * flat_args.training_num_gpus_per_node

    defaults = {
        "colocate": False,
        "debug_train_only": False,
        "debug_inference_only": False,
        "dp_size": None,
        "save_debug_train_data": None,
    }
    for key, value in defaults.items():
        if not hasattr(flat_args, key) or getattr(flat_args, key) is None:
            setattr(flat_args, key, value)

    _resolve_batch_size(flat_args)

    return flat_args


def _resolve_batch_size(args):
    """Derive per_dp_rank_batch_size, dispatch_batch_size, and global_batch_size."""
    dp_size = (
        getattr(args, "dp_size", None) or args.training_num_nodes * args.training_num_gpus_per_node
    )
    sp_size = getattr(args, "sp_size", None)
    if sp_size is not None and sp_size != 1:
        raise NotImplementedError(f"Sequence parallel is not yet supported (got sp_size={sp_size})")
    sp_size = sp_size or 1
    accumulation_steps = getattr(args, "draft_accumulation_steps", 1)
    args.per_dp_rank_batch_size = args.micro_batch_size * sp_size
    args.dispatch_batch_size = args.per_dp_rank_batch_size * dp_size
    args.global_batch_size = args.dispatch_batch_size * accumulation_steps

    eval_mbs = getattr(args, "eval_micro_batch_size", None) or args.micro_batch_size
    args.eval_per_dp_rank_batch_size = eval_mbs * sp_size
    args.eval_dispatch_batch_size = args.eval_per_dp_rank_batch_size * dp_size


def _get_draft_model_config(args):
    """Resolve draft model config from args or auto-generate from target model."""
    from torchspec import AutoDraftModelConfig
    from torchspec.config.utils import generate_draft_model_config

    draft_config_path = getattr(args, "draft_model_config", None)
    if draft_config_path is not None:
        return AutoDraftModelConfig.from_file(draft_config_path)

    config_dict = generate_draft_model_config(
        target_model_path=args.target_model_path,
        cache_dir=getattr(args, "model_download_dir", None),
    )
    return AutoDraftModelConfig.from_dict(config_dict)


def _load_eval_dataset(args):
    """Load eval dataset using the same pipeline as training data."""
    eval_data_path = getattr(args, "eval_data_path", None)
    if not eval_data_path:
        return None

    eval_args = copy.copy(args)
    eval_args.train_data_path = eval_data_path
    eval_prompt_key = getattr(args, "eval_prompt_key", None)
    if eval_prompt_key:
        eval_args.prompt_key = eval_prompt_key
    eval_dataset = load_conversation_dataset(eval_args)
    logger.info(f"Loaded {len(eval_dataset)} eval samples from {eval_data_path}")
    return eval_dataset


def train_async_no_generation(args):
    """Entry point for Eagle3 distillation training without generation.

    Uses distributed HFEngine Ray actors with placement groups for multi-node deployment.
    Engines are scheduled across nodes and communicate via Ray.
    Each engine stores tensors in mooncake and returns keys to AsyncInferenceManager.

    Mooncake master is launched as a Ray actor to handle tensor storage coordination.
    """
    init_tracking(args)

    dataset = load_conversation_dataset(args)
    eval_dataset = _load_eval_dataset(args)

    auto_calculate_training_steps(args, len(dataset))

    draft_model_config = _get_draft_model_config(args)
    args.draft_model_config_obj = draft_model_config

    # Generate vocab mapping if vocab pruning is enabled
    vocab_mapping = None
    draft_vocab_size = getattr(draft_model_config, "draft_vocab_size", None)
    vocab_size = draft_model_config.vocab_size
    if draft_vocab_size is not None and draft_vocab_size != vocab_size:
        vocab_mapping = generate_vocab_mapping(
            prompts=dataset,
            target_vocab_size=vocab_size,
            draft_vocab_size=draft_vocab_size,
        )
        logger.info(
            f"Generated vocab mapping: d2t={vocab_mapping[0].shape}, t2d={vocab_mapping[1].shape}"
        )

    pgs = create_placement_groups(args)

    launch_mooncake_master(args)

    from torchspec.training.trainer_actor import TrainerActor

    mooncake_config = build_mooncake_config(args)

    # Phase 1: Create actors and fire init calls (non-blocking)
    train_group = allocate_train_group(
        args=args,
        num_nodes=args.training_num_nodes,
        num_gpus_per_node=args.training_num_gpus_per_node,
        pg=pgs["training"],
        training_class=TrainerActor,
    )
    train_init_refs = train_group.async_init(
        args, role="training", mooncake_config=mooncake_config, with_ref=False
    )

    inference_engines, engine_init_refs = prepare_inference_engines(
        args, pgs["inference"], mooncake_config
    )

    # Phase 2: Wait for all init to complete concurrently
    n_train = len(train_init_refs)
    logger.info(
        f"Waiting for {n_train} training actors and {len(engine_init_refs)} "
        f"inference engines to initialize in parallel..."
    )
    all_results = ray.get(train_init_refs + engine_init_refs)

    train_results = all_results[:n_train]
    assert len(set(train_results)) == 1
    logger.info(
        f"All {n_train} training actors and {len(engine_init_refs)} inference engines initialized"
    )

    if vocab_mapping is not None:
        train_group.set_vocab_buffers(*vocab_mapping)
        logger.info("Loaded vocab mapping into training actors")

    controller, inference_manager = setup_async_training_with_engines(
        args, train_group, mooncake_config, inference_engines
    )

    dataset_size = len(dataset)
    dataset_ref = ray.put(dataset)
    del dataset

    run_training_loop(
        args,
        dataset_ref,
        controller,
        inference_manager,
        train_group,
        inference_engines=inference_engines,
        eval_dataset=eval_dataset,
        dataset_size=dataset_size,
    )


def _detect_nested_config() -> bool:
    """Detect if using nested config format by checking for nested YAML structure."""
    for i, arg in enumerate(sys.argv[1:]):
        if arg in ("--config", "-c") and i + 1 < len(sys.argv) - 1:
            config_path = sys.argv[i + 2]
            if os.path.exists(config_path):
                with open(config_path) as f:
                    data = yaml.safe_load(f) or {}
                return any(key in data for key in ("model", "training", "inference"))
    return False


if __name__ == "__main__":
    if _detect_nested_config():
        logger.info("Detected nested config format, using parse_nested_config()")
        args = parse_nested_config()
    else:
        raise NotImplementedError(
            "Flat config format is not supported in this version. "
            "Please use nested YAML config with --config."
        )

    train_async_no_generation(args)
