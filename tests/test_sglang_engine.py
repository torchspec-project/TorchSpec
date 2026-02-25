"""Standalone integration script that tests Mooncake hidden states collection behavior."""

import os

import sglang as sgl
import torch
from transformers import AutoTokenizer

from torchspec.transfer.mooncake import EagleMooncakeStore, MooncakeConfig

os.environ["MOONCAKE_MASTER_HOST"] = "0.0.0.0"
os.environ["MOONCAKE_MASTER_PORT"] = "50051"
os.environ["MOONCAKE_METADATA_PORT"] = "8090"

if __name__ == "__main__":
    model_path = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    eos_token_id = tokenizer.eos_token_id

    input_ids_list = [
        [1, 2345, 6789],
        [100, 200, 300, 400],
        [500, 600],
    ]

    engine = sgl.Engine(
        model_path=model_path,
        disable_cuda_graph=True,
        enable_return_hidden_states=True,
        enable_aux_hidden_states=True,
        aux_hidden_state_layer_ids=[2, 4, 6],
        enable_spec_training_mooncake=True,
        log_level="info",
        tp_size=4,
    )

    results = engine.generate(
        input_ids=input_ids_list,
        spec_training_data_id=["data_id_1", "data_id_2", "data_id_3"],
        spec_training_prompt_length=[1, 2, 1],
        spec_training_response_length=[5, 10, 8],
        sampling_params={"max_new_tokens": 32},
        return_hidden_states=True,
    )

    print("=== Batch Results ===")
    all_keys = []
    seq_lens = []
    for i, result in enumerate(results):
        output_ids = result["output_ids"]
        hidden_states = result["meta_info"].get("hidden_states")
        mooncake_keys = result["meta_info"].get("spec_training_mooncake_store_keys")

        print(f"\n--- Request {i} ---")
        print(f"output_ids: {output_ids}")
        print(f"num tokens generated: {len(output_ids)}")
        print(f"spec_training_data_id: {result['meta_info'].get('spec_training_data_id')}")

        print(f"\n  Hidden states in meta_info: {hidden_states}")
        assert hidden_states is None, "hidden_states should be None when using mooncake"

        print(f"\n  Mooncake store keys: {mooncake_keys}")
        assert mooncake_keys and len(mooncake_keys) > 0, "mooncake_store_keys should not be empty"

        all_keys.extend(mooncake_keys)
        seq_lens.append(len(input_ids_list[i]))

        print(f"\n  All meta_info keys: {list(result['meta_info'].keys())}")

    print("\n=== Fetching data from Mooncake Store ===")
    mooncake_config = MooncakeConfig.from_env()
    mooncake_store = EagleMooncakeStore(mooncake_config)
    mooncake_store.setup(device="cuda")

    hidden_dim = 12288
    last_hidden_dim = 4096

    for i, key in enumerate(all_keys):
        seq_len = seq_lens[i]
        shapes = {
            "hidden_states": (seq_len, hidden_dim),
            "loss_mask": (seq_len,),
            "input_ids": (seq_len,),
            "last_hidden_states": (seq_len, last_hidden_dim),
        }
        dtypes = {
            "hidden_states": torch.bfloat16,
            "loss_mask": torch.long,
            "input_ids": torch.long,
            "last_hidden_states": torch.bfloat16,
        }

        data = mooncake_store.get(key, shapes=shapes, dtypes=dtypes, device="cuda")
        print(f"\n  Key: {key}")
        print(
            f"    hidden_states: shape={data.hidden_states.shape}, dtype={data.hidden_states.dtype}"
        )
        print(f"    loss_mask: {data.loss_mask.tolist()}")
        print(f"    input_ids: {data.input_ids.tolist()}")
        print(f"    last_hidden_states: shape={data.last_hidden_states.shape}")

    print("\n✓ Test completed - hidden states sent to mooncake and retrieved successfully")
    engine.shutdown()
