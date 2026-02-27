"""Tests for saving resolved config to output_dir."""

import os

import pytest
from omegaconf import MISSING, OmegaConf

from torchspec.config.train_config import Config, load_config, save_config

# ---------------------------------------------------------------------------
# Unit tests for save_config / load_config round-trip
# ---------------------------------------------------------------------------


def test_save_resolved_config_basic(tmp_path):
    """Config with output_dir saves to output_dir/config.yaml and round-trips correctly."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    config = OmegaConf.structured(Config)
    config.output_dir = str(output_dir)
    config.model.target_model_path = "test-model"
    config.dataset.train_data_path = "test-data"
    config.training.micro_batch_size = 8

    save_path = str(output_dir / "config.yaml")
    save_config(config, save_path)

    reloaded = load_config(config_path=save_path)
    assert reloaded.output_dir == str(output_dir)
    assert reloaded.model.target_model_path == "test-model"
    assert reloaded.dataset.train_data_path == "test-data"
    assert reloaded.training.micro_batch_size == 8


def test_save_resolved_config_with_cli_overrides(tmp_path):
    """CLI dotlist overrides are persisted in the saved config."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    base = OmegaConf.structured(Config)
    base.output_dir = str(output_dir)
    base.model.target_model_path = "base-model"
    base.dataset.train_data_path = "base-data"

    config = load_config(
        cli_args=["training.micro_batch_size=16", "training.learning_rate=3e-5"],
        base_config=base,
    )

    save_path = str(output_dir / "config.yaml")
    save_config(config, save_path)

    reloaded = load_config(config_path=save_path)
    assert reloaded.training.micro_batch_size == 16
    assert reloaded.training.learning_rate == 3e-5
    assert reloaded.model.target_model_path == "base-model"


def test_save_resolved_config_empty_output_dir():
    """When output_dir is empty, the OmegaConf.select guard evaluates to falsy."""
    config = OmegaConf.structured(Config)
    config.output_dir = ""

    assert not OmegaConf.select(config, "output_dir")


def test_save_resolved_config_missing_output_dir():
    """When output_dir is MISSING (???), OmegaConf.select returns None."""
    config = OmegaConf.structured(Config)
    config.output_dir = MISSING

    assert OmegaConf.select(config, "output_dir") is None


def test_save_config_overwrites_existing(tmp_path):
    """Saving config twice overwrites the previous file."""
    config = OmegaConf.structured(Config)
    config.output_dir = str(tmp_path)
    config.training.micro_batch_size = 4

    save_path = str(tmp_path / "config.yaml")
    save_config(config, save_path)

    config.training.micro_batch_size = 8
    save_config(config, save_path)

    reloaded = load_config(config_path=save_path)
    assert reloaded.training.micro_batch_size == 8


# ---------------------------------------------------------------------------
# Integration tests for parse_config()
# ---------------------------------------------------------------------------


def _write_minimal_config(path, output_dir="", **overrides):
    """Write a minimal valid YAML config to *path*."""
    lines = [
        "model:",
        "  target_model_path: test-model",
        "dataset:",
        "  train_data_path: test-data",
        "training:",
        "  training_num_gpus_per_node: 1",
        "  training_num_nodes: 1",
        f"output_dir: '{output_dir}'",
    ]
    for dotkey, val in overrides.items():
        # Only supports top-level overrides for simplicity
        lines.append(f"{dotkey}: {val}")
    path.write_text("\n".join(lines))


def test_parse_config_saves_to_output_dir(tmp_path, monkeypatch):
    """parse_config() saves config.yaml into output_dir and returns valid flat_args."""
    output_dir = tmp_path / "out"
    config_file = tmp_path / "test.yaml"
    _write_minimal_config(config_file, output_dir=str(output_dir))

    monkeypatch.setattr("sys.argv", ["prog", "--config", str(config_file)])

    from torchspec.train_entry import parse_config

    args = parse_config()

    assert (output_dir / "config.yaml").exists()
    assert args.target_model_path == "test-model"

    reloaded = load_config(config_path=str(output_dir / "config.yaml"))
    assert reloaded.output_dir == str(output_dir)


def test_parse_config_creates_nested_dir(tmp_path, monkeypatch):
    """parse_config() creates nested output_dir automatically."""
    output_dir = tmp_path / "a" / "b" / "c"
    config_file = tmp_path / "test.yaml"
    _write_minimal_config(config_file, output_dir=str(output_dir))

    monkeypatch.setattr("sys.argv", ["prog", "--config", str(config_file)])

    from torchspec.train_entry import parse_config

    parse_config()

    assert (output_dir / "config.yaml").exists()


def test_parse_config_no_save_when_empty_output_dir(tmp_path, monkeypatch):
    """parse_config() does not save config when output_dir is empty."""
    config_file = tmp_path / "test.yaml"
    _write_minimal_config(config_file, output_dir="")

    monkeypatch.setattr("sys.argv", ["prog", "--config", str(config_file)])

    from torchspec.train_entry import parse_config

    args = parse_config()

    assert args.target_model_path == "test-model"
    # No config.yaml should be created anywhere in tmp_path
    assert not list(tmp_path.glob("**/config.yaml"))


def test_parse_config_print_config_only_exits(tmp_path, monkeypatch):
    """--print-config-only prints config and exits without saving."""
    output_dir = tmp_path / "out"
    config_file = tmp_path / "test.yaml"
    _write_minimal_config(config_file, output_dir=str(output_dir))

    monkeypatch.setattr("sys.argv", ["prog", "--config", str(config_file), "--print-config-only"])

    from torchspec.train_entry import parse_config

    with pytest.raises(SystemExit) as exc_info:
        parse_config()

    assert exc_info.value.code == 0
    # Config should NOT be saved because exit happens before save
    assert not output_dir.exists()


def test_parse_config_cli_overrides_saved(tmp_path, monkeypatch):
    """CLI overrides are reflected in the saved config.yaml."""
    output_dir = tmp_path / "out"
    config_file = tmp_path / "test.yaml"
    _write_minimal_config(config_file, output_dir=str(output_dir))

    monkeypatch.setattr(
        "sys.argv",
        ["prog", "--config", str(config_file), "training.micro_batch_size=32"],
    )

    from torchspec.train_entry import parse_config

    args = parse_config()

    assert args.micro_batch_size == 32

    reloaded = load_config(config_path=str(output_dir / "config.yaml"))
    assert reloaded.training.micro_batch_size == 32


def test_parse_config_survives_readonly_output_dir(tmp_path, monkeypatch):
    """parse_config() logs warning but does not crash when output_dir is not writable."""
    output_dir = tmp_path / "readonly"
    output_dir.mkdir()
    os.chmod(str(output_dir), 0o444)

    config_file = tmp_path / "test.yaml"
    _write_minimal_config(config_file, output_dir=str(output_dir))

    monkeypatch.setattr("sys.argv", ["prog", "--config", str(config_file)])

    from torchspec.train_entry import parse_config

    try:
        args = parse_config()
        # Should succeed — config save failure is non-fatal
        assert args.target_model_path == "test-model"
    finally:
        os.chmod(str(output_dir), 0o755)
    # Verify no config.yaml was written (check after restoring permissions)
    assert not (output_dir / "config.yaml").exists()
