"""Tests for HFEngine Ray actor wrapper."""

from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
import torch


@dataclass
class MockArgs:
    """Mock args for HFEngine initialization."""

    target_model_path: str = "/fake/model/path"
    max_seq_length: int = 8192
    trust_remote_code: bool = True
    aux_hidden_states_layers: Optional[list] = None


def _import_hf_engine():
    """Import HFEngine, skipping test if dependencies unavailable."""
    try:
        from torchspec.inference.engine.hf_engine import HFEngine

        return HFEngine
    except ImportError as e:
        pytest.skip(f"HFEngine import failed (missing deps): {e}")


def _get_engine_module():
    """Get the hf_engine module, skipping test if unavailable."""
    try:
        import torchspec.inference.engine.hf_engine as engine_module

        return engine_module
    except ImportError as e:
        pytest.skip(f"HFEngine module import failed (missing deps): {e}")


class TestHFEngineInit:
    """Tests for HFEngine initialization."""

    def test_init_stores_config(self):
        """Test constructor stores configuration without loading model."""
        HFEngine = _import_hf_engine()

        args = MockArgs()
        engine = HFEngine(args, rank=0, base_gpu_id=0)

        assert engine.args is args
        assert engine.rank == 0
        assert engine.base_gpu_id == 0
        assert engine._engine is None
        assert engine._mooncake_config is None

    def test_init_with_different_ranks(self):
        """Test initialization with various rank values."""
        HFEngine = _import_hf_engine()

        args = MockArgs()

        for rank in [0, 1, 7]:
            engine = HFEngine(args, rank=rank)
            assert engine.rank == rank

    def test_init_without_base_gpu_id(self):
        """Test initialization without specifying base_gpu_id."""
        HFEngine = _import_hf_engine()

        args = MockArgs()
        engine = HFEngine(args, rank=0, base_gpu_id=None)

        assert engine.base_gpu_id is None


class TestHFEngineHealthCheck:
    """Tests for HFEngine health check."""

    def test_health_check_before_init(self):
        """Test health_check returns False before initialization."""
        HFEngine = _import_hf_engine()

        args = MockArgs()
        engine = HFEngine(args, rank=0)

        assert engine.health_check() is False

    def test_health_check_after_init(self):
        """Test health_check returns True after initialization."""
        HFEngine = _import_hf_engine()

        args = MockArgs()
        engine = HFEngine(args, rank=0)
        engine._engine = MagicMock()

        assert engine.health_check() is True


class TestHFEngineStatus:
    """Tests for HFEngine status reporting."""

    def test_get_status_before_init(self):
        """Test get_status before initialization."""
        HFEngine = _import_hf_engine()

        args = MockArgs()
        engine = HFEngine(args, rank=1, base_gpu_id=2)

        status = engine.get_status()

        assert status["rank"] == 1
        assert status["initialized"] is False
        assert status["base_gpu_id"] == 2

    def test_get_status_after_init(self):
        """Test get_status after initialization."""
        HFEngine = _import_hf_engine()

        args = MockArgs()
        engine = HFEngine(args, rank=3, base_gpu_id=4)
        engine._engine = MagicMock()

        status = engine.get_status()

        assert status["rank"] == 3
        assert status["initialized"] is True
        assert status["base_gpu_id"] == 4


class TestHFEngineMooncakeConfig:
    """Tests for HFEngine mooncake configuration."""

    def test_get_mooncake_config_returns_none_initially(self):
        """Test mooncake config is None before init."""
        HFEngine = _import_hf_engine()

        args = MockArgs()
        engine = HFEngine(args, rank=0)

        assert engine.get_mooncake_config() is None

    def test_get_mooncake_config_after_init(self):
        """Test mooncake config is stored after init."""
        HFEngine = _import_hf_engine()
        from torchspec.config.mooncake_config import MooncakeConfig

        args = MockArgs()
        engine = HFEngine(args, rank=0)

        mooncake_config = MooncakeConfig(master_server_address="localhost:50051")
        engine._mooncake_config = mooncake_config

        assert engine.get_mooncake_config() is mooncake_config


class TestHFEngineSetup:
    """Tests for HFEngine init() method."""

    def test_init_creates_engine_without_mooncake(self):
        """Test init() creates engine without mooncake config."""
        HFEngine = _import_hf_engine()

        args = MockArgs()
        engine = HFEngine(args, rank=0, base_gpu_id=None)

        mock_inference_engine = MagicMock()
        mock_engine_class = MagicMock()
        mock_engine_class.from_pretrained.return_value = mock_inference_engine

        with patch(
            "torchspec.inference.engine.hf_runner.HFRunner",
            mock_engine_class,
        ):
            engine.init(mooncake_config=None)

        assert engine._engine is mock_inference_engine
        assert engine._mooncake_config is None
        mock_engine_class.from_pretrained.assert_called_once()

    def test_init_creates_engine_with_mooncake(self):
        """Test init() creates engine with mooncake config."""
        HFEngine = _import_hf_engine()
        from torchspec.config.mooncake_config import MooncakeConfig

        args = MockArgs()
        engine = HFEngine(args, rank=0, base_gpu_id=None)

        mock_inference_engine = MagicMock()
        mock_engine_class = MagicMock()
        mock_engine_class.from_pretrained.return_value = mock_inference_engine

        mooncake_config = MooncakeConfig(master_server_address="localhost:50051")

        with patch(
            "torchspec.inference.engine.hf_runner.HFRunner",
            mock_engine_class,
        ):
            with patch(
                "torchspec.transfer.mooncake.utils.check_mooncake_master_available",
            ):
                engine.init(mooncake_config=mooncake_config)

        assert engine._engine is mock_inference_engine
        assert engine._mooncake_config is mooncake_config
        # Verify from_pretrained was called with the MooncakeConfig object directly
        call_kwargs = mock_engine_class.from_pretrained.call_args[1]
        assert call_kwargs["mooncake_config"] is mooncake_config

    def test_init_sets_cuda_device_with_base_gpu_id(self):
        """Test init() sets CUDA device when base_gpu_id is provided."""
        HFEngine = _import_hf_engine()

        args = MockArgs()
        engine = HFEngine(args, rank=0, base_gpu_id=2)

        mock_engine_class = MagicMock()
        mock_engine_class.from_pretrained.return_value = MagicMock()

        with patch(
            "torchspec.inference.engine.hf_runner.HFRunner",
            mock_engine_class,
        ):
            with patch("torch.cuda.set_device") as mock_set_device:
                with patch("torchspec.ray.ray_actor._to_local_gpu_id", return_value=0):
                    engine.init(mooncake_config=None)

        mock_set_device.assert_called_once_with(0)


class TestHFEngineGenerate:
    """Tests for HFEngine generate() method."""

    def test_generate_raises_if_not_initialized(self):
        """Test generate() raises RuntimeError if engine not initialized."""
        HFEngine = _import_hf_engine()

        args = MockArgs()
        engine = HFEngine(args, rank=0)

        with pytest.raises(RuntimeError, match="not initialized"):
            engine.generate(
                data_id="test",
                input_ids_ref=[torch.tensor([1, 2, 3])],
                packed_loss_mask_list=["0,3"],
            )

    def test_generate_with_list_inputs(self):
        """Test generate() works with list inputs."""
        HFEngine = _import_hf_engine()

        args = MockArgs()
        engine = HFEngine(args, rank=0)

        mock_inner_engine = MagicMock()
        mock_inner_engine.generate.return_value = [{"result": "data"}]
        engine._engine = mock_inner_engine

        input_ids = [torch.tensor([1, 2, 3])]
        packed_loss_mask = ["0,3"]

        result = engine.generate(
            data_id="test",
            input_ids_ref=input_ids,
            packed_loss_mask_list=packed_loss_mask,
        )

        mock_inner_engine.generate.assert_called_once_with(
            data_id="test",
            input_ids_list=input_ids,
            packed_loss_mask_list=packed_loss_mask,
        )
        assert result == [{"result": "data"}]


class TestHFEngineShutdown:
    """Tests for HFEngine shutdown() method."""

    def test_shutdown_cleans_up_engine(self):
        """Test shutdown() calls engine shutdown and clears reference."""
        HFEngine = _import_hf_engine()

        args = MockArgs()
        engine = HFEngine(args, rank=0)

        mock_inner_engine = MagicMock()
        engine._engine = mock_inner_engine

        engine.shutdown()

        mock_inner_engine.shutdown.assert_called_once()
        assert engine._engine is None

    def test_shutdown_handles_no_engine(self):
        """Test shutdown() works when engine is None."""
        HFEngine = _import_hf_engine()

        args = MockArgs()
        engine = HFEngine(args, rank=0)

        engine.shutdown()

        assert engine._engine is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
