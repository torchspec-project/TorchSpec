"""Tests for HFRunner mooncake store initialization."""

from unittest.mock import patch

import torch

from torchspec.config.inference_config import HFInferenceConfig
from torchspec.config.mooncake_config import MooncakeConfig
from torchspec.inference.engine.hf_runner import HFRunner


class MockMooncakeStore:
    """Mock mooncake store for testing."""

    def __init__(self, config: MooncakeConfig):
        self.config = config
        self._setup_called = False
        self._closed = False

    def setup(self, device):
        self._setup_called = True
        self._device = device

    def close(self):
        self._closed = True

    def put(self, key, **tensors):
        return {k: v.shape for k, v in tensors.items() if v is not None}

    def get(self, key, shapes, dtypes, device):
        return {
            k: torch.zeros(s, dtype=dtypes.get(k, torch.float32), device=device)
            for k, s in shapes.items()
        }


class TestHFRunnerInitMooncakeStore:
    """Tests for mooncake store initialization in HFRunner."""

    def test_init_mooncake_store_with_config(self):
        """Test init_mooncake_store creates store from config."""
        mooncake_config = MooncakeConfig(
            master_server_address="localhost:50051",
            metadata_server="http://localhost:8090/metadata",
        )

        config = HFInferenceConfig(
            model_path="/fake/path",
            mooncake_config=mooncake_config,
        )

        engine = HFRunner(config=config)

        assert engine.mooncake_store is None

        with patch(
            "torchspec.inference.engine.hf_runner.EagleMooncakeStore",
            MockMooncakeStore,
        ):
            with patch("torch.cuda.current_device", return_value=0):
                store = engine.init_mooncake_store()

        assert engine.mooncake_store is not None
        assert engine.mooncake_store._setup_called
        assert store.config == mooncake_config

    def test_init_mooncake_store_with_explicit_config(self):
        """Test init_mooncake_store with explicitly passed config."""
        config = HFInferenceConfig(model_path="/fake/path")
        engine = HFRunner(config=config)

        explicit_config = MooncakeConfig(
            master_server_address="explicit:50051",
        )

        with patch(
            "torchspec.inference.engine.hf_runner.EagleMooncakeStore",
            MockMooncakeStore,
        ):
            with patch("torch.cuda.current_device", return_value=0):
                store = engine.init_mooncake_store(explicit_config)

        assert store.config.master_server_address == "explicit:50051"

    def test_init_mooncake_store_raises_without_config(self):
        """Test init_mooncake_store raises ValueError without config."""
        config = HFInferenceConfig(model_path="/fake/path")
        engine = HFRunner(config=config)

        try:
            engine.init_mooncake_store()
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "mooncake_config must be provided" in str(e)

    def test_setup_initializes_mooncake_store_if_configured(self):
        """Test setup() calls init_mooncake_store when config is provided."""
        mooncake_config = MooncakeConfig(
            master_server_address="localhost:50051",
        )

        config = HFInferenceConfig(
            model_path="/fake/path",
            mooncake_config=mooncake_config,
        )

        engine = HFRunner(config=config)

        with patch.object(engine, "_setup_target_model"):
            with patch(
                "torchspec.inference.engine.hf_runner.EagleMooncakeStore",
                MockMooncakeStore,
            ):
                with patch("torch.cuda.current_device", return_value=0):
                    engine.setup()

        assert engine.mooncake_store is not None
        assert engine._initialized

    def test_setup_skips_mooncake_if_not_configured(self):
        """Test setup() doesn't init mooncake store when no config."""
        config = HFInferenceConfig(model_path="/fake/path")
        engine = HFRunner(config=config)

        with patch.object(engine, "_setup_target_model"):
            engine.setup()

        assert engine.mooncake_store is None
        assert engine._initialized

    def test_setup_skips_mooncake_if_already_provided(self):
        """Test setup() doesn't reinit mooncake store if already provided."""
        mooncake_config = MooncakeConfig(
            master_server_address="localhost:50051",
        )
        config = HFInferenceConfig(
            model_path="/fake/path",
            mooncake_config=mooncake_config,
        )

        existing_store = MockMooncakeStore(mooncake_config)
        engine = HFRunner(config=config, mooncake_store=existing_store)

        with patch.object(engine, "_setup_target_model"):
            engine.setup()

        assert engine.mooncake_store is existing_store

    def test_shutdown_closes_mooncake_store(self):
        """Test shutdown() closes the mooncake store."""
        mooncake_config = MooncakeConfig(master_server_address="localhost:50051")
        config = HFInferenceConfig(
            model_path="/fake/path",
            mooncake_config=mooncake_config,
        )

        engine = HFRunner(config=config)

        mock_store = MockMooncakeStore(mooncake_config)
        engine.mooncake_store = mock_store
        engine._initialized = True

        engine.shutdown()

        assert mock_store._closed
        assert engine.mooncake_store is None
        assert not engine._initialized


class TestHFInferenceConfigMooncakeConfig:
    """Tests for MooncakeConfig in HFInferenceConfig."""

    def test_config_includes_mooncake_config(self):
        """Test HFInferenceConfig can hold MooncakeConfig."""
        mooncake_config = MooncakeConfig(
            master_server_address="test:50051",
            metadata_server="http://test:8090/metadata",
            local_hostname="testhost",
        )

        config = HFInferenceConfig(
            model_path="/path/to/model",
            mooncake_config=mooncake_config,
        )

        assert config.mooncake_config is not None
        assert config.mooncake_config.master_server_address == "test:50051"
        assert config.mooncake_config.local_hostname == "testhost"

    def test_config_mooncake_config_defaults_to_none(self):
        """Test mooncake_config defaults to None."""
        config = HFInferenceConfig(model_path="/path/to/model")
        assert config.mooncake_config is None
