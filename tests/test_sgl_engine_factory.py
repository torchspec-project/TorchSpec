"""Tests for SGLang engine factory: topology, dist_init_addr negotiation, and tp_size validation."""

from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_args(**overrides):
    """Build a minimal args Namespace for _init_sgl_engines."""
    defaults = dict(
        target_model_path="/fake/model",
        inference_num_gpus=8,
        inference_num_gpus_per_engine=8,
        inference_num_gpus_per_node=8,
        sglang_nnodes=1,
        sglang_dist_init_addr=None,
        sglang_init_timeout=10,
        sglang_tp_size=None,
        sglang_pp_size=1,
        aux_hidden_states_layers=None,
        sglang_mem_fraction_static=0.8,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def _make_pg(num_bundles):
    """Create a fake placement group tuple with *num_bundles* bundles.

    Returns (pg_obj, reordered_bundle_indices, reordered_gpu_ids) where
    indices and gpu_ids are simple identity lists.
    """
    pg_obj = MagicMock(name="pg_obj")
    indices = list(range(num_bundles))
    gpu_ids = list(range(num_bundles))
    return (pg_obj, indices, gpu_ids)


def _build_mock_ray_infrastructure():
    """Set up mock Ray so _init_sgl_engines can run without a real cluster.

    Returns (engines, init_calls, mock_ray_remote) where:
      - engines: list collecting every actor created via .remote()
      - init_calls: list of (engine, kwargs) for every engine.init.remote() call
    """
    engines = []
    init_calls = []

    def make_engine(**constructor_kwargs):
        engine = MagicMock(name=f"engine_{len(engines)}")
        engine._constructor_kwargs = constructor_kwargs

        # .init.remote() → record call, return a sentinel ObjectRef
        def _init_remote(**kw):
            ref = MagicMock(name=f"init_ref_{len(init_calls)}")
            init_calls.append((engine, kw))
            return ref

        engine.init.remote.side_effect = _init_remote

        # get_node_ip / find_free_port → return ObjectRefs resolved by ray.get
        engine.get_node_ip.remote.return_value = MagicMock(name="ip_ref")
        engine.find_free_port.remote.return_value = MagicMock(name="port_ref")

        engines.append(engine)
        return engine

    # ray.remote(SglEngine) → SglRayActor
    mock_sgl_ray_actor = MagicMock(name="SglRayActor")

    # SglRayActor.options(...).remote(**kwargs) → make_engine(**kwargs)
    mock_sgl_ray_actor.options.return_value.remote.side_effect = make_engine

    def mock_ray_remote(cls):
        return mock_sgl_ray_actor

    return engines, init_calls, mock_ray_remote


# ---------------------------------------------------------------------------
# Topology tests for _init_sgl_engines
# ---------------------------------------------------------------------------


class TestSglEngineTopology:
    """Test engine count, replica count, and node_rank assignment."""

    def _call(self, args, pg, mooncake_config=None, ray_get_side_effect=None):
        engines, init_calls, mock_ray_remote = _build_mock_ray_infrastructure()

        with (
            patch("torchspec.inference.factory.ray") as mock_ray,
            patch("torchspec.inference.factory.get_torchspec_env_vars", return_value={}),
            patch("torchspec.inference.factory._alive_worker_engines", []) as alive_workers,
        ):
            mock_ray.remote.side_effect = mock_ray_remote
            mock_ray.get.side_effect = ray_get_side_effect or (lambda refs, **kw: None)

            from torchspec.inference.factory import _init_sgl_engines

            result = _init_sgl_engines(args, pg, mooncake_config)

        return result, engines, init_calls, alive_workers

    # -- Single-node cases --------------------------------------------------

    def test_single_node_4_engines(self):
        """8 GPUs, 2 GPUs/engine → 4 engines, all returned."""
        args = _make_args(inference_num_gpus=8, inference_num_gpus_per_engine=2)
        pg = _make_pg(8)

        result, engines, init_calls, _ = self._call(args, pg)

        assert len(engines) == 4
        assert len(result) == 4
        # All node_rank == 0
        for e in engines:
            assert e._constructor_kwargs["node_rank"] == 0

    def test_single_node_1_engine(self):
        """8 GPUs, 8 GPUs/engine → 1 engine."""
        args = _make_args(inference_num_gpus=8, inference_num_gpus_per_engine=8)
        pg = _make_pg(8)

        result, engines, *_ = self._call(args, pg)

        assert len(engines) == 1
        assert len(result) == 1

    def test_single_node_gpu_ids(self):
        """base_gpu_id should be bundle_offset = i * gpus_per_engine."""
        args = _make_args(inference_num_gpus=8, inference_num_gpus_per_engine=2)
        pg = _make_pg(8)

        _, engines, *_ = self._call(args, pg)

        for i, e in enumerate(engines):
            assert e._constructor_kwargs["base_gpu_id"] == i * 2

    def test_single_node_no_dist_init_addr(self):
        """Single-node engines should NOT receive dist_init_addr."""
        args = _make_args(inference_num_gpus=8, inference_num_gpus_per_engine=2)
        pg = _make_pg(8)

        _, _, init_calls, _ = self._call(args, pg)

        for _, kw in init_calls:
            assert kw["dist_init_addr"] is None

    # -- Multi-node, single replica -----------------------------------------

    def test_multi_node_single_replica_topology(self):
        """3 nodes × 8 GPUs = 24 total, 1 replica → 3 engines, 1 head + 2 workers."""
        args = _make_args(
            inference_num_gpus=24,
            inference_num_gpus_per_node=8,
            sglang_nnodes=3,
        )
        pg = _make_pg(24)

        def ray_get_side_effect(refs, **kw):
            if isinstance(refs, list) and len(refs) == 2:
                return ("10.0.0.1", 12345)
            return None

        result, engines, init_calls, alive_workers = self._call(
            args, pg, ray_get_side_effect=ray_get_side_effect
        )

        assert len(engines) == 3
        assert len(result) == 1  # only head
        assert len(alive_workers) == 2  # 2 workers stored

        # node_rank assignment: 0, 1, 2
        for i, e in enumerate(engines):
            assert e._constructor_kwargs["node_rank"] == i

    def test_multi_node_dist_init_addr_negotiated(self):
        """Multi-node auto-negotiation calls get_node_ip + find_free_port on head engine."""
        args = _make_args(
            inference_num_gpus=24,
            inference_num_gpus_per_node=8,
            sglang_nnodes=3,
        )
        pg = _make_pg(24)

        def ray_get_side_effect(refs, **kw):
            if isinstance(refs, list) and len(refs) == 2:
                return ("10.0.0.1", 12345)
            return None

        _, engines, init_calls, _ = self._call(args, pg, ray_get_side_effect=ray_get_side_effect)

        # Head engine (index 0) should have get_node_ip/find_free_port called
        engines[0].get_node_ip.remote.assert_called_once()
        engines[0].find_free_port.remote.assert_called_once()

        # All 3 engines should receive the same dist_init_addr
        for _, kw in init_calls:
            assert kw["dist_init_addr"] == "10.0.0.1:12345"

    def test_multi_node_configured_addr_override(self):
        """When sglang_dist_init_addr is set and single replica, use it directly."""
        args = _make_args(
            inference_num_gpus=24,
            inference_num_gpus_per_node=8,
            sglang_nnodes=3,
            sglang_dist_init_addr="192.168.1.1:9999",
        )
        pg = _make_pg(24)

        _, engines, init_calls, _ = self._call(args, pg)

        # Should NOT call get_node_ip/find_free_port
        engines[0].get_node_ip.remote.assert_not_called()
        engines[0].find_free_port.remote.assert_not_called()

        # All engines get the configured addr
        for _, kw in init_calls:
            assert kw["dist_init_addr"] == "192.168.1.1:9999"

    # -- Multi-node, multiple replicas --------------------------------------

    def test_multi_node_multi_replica_topology(self):
        """2 replicas × 3 nodes × 8 GPUs = 48 total → 6 engines, 2 heads + 4 workers."""
        args = _make_args(
            inference_num_gpus=48,
            inference_num_gpus_per_node=8,
            sglang_nnodes=3,
        )
        pg = _make_pg(48)

        call_count = [0]

        def ray_get_side_effect(refs, **kw):
            if isinstance(refs, list) and len(refs) == 2:
                call_count[0] += 1
                return (f"10.0.{call_count[0]}.1", 10000 + call_count[0])
            return None

        result, engines, init_calls, alive_workers = self._call(
            args, pg, ray_get_side_effect=ray_get_side_effect
        )

        assert len(engines) == 6
        assert len(result) == 2  # 2 heads
        assert len(alive_workers) == 4  # 4 workers

        # node_rank pattern: [0,1,2, 0,1,2]
        expected_node_ranks = [0, 1, 2, 0, 1, 2]
        for i, e in enumerate(engines):
            assert e._constructor_kwargs["node_rank"] == expected_node_ranks[i]

    def test_multi_node_multi_replica_per_replica_addr(self):
        """Each replica gets its own dist_init_addr."""
        args = _make_args(
            inference_num_gpus=48,
            inference_num_gpus_per_node=8,
            sglang_nnodes=3,
        )
        pg = _make_pg(48)

        call_count = [0]

        def ray_get_side_effect(refs, **kw):
            if isinstance(refs, list) and len(refs) == 2:
                call_count[0] += 1
                return (f"10.0.{call_count[0]}.1", 10000 + call_count[0])
            return None

        _, engines, init_calls, _ = self._call(args, pg, ray_get_side_effect=ray_get_side_effect)

        # Negotiation called on engine[0] (replica 0 head) and engine[3] (replica 1 head)
        engines[0].get_node_ip.remote.assert_called_once()
        engines[3].get_node_ip.remote.assert_called_once()
        # Worker engines should NOT be used for negotiation
        for idx in [1, 2, 4, 5]:
            engines[idx].get_node_ip.remote.assert_not_called()

        # Replica 0 engines (0,1,2) get addr from first negotiation
        # Replica 1 engines (3,4,5) get addr from second negotiation
        for i in range(3):
            assert init_calls[i][1]["dist_init_addr"] == "10.0.1.1:10001"
        for i in range(3, 6):
            assert init_calls[i][1]["dist_init_addr"] == "10.0.2.1:10002"

    def test_multi_node_multi_replica_configured_addr_ignored(self):
        """With multiple replicas, configured addr is ignored (auto-negotiate each)."""
        args = _make_args(
            inference_num_gpus=48,
            inference_num_gpus_per_node=8,
            sglang_nnodes=3,
            sglang_dist_init_addr="192.168.1.1:9999",
        )
        pg = _make_pg(48)

        call_count = [0]

        def ray_get_side_effect(refs, **kw):
            if isinstance(refs, list) and len(refs) == 2:
                call_count[0] += 1
                return (f"10.0.{call_count[0]}.1", 10000 + call_count[0])
            return None

        _, engines, init_calls, _ = self._call(args, pg, ray_get_side_effect=ray_get_side_effect)

        # Should auto-negotiate despite configured_addr, because num_replicas > 1
        engines[0].get_node_ip.remote.assert_called_once()
        engines[3].get_node_ip.remote.assert_called_once()


# ---------------------------------------------------------------------------
# tp_size / pp_size validation in SglEngine.init()
# ---------------------------------------------------------------------------


class TestSglEngineTpSizeValidation:
    """Test tp_size and pp_size assertions in SglEngine.init()."""

    def _make_engine(self, num_gpus_per_engine=8, node_rank=0, **args_overrides):
        from torchspec.inference.engine.sgl_engine import SglEngine

        args = _make_args(**args_overrides)
        return SglEngine(
            args=args,
            rank=0,
            base_gpu_id=0,
            num_gpus_per_engine=num_gpus_per_engine,
            node_rank=node_rank,
        )

    def _init_engine(self, engine):
        """Call engine.init() with all heavy deps mocked out."""
        mock_sgl = MagicMock()
        mock_sgl.Engine.return_value = MagicMock(model_config=MagicMock(hidden_size=4096))
        with (
            patch.object(engine, "setup_gpu", return_value=0),
            patch("torchspec.inference.engine.sgl_engine.sgl", mock_sgl),
            patch(
                "torchspec.inference.engine.sgl_engine.get_default_eagle3_aux_layer_ids",
                return_value=[0, 1, 2],
            ),
            patch.object(engine, "_get_hidden_size_from_engine", return_value=4096),
        ):
            engine.init(mooncake_config=None)
        return mock_sgl

    def test_pp_size_must_be_1(self):
        """pp_size != 1 should raise AssertionError."""
        engine = self._make_engine(sglang_pp_size=2)

        with pytest.raises(AssertionError, match="pp_size must be 1"):
            self._init_engine(engine)

    def test_tp_size_correct_single_node(self):
        """sglang_tp_size=8, nnodes=1, gpus_per_engine=8 → OK, tp_size=8 passed to sgl.Engine."""
        engine = self._make_engine(
            num_gpus_per_engine=8,
            sglang_tp_size=8,
            sglang_nnodes=1,
        )
        mock_sgl = self._init_engine(engine)
        kwargs = mock_sgl.Engine.call_args[1]
        assert kwargs["tp_size"] == 8
        assert kwargs["nnodes"] == 1

    def test_tp_size_correct_multi_node(self):
        """sglang_tp_size=24, nnodes=3, gpus_per_engine=8 → OK, tp_size=24 passed to sgl.Engine."""
        engine = self._make_engine(
            num_gpus_per_engine=8,
            sglang_tp_size=24,
            sglang_nnodes=3,
        )
        mock_sgl = self._init_engine(engine)
        kwargs = mock_sgl.Engine.call_args[1]
        assert kwargs["tp_size"] == 24
        assert kwargs["nnodes"] == 3

    def test_tp_size_wrong_single_node(self):
        """sglang_tp_size=4 with gpus_per_engine=8 → AssertionError."""
        engine = self._make_engine(
            num_gpus_per_engine=8,
            sglang_tp_size=4,
            sglang_nnodes=1,
        )
        with pytest.raises(AssertionError, match="sglang_tp_size"):
            self._init_engine(engine)

    def test_tp_size_wrong_multi_node(self):
        """sglang_tp_size=8 with nnodes=3, gpus_per_engine=8 → should be 24."""
        engine = self._make_engine(
            num_gpus_per_engine=8,
            sglang_tp_size=8,
            sglang_nnodes=3,
        )
        with pytest.raises(AssertionError, match="sglang_tp_size.*8.*must equal.*24"):
            self._init_engine(engine)

    def test_tp_size_defaults_when_not_configured(self):
        """When sglang_tp_size is None, no assertion, tp_size = nnodes * gpus_per_engine."""
        engine = self._make_engine(
            num_gpus_per_engine=8,
            sglang_tp_size=None,
            sglang_nnodes=3,
        )
        mock_sgl = self._init_engine(engine)
        kwargs = mock_sgl.Engine.call_args[1]
        assert kwargs["tp_size"] == 24  # 3 * 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
