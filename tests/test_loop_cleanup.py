from types import SimpleNamespace
from unittest import mock

import pytest

from torchspec.controller import eval as eval_utils
from torchspec.controller import loop


class TestTrainingLoopCleanup:
    def test_safe_cleanup_stops_inference_and_shutdowns_mooncake_actor(self):
        args = SimpleNamespace(_mooncake_master_actor=mock.MagicMock())
        inference_manager = mock.MagicMock()
        inference_future = object()

        stop_ref = inference_manager.stop.remote.return_value
        shutdown_ref = args._mooncake_master_actor.shutdown.remote.return_value

        with mock.patch("torchspec.controller.loop.ray.get") as mock_ray_get:
            loop._safe_training_cleanup(args, inference_manager, inference_future)

        mock_ray_get.assert_any_call(stop_ref)
        mock_ray_get.assert_any_call(inference_future)
        mock_ray_get.assert_any_call(shutdown_ref, timeout=10)

    def test_safe_cleanup_skips_mooncake_shutdown_when_actor_missing(self):
        args = SimpleNamespace()
        inference_manager = mock.MagicMock()
        inference_future = object()

        with mock.patch("torchspec.controller.loop.ray.get"):
            loop._safe_training_cleanup(args, inference_manager, inference_future)

        inference_manager.stop.remote.assert_called_once()

    def test_run_training_loop_finally_runs_cleanup_on_success(self):
        args = SimpleNamespace(_mooncake_master_actor=mock.MagicMock())
        inference_manager = mock.MagicMock()

        def _fake_impl(
            args,
            controller,
            inference_manager,
            train_group,
            inference_engines=None,
            dataset_size=None,
            eval_dataset_size=None,
            cleanup_state=None,
        ):
            cleanup_state["inference_future"] = "future-ok"
            return "ok"

        with (
            mock.patch("torchspec.controller.loop._run_training_loop_impl", side_effect=_fake_impl),
            mock.patch("torchspec.controller.loop._safe_training_cleanup") as mock_cleanup,
        ):
            result = loop.run_training_loop(args, object(), inference_manager, object())

        assert result == "ok"
        mock_cleanup.assert_called_once_with(
            args=args,
            inference_manager=inference_manager,
            inference_future="future-ok",
        )

    def test_run_training_loop_finally_runs_cleanup_on_exception(self):
        args = SimpleNamespace(_mooncake_master_actor=mock.MagicMock())
        inference_manager = mock.MagicMock()

        def _fake_impl(
            args,
            controller,
            inference_manager,
            train_group,
            inference_engines=None,
            dataset_size=None,
            eval_dataset_size=None,
            cleanup_state=None,
        ):
            cleanup_state["inference_future"] = "future-error"
            raise RuntimeError("training failed")

        with (
            mock.patch("torchspec.controller.loop._run_training_loop_impl", side_effect=_fake_impl),
            mock.patch("torchspec.controller.loop._safe_training_cleanup") as mock_cleanup,
        ):
            with pytest.raises(RuntimeError, match="training failed"):
                loop.run_training_loop(args, object(), inference_manager, object())

        mock_cleanup.assert_called_once_with(
            args=args,
            inference_manager=inference_manager,
            inference_future="future-error",
        )


def test_generate_eval_cache_waits_for_eval_dispatch_completion():
    controller = mock.MagicMock()
    train_group = mock.MagicMock()

    # First dispatch loop: one required full eval batch is available.
    # Completion loop: eval is still in flight once, then becomes complete.
    controller.try_dispatch_eval_batch.remote.side_effect = [True, False]
    controller.is_eval_dispatch_complete.remote.side_effect = [False, True]

    with (
        mock.patch("torchspec.controller.eval.ray.get", side_effect=lambda x: x),
        mock.patch("torchspec.controller.eval.time.sleep") as mock_sleep,
    ):
        eval_utils.generate_eval_cache(
            controller=controller,
            train_group=train_group,
            num_dispatches=1,
            samples_per_rank=4,
            eval_cache_path=None,
        )

    train_group.cache_eval_samples.assert_called_once_with(4)
    controller.finalize_eval_dispatch.remote.assert_called_once()
    assert controller.is_eval_dispatch_complete.remote.call_count >= 2
    mock_sleep.assert_called_once_with(0.1)


def test_generate_eval_cache_times_out_when_eval_never_arrives():
    controller = mock.MagicMock()
    train_group = mock.MagicMock()

    controller.try_dispatch_eval_batch.remote.return_value = False
    with (
        mock.patch("torchspec.controller.eval.ray.get", side_effect=lambda x: x),
        mock.patch("torchspec.controller.eval.time.sleep") as mock_sleep,
        mock.patch("torchspec.controller.eval.EVAL_CACHE_IDLE_TIMEOUT", 0.0),
    ):
        with pytest.raises(TimeoutError, match="Timed out while waiting for eval cache generation"):
            eval_utils.generate_eval_cache(
                controller=controller,
                train_group=train_group,
                num_dispatches=1,
                samples_per_rank=4,
                eval_cache_path=None,
            )

    train_group.cache_eval_samples.assert_not_called()
    controller.finalize_eval_dispatch.remote.assert_not_called()
    mock_sleep.assert_not_called()
