"""Tests for mooncake master Ray actor and launcher."""

import os
import subprocess
import tempfile
from argparse import Namespace
from unittest import mock

from torchspec.transfer.mooncake.utils import (
    MooncakeMaster,
    launch_mooncake_master,
    resolve_mooncake_master_bin,
)


class TestResolveMooncakeMasterBin:
    def test_uses_mooncake_build_dir_env(self, monkeypatch):
        monkeypatch.setenv("MOONCAKE_BUILD_DIR", "/custom/build")
        result = resolve_mooncake_master_bin()
        assert result == "/custom/build/mooncake-store/src/mooncake_master"

    def test_uses_system_path_if_available(self, monkeypatch):
        monkeypatch.delenv("MOONCAKE_BUILD_DIR", raising=False)
        with mock.patch("shutil.which", return_value="/usr/bin/mooncake_master"):
            result = resolve_mooncake_master_bin()
        assert result == "/usr/bin/mooncake_master"

    def test_falls_back_to_default_path(self, monkeypatch):
        monkeypatch.delenv("MOONCAKE_BUILD_DIR", raising=False)
        with mock.patch("shutil.which", return_value=None):
            result = resolve_mooncake_master_bin()
        assert "build/mooncake-store/src/mooncake_master" in result


class TestMooncakeMaster:
    """Tests for the MooncakeMaster class (tested as plain class, no Ray)."""

    def test_start_launches_subprocess(self):
        actor = MooncakeMaster()
        mock_process = mock.MagicMock()
        mock_process.poll.return_value = None
        mock_process.pid = 12345
        mock_process.stdout = iter([])
        mock_process.stderr = iter([])

        with (
            mock.patch(
                "torchspec.transfer.mooncake.utils.resolve_mooncake_master_bin",
                return_value="/usr/bin/mooncake_master",
            ),
            mock.patch("os.path.exists", return_value=True),
            mock.patch("subprocess.Popen", return_value=mock_process) as mock_popen,
            mock.patch("time.sleep"),
            mock.patch("torchspec.ray.ray_actor.get_current_node_ip", return_value="10.0.0.1"),
        ):
            info = actor.start(50051, 8090, "0.0.0.0")

        cmd = mock_popen.call_args[0][0]
        assert cmd[0] == "/usr/bin/mooncake_master"
        assert "--port=50051" in cmd
        assert "--http_metadata_server_port=8090" in cmd
        assert "--http_metadata_server_host=0.0.0.0" in cmd
        assert "--enable_http_metadata_server=true" in cmd
        assert info["master_addr"] == "10.0.0.1:50051"
        assert info["metadata_port"] == 8090

    def test_start_raises_on_missing_binary(self):
        actor = MooncakeMaster()
        with (
            mock.patch(
                "torchspec.transfer.mooncake.utils.resolve_mooncake_master_bin",
                return_value="/nonexistent/mooncake_master",
            ),
            mock.patch("os.path.exists", return_value=False),
        ):
            try:
                actor.start(50051, 8090)
                assert False, "Should have raised FileNotFoundError"
            except FileNotFoundError:
                pass

    def test_start_raises_on_process_failure(self):
        actor = MooncakeMaster()
        mock_process = mock.MagicMock()
        mock_process.poll.return_value = 1
        mock_process.returncode = 1
        mock_process.stdout = iter([])
        mock_process.stderr = iter([])

        with (
            mock.patch(
                "torchspec.transfer.mooncake.utils.resolve_mooncake_master_bin",
                return_value="/usr/bin/mooncake_master",
            ),
            mock.patch("os.path.exists", return_value=True),
            mock.patch("subprocess.Popen", return_value=mock_process),
            mock.patch("time.sleep"),
        ):
            try:
                actor.start(50051, 8090)
                assert False, "Should have raised RuntimeError"
            except RuntimeError as e:
                assert "failed to start" in str(e)

    def test_health_check_returns_true_when_running(self):
        actor = MooncakeMaster()
        mock_process = mock.MagicMock()
        mock_process.poll.return_value = None
        mock_process.pid = 12345
        actor._process = mock_process
        assert actor.health_check() is True
        actor._process = None  # prevent __del__ from calling os.killpg

    def test_health_check_returns_false_when_dead(self):
        actor = MooncakeMaster()
        mock_process = mock.MagicMock()
        mock_process.poll.return_value = 1
        actor._process = mock_process
        assert actor.health_check() is False

    def test_health_check_returns_false_when_no_process(self):
        actor = MooncakeMaster()
        assert actor.health_check() is False

    def test_get_info_returns_stored_info(self):
        actor = MooncakeMaster()
        actor._info = {"master_addr": "10.0.0.1:50051", "metadata_port": 8090}
        assert actor.get_info() == {"master_addr": "10.0.0.1:50051", "metadata_port": 8090}

    def test_del_terminates_process(self):
        actor = MooncakeMaster()
        mock_process = mock.MagicMock()
        mock_process.poll.return_value = None
        mock_process.pid = 12345
        actor._process = mock_process
        with mock.patch("os.killpg") as mock_killpg:
            actor.__del__()
            mock_killpg.assert_called_once_with(12345, mock.ANY)
        mock_process.wait.assert_called_once_with(timeout=5)

    def test_del_kills_on_timeout(self):
        actor = MooncakeMaster()
        mock_process = mock.MagicMock()
        mock_process.poll.return_value = None
        mock_process.pid = 12345
        mock_process.wait.side_effect = subprocess.TimeoutExpired("cmd", 5)
        actor._process = mock_process
        with mock.patch("os.killpg") as mock_killpg:
            actor.__del__()
            assert mock_killpg.call_count == 2  # SIGTERM then SIGKILL


class TestLaunchMooncakeMaster:
    def _make_mock_ray_remote(self, mock_actor):
        """Create a mock for ray.remote that yields mock_actor at the end of the chain.

        Matches: ray.remote(num_cpus=0)(MooncakeMaster).options(name=...).remote()
        """
        # ray.remote(num_cpus=0) returns a decorator
        mock_decorator = mock.MagicMock()
        # decorator(MooncakeMaster) returns RemoteActorClass
        mock_remote_actor_cls = mock.MagicMock()
        mock_decorator.return_value = mock_remote_actor_cls
        # RemoteActorClass.options(name=...).remote() returns the actor handle
        mock_remote_actor_cls.options.return_value.remote.return_value = mock_actor
        return mock_decorator, mock_remote_actor_cls

    def test_creates_named_actor_and_writes_back_to_args(self):
        args = Namespace(
            mooncake_master_server_address="localhost:50051",
            mooncake_metadata_port=8090,
            mooncake_http_host="0.0.0.0",
        )

        mock_actor = mock.MagicMock()
        mock_actor.start.remote.return_value = mock.MagicMock()
        mock_decorator, mock_remote_actor_cls = self._make_mock_ray_remote(mock_actor)

        with (
            mock.patch(
                "torchspec.transfer.mooncake.utils.resolve_mooncake_master_bin",
                return_value="/usr/bin/mooncake_master",
            ),
            mock.patch("os.path.exists", return_value=True),
            mock.patch("ray.remote", return_value=mock_decorator) as mock_ray_remote,
            mock.patch(
                "ray.get",
                return_value={"master_addr": "10.0.0.1:50051", "metadata_port": 8090},
            ),
            mock.patch("atexit.register"),
            mock.patch(
                "torchspec.ray.ray_actor.node_affinity_for_ip",
                return_value=None,
            ),
        ):
            result = launch_mooncake_master(args)

        mock_ray_remote.assert_called_once()
        call_kwargs = mock_ray_remote.call_args[1]
        assert call_kwargs["num_cpus"] == 0
        assert "env_vars" in call_kwargs["runtime_env"]
        assert result is mock_actor
        assert args.mooncake_master_server_address == "10.0.0.1:50051"
        assert args.mooncake_metadata_port == 8090
        assert args._mooncake_master_actor is mock_actor

    def test_auto_resolves_addr_and_port(self):
        args = Namespace()

        mock_actor = mock.MagicMock()
        mock_actor.start.remote.return_value = mock.MagicMock()
        mock_decorator, _ = self._make_mock_ray_remote(mock_actor)

        with (
            mock.patch(
                "torchspec.transfer.mooncake.utils.resolve_mooncake_master_bin",
                return_value="/usr/bin/mooncake_master",
            ),
            mock.patch("os.path.exists", return_value=True),
            mock.patch("torchspec.ray.ray_actor.get_current_node_ip", return_value="10.0.0.1"),
            mock.patch(
                "torchspec.ray.ray_actor.get_free_port",
                side_effect=[55000, 8500],
            ),
            mock.patch("ray.remote", return_value=mock_decorator),
            mock.patch(
                "ray.get",
                return_value={"master_addr": "10.0.0.1:55000", "metadata_port": 8500},
            ),
            mock.patch("atexit.register"),
            mock.patch(
                "torchspec.ray.ray_actor.node_affinity_for_ip",
                return_value=None,
            ),
        ):
            result = launch_mooncake_master(args)

        assert result is mock_actor
        assert args.mooncake_master_server_address == "10.0.0.1:55000"
        assert args.mooncake_metadata_port == 8500

    def test_returns_none_when_binary_not_found(self):
        args = Namespace(mooncake_master_server_address="localhost:50051")

        with (
            mock.patch(
                "torchspec.transfer.mooncake.utils.resolve_mooncake_master_bin",
                return_value="/nonexistent/path/mooncake_master",
            ),
            mock.patch("os.path.exists", return_value=False),
            mock.patch(
                "torchspec.ray.ray_actor.node_affinity_for_ip",
                return_value=None,
            ),
        ):
            result = launch_mooncake_master(args)
            assert result is None

    def test_returns_none_on_start_failure(self):
        args = Namespace(
            mooncake_master_server_address="localhost:50051",
            mooncake_metadata_port=8090,
            mooncake_http_host="0.0.0.0",
        )

        mock_actor = mock.MagicMock()
        mock_actor.start.remote.return_value = mock.MagicMock()
        mock_decorator, _ = self._make_mock_ray_remote(mock_actor)

        with (
            mock.patch(
                "torchspec.transfer.mooncake.utils.resolve_mooncake_master_bin",
                return_value="/usr/bin/mooncake_master",
            ),
            mock.patch("os.path.exists", return_value=True),
            mock.patch("ray.remote", return_value=mock_decorator),
            mock.patch("ray.get", side_effect=RuntimeError("start failed")),
            mock.patch(
                "torchspec.ray.ray_actor.node_affinity_for_ip",
                return_value=None,
            ),
        ):
            result = launch_mooncake_master(args)

        assert result is None


class TestMooncakeMasterIntegration:
    """Integration tests using a real subprocess (dummy script)."""

    def test_launches_real_subprocess(self):
        """Test that the actor class can launch and manage a real subprocess."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write("#!/bin/bash\n")
            f.write('echo "started"\n')
            f.write("while true; do sleep 1; done\n")
            script_path = f.name

        os.chmod(script_path, 0o755)

        try:
            actor = MooncakeMaster()

            with (
                mock.patch(
                    "torchspec.transfer.mooncake.utils.resolve_mooncake_master_bin",
                    return_value=script_path,
                ),
                mock.patch("torchspec.ray.ray_actor.get_current_node_ip", return_value="127.0.0.1"),
            ):
                info = actor.start(50051, 8090, "0.0.0.0")

            assert info["master_addr"] == "127.0.0.1:50051"
            assert info["metadata_port"] == 8090
            assert actor.health_check() is True
            assert actor.get_info() == info

            # Cleanup
            actor.shutdown()
            assert actor._process is None

        finally:
            os.unlink(script_path)
