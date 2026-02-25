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

from typing import List, Optional

import torch

from torchspec.utils.logging import logger


class HostBuffer:
    """
    RDMA-registered host buffer using pinned CPU memory.

    Provides a reusable buffer for zero-copy transfers to Mooncake Store.
    """

    def __init__(self, size: int):
        self.size = size
        # Allocate pinned CPU memory for RDMA compatibility
        self._tensor = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        self._ptr = self._tensor.data_ptr()

    @property
    def ptr(self) -> int:
        return self._ptr

    def copy_from_tensor(self, tensor: torch.Tensor, offset: int = 0) -> int:
        """
        Copy tensor data into this host buffer at the given offset.

        Returns the number of bytes copied.
        """
        tensor = tensor.contiguous()
        nbytes = tensor.numel() * tensor.element_size()

        if offset + nbytes > self.size:
            raise ValueError(f"Buffer overflow: need {offset + nbytes}, have {self.size}")

        # Get a view into our buffer at the right offset
        host_view = self._tensor[offset : offset + nbytes]

        # PyTorch .copy_() handles CUDA→pinned-CPU directly in one DMA.
        # No need for .cpu() which would create an intermediate unpinned copy.
        host_view.copy_(tensor.view(torch.uint8).view(-1))

        return nbytes

    def free(self) -> None:
        if self._tensor is not None:
            del self._tensor
            self._tensor = None
            self._ptr = 0

    def __del__(self):
        self.free()


class HostBufferPool:
    """
    Pool of RDMA-registered host buffers for tensor storage.
    """

    def __init__(self, buffer_size: int = 4 * 1024**3, pool_size: int = 2):
        self.buffer_size = buffer_size
        self.pool_size = pool_size
        self._buffers: List[HostBuffer] = []
        self._current_idx = 0

    def initialize(self) -> None:
        """Pre-allocate all buffers."""
        for _ in range(self.pool_size):
            self._buffers.append(HostBuffer(self.buffer_size))
        logger.info(
            "Initialized host buffer pool: %s x %.1fGB",
            self.pool_size,
            self.buffer_size / (1024**3),
        )

    def get_buffer(self) -> HostBuffer:
        """Get the next buffer (round-robin)."""
        if not self._buffers:
            self.initialize()
        buf = self._buffers[self._current_idx]
        self._current_idx = (self._current_idx + 1) % len(self._buffers)
        return buf

    def shutdown(self) -> None:
        for buf in self._buffers:
            buf.free()
        self._buffers.clear()


class _GPUBuffer:
    """Base class for RDMA-registered GPU buffers (send and receive).

    Provides allocation, pointer access, free, and lazy initialization.
    Subclasses add direction-specific methods (copy_from_tensor, get_slice).
    """

    _label: str = "GPU"  # overridden by subclasses for log messages

    def __init__(self, size: int, device: torch.device = None):
        self.size = size
        self.device = torch.device(device) if device is not None else torch.device("cuda")
        self._tensor: Optional[torch.Tensor] = None
        self._ptr: int = 0
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return
        self._tensor = torch.empty(self.size, dtype=torch.uint8, device=self.device)
        self._ptr = self._tensor.data_ptr()
        self._initialized = True
        logger.info(
            "Initialized %s buffer: %.1fMB on %s",
            self._label,
            self.size / (1024**2),
            self.device,
        )

    @property
    def ptr(self) -> int:
        return self._ptr

    def free(self) -> None:
        if self._tensor is not None:
            del self._tensor
            self._tensor = None
            self._ptr = 0
            self._initialized = False

    def __del__(self):
        self.free()


class GPUReceiveBuffer(_GPUBuffer):
    """RDMA-registered GPU buffer for receiving tensors via GPU Direct RDMA.

    Allocates contiguous GPU memory that can be registered with Mooncake
    for direct RDMA transfers without CPU involvement.
    """

    _label = "GPU receive"

    def get_slice(self, offset: int, size: int) -> torch.Tensor:
        """Get a slice of the buffer as a tensor view."""
        if not self._initialized:
            raise RuntimeError("GPU buffer not initialized")
        return self._tensor[offset : offset + size]


class GPUSendBuffer(_GPUBuffer):
    """RDMA-registered GPU buffer for sending tensors via GPU Direct RDMA.

    Pre-allocates contiguous GPU memory registered with Mooncake so that
    put operations can stage tensors with a fast GPU-to-GPU copy (~5ms for 1.2GB)
    instead of GPU-to-CPU (~150ms). The NIC reads directly from GPU memory via
    nvidia_peermem (GPU Direct RDMA).
    """

    _label = "GPU send"

    def copy_from_tensor(self, tensor: torch.Tensor, offset: int = 0) -> int:
        """Copy tensor data into this GPU buffer at the given offset.

        The source tensor must be on the same CUDA device. This is a fast
        GPU-to-GPU copy within device memory.

        Returns the number of bytes copied.
        """
        if not self._initialized:
            raise RuntimeError("GPU send buffer not initialized")

        tensor = tensor.contiguous()
        nbytes = tensor.numel() * tensor.element_size()

        if offset + nbytes > self.size:
            raise ValueError(f"Buffer overflow: need {offset + nbytes}, have {self.size}")

        gpu_view = self._tensor[offset : offset + nbytes]
        gpu_view.copy_(tensor.view(torch.uint8).view(-1))

        return nbytes
