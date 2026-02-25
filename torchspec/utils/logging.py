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

import logging
import os

import torch.distributed as dist

from torchspec.utils import wandb as wandb_utils


def _get_logger_level():
    level_str = os.getenv("TORCHSPEC_LOG_LEVEL", "INFO").upper()
    try:
        log_level = getattr(logging, level_str)
    except ValueError:
        logging.warning("Invalid log level: %s, defaulting to WARNING", level_str)
    return log_level


def setup_logger(log_level=None, actor_name=None, ip_addr=None):
    logger_name = "TorchSpec" if actor_name is None else f"TorchSpec-{actor_name}"
    _logger = logging.getLogger(logger_name)
    _logger.handlers.clear()
    _logger.propagate = False
    if log_level is None:
        log_level = _get_logger_level()
    _logger.setLevel(log_level)
    handler = logging.StreamHandler()
    if ip_addr is None:
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(filename)s:%(lineno)d %(levelname)s %(message)s")
        )
    else:
        rank = os.environ.get("RANK", 0)
        handler.setFormatter(
            logging.Formatter(
                f"[%(asctime)s{ip_addr} RANK:{rank}] %(filename)s:%(lineno)d %(levelname)s %(message)s"
            )
        )
    handler.setLevel(log_level)
    _logger.addHandler(handler)
    return _logger


logger = setup_logger()


def print_with_rank(message):
    if dist.is_available() and dist.is_initialized():
        logger.info(f"rank {dist.get_rank()}: {message}")
    else:
        logger.info(f"non-distributed: {message}")


def print_on_rank0(message):
    if dist.get_rank() == 0:
        logger.info(message)


def init_tracking(args, primary: bool = True, **kwargs):
    if primary:
        wandb_utils.init_wandb_primary(args, **kwargs)
    else:
        wandb_utils.init_wandb_secondary(args, **kwargs)
