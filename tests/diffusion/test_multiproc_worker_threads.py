# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The diffusion executor sizes worker Torch threads by the local worker count, for the worker spawns only."""

import inspect
import os
from types import SimpleNamespace

import pytest
import torch

import vllm_omni.diffusion.executor.multiproc_executor as mpe

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_KEYS = ("OMP_NUM_THREADS", "VLLM_OMP_NUM_THREADS_SET_BY_VLLM")


class _Stop(Exception):
    pass


def _fake_vllm_helper(cores: int):
    """Mimics vLLM's set_multiprocessing_worker_envs (signature checked against the real one)."""
    real_sig = inspect.signature(mpe.set_multiprocessing_worker_envs)
    calls = []

    def helper(*args, **kwargs):
        bound = real_sig.bind(*args, **kwargs)  # TypeError if the call does not fit the real signature
        bound.apply_defaults()
        n = bound.arguments["local_world_size"]
        calls.append(n)
        if "OMP_NUM_THREADS" in os.environ:  # vLLM honours an existing value and returns early
            return
        threads = max(1, cores // n)
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["VLLM_OMP_NUM_THREADS_SET_BY_VLLM"] = "1"
        torch.set_num_threads(threads)

    return helper, calls


def _launch(monkeypatch, num_gpus: int, cores: int = 16):
    helper, calls = _fake_vllm_helper(cores)
    monkeypatch.setattr(mpe, "set_multiprocessing_worker_envs", helper)
    monkeypatch.setattr(mpe.mp, "set_start_method", lambda *a, **k: None)
    seen_at_start = []

    class _Proc:
        def __init__(self, *a, **k):
            pass

        def start(self):
            seen_at_start.append({k: os.environ.get(k) for k in _KEYS})

    class _Reader:
        def recv(self):
            raise _Stop

    class _Writer:
        def close(self):
            pass

    monkeypatch.setattr(mpe.mp, "Process", _Proc)
    monkeypatch.setattr(mpe.mp, "Pipe", lambda duplex=False: (_Reader(), _Writer()))
    executor = object.__new__(mpe.MultiprocDiffusionExecutor)
    executor.od_config = SimpleNamespace(num_gpus=num_gpus, worker_extension_cls=None, custom_pipeline_args=None)
    with pytest.raises(_Stop):
        executor._launch_workers(broadcast_handle=None, wake_events=[None] * num_gpus)
    return calls, seen_at_start


@pytest.fixture
def clean_env(monkeypatch):
    for k in _KEYS:
        monkeypatch.delenv(k, raising=False)
    threads = torch.get_num_threads()
    yield
    torch.set_num_threads(threads)


@pytest.mark.parametrize("num_gpus", [1, 8])
def test_workers_see_threads_sized_by_local_worker_count(monkeypatch, clean_env, num_gpus):
    calls, seen = _launch(monkeypatch, num_gpus, cores=16)
    # The default (no argument) means local_world_size=1: every worker would get the full core count.
    assert calls == [num_gpus]
    assert len(seen) == num_gpus
    assert all(s["OMP_NUM_THREADS"] == str(16 // num_gpus) for s in seen)


def test_parent_env_and_threads_are_restored(monkeypatch, clean_env):
    before = torch.get_num_threads()
    _launch(monkeypatch, 8, cores=16)
    assert all(k not in os.environ for k in _KEYS)
    assert torch.get_num_threads() == before


def test_user_provided_omp_num_threads_is_kept(monkeypatch, clean_env):
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    _, seen = _launch(monkeypatch, 8, cores=16)
    assert all(s["OMP_NUM_THREADS"] == "3" for s in seen)
    assert os.environ["OMP_NUM_THREADS"] == "3"
    assert "VLLM_OMP_NUM_THREADS_SET_BY_VLLM" not in os.environ
