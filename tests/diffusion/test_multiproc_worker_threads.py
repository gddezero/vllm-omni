# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The diffusion executor must size worker Torch threads by the local worker count."""

from types import SimpleNamespace

import pytest

import vllm_omni.diffusion.executor.multiproc_executor as mpe


class _StopBeforeSpawn(Exception):
    pass


@pytest.mark.parametrize("num_gpus", [1, 8])
def test_launch_workers_passes_local_worker_count(monkeypatch, num_gpus):
    seen = []
    monkeypatch.setattr(mpe, "set_multiprocessing_worker_envs", lambda *a, **k: seen.append((a, k)))

    def stop(*_a, **_k):
        raise _StopBeforeSpawn

    monkeypatch.setattr(mpe.mp, "set_start_method", stop)
    executor = object.__new__(mpe.MultiprocDiffusionExecutor)
    executor.od_config = SimpleNamespace(num_gpus=num_gpus, worker_extension_cls=None, custom_pipeline_args=None)

    with pytest.raises(_StopBeforeSpawn):
        executor._launch_workers(broadcast_handle=None, wake_events=[None] * num_gpus)

    # Default (no argument) means local_world_size=1: every worker would get the full core count.
    assert seen == [((num_gpus,), {})]
