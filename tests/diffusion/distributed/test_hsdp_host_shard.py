# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Host-side HSDP sharding gives the same shards and outputs as the default path.

Two CPU ranks (gloo). The default path is ``fully_shard`` on host parameters;
the host-shard path swaps them to ``meta``, lets FSDP2 build meta shards and
loads each rank's slice with ``load_state_dict(assign=True)``. Dim 0 of some
weights does not divide by the world size, so FSDP2's uneven (padded) shards
are covered, as are a tied parameter and an unsharded buffer.
"""

import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

pytestmark = [pytest.mark.diffusion, pytest.mark.parallel, pytest.mark.cpu, pytest.mark.core_model]

WORLD = 2


class _Block(nn.Module):
    def __init__(self, din: int, dout: int) -> None:
        super().__init__()
        self.proj = nn.Linear(din, dout)
        self.register_buffer("scale", torch.linspace(0.5, 1.5, dout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x) * self.scale


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inp = _Block(4, 5)  # 5 rows over 2 ranks: uneven
        self.mid = _Block(5, 3)  # 3 rows over 2 ranks: uneven
        self.out = nn.Linear(3, 5)
        self.head = nn.Linear(3, 5)
        self.head.weight = self.out.weight  # tied

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.mid(self.inp(x))
        return self.out(h) + self.head(h)


def _shard(model: nn.Module, mesh, host: bool) -> None:
    from torch.distributed.fsdp import fully_shard

    from vllm_omni.diffusion.distributed import hsdp

    kwargs = {"mesh": mesh, "reshard_after_forward": True}
    for child in (model.inp, model.mid):
        hsdp._fully_shard_from_host(child, dict(kwargs)) if host else fully_shard(child, **kwargs)
    hsdp._fully_shard_from_host(model, dict(kwargs)) if host else fully_shard(model, **kwargs)


def _worker(rank: int, init_file: str, out_dir: str) -> None:
    dist.init_process_group("gloo", rank=rank, world_size=WORLD, init_method=f"file://{init_file}")
    from torch.distributed import init_device_mesh

    mesh = init_device_mesh("cpu", (WORLD,))
    torch.manual_seed(0)
    ref = _Model()
    state = {k: v.clone() for k, v in ref.state_dict().items()}
    host_model = _Model()
    host_model.load_state_dict(state)
    tied_before = host_model.head.weight is host_model.out.weight
    _shard(ref, mesh, host=False)
    _shard(host_model, mesh, host=True)
    result = {"tied_before": tied_before, "tied_after": host_model.head.weight is host_model.out.weight}
    ref_params = dict(ref.named_parameters())
    result["same_param_names"] = sorted(ref_params) == sorted(n for n, _ in host_model.named_parameters())
    for name, param in host_model.named_parameters():
        a, b = ref_params[name].to_local(), param.to_local()
        result[f"local:{name}"] = (tuple(a.shape) == tuple(b.shape)) and torch.equal(a, b)
        result[f"placements:{name}"] = ref_params[name].placements == param.placements
    for name, buf in host_model.named_buffers():
        result[f"buffer:{name}"] = torch.equal(buf, dict(ref.named_buffers())[name])
    x = torch.randn(7, 4, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        result["forward"] = torch.equal(ref(x), host_model(x))
        result["forward_again"] = torch.equal(ref(x), host_model(x))
    result["no_meta"] = all(t.device.type != "meta" for t in host_model.state_dict().values())
    torch.save(result, os.path.join(out_dir, f"r{rank}.pt"))
    dist.destroy_process_group()


def _worker_entry(rank: int, init_file: str, out_dir: str) -> None:
    """Production entry (shard_model -> _fully_shard_placing) on a 2-D (replicate=1, shard=2) mesh, TPU branch forced."""
    os.environ["VLLM_OMNI_HSDP_HOST_SHARD"] = "1"
    dist.init_process_group("gloo", rank=rank, world_size=WORLD, init_method=f"file://{init_file}")
    from torch.distributed import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    from vllm_omni.diffusion.distributed import hsdp

    calls = []
    real = hsdp._fully_shard_from_host
    hsdp._fully_shard_from_host = lambda m, k: (calls.append(type(m).__name__), real(m, k))[1]
    hsdp._fsdp_cannot_move_from_cpu = lambda device_type: True
    hsdp._place_states_fsdp_cannot_move = lambda *a, **k: None  # FSDP2 can move CPU states itself on CPU
    mesh = init_device_mesh("cpu", (1, WORLD), mesh_dim_names=("replicate", "shard"))
    mp_policy = MixedPrecisionPolicy()
    torch.manual_seed(0)
    ref, test = _Model(), _Model()
    test.load_state_dict({k: v.clone() for k, v in ref.state_dict().items()})
    for child in (ref.inp, ref.mid):
        fully_shard(child, mesh=mesh, mp_policy=mp_policy)
    fully_shard(ref, mesh=mesh, mp_policy=mp_policy)
    hsdp.shard_model(test, hsdp_shard_conditions=[lambda n, m: n in ("inp", "mid")], mesh=mesh,
                     reshard_after_forward=True, mp_policy=mp_policy)
    ref_params = dict(ref.named_parameters())
    result = {"host_calls": list(calls),  # copy: calls is cleared below for the meta case
              "local": all(torch.equal(ref_params[n].to_local(), p.to_local()) for n, p in test.named_parameters())}
    x = torch.randn(7, 4, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        result["forward"] = torch.equal(ref(x), test(x))
    # A module whose parameters are already on meta (e.g. a pre-sharded load) takes the default path.
    with torch.device("meta"):
        on_meta = _Block(4, 5)
    calls.clear()
    try:
        hsdp._fully_shard_placing(on_meta, {"mesh": mesh, "mp_policy": mp_policy})
        result["meta_ok"] = not calls and all(p.device.type == "meta" for p in on_meta.parameters())
    except Exception as e:  # noqa: BLE001
        result["meta_ok"] = repr(e)
    torch.save(result, os.path.join(out_dir, f"e{rank}.pt"))
    dist.destroy_process_group()


@pytest.fixture(scope="module")
def entry_results():
    with tempfile.TemporaryDirectory() as d:
        mp.spawn(_worker_entry, args=(os.path.join(d, "init"), d), nprocs=WORLD, join=True)
        yield [torch.load(os.path.join(d, f"e{r}.pt")) for r in range(WORLD)]


def test_production_entry_on_2d_mesh_uses_host_shard_and_matches(entry_results):
    for rank, res in enumerate(entry_results):
        assert res["host_calls"] == ["_Block", "_Block", "_Model"], f"rank {rank}: {res['host_calls']}"
        assert res["local"] and res["forward"], f"rank {rank}"


def test_states_already_on_meta_take_the_default_path(entry_results):
    for rank, res in enumerate(entry_results):
        assert res["meta_ok"] is True, f"rank {rank}: {res['meta_ok']}"


@pytest.fixture(scope="module")
def results():
    with tempfile.TemporaryDirectory() as d:
        init_file = os.path.join(d, "init")
        mp.spawn(_worker, args=(init_file, d), nprocs=WORLD, join=True)
        yield [torch.load(os.path.join(d, f"r{r}.pt")) for r in range(WORLD)]


def test_host_shard_matches_default_shards(results):
    for rank, res in enumerate(results):
        bad = [k for k, v in res.items() if k.startswith(("local:", "placements:", "buffer:")) and not v]
        assert not bad, f"rank {rank}: {bad}"
        assert any(k.startswith("local:") for k in res), "no parameters compared"


def test_host_shard_forward_is_bitwise_equal(results):
    for rank, res in enumerate(results):
        assert res["forward"] and res["forward_again"], f"rank {rank}"


def test_host_shard_leaves_no_meta_tensors(results):
    for res in results:
        assert res["no_meta"]


def test_host_shard_keeps_tied_parameters_tied(results):
    for rank, res in enumerate(results):
        assert res["tied_before"], "the test model must start tied"
        assert res["tied_after"], f"rank {rank}: tied parameter split into two"
        assert res["same_param_names"], f"rank {rank}: parameter names differ from the default path"

