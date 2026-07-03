"""Security-hardening tests: malformed payload rejection and sandbox enforcement.

These assert the environment fails *safely* on hostile input:

* the native task reader rejects truncated / negative-count / oversized-count
  payloads with a clean error instead of an out-of-bounds read or a giant
  allocation, and
* the bubblewrap + seccomp + rlimit sandbox contains a model-authored solver
  that tries to open a socket, exec a program, exhaust memory, or spin forever.

The sandbox tests need ``bwrap`` and unprivileged user namespaces; they skip
cleanly where those are unavailable (e.g. some CI runners).
"""

from __future__ import annotations

import struct
import subprocess
import tempfile
from pathlib import Path

import pytest

from src.decoder.rl.curriculum import TaskSpec
from src.decoder.rl.harness import (
    HarnessOptions,
    Sandbox,
    _INCLUDE_DIR,
    _MAGIC,
    _SDK_SOURCE,
    build_harness,
    bwrap_available,
    harness_available,
)
from src.decoder.rl.solver_env import SolverAuthoringEnv

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_METADATA = ROOT / "src" / "testing_setup" / "lorem_ipsum_aes_xts.json"

requires_toolchain = pytest.mark.skipif(
    not harness_available(), reason="C++ toolchain to build the harness is unavailable"
)
requires_sandbox = pytest.mark.skipif(
    not (harness_available() and bwrap_available()),
    reason="bwrap (bubblewrap) with user namespaces is unavailable",
)

_TRIVIAL_SOLVER = (
    '#include "solver_sdk.hpp"\n'
    "using namespace aes_xts_decoder::sdk;\n"
    "struct S : SolverBase { S(const SolverContext& c) : SolverBase(c) {} };\n"
    'extern "C" ISolver* create_solver(const SolverContext& c){ return new S(c); }\n'
)


def _run_harness_on_payload(payload: bytes) -> tuple[int, str, str]:
    """Invoke the harness binary directly on a raw task payload (unsandboxed).

    Returns (returncode, stdout, stderr). We bypass the Python serializer on
    purpose: the point is to feed the C++ reader bytes it must defend against.
    """

    binary = build_harness()
    with tempfile.TemporaryDirectory() as tmp:
        task_path = Path(tmp) / "tasks.bin"
        source_path = Path(tmp) / "solver.cpp"
        stderr_path = Path(tmp) / "compile.stderr"
        task_path.write_bytes(payload)
        source_path.write_text(_TRIVIAL_SOLVER, encoding="utf-8")
        cmd = [
            str(binary), str(task_path), str(source_path),
            str(_INCLUDE_DIR), str(_SDK_SOURCE),
            "30", "5", "1", "10", tmp, str(stderr_path),
            "0", "0", "0", "0", "0", "1",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return result.returncode, result.stdout, result.stderr


def _valid_single_task_header() -> bytes:
    """A structurally valid magic + count + start of one circuit config block."""
    out = bytearray(_MAGIC)
    out += struct.pack("<i", 1)  # num_tasks
    out += struct.pack("<q", 10)  # value_count
    out += struct.pack("<i", 2)  # aes_rounds
    out += struct.pack("<qqqqqq", 0, 10, 10, 32, 42, 32)  # start/count fields
    return bytes(out)


# ---------------------------------------------------------------------------
# Malformed / hostile task payloads
# ---------------------------------------------------------------------------


@requires_toolchain
def test_stale_axh1_magic_is_rejected() -> None:
    rc, _out, err = _run_harness_on_payload(b"AXH1" + struct.pack("<i", 1) + b"\x00" * 40)
    assert rc == 3
    assert "bad task magic" in err


@requires_toolchain
def test_truncated_payload_is_rejected() -> None:
    rc, _out, err = _run_harness_on_payload(_MAGIC + b"\x01")
    assert rc == 3
    assert "magic" in err  # too short for a valid magic+count header


@requires_toolchain
def test_negative_task_count_is_rejected() -> None:
    rc, _out, err = _run_harness_on_payload(_MAGIC + struct.pack("<i", -5))
    assert rc == 3
    assert "count" in err


@requires_toolchain
def test_payload_cut_mid_circuit_is_rejected() -> None:
    payload = _MAGIC + struct.pack("<i", 1) + struct.pack("<q", 100)
    rc, _out, err = _run_harness_on_payload(payload)
    assert rc == 3
    assert "unexpected end" in err


@requires_toolchain
def test_hostile_array_length_is_rejected_without_oom() -> None:
    # A ~2e9 element opcode array that would allocate ~8GB if trusted.
    payload = bytearray(_valid_single_task_header())
    payload += struct.pack("<dddd", 1, 1, 1, 1)  # weights (ascii/consistency/goal/coupling)
    payload += struct.pack("<ddd", 0, 0, 0)  # rho (ascii/consistency/goal)
    payload += struct.pack("<Q", 7)  # seed
    payload += struct.pack("<i", 2_000_000_000)  # opcodes length prefix
    rc, _out, err = _run_harness_on_payload(bytes(payload))
    assert rc == 3
    assert "exceeds" in err


@requires_toolchain
def test_negative_value_count_is_rejected() -> None:
    payload = bytearray(_MAGIC)
    payload += struct.pack("<i", 1)  # num_tasks
    payload += struct.pack("<q", -1)  # value_count negative
    payload += struct.pack("<i", 2)
    payload += struct.pack("<qqqqqq", 0, 10, 10, 32, 42, 32)
    rc, _out, err = _run_harness_on_payload(bytes(payload))
    assert rc == 3
    assert "negative" in err


# ---------------------------------------------------------------------------
# Sandbox behaviour (bwrap + seccomp + rlimits)
# ---------------------------------------------------------------------------


def _spec() -> TaskSpec:
    return TaskSpec(metadata_path=FIXTURE_METADATA, block_count=1, rounds_ladder=(1,), seed=7)


def _sandbox_env() -> SolverAuthoringEnv:
    opts = HarnessOptions(
        run_timeout=5.0, compile_timeout=60.0, epochs=1, sweeps_per_epoch=50,
        sandbox=Sandbox(enabled=True),
    )
    return SolverAuthoringEnv(options=opts)


def _hostile(body_in_ctor: str, extra_include: str = "") -> str:
    return (
        f"{extra_include}"
        '#include "solver_sdk.hpp"\n'
        "using namespace aes_xts_decoder::sdk;\n"
        "struct Bad : SolverBase {\n"
        f"  Bad(const SolverContext& c) : SolverBase(c) {{ {body_in_ctor} }}\n"
        "};\n"
        'extern "C" ISolver* create_solver(const SolverContext& c){ return new Bad(c); }\n'
    )


@requires_sandbox
def test_sandbox_runs_legit_solver() -> None:
    env = _sandbox_env()
    src = (ROOT / "examples" / "solvers" / "baseline" / "solver.cpp").read_text()
    bd = env.score_source(src, _spec())
    assert bd.compile_ok, bd.compile_stderr
    assert bd.per_task[0]["ran"] is True


@requires_sandbox
def test_sandbox_kills_socket_syscall() -> None:
    env = _sandbox_env()
    src = _hostile("volatile int fd = ::socket(2,1,0); (void)fd;", "#include <sys/socket.h>\n")
    bd = env.score_source(src, _spec())
    assert bd.compile_ok, bd.compile_stderr
    task = bd.per_task[0]
    assert task["crashed"] is True
    assert "seccomp" in task["error"]


@requires_sandbox
def test_sandbox_kills_exec_syscall() -> None:
    env = _sandbox_env()
    src = _hostile(
        'char* a[]={(char*)"/bin/echo",0}; ::execv("/bin/echo",a);',
        "#include <unistd.h>\n",
    )
    bd = env.score_source(src, _spec())
    assert bd.compile_ok, bd.compile_stderr
    task = bd.per_task[0]
    assert task["crashed"] is True
    assert "seccomp" in task["error"]


@requires_sandbox
def test_sandbox_times_out_infinite_loop() -> None:
    env = _sandbox_env()
    src = _hostile("while (true) {}")
    bd = env.score_source(src, _spec())
    assert bd.compile_ok, bd.compile_stderr
    assert bd.per_task[0]["timed_out"] is True


@requires_sandbox
def test_sandbox_contains_memory_bomb() -> None:
    env = _sandbox_env()
    # RLIMIT_AS makes the allocation fail; the process is killed/aborts or is
    # reaped by the wall-clock guard -- in no case does it OOM the host.
    src = _hostile(
        "std::vector<char*> v; for(;;){ char* p=new char[64*1024*1024];"
        " for(std::size_t i=0;i<64*1024*1024;i+=4096)p[i]=1; v.push_back(p);} ",
        "#include <vector>\n",
    )
    bd = env.score_source(src, _spec())
    assert bd.compile_ok, bd.compile_stderr
    task = bd.per_task[0]
    assert task["ran"] is False
    assert task["crashed"] or task["timed_out"]


def test_disabled_sandbox_requires_explicit_optin() -> None:
    from src.decoder.rl.harness import SandboxUnavailable, run_solver_on_tasks

    spec = _spec()
    tasks = spec.build_flat_tasks()
    opts = HarnessOptions(sandbox=Sandbox(enabled=False, allow_unsandboxed=False))
    with pytest.raises(SandboxUnavailable):
        run_solver_on_tasks(_TRIVIAL_SOLVER, tasks, opts)
