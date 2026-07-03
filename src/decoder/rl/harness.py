"""Python driver for the C++ compile-and-run harness.

This module owns the boundary between the Python environment and the native
harness: it flattens a :class:`~src.decoder.circuit.Circuit` into the wire
format the harness reads, builds the standalone harness binary on demand (and
caches it), and parses the harness output back into per-task measurements.

Keeping the harness a subprocess (rather than a Cython symbol) is deliberate:
authored solvers are compiled on demand and run in a forked child under a
wall-clock budget, so a hang or crash is a bounded, shaped failure rather than a
trainer outage.
"""

from __future__ import annotations

import hashlib
import os
import re
import shlex
import shutil
import struct
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from ..circuit import Circuit
from ..constraints import classify_constraints

_NATIVE_DIR = Path(__file__).resolve().parent / "native"
_INCLUDE_DIR = _NATIVE_DIR / "include"
_SDK_SOURCE = _NATIVE_DIR / "sdk" / "solver_sdk.cpp"
_HARNESS_DIR = _NATIVE_DIR / "harness"
_HARNESS_SOURCES = (
    _HARNESS_DIR / "harness_main.cpp",
    _HARNESS_DIR / "harness.cpp",
    _SDK_SOURCE,
)
_BUILD_DIR = _HARNESS_DIR / "_build"

_DATA_ROUND_RE = re.compile(r":data:round_(\d+):")

# Wire-format magic; kept in lockstep with kMagic in harness_main.cpp. Bumped to
# AXH2 when the per-task config block gained the augmented-Lagrangian penalty
# (rho) coefficients, so a stale harness binary refuses a newer payload instead
# of silently misreading it.
_MAGIC = b"AXH2"

# The int32 length prefixes in the wire format are signed; guard against a
# caller handing us an array the harness's bounds-checked reader would reject.
_INT32_MAX = 2 ** 31 - 1


class HarnessUnavailable(RuntimeError):
    """Raised when the C++ toolchain needed to build the harness is missing."""


class SandboxUnavailable(RuntimeError):
    """Raised when sandboxed execution is required but ``bwrap`` is missing.

    Refusing to run untrusted, model-authored C++ without isolation is a
    deliberate fail-loud: a CI box without ``bwrap`` should error rather than
    silently compiling and executing arbitrary code with full ambient
    privileges. Pass ``Sandbox(allow_unsandboxed=True)`` to opt out explicitly.
    """


# ---------------------------------------------------------------------------
# Sandbox policy for untrusted solver compile + execution
# ---------------------------------------------------------------------------

_GIB = 1024 ** 3
_MIB = 1024 ** 2


@dataclass
class Sandbox:
    """Isolation policy for compiling and running model-authored solvers.

    The harness runs untrusted C++: it compiles a model-authored translation
    unit and ``dlopen``s the result. Two layers of defence are configured here:

    * an outer ``bubblewrap`` (``bwrap``) jail giving the harness a fresh set of
      namespaces (no network, private PID/IPC/UTS, a read-only view of the
      toolchain, and a writable scratch dir only), and
    * inner ``setrlimit`` caps plus a ``seccomp-bpf`` syscall whitelist installed
      by the harness child itself (see ``harness.cpp``), which bound memory/CPU
      and kill hostile syscalls even against static initializers that run at
      ``dlopen`` time.

    ``address_space_bytes`` / ``open_files`` / ``file_size_bytes`` map to
    ``RLIMIT_AS`` / ``RLIMIT_NOFILE`` / ``RLIMIT_FSIZE``. ``cpu_seconds`` and
    ``processes`` map to ``RLIMIT_CPU`` / ``RLIMIT_NPROC`` and are left untouched
    when ``0`` (``RLIMIT_NPROC`` interferes with OpenMP threading, so it stays
    ``0`` by default and the seccomp filter blocks ``fork``/``execve`` instead).
    """

    enabled: bool = True
    no_network: bool = True
    seccomp: bool = True
    address_space_bytes: int = 2 * _GIB
    cpu_seconds: int = 0
    open_files: int = 256
    processes: int = 0
    file_size_bytes: int = 256 * _MIB
    allow_unsandboxed: bool = False

    def rlimit_args(self) -> list[str]:
        """Positional rlimit/seccomp args appended to the harness command line."""
        return [
            str(int(self.address_space_bytes)),
            str(int(self.cpu_seconds)),
            str(int(self.open_files)),
            str(int(self.processes)),
            str(int(self.file_size_bytes)),
            "1" if self.seccomp else "0",
        ]


def bwrap_available() -> bool:
    """True if the ``bwrap`` binary is on PATH."""
    return shutil.which("bwrap") is not None


_BWRAP_RO_TRY = (
    "/bin",
    "/sbin",
    "/lib",
    "/lib64",
    "/etc/alternatives",
    "/etc/ld.so.cache",
    "/etc/ld.so.conf",
    "/etc/ld.so.conf.d",
    "/usr/lib/gcc",
)


def _sandbox_prefix(sandbox: Sandbox, workdir: str, ro_paths: list[Path]) -> list[str]:
    """Build the ``bwrap`` argv prefix that isolates a harness invocation.

    ``workdir`` (the writable scratch dir the harness compiles/builds inside) is
    bound read-write; the toolchain and the read-only ``ro_paths`` (SDK include
    dir, sdk source, harness binary) are bound read-only. All binds are identity
    (same path inside and out) so the paths handed to the harness are valid in
    both namespaces. Networking is dropped unless ``no_network`` is False.
    """

    args = [
        "bwrap",
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--proc", "/proc",
        "--dev", "/dev",
        "--tmpfs", "/tmp",
        "--ro-bind", "/usr", "/usr",
    ]
    if not sandbox.no_network:
        args.append("--share-net")
    for path in _BWRAP_RO_TRY:
        args += ["--ro-bind-try", path, path]
    seen: set[str] = set()
    for ro in ro_paths:
        resolved = str(Path(ro).resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        args += ["--ro-bind-try", resolved, resolved]
    # Writable scratch dir + point the compiler's intermediate files at it so we
    # never need a writable /tmp shared with the host.
    args += ["--bind", workdir, workdir]
    args += ["--setenv", "TMPDIR", workdir]
    args += ["--setenv", "PATH", "/usr/bin:/bin"]
    args.append("--")
    return args


def _wrap_command(
    command: list[str],
    sandbox: Sandbox,
    workdir: str,
    ro_paths: list[Path],
) -> list[str]:
    """Return ``command`` wrapped in the sandbox, or the bare command if opted out.

    Raises :class:`SandboxUnavailable` when isolation is requested but ``bwrap``
    is not installed and ``allow_unsandboxed`` was not set.
    """

    if not sandbox.enabled:
        if not sandbox.allow_unsandboxed:
            raise SandboxUnavailable(
                "sandbox is disabled but allow_unsandboxed is False; refusing to "
                "run untrusted solver code. Set Sandbox(enabled=True) to isolate "
                "with bwrap, or Sandbox(enabled=False, allow_unsandboxed=True) to "
                "explicitly accept the risk."
            )
        return command
    if not bwrap_available():
        if sandbox.allow_unsandboxed:
            return command
        raise SandboxUnavailable(
            "bwrap (bubblewrap) is not installed, so untrusted solver code cannot "
            "be isolated. Install bubblewrap, or pass "
            "Sandbox(allow_unsandboxed=True) to run without isolation (unsafe)."
        )
    return _sandbox_prefix(sandbox, workdir, ro_paths) + command


# ---------------------------------------------------------------------------
# Round tagging
# ---------------------------------------------------------------------------


def constraint_round_indices(circuit: Circuit, aes_rounds: int) -> list[int]:
    """Assign each constraint the AES data-path round it belongs to.

    Data-path consistency constraints (``block_*:data:round_r:*``) are tagged
    with their round ``r`` in ``1..aes_rounds``; everything else -- ASCII, the
    key-2 tweak schedule, round-key derivation, prewhitening, the cipher XOR and
    the goal equalities -- is tagged round ``0`` (a structural/boundary bucket
    that the staircase ignores). This makes the per-round staircase measure
    exactly "how many consecutive AES rounds of the data path are internally
    consistent", which is the dense ``1,2,3,...,n`` signal the environment
    rewards.
    """

    rounds: list[int] = []
    for constraint in circuit.constraints:
        match = _DATA_ROUND_RE.search(constraint.label)
        if match:
            value = int(match.group(1))
            rounds.append(value if 1 <= value <= aes_rounds else 0)
        else:
            rounds.append(0)
    return rounds


# ---------------------------------------------------------------------------
# Flattening + serialization
# ---------------------------------------------------------------------------


@dataclass
class TaskWeights:
    # Per-class Lagrange multipliers (linear term).
    ascii_weight: float = 1.0
    consistency_weight: float = 1.0
    goal_weight: float = 1.0
    # Per-class quadratic penalty coefficients (rho) of the augmented Lagrangian
    # Energy = sum_i [ lambda_i r_i + 1/2 rho_i r_i^2 ].
    ascii_rho: float = 0.0
    consistency_rho: float = 0.0
    goal_rho: float = 0.0
    # Scales both the consistency lambda and rho (inter-round coupling ramp).
    coupling_weight: float = 1.0
    seed: int = 0


@dataclass
class FlatCircuit:
    """A circuit flattened into the harness wire format."""

    aes_rounds: int
    value_count: int
    plaintext_start: int
    plaintext_count: int
    key1_start: int
    key1_count: int
    key2_start: int
    key2_count: int
    opcodes: list[int]
    outputs: list[int]
    input_offsets: list[int]
    input_counts: list[int]
    inputs: list[int]
    immediate_offsets: list[int]
    immediate_counts: list[int]
    immediates: list[int]
    const_offsets: list[int]
    const_counts: list[int]
    constants: bytes
    constraint_kinds: list[int]
    constraint_left: list[int]
    constraint_right: list[int]
    constraint_classes: list[int]
    constraint_rounds: list[int]
    wire_value_ids: list[int]
    wire_offsets: list[int]
    xts_block_sectors: list[int]
    xts_block_indices: list[int]
    xts_block_offsets: list[int]
    xts_block_targets: bytes
    value_widths: list[int]
    constraint_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.constraint_count = len(self.constraint_kinds)


def reference_energy(
    residuals: list[float],
    classes: list[int],
    weights: TaskWeights,
) -> float:
    """Pure-Python mirror of ``SdkCircuit::Energy`` for cross-checking the SDK.

    Computes the augmented Lagrangian ``sum_i [ lambda_i r_i + 1/2 rho_i r_i^2 ]``
    with the same per-class multipliers/penalties and coupling scaling the C++
    uses, so a test can assert the native energy matches the reference exactly.
    """

    from ..constraints import ConstraintClass

    energy = 0.0
    for residual, cls in zip(residuals, classes):
        r = float(residual)
        if r == 0.0:
            continue
        if cls == ConstraintClass.ASCII:
            lam = weights.ascii_weight
            rho = weights.ascii_rho
        elif cls == ConstraintClass.CONSISTENCY:
            lam = weights.consistency_weight * weights.coupling_weight
            rho = weights.consistency_rho * weights.coupling_weight
        else:
            lam = weights.goal_weight
            rho = weights.goal_rho
        energy += lam * r + 0.5 * rho * r * r
    return energy


def flatten_circuit(circuit: Circuit, aes_rounds: int) -> FlatCircuit:
    opcodes: list[int] = []
    outputs: list[int] = []
    input_offsets: list[int] = []
    input_counts: list[int] = []
    inputs: list[int] = []
    immediate_offsets: list[int] = []
    immediate_counts: list[int] = []
    immediates: list[int] = []
    for op in circuit.ops:
        opcodes.append(int(op.opcode))
        outputs.append(op.output)
        input_offsets.append(len(inputs))
        input_counts.append(len(op.inputs))
        inputs.extend(op.inputs)
        immediate_offsets.append(len(immediates))
        immediate_counts.append(len(op.immediates))
        immediates.extend(op.immediates)

    const_offsets: list[int] = []
    const_counts: list[int] = []
    constants = bytearray()
    for data in circuit.constants:
        const_offsets.append(len(constants))
        const_counts.append(len(data))
        constants.extend(data)

    classes = classify_constraints(circuit)
    rounds = constraint_round_indices(circuit, aes_rounds)
    constraint_kinds: list[int] = []
    constraint_left: list[int] = []
    constraint_right: list[int] = []
    for constraint in circuit.constraints:
        constraint_kinds.append(int(constraint.kind))
        constraint_left.append(constraint.left)
        constraint_right.append(-1 if constraint.right is None else constraint.right)

    wire_value_ids = list(circuit.wire_offsets.keys())
    wire_offsets = [circuit.wire_offsets[value_id] for value_id in wire_value_ids]

    sectors: list[int] = []
    indices: list[int] = []
    offsets: list[int] = []
    targets = bytearray()
    for block in circuit.xts_blocks:
        sectors.append(block.sector_number)
        indices.append(block.block_index_in_sector)
        offsets.append(block.plaintext_offset)
        targets.extend(block.ciphertext[:16].ljust(16, b"\x00"))

    plaintext_start, plaintext_count = circuit.input_ranges.get("plaintext", (0, 0))
    key1_start, key1_count = circuit.input_ranges.get("key1", (0, 0))
    key2_start, key2_count = circuit.input_ranges.get("key2", (0, 0))

    return FlatCircuit(
        aes_rounds=aes_rounds,
        value_count=len(circuit.values),
        plaintext_start=plaintext_start,
        plaintext_count=plaintext_count,
        key1_start=key1_start,
        key1_count=key1_count,
        key2_start=key2_start,
        key2_count=key2_count,
        opcodes=opcodes,
        outputs=outputs,
        input_offsets=input_offsets,
        input_counts=input_counts,
        inputs=inputs,
        immediate_offsets=immediate_offsets,
        immediate_counts=immediate_counts,
        immediates=immediates,
        const_offsets=const_offsets,
        const_counts=const_counts,
        constants=bytes(constants),
        constraint_kinds=constraint_kinds,
        constraint_left=constraint_left,
        constraint_right=constraint_right,
        constraint_classes=[int(c) for c in classes],
        constraint_rounds=rounds,
        wire_value_ids=wire_value_ids,
        wire_offsets=wire_offsets,
        xts_block_sectors=sectors,
        xts_block_indices=indices,
        xts_block_offsets=offsets,
        xts_block_targets=bytes(targets),
        value_widths=[value.width for value in circuit.values],
    )


def _check_len(n: int) -> int:
    if n > _INT32_MAX:
        raise ValueError(f"array length {n} exceeds wire-format int32 limit")
    return n


def _pack_int32_array(values: list[int]) -> bytes:
    _check_len(len(values))
    return struct.pack("<i", len(values)) + struct.pack(f"<{len(values)}i", *values)


def _pack_uint16_array(values: list[int]) -> bytes:
    _check_len(len(values))
    return struct.pack("<i", len(values)) + struct.pack(f"<{len(values)}H", *values)


def _pack_byte_array(data: bytes) -> bytes:
    _check_len(len(data))
    return struct.pack("<i", len(data)) + bytes(data)


def _serialize_task(flat: FlatCircuit, weights: TaskWeights) -> bytes:
    out = bytearray()
    out += struct.pack("<q", flat.value_count)
    out += struct.pack("<i", flat.aes_rounds)
    out += struct.pack("<qqqqqq", flat.plaintext_start, flat.plaintext_count,
                       flat.key1_start, flat.key1_count, flat.key2_start, flat.key2_count)
    out += struct.pack("<dddd", weights.ascii_weight, weights.consistency_weight,
                       weights.goal_weight, weights.coupling_weight)
    out += struct.pack("<ddd", weights.ascii_rho, weights.consistency_rho,
                       weights.goal_rho)
    out += struct.pack("<Q", weights.seed & 0xFFFFFFFFFFFFFFFF)
    for arr in (
        flat.opcodes, flat.outputs, flat.input_offsets, flat.input_counts, flat.inputs,
        flat.immediate_offsets, flat.immediate_counts, flat.immediates,
        flat.const_offsets, flat.const_counts, flat.constraint_kinds,
        flat.constraint_left, flat.constraint_right, flat.constraint_classes,
        flat.constraint_rounds, flat.wire_value_ids, flat.wire_offsets,
        flat.xts_block_sectors, flat.xts_block_indices, flat.xts_block_offsets,
    ):
        out += _pack_int32_array(arr)
    out += _pack_byte_array(flat.constants)
    out += _pack_byte_array(flat.xts_block_targets)
    out += _pack_uint16_array(flat.value_widths)
    return bytes(out)


def serialize_tasks(tasks: list[tuple[FlatCircuit, TaskWeights]]) -> bytes:
    out = bytearray(_MAGIC)
    out += struct.pack("<i", len(tasks))
    for flat, weights in tasks:
        out += _serialize_task(flat, weights)
    return bytes(out)


# ---------------------------------------------------------------------------
# Building / invoking the harness binary
# ---------------------------------------------------------------------------


def _sources_fingerprint() -> str:
    hasher = hashlib.sha256()
    for source in _HARNESS_SOURCES + (_INCLUDE_DIR / "solver_sdk.hpp", _HARNESS_DIR / "harness.hpp"):
        hasher.update(Path(source).read_bytes())
    return hasher.hexdigest()[:16]


def _default_compiler() -> str:
    import os

    return os.environ.get("CXX", "c++")


def harness_available() -> bool:
    try:
        build_harness()
        return True
    except HarnessUnavailable:
        return False


def build_harness(compiler: str | None = None) -> Path:
    """Compile (and cache) the standalone harness binary; return its path."""

    compiler = compiler or _default_compiler()
    _BUILD_DIR.mkdir(parents=True, exist_ok=True)
    binary = _BUILD_DIR / f"harness_{_sources_fingerprint()}"
    if binary.exists():
        return binary

    command = [
        *shlex.split(compiler), "-std=c++17", "-O2", "-fopenmp",
        f"-I{_INCLUDE_DIR}", f"-I{_HARNESS_DIR}",
        *[str(source) for source in _HARNESS_SOURCES],
        "-o", str(binary), "-ldl",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:  # pragma: no cover - depends on environment
        raise HarnessUnavailable(f"C++ compiler {compiler!r} not found") from exc
    if result.returncode != 0 or not binary.exists():
        raise HarnessUnavailable(
            f"failed to build harness binary:\n{result.stderr}"
        )
    return binary


@dataclass
class TaskResult:
    aes_rounds: int
    ran: bool
    feasible: bool
    timed_out: bool
    crashed: bool
    energy: float
    best_hamming: float
    round_staircase: float
    seconds: float
    per_round: list[float]
    error: str
    constraint_count: int
    # Residual mass per constraint class [ascii, consistency, goal]; consumed by
    # the method-of-multipliers dual update in the curriculum.
    per_class: list[float] = field(default_factory=list)


@dataclass
class HarnessRun:
    compile_ok: bool
    compile_seconds: float
    compile_stderr: str
    tasks: list[TaskResult]


@dataclass
class HarnessOptions:
    compile_timeout: float = 60.0
    run_timeout: float = 10.0
    epochs: int = 1
    sweeps_per_epoch: int = 4000
    compiler: str | None = None
    sandbox: Sandbox = field(default_factory=Sandbox)


def run_solver_on_tasks(
    source: str,
    tasks: list[tuple[FlatCircuit, TaskWeights]],
    options: HarnessOptions | None = None,
) -> HarnessRun:
    options = options or HarnessOptions()
    binary = build_harness(options.compiler)

    with tempfile.TemporaryDirectory(prefix="aes_harness_") as tmp:
        tmp_path = Path(tmp)
        task_path = tmp_path / "tasks.bin"
        source_path = tmp_path / "solver.cpp"
        stderr_path = tmp_path / "compile.stderr"
        task_path.write_bytes(serialize_tasks(tasks))
        source_path.write_text(source, encoding="utf-8")

        command = [
            str(binary), str(task_path), str(source_path),
            str(_INCLUDE_DIR), str(_SDK_SOURCE),
            str(options.compile_timeout), str(options.run_timeout),
            str(options.epochs), str(options.sweeps_per_epoch),
            tmp, str(stderr_path),
            *options.sandbox.rlimit_args(),
        ]
        command = _wrap_command(
            command,
            options.sandbox,
            tmp,
            ro_paths=[binary, _INCLUDE_DIR, _SDK_SOURCE, _HARNESS_DIR],
        )
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise HarnessUnavailable(
                f"harness binary failed (code {result.returncode}):\n{result.stderr}"
            )

        compile_ok = False
        compile_seconds = 0.0
        parsed: list[TaskResult] = []
        for line in result.stdout.splitlines():
            fields = line.split("\t")
            if fields[0] == "COMPILE":
                compile_ok = fields[1] == "1"
                compile_seconds = float(fields[2])
            elif fields[0] == "TASK":
                index = int(fields[1])
                per_round = [float(x) for x in fields[10].split(",") if x] if len(fields) > 10 else []
                per_class = [float(x) for x in fields[11].split(",") if x] if len(fields) > 11 else []
                parsed.append(
                    TaskResult(
                        aes_rounds=tasks[index][0].aes_rounds,
                        ran=fields[2] == "1",
                        feasible=fields[3] == "1",
                        timed_out=fields[4] == "1",
                        crashed=fields[5] == "1",
                        energy=float(fields[6]),
                        best_hamming=float(fields[7]),
                        round_staircase=float(fields[8]),
                        seconds=float(fields[9]),
                        per_round=per_round,
                        error=fields[12] if len(fields) > 12 else "",
                        constraint_count=tasks[index][0].constraint_count,
                        per_class=per_class,
                    )
                )
        compile_stderr = stderr_path.read_text(encoding="utf-8") if stderr_path.exists() else ""
        return HarnessRun(
            compile_ok=compile_ok,
            compile_seconds=compile_seconds,
            compile_stderr=compile_stderr,
            tasks=parsed,
        )


@dataclass
class ProbeResult:
    hamming: float
    energy: float
    per_round: list[float]
    residuals: list[float]


def probe_assignment(
    circuit: Circuit,
    aes_rounds: int,
    plaintext: bytes,
    key1: bytes,
    key2: bytes,
    weights: TaskWeights | None = None,
    compiler: str | None = None,
) -> ProbeResult:
    """Score one fixed assignment through the SDK, for cross-checking Python.

    Returns the SDK's hard hamming score, augmented-Lagrangian energy, per-round
    residual buckets and full residual vector so a test can assert they match
    ``circuit.evaluate`` on the same assignment.
    """

    weights = weights or TaskWeights()
    binary = build_harness(compiler)
    flat = flatten_circuit(circuit, aes_rounds)
    with tempfile.TemporaryDirectory(prefix="aes_probe_") as tmp:
        tmp_path = Path(tmp)
        task_path = tmp_path / "task.bin"
        assignment_path = tmp_path / "assignment.bin"
        task_path.write_bytes(serialize_tasks([(flat, weights)]))
        assignment_path.write_bytes(
            _pack_byte_array(plaintext) + _pack_byte_array(key1) + _pack_byte_array(key2)
        )
        result = subprocess.run(
            [str(binary), "--probe", str(task_path), str(assignment_path)],
            capture_output=True, text=True, check=False,
        )
    if result.returncode != 0:
        raise HarnessUnavailable(f"probe failed:\n{result.stderr}")
    for line in result.stdout.splitlines():
        fields = line.split("\t")
        if fields[0] == "PROBE":
            per_round = [float(x) for x in fields[3].split(",") if x]
            residuals = [float(x) for x in fields[4].split(",") if x]
            return ProbeResult(float(fields[1]), float(fields[2]), per_round, residuals)
    raise HarnessUnavailable("probe produced no PROBE line")


def evaluate_solver_source(source: str, task_spec, options: HarnessOptions | None = None):
    """Compile and score ``source`` against ``task_spec``; return a reward breakdown.

    ``task_spec`` is a :class:`~src.decoder.rl.curriculum.TaskSpec`. This is the
    single entry point the RL environment (and any external trainer) calls.
    """

    from .curriculum import TaskSpec  # local import to avoid a cycle
    from .reward import Rubric

    if not isinstance(task_spec, TaskSpec):
        raise TypeError("task_spec must be a TaskSpec")

    tasks = task_spec.build_flat_tasks()
    run = run_solver_on_tasks(source, tasks, options)
    return Rubric().score(run, task_spec)
