# Security & sandbox model

The environment compiles and runs **untrusted, model-authored C++**. A policy's
output is a translation unit that the harness compiles into a shared object and
`dlopen`s. That is arbitrary code execution by construction, so isolation is a
first-class concern, not an afterthought. This document describes the layered
sandbox, its guarantees, and its limits.

## Threat model

The adversary is the model completion: it can emit C++ that (a) does hostile
work at **compile time** (huge template expansion, `#include` tricks, `#pragma`
side effects), (b) runs hostile code in a **static initializer** that executes at
`dlopen` time before any of our code runs, or (c) runs hostile code during the
solver's `RunEpoch`. We defend compile and run separately because both are
attacker-controlled.

Trusted inputs (never sandboxed against): the harness sources themselves, the
SDK, the compiler binary path (an environment/config value like `c++`), and the
serialized task payload — though the payload is still **bounds-checked** as
defence against corruption (see below).

## The layers

### 1. Bubblewrap (`bwrap`) outer jail

The Python harness wraps each harness invocation in a rootless `bwrap` sandbox
(`src/decoder/rl/harness.py`, `Sandbox` + `_sandbox_prefix`):

- `--unshare-all` — new user/PID/IPC/UTS/cgroup **and network** namespaces
  (no network unless `Sandbox(no_network=False)`).
- `--ro-bind /usr` and the toolchain dirs — the compiler and shared libraries
  are read-only; the SDK include dir, sdk source and harness binary are bound
  read-only.
- `--bind <workdir>` — a single writable scratch dir (the per-rollout temp dir),
  with `TMPDIR` pointed at it so the compiler's intermediate files never touch a
  shared `/tmp`.
- `--die-with-parent`, `--new-session` — the jail dies with the trainer and
  cannot re-acquire a controlling terminal.

### 2. `setrlimit` resource caps (inner)

Inside the forked child, before it runs attacker code, the harness applies
`setrlimit` (`harness.cpp` `ApplyRlimits`): `RLIMIT_AS` (address space, default
2 GiB), `RLIMIT_NOFILE`, `RLIMIT_FSIZE`, and optionally `RLIMIT_CPU` /
`RLIMIT_NPROC`. `RLIMIT_NPROC` is left untouched by default because it counts
OpenMP threads too; the seccomp filter handles process creation instead.

### 3. seccomp-bpf syscall whitelist (inner, before `dlopen`)

The solver child installs a seccomp-bpf filter (`harness.cpp` `InstallSeccomp`)
**before** `dlopen`, so a hostile static initializer is already confined. The
filter:

- allows the syscalls a compute-only solver needs (memory management, OpenMP
  threading via `clone`/`futex`, reading the already-built `.so`, writing its
  result to the pipe, timers/RNG);
- **kills the process** (`SECCOMP_RET_KILL_PROCESS`) on unambiguously hostile
  syscalls: `execve`/`execveat`, all networking (`socket`/`connect`/…),
  `ptrace`, `process_vm_*`, `fork`/`vfork`, `mount`, `reboot`, and non-x86_64 /
  x32 ABI syscalls (to prevent syscall-number confusion);
- denies everything else with `EPERM` so benign probes degrade cleanly instead
  of crashing.

A solver killed by seccomp surfaces as `crashed=True` with
`"solver killed by seccomp (disallowed syscall)"`.

### 4. Wall-clock timeout + process-group kill

The parent enforces a wall-clock budget with `select()` on the result pipe;
on overrun it `SIGKILL`s the child's **process group**, so anything the solver
forked dies too. Compile timeouts use the same pattern with `waitpid`.

## Shell-free execution

There is no `std::system`/shell anywhere in the hot path: compilation is a
`fork`/`execvp` of the compiler with an argv vector (no string interpolation of
paths), and workdir cleanup is an `nftw` recursive unlink, not `rm -rf`.

## Input validation

The binary task reader (`harness_main.cpp` `Reader`) bounds-checks every read:
length prefixes must be non-negative and fit in the remaining bytes, the magic
(`AXH2`) is verified against the file size, and a stale/old-format payload is
rejected with a clean non-zero exit rather than an out-of-bounds read or a giant
allocation.

## Fail-loud policy

If `bwrap` is not installed, the environment **refuses to run** untrusted code
unless you explicitly opt out with `Sandbox(allow_unsandboxed=True)`. A CI box
without bubblewrap therefore errors loudly instead of silently executing
arbitrary code with ambient privileges.

```python
from src.decoder.rl.harness import HarnessOptions, Sandbox

# Default: isolate with bwrap; error if bwrap is missing.
opts = HarnessOptions(sandbox=Sandbox(enabled=True))

# Explicit, audited opt-out (e.g. a already-isolated CI container):
opts = HarnessOptions(sandbox=Sandbox(enabled=False, allow_unsandboxed=True))
```

## Guarantees and limits

**Guarantees.** No network egress; no writes outside the per-rollout workdir; no
`exec` of other programs; bounded memory, file size and wall-clock; hostile
static initializers are confined; malformed payloads cannot cause OOB reads or
unbounded allocations; a hang or crash is a shaped reward, not a trainer outage.

**Limits.** `bwrap` needs unprivileged user namespaces (some hosts disable
them; see troubleshooting). seccomp path-based filtering is not possible in BPF,
so "no writes outside the workdir" is enforced by the bwrap filesystem view, not
by seccomp. A fork bomb is bounded by the PID namespace + wall-clock + memory
cap rather than a hard `RLIMIT_NPROC` (kept off for OpenMP); the HTTP adapters,
which run as isolated services, are the recommended deployment shape for
hostile-scale workloads.
