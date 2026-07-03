// Compile-and-run harness for authored solver plugins.
//
// The harness is the executable verifier of the RL environment: it compiles a
// model-authored solver source string into a shared object, then runs it under
// a wall-clock budget in a forked child so a hang or crash becomes a bounded,
// shaped failure instead of taking down the trainer. Reward *shaping* and the
// round curriculum live in Python (src/decoder/rl); this layer only compiles
// and runs, returning the raw Lagrangian / per-round measurements.

#pragma once

#include "solver_sdk.hpp"

#include <string>
#include <vector>

namespace aes_xts_decoder {
namespace harness {

// Resource limits applied (via setrlimit) inside the forked child before it
// hands control to attacker-controlled code (the compiler, or the solver .so).
// A value of 0 means "leave this limit untouched". These are defence-in-depth
// on top of the outer bubblewrap sandbox the Python layer wraps us in.
struct ResourceLimits {
  uint64_t address_space_bytes = 0;  // RLIMIT_AS
  uint64_t cpu_seconds = 0;          // RLIMIT_CPU (hard SIGKILL, soft SIGXCPU)
  uint64_t open_files = 0;           // RLIMIT_NOFILE
  uint64_t processes = 0;            // RLIMIT_NPROC (0 keeps OpenMP threading)
  uint64_t file_size_bytes = 0;      // RLIMIT_FSIZE
};

struct CompileOptions {
  std::string compiler = "c++";
  std::string include_dir;   // dir containing solver_sdk.hpp
  std::string sdk_source;    // path to solver_sdk.cpp
  std::string extra_flags;   // additional compiler flags
  std::string workdir_base;  // where temp build dirs are created ("" -> /tmp)
  double timeout_seconds = 60.0;
  ResourceLimits limits;     // rlimits for the compiler child
};

struct CompileResult {
  bool ok = false;
  std::string so_path;
  std::string workdir;
  std::string stderr_text;
  double seconds = 0.0;
};

struct RunOptions {
  double timeout_seconds = 10.0;
  std::size_t epochs = 1;
  std::size_t sweeps_per_epoch = 1000;
  ResourceLimits limits;         // rlimits for the solver child
  bool seccomp = true;           // install a syscall whitelist before dlopen
};

struct RunResult {
  bool ran = false;
  bool timed_out = false;
  bool crashed = false;
  bool feasible = false;
  double energy = 0.0;
  double best_hamming = 0.0;
  double round_staircase = 0.0;
  std::vector<double> per_round;
  std::vector<double> per_class;  // [ascii, consistency, goal] residual mass
  double seconds = 0.0;
  std::string error;
};

// Build the deterministic initial assignment (random ASCII plaintext + random
// keys, wires derived) a solver starts from, given the config seed. Exposed so
// the probe path can score the exact same assignment the runner would.
sdk::AssignmentState MakeInitial(const sdk::SdkCircuit& circuit, const sdk::SolverConfig& config);

// Compile a solver source string into a shared object. Never throws; on
// failure returns ok=false with the compiler's stderr in stderr_text.
CompileResult CompileSolver(const std::string& source, const CompileOptions& options);

// Run a compiled solver on one flattened circuit/task under a budget. Forks a
// child so timeouts and crashes are bounded. Never throws.
RunResult RunSolver(const std::string& so_path,
                    const sdk::CircuitBuffers& buffers,
                    const sdk::SolverConfig& config,
                    const RunOptions& options);

// Remove a compile workdir (best effort).
void CleanupWorkdir(const std::string& workdir);

}  // namespace harness
}  // namespace aes_xts_decoder
