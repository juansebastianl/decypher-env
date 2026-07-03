#include "harness.hpp"

#include <dlfcn.h>
#include <fcntl.h>
#include <ftw.h>
#include <linux/audit.h>
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/select.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <sstream>
#include <vector>

namespace aes_xts_decoder {
namespace harness {

namespace {

double NowSeconds() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}

std::string ReadFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return "";
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

// Split a command string on whitespace into argv tokens. The solver compiler is
// a trusted, environment-controlled value (e.g. "c++" or "ccache g++"), never
// model input, so a simple whitespace split is sufficient and avoids a shell.
std::vector<std::string> SplitWhitespace(const std::string& text) {
  std::vector<std::string> out;
  std::istringstream ss(text);
  std::string token;
  while (ss >> token) out.push_back(token);
  return out;
}

// Apply setrlimit for each nonzero limit. Called in the forked child, before it
// runs attacker-controlled code, as defence-in-depth atop the outer bubblewrap
// sandbox. Returns false if any requested limit could not be set.
bool ApplyRlimits(const ResourceLimits& limits) {
  auto set_one = [](int resource, uint64_t value) -> bool {
    if (value == 0) return true;  // 0 => leave untouched
    struct rlimit rl;
    rl.rlim_cur = static_cast<rlim_t>(value);
    rl.rlim_max = static_cast<rlim_t>(value);
    return setrlimit(resource, &rl) == 0;
  };
  bool ok = true;
  ok &= set_one(RLIMIT_AS, limits.address_space_bytes);
  ok &= set_one(RLIMIT_CPU, limits.cpu_seconds);
  ok &= set_one(RLIMIT_NOFILE, limits.open_files);
  ok &= set_one(RLIMIT_NPROC, limits.processes);
  ok &= set_one(RLIMIT_FSIZE, limits.file_size_bytes);
  return ok;
}

// Install a seccomp-bpf syscall filter that whitelists the syscalls a
// compute-only solver plugin needs (arithmetic, memory management, OpenMP
// threading, loading the already-built .so, and writing its result to the
// pipe), kills the process on unambiguously hostile syscalls (execve, network,
// ptrace, fork), and denies everything else with EPERM so benign probes degrade
// gracefully instead of crashing. Must be installed *before* dlopen so hostile
// static initializers in the model-authored .so are already confined.
bool InstallSeccomp() {
  // Syscalls a legitimate compute-only solver needs. dlopen/dlsym of the
  // already-present .so needs openat/read/mmap/mprotect/close/newfstatat;
  // libgomp threading needs clone/futex/mmap/rseq/sched_*; the result is
  // written to an already-open pipe fd with write.
  const int allow[] = {
      SYS_read, SYS_write, SYS_close, SYS_fstat, SYS_newfstatat, SYS_lseek,
      SYS_mmap, SYS_mprotect, SYS_munmap, SYS_brk, SYS_madvise, SYS_mremap,
      SYS_rt_sigaction, SYS_rt_sigprocmask, SYS_rt_sigreturn, SYS_sigaltstack,
      SYS_futex, SYS_set_robust_list, SYS_get_robust_list, SYS_set_tid_address,
      SYS_clone, SYS_clone3, SYS_gettid, SYS_getpid, SYS_getppid,
      SYS_getuid, SYS_geteuid, SYS_getgid, SYS_getegid,
      SYS_sched_getaffinity, SYS_sched_setaffinity, SYS_sched_yield,
      SYS_sched_getparam, SYS_sched_get_priority_max, SYS_sched_get_priority_min,
      SYS_getcpu, SYS_nanosleep, SYS_clock_nanosleep, SYS_clock_gettime,
      SYS_gettimeofday, SYS_time, SYS_exit, SYS_exit_group, SYS_arch_prctl,
      SYS_prctl, SYS_openat, SYS_open, SYS_readlink, SYS_readlinkat,
      SYS_access, SYS_faccessat, SYS_faccessat2, SYS_getdents64, SYS_getrandom,
      SYS_getrlimit, SYS_prlimit64, SYS_rseq, SYS_membarrier, SYS_tgkill,
      SYS_restart_syscall, SYS_pread64, SYS_pwrite64, SYS_dup, SYS_dup2,
      SYS_epoll_create1, SYS_pipe2, SYS_ppoll, SYS_poll, SYS_futex_waitv,
  };
  // Unambiguously hostile syscalls: kill the whole process on any attempt.
  const int kill_list[] = {
      SYS_execve, SYS_execveat, SYS_socket, SYS_socketpair, SYS_connect,
      SYS_bind, SYS_listen, SYS_accept, SYS_accept4, SYS_sendto, SYS_recvfrom,
      SYS_sendmsg, SYS_recvmsg, SYS_ptrace, SYS_process_vm_readv,
      SYS_process_vm_writev, SYS_fork, SYS_vfork, SYS_kill, SYS_mount,
      SYS_umount2, SYS_reboot, SYS_ptrace,
  };

  std::vector<struct sock_filter> prog;
  // Reject any non-x86_64 arch (blocks i386/x32 syscall-number confusion).
  prog.push_back(BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
                          offsetof(struct seccomp_data, arch)));
  prog.push_back(BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, AUDIT_ARCH_X86_64, 1, 0));
  prog.push_back(BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS));
  // Load the syscall number.
  prog.push_back(BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
                          offsetof(struct seccomp_data, nr)));
  // Kill x32 ABI calls (nr has the 0x40000000 bit set).
  prog.push_back(BPF_JUMP(BPF_JMP | BPF_JGE | BPF_K, 0x40000000u, 0, 1));
  prog.push_back(BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS));
  // Allowed syscalls: return ALLOW inline.
  for (int nr : allow) {
    prog.push_back(BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K,
                            static_cast<__u32>(nr), 0, 1));
    prog.push_back(BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW));
  }
  // Hostile syscalls: kill the process.
  for (int nr : kill_list) {
    prog.push_back(BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K,
                            static_cast<__u32>(nr), 0, 1));
    prog.push_back(BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS));
  }
  // Default: deny with EPERM (benign probes see a clean failure, not a crash).
  prog.push_back(BPF_STMT(BPF_RET | BPF_K,
                          SECCOMP_RET_ERRNO | (EPERM & SECCOMP_RET_DATA)));

  struct sock_fprog fprog;
  fprog.len = static_cast<unsigned short>(prog.size());
  fprog.filter = prog.data();

  if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0) return false;
  if (prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &fprog, 0, 0) != 0) return false;
  return true;
}

enum class WaitOutcome { kExited, kSignaled, kTimedOut };

struct WaitStatus {
  WaitOutcome outcome = WaitOutcome::kExited;
  int code = 0;  // exit code (kExited) or signal number (kSignaled)
};

// Wait for a child until a wall-clock deadline, SIGKILLing it if it overruns.
// Used for the compile step (the run step waits on its result pipe instead).
WaitStatus WaitWithTimeout(pid_t child, double deadline) {
  for (;;) {
    int status = 0;
    pid_t r = waitpid(child, &status, WNOHANG);
    if (r == child) {
      WaitStatus ws;
      if (WIFEXITED(status)) {
        ws.outcome = WaitOutcome::kExited;
        ws.code = WEXITSTATUS(status);
      } else {
        ws.outcome = WaitOutcome::kSignaled;
        ws.code = WIFSIGNALED(status) ? WTERMSIG(status) : 0;
      }
      return ws;
    }
    if (r < 0 && errno != EINTR) {
      WaitStatus ws;
      ws.outcome = WaitOutcome::kSignaled;
      ws.code = 0;
      return ws;
    }
    if (NowSeconds() >= deadline) {
      // Kill the whole process group so any children the child forked (e.g. the
      // compiler's cc1plus/as/ld) die too, not just the direct child.
      kill(-child, SIGKILL);
      kill(child, SIGKILL);
      int discard = 0;
      waitpid(child, &discard, 0);
      WaitStatus ws;
      ws.outcome = WaitOutcome::kTimedOut;
      return ws;
    }
    struct timespec nap;
    nap.tv_sec = 0;
    nap.tv_nsec = 5 * 1000 * 1000;  // 5ms
    nanosleep(&nap, nullptr);
  }
}

}  // namespace

// Build a deterministic initial assignment (random ASCII plaintext + random
// keys) from the config seed, so a rollout is reproducible.
sdk::AssignmentState MakeInitial(const sdk::SdkCircuit& circuit,
                                 const sdk::SolverConfig& config) {
  const sdk::CircuitBuffers& b = circuit.buffers();
  sdk::AssignmentState a;
  a.plaintext.assign(b.plaintext_count, 0);
  a.key1.assign(b.key1_count, 0);
  a.key2.assign(b.key2_count, 0);
  uint64_t rng = config.seed ? config.seed : 0x1234567890ABCDEFULL;
  for (auto& c : a.plaintext) c = sdk::RandomTextAscii(&rng);
  for (auto& c : a.key1) c = static_cast<uint8_t>(sdk::RngRange(&rng, 256));
  for (auto& c : a.key2) c = static_cast<uint8_t>(sdk::RngRange(&rng, 256));
  if (circuit.wire_count() > 0) circuit.DeriveWires(&a);
  return a;
}

namespace {

// Binary payload written by the child over the result pipe.
struct WirePayload {
  double energy;
  double best_hamming;
  double round_staircase;
  double per_class[3];  // [ascii, consistency, goal] residual mass
  int32_t feasible;
  int32_t per_round_count;
};

}  // namespace

CompileResult CompileSolver(const std::string& source, const CompileOptions& options) {
  CompileResult result;
  double started = NowSeconds();

  std::string base = options.workdir_base.empty() ? "/tmp" : options.workdir_base;
  std::string tmpl = base + "/aes_solver_XXXXXX";
  std::vector<char> buf(tmpl.begin(), tmpl.end());
  buf.push_back('\0');
  char* dir = mkdtemp(buf.data());
  if (dir == nullptr) {
    result.stderr_text = "failed to create temp workdir";
    return result;
  }
  result.workdir = dir;
  std::string source_path = result.workdir + "/solver.cpp";
  result.so_path = result.workdir + "/solver.so";
  std::string stderr_path = result.workdir + "/compile.stderr";

  {
    std::ofstream out(source_path, std::ios::binary);
    out << source;
  }

  // Build the compiler argv (no shell: avoids path-injection and shell parsing
  // of any attacker-influenced string).
  std::vector<std::string> argv_str = SplitWhitespace(options.compiler);
  if (argv_str.empty()) argv_str.push_back("c++");
  for (const char* flag : {"-std=c++17", "-O2", "-fPIC", "-shared", "-fopenmp"})
    argv_str.push_back(flag);
  for (const std::string& extra : SplitWhitespace(options.extra_flags))
    argv_str.push_back(extra);
  if (!options.include_dir.empty()) argv_str.push_back("-I" + options.include_dir);
  argv_str.push_back(source_path);
  argv_str.push_back(options.sdk_source);
  argv_str.push_back("-o");
  argv_str.push_back(result.so_path);

  std::vector<char*> argv;
  argv.reserve(argv_str.size() + 1);
  for (std::string& s : argv_str) argv.push_back(s.data());
  argv.push_back(nullptr);

  pid_t child = fork();
  if (child < 0) {
    result.stderr_text = "fork() failed for compiler";
    return result;
  }
  if (child == 0) {
    // ---- Child: redirect stderr to file, apply rlimits, exec the compiler. ----
    // Own process group so a timeout can SIGKILL the whole compiler subtree
    // (cc1plus/as/ld), not just this driver process.
    setpgid(0, 0);
    int efd = open(stderr_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (efd >= 0) {
      dup2(efd, STDERR_FILENO);
      if (efd != STDERR_FILENO) close(efd);
    }
    int nfd = open("/dev/null", O_WRONLY);
    if (nfd >= 0) {
      dup2(nfd, STDOUT_FILENO);
      if (nfd != STDOUT_FILENO) close(nfd);
    }
    // The compiler driver forks cc1plus/as/ld, so leave RLIMIT_NPROC untouched
    // for the compile step; still cap memory, CPU, file size and fds.
    ResourceLimits climits = options.limits;
    climits.processes = 0;
    ApplyRlimits(climits);
    execvp(argv[0], argv.data());
    _exit(127);  // exec failed
  }

  // ---- Parent: enforce the compile timeout with a wall-clock deadline. ----
  double timeout = options.timeout_seconds > 0.0 ? options.timeout_seconds : 1e9;
  WaitStatus ws = WaitWithTimeout(child, started + timeout);
  result.seconds = NowSeconds() - started;
  result.stderr_text = ReadFile(stderr_path);

  if (ws.outcome == WaitOutcome::kTimedOut) {
    result.ok = false;
    if (result.stderr_text.empty()) result.stderr_text = "compile exceeded time budget";
    else result.stderr_text += "\ncompile exceeded time budget";
    return result;
  }
  if (ws.outcome == WaitOutcome::kSignaled) {
    result.ok = false;
    if (result.stderr_text.empty())
      result.stderr_text = "compiler killed by signal " + std::to_string(ws.code);
    return result;
  }
  result.ok = (ws.code == 0);
  if (!result.ok && result.stderr_text.empty()) {
    result.stderr_text = "compiler exited with code " + std::to_string(ws.code);
  }
  return result;
}

RunResult RunSolver(const std::string& so_path,
                    const sdk::CircuitBuffers& buffers,
                    const sdk::SolverConfig& config,
                    const RunOptions& options) {
  RunResult result;
  double started = NowSeconds();

  int pipefd[2];
  if (pipe(pipefd) != 0) {
    result.error = "pipe() failed";
    return result;
  }

  pid_t child = fork();
  if (child < 0) {
    close(pipefd[0]);
    close(pipefd[1]);
    result.error = "fork() failed";
    return result;
  }

  if (child == 0) {
    // ---- Child: confine, load and run the plugin, write results, _exit. ----
    close(pipefd[0]);
    int wfd = pipefd[1];

    // Own process group so a timeout can SIGKILL the whole subtree, including
    // anything the solver forked (a compute-only solver should fork nothing).
    setpgid(0, 0);
    // Resource caps first, then the syscall filter, *before* dlopen so hostile
    // static initializers in the model-authored .so are already sandboxed.
    ApplyRlimits(options.limits);
    if (options.seccomp && !InstallSeccomp()) _exit(45);

    void* handle = dlopen(so_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) _exit(41);
    using CreateFn = sdk::ISolver* (*)(const sdk::SolverContext&);
    CreateFn create = reinterpret_cast<CreateFn>(dlsym(handle, "create_solver"));
    if (create == nullptr) _exit(42);

    sdk::SdkCircuit circuit(buffers);
    sdk::SolverConfig cfg = config;
    sdk::SolverContext context{circuit, cfg, MakeInitial(circuit, cfg)};

    sdk::ISolver* solver = create(context);
    if (solver == nullptr) _exit(43);

    bool feasible = false;
    for (std::size_t e = 0; e < options.epochs; ++e) {
      solver->RunEpoch(options.sweeps_per_epoch);
      std::vector<sdk::AssignmentState> drained = solver->DrainFeasible(0);
      if (!drained.empty()) feasible = true;
    }
    sdk::AssignmentState best = solver->CurrentAssignment();
    sdk::ScoreData score = circuit.Score(best);
    if (score.feasible()) feasible = true;

    WirePayload payload;
    payload.energy = circuit.Energy(score, cfg);
    payload.best_hamming = score.hamming_score;
    payload.round_staircase = circuit.RoundStaircase(score);
    std::vector<double> per_class = circuit.PerClassResiduals(score);
    for (int i = 0; i < 3; ++i)
      payload.per_class[i] = (i < static_cast<int>(per_class.size())) ? per_class[i] : 0.0;
    payload.feasible = feasible ? 1 : 0;
    std::vector<double> per_round = circuit.PerRoundResiduals(score);
    payload.per_round_count = static_cast<int32_t>(per_round.size());

    ssize_t ignore = 0;
    ignore += write(wfd, &payload, sizeof(payload));
    if (!per_round.empty())
      ignore += write(wfd, per_round.data(), per_round.size() * sizeof(double));
    (void)ignore;
    close(wfd);
    delete solver;
    _exit(0);
  }

  // ---- Parent: read child output under a wall-clock deadline. ----
  close(pipefd[1]);
  int rfd = pipefd[0];
  std::string data;
  double deadline = started + options.timeout_seconds;
  bool timed_out = false;
  for (;;) {
    double remaining = deadline - NowSeconds();
    if (remaining <= 0.0) {
      timed_out = true;
      break;
    }
    fd_set set;
    FD_ZERO(&set);
    FD_SET(rfd, &set);
    struct timeval tv;
    tv.tv_sec = static_cast<long>(remaining);
    tv.tv_usec = static_cast<long>((remaining - tv.tv_sec) * 1e6);
    int ready = select(rfd + 1, &set, nullptr, nullptr, &tv);
    if (ready < 0) {
      if (errno == EINTR) continue;
      break;
    }
    if (ready == 0) {
      timed_out = true;
      break;
    }
    char chunk[4096];
    ssize_t n = read(rfd, chunk, sizeof(chunk));
    if (n <= 0) break;  // EOF: child finished (or closed pipe)
    data.append(chunk, static_cast<std::size_t>(n));
  }
  close(rfd);

  if (timed_out) {
    kill(-child, SIGKILL);  // whole process group (solver + any forks)
    kill(child, SIGKILL);
  }
  int status = 0;
  waitpid(child, &status, 0);
  result.seconds = NowSeconds() - started;

  if (timed_out) {
    result.timed_out = true;
    result.error = "solver exceeded wall-clock budget";
    return result;
  }
  if (!(WIFEXITED(status) && WEXITSTATUS(status) == 0)) {
    result.crashed = true;
    if (WIFEXITED(status)) {
      int code = WEXITSTATUS(status);
      if (code == 41) result.error = "dlopen failed";
      else if (code == 42) result.error = "create_solver symbol missing";
      else if (code == 43) result.error = "create_solver returned null";
      else if (code == 45) result.error = "seccomp sandbox setup failed";
      else result.error = "solver exited with code " + std::to_string(code);
    } else if (WIFSIGNALED(status)) {
      int sig = WTERMSIG(status);
      if (sig == SIGSYS)
        result.error = "solver killed by seccomp (disallowed syscall)";
      else
        result.error = "solver killed by signal " + std::to_string(sig);
    } else {
      result.error = "solver did not exit cleanly";
    }
    return result;
  }

  WirePayload payload;
  if (data.size() < sizeof(payload)) {
    result.crashed = true;
    result.error = "incomplete result payload";
    return result;
  }
  std::memcpy(&payload, data.data(), sizeof(payload));
  result.ran = true;
  result.energy = payload.energy;
  result.best_hamming = payload.best_hamming;
  result.round_staircase = payload.round_staircase;
  result.feasible = payload.feasible != 0;
  result.per_class.assign(payload.per_class, payload.per_class + 3);
  std::size_t offset = sizeof(payload);
  for (int32_t i = 0; i < payload.per_round_count; ++i) {
    double value = 0.0;
    if (offset + sizeof(double) <= data.size()) {
      std::memcpy(&value, data.data() + offset, sizeof(double));
      offset += sizeof(double);
    }
    result.per_round.push_back(value);
  }
  return result;
}

namespace {

int RemoveEntry(const char* path, const struct stat*, int, struct FTW*) {
  return remove(path);
}

}  // namespace

void CleanupWorkdir(const std::string& workdir) {
  if (workdir.empty()) return;
  // Depth-first recursive unlink without invoking a shell.
  nftw(workdir.c_str(), RemoveEntry, 16, FTW_DEPTH | FTW_PHYS | FTW_MOUNT);
}

}  // namespace harness
}  // namespace aes_xts_decoder
