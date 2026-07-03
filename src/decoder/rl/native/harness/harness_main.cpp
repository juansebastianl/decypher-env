// Standalone harness CLI.
//
// This binary is compiled on demand by src/decoder/rl/harness.py. It reads a
// binary task file (one or more flattened circuits produced by Python), compiles
// the supplied solver source once, runs it on every task under a budget, and
// prints tab-separated results to stdout. Keeping this out of the Cython
// extension is deliberate: the environment compiles solvers (both examples and
// model output) on demand rather than baking them into the build.
//
// Wire format is documented in src/decoder/rl/harness.py (kept in lockstep).

#include "harness.hpp"
#include "solver_sdk.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace aes_xts_decoder;

namespace {

// Wire-format magic. Bumped whenever the per-task layout changes so a stale
// harness binary cannot silently misread a newer payload. Kept in lockstep with
// _MAGIC in src/decoder/rl/harness.py.
constexpr char kMagic[4] = {'A', 'X', 'H', '2'};

// Bounds-checked reader over an untrusted binary payload. Every read validates
// that enough bytes remain before touching memory, and every length prefix is
// validated to be non-negative and to fit in the remaining bytes, so a
// malformed or hostile file yields a clean parse error (ok=false) instead of an
// out-of-bounds read or an unbounded allocation. Once a read fails, ok stays
// false and subsequent reads are no-ops returning defaults.
struct Reader {
  const std::vector<uint8_t>& data;
  std::size_t pos = 0;
  bool ok = true;
  std::string error;

  explicit Reader(const std::vector<uint8_t>& d) : data(d) {}

  void Fail(const char* msg) {
    if (ok) {
      ok = false;
      error = msg;
    }
  }

  // True if `need` more bytes are available from the current position.
  bool Ensure(std::size_t need) {
    if (!ok) return false;
    if (need > data.size() || pos > data.size() - need) {
      Fail("unexpected end of task payload");
      return false;
    }
    return true;
  }

  template <typename T>
  T Scalar() {
    T value{};
    if (!Ensure(sizeof(T))) return value;
    std::memcpy(&value, data.data() + pos, sizeof(T));
    pos += sizeof(T);
    return value;
  }

  // Validate an int32 length prefix for an array of `elem_size`-byte elements.
  // Returns the count (>= 0) or 0 on failure (setting ok=false).
  std::size_t ArrayCount(std::size_t elem_size) {
    int32_t count = Scalar<int32_t>();
    if (!ok) return 0;
    if (count < 0) {
      Fail("negative array length in task payload");
      return 0;
    }
    std::size_t n = static_cast<std::size_t>(count);
    std::size_t remaining = data.size() - pos;
    if (elem_size != 0 && n > remaining / elem_size) {
      Fail("array length exceeds task payload size");
      return 0;
    }
    return n;
  }

  std::vector<int32_t> Int32Array() {
    std::size_t count = ArrayCount(sizeof(int32_t));
    std::vector<int32_t> out(count);
    if (ok && count > 0) {
      std::memcpy(out.data(), data.data() + pos, count * sizeof(int32_t));
      pos += count * sizeof(int32_t);
    }
    return out;
  }

  std::vector<uint8_t> ByteArray() {
    std::size_t count = ArrayCount(sizeof(uint8_t));
    std::vector<uint8_t> out(count);
    if (ok && count > 0) {
      std::memcpy(out.data(), data.data() + pos, count);
      pos += count;
    }
    return out;
  }

  std::vector<uint16_t> Uint16Array() {
    std::size_t count = ArrayCount(sizeof(uint16_t));
    std::vector<uint16_t> out(count);
    if (ok && count > 0) {
      std::memcpy(out.data(), data.data() + pos, count * sizeof(uint16_t));
      pos += count * sizeof(uint16_t);
    }
    return out;
  }

  // Validate a non-negative count field that will be used as an allocation size
  // or index bound (e.g. value_count). Bounds it by the remaining payload so a
  // hostile huge value can't drive a giant allocation downstream.
  std::size_t CountField(const char* what) {
    int64_t raw = Scalar<int64_t>();
    if (!ok) return 0;
    if (raw < 0) {
      Fail(what);
      return 0;
    }
    return static_cast<std::size_t>(raw);
  }
};

template <typename T>
std::string NumberCsv(const std::vector<T>& values) {
  std::string out;
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i) out += ",";
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.10g", static_cast<double>(values[i]));
    out += buf;
  }
  return out;
}

// Read one flattened circuit + its per-task solver config from the reader.
sdk::CircuitBuffers ReadCircuit(Reader& r, sdk::SolverConfig* cfg) {
  sdk::CircuitBuffers b;
  b.value_count = r.CountField("negative value_count in task payload");
  b.aes_rounds = r.Scalar<int32_t>();
  b.plaintext_start = r.CountField("negative plaintext_start in task payload");
  b.plaintext_count = r.CountField("negative plaintext_count in task payload");
  b.key1_start = r.CountField("negative key1_start in task payload");
  b.key1_count = r.CountField("negative key1_count in task payload");
  b.key2_start = r.CountField("negative key2_start in task payload");
  b.key2_count = r.CountField("negative key2_count in task payload");
  cfg->ascii_weight = r.Scalar<double>();
  cfg->consistency_weight = r.Scalar<double>();
  cfg->goal_weight = r.Scalar<double>();
  cfg->coupling_weight = r.Scalar<double>();
  cfg->ascii_rho = r.Scalar<double>();
  cfg->consistency_rho = r.Scalar<double>();
  cfg->goal_rho = r.Scalar<double>();
  cfg->seed = r.Scalar<uint64_t>();
  cfg->aes_rounds = b.aes_rounds;
  b.opcodes = r.Int32Array();
  b.outputs = r.Int32Array();
  b.input_offsets = r.Int32Array();
  b.input_counts = r.Int32Array();
  b.inputs = r.Int32Array();
  b.immediate_offsets = r.Int32Array();
  b.immediate_counts = r.Int32Array();
  b.immediates = r.Int32Array();
  b.const_offsets = r.Int32Array();
  b.const_counts = r.Int32Array();
  b.constraint_kinds = r.Int32Array();
  b.constraint_left = r.Int32Array();
  b.constraint_right = r.Int32Array();
  b.constraint_classes = r.Int32Array();
  b.constraint_rounds = r.Int32Array();
  b.wire_value_ids = r.Int32Array();
  b.wire_offsets = r.Int32Array();
  b.xts_block_sectors = r.Int32Array();
  b.xts_block_indices = r.Int32Array();
  b.xts_block_offsets = r.Int32Array();
  b.constants = r.ByteArray();
  b.xts_block_targets = r.ByteArray();
  b.value_widths = r.Uint16Array();
  return b;
}

std::vector<uint8_t> ReadWholeFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

// probe mode: score one fixed assignment through the SDK and print residuals so
// a test can cross-check against the Python circuit.evaluate reference.
int RunProbe(const std::string& task_path, const std::string& assignment_path) {
  std::vector<uint8_t> data = ReadWholeFile(task_path);
  if (data.size() < 8 || std::memcmp(data.data(), kMagic, 4) != 0) {
    std::fprintf(stderr, "bad task magic\n");
    return 3;
  }
  Reader r(data);
  r.pos = 4;  // skip magic
  int32_t num_tasks = r.Scalar<int32_t>();
  if (!r.ok || num_tasks < 1) return 3;
  sdk::SolverConfig cfg;
  sdk::CircuitBuffers b = ReadCircuit(r, &cfg);
  if (!r.ok) {
    std::fprintf(stderr, "%s\n", r.error.c_str());
    return 3;
  }
  sdk::SdkCircuit circuit(b);

  std::vector<uint8_t> adata = ReadWholeFile(assignment_path);
  Reader ar(adata);
  sdk::AssignmentState a;
  a.plaintext = ar.ByteArray();
  a.key1 = ar.ByteArray();
  a.key2 = ar.ByteArray();
  if (!ar.ok) {
    std::fprintf(stderr, "%s\n", ar.error.c_str());
    return 3;
  }
  if (circuit.wire_count() > 0) circuit.DeriveWires(&a);

  sdk::ScoreData score = circuit.Score(a);
  std::vector<double> per_round = circuit.PerRoundResiduals(score);
  std::printf("PROBE\t%.10g\t%.10g\t%s\t%s\n", score.hamming_score,
              circuit.Energy(score, cfg), NumberCsv(per_round).c_str(),
              NumberCsv(score.residuals).c_str());
  return 0;
}

}  // namespace

// Parse a uint64 CLI arg, tolerating empty/garbage as 0 ("unset").
uint64_t ParseU64(const char* s) {
  if (s == nullptr || *s == '\0') return 0;
  errno = 0;
  char* end = nullptr;
  unsigned long long v = std::strtoull(s, &end, 10);
  if (errno != 0 || end == s) return 0;
  return static_cast<uint64_t>(v);
}

int main(int argc, char** argv) {
  if (argc >= 4 && std::string(argv[1]) == "--probe") {
    return RunProbe(argv[2], argv[3]);
  }
  if (argc < 11) {
    std::fprintf(stderr,
                 "usage: harness <task.bin> <source> <include_dir> <sdk_source> "
                 "<compile_timeout> <run_timeout> <epochs> <sweeps> <workdir_base> "
                 "<stderr_out> [as_bytes] [cpu_seconds] [open_files] [processes] "
                 "[file_size_bytes] [seccomp]\n");
    return 2;
  }
  std::string task_path = argv[1];
  std::string source_path = argv[2];
  std::string include_dir = argv[3];
  std::string sdk_source = argv[4];
  double compile_timeout = std::stod(argv[5]);
  double run_timeout = std::stod(argv[6]);
  std::size_t epochs = static_cast<std::size_t>(std::stoul(argv[7]));
  std::size_t sweeps = static_cast<std::size_t>(std::stoul(argv[8]));
  std::string workdir_base = argv[9];
  std::string stderr_out = argv[10];

  // Optional resource-limit / seccomp args (0 or absent => leave untouched).
  harness::ResourceLimits limits;
  limits.address_space_bytes = argc > 11 ? ParseU64(argv[11]) : 0;
  limits.cpu_seconds = argc > 12 ? ParseU64(argv[12]) : 0;
  limits.open_files = argc > 13 ? ParseU64(argv[13]) : 0;
  limits.processes = argc > 14 ? ParseU64(argv[14]) : 0;
  limits.file_size_bytes = argc > 15 ? ParseU64(argv[15]) : 0;
  bool seccomp = argc > 16 ? (ParseU64(argv[16]) != 0) : true;

  std::ifstream source_in(source_path, std::ios::binary);
  std::string source((std::istreambuf_iterator<char>(source_in)), std::istreambuf_iterator<char>());

  harness::CompileOptions co;
  co.include_dir = include_dir;
  co.sdk_source = sdk_source;
  co.workdir_base = workdir_base;
  co.timeout_seconds = compile_timeout;
  co.limits = limits;
  harness::CompileResult cr = harness::CompileSolver(source, co);

  {
    std::ofstream err_out(stderr_out, std::ios::binary);
    err_out << cr.stderr_text;
  }
  std::printf("COMPILE\t%d\t%.4f\n", cr.ok ? 1 : 0, cr.seconds);
  if (!cr.ok) {
    return 0;
  }

  std::ifstream in(task_path, std::ios::binary);
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  if (data.size() < 8 || std::memcmp(data.data(), kMagic, 4) != 0) {
    std::fprintf(stderr, "bad task magic\n");
    return 3;
  }
  Reader r(data);
  r.pos = 4;
  int32_t num_tasks = r.Scalar<int32_t>();
  if (!r.ok || num_tasks < 0) {
    std::fprintf(stderr, "bad task count\n");
    return 3;
  }

  for (int32_t t = 0; t < num_tasks; ++t) {
    sdk::SolverConfig cfg;
    sdk::CircuitBuffers b = ReadCircuit(r, &cfg);
    if (!r.ok) {
      std::fprintf(stderr, "%s\n", r.error.c_str());
      return 3;
    }

    harness::RunOptions ro;
    ro.timeout_seconds = run_timeout;
    ro.epochs = epochs;
    ro.sweeps_per_epoch = sweeps;
    ro.limits = limits;
    ro.seccomp = seccomp;
    harness::RunResult rr = harness::RunSolver(cr.so_path, b, cfg, ro);

    std::printf("TASK\t%d\t%d\t%d\t%d\t%d\t%.10g\t%.10g\t%.10g\t%.4f\t%s\t%s\t%s\n",
                t, rr.ran ? 1 : 0, rr.feasible ? 1 : 0, rr.timed_out ? 1 : 0, rr.crashed ? 1 : 0,
                rr.energy, rr.best_hamming, rr.round_staircase, rr.seconds,
                NumberCsv(rr.per_round).c_str(), NumberCsv(rr.per_class).c_str(),
                rr.error.c_str());
  }

  harness::CleanupWorkdir(cr.workdir);
  return 0;
}
