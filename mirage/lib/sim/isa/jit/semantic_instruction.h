#ifndef MIRAGE_SIM_ISA_JIT_SEMANTIC_INSTRUCTION_H_
#define MIRAGE_SIM_ISA_JIT_SEMANTIC_INSTRUCTION_H_

#include <cstdint>
#include <string_view>
#include <vector>

namespace mirage::sim::isa::jit {

enum class SemanticFamily : std::uint8_t {
  kScalar,
  kVector,
  kMemory,
  kBranch,
  kBarrier,
  kTensorMemory,
  kTranspose,
  kWmma,
  kSwmmac,
  kMfma,
  kFp8Bf8,
  kScalePaired,
  kOther,
};

enum class BarrierKind : std::uint8_t {
  kNone,
  kMonolithic,
  kSplitSignal,
  kSplitWait,
};

enum class MemorySpace : std::uint8_t {
  kNone,
  kGlobal,
  kLds,
  kScratch,
  kFlat,
  kTensor,
};

struct SemanticInstruction {
  std::string_view opcode;
  SemanticFamily family = SemanticFamily::kOther;

  std::uint8_t source_arch = 0;
  std::uint8_t target_arch = 0;

  bool wave_size_sensitive = false;

  MemorySpace memory_space = MemorySpace::kNone;

  BarrierKind barrier_kind = BarrierKind::kNone;
  std::uint8_t barrier_id = 0;

  bool is_approximate = false;

  std::string_view normalized_operation;
};

using SemanticProgram = std::vector<SemanticInstruction>;

}  // namespace mirage::sim::isa::jit

#endif  // MIRAGE_SIM_ISA_JIT_SEMANTIC_INSTRUCTION_H_
