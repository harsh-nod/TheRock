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

// Matrix fragment layout describes the register-level data layout for
// matrix multiply-accumulate instructions (WMMA, SWMMAC, MFMA).
enum class MatrixFragmentLayout : std::uint8_t {
  kNone,
  kRowMajor,
  kColMajor,
  kAccumulator,
};

// Element type for matrix and FP8/BF8/F4 family classification.
enum class ElementType : std::uint8_t {
  kNone,
  kF16,
  kBF16,
  kF32,
  kF64,
  kI8,
  kI32,
  kFP8_E4M3,
  kFP8_E5M2,
  kBF8,
  kF4,
};

// Tensor descriptor role for tensor memory instructions (gfx1250).
enum class TensorDescRole : std::uint8_t {
  kNone,
  kLoad,
  kStore,
  kAtomic,
};

// Lowering prerequisite flags.  Describes what runtime or backend
// capabilities are needed before semantic lowering can produce
// executable code for a given instruction.
struct LoweringPrerequisites {
  bool requires_accvgpr_support = false;
  bool requires_tensor_core = false;
  bool requires_split_barrier_runtime = false;
  bool requires_fp8_hardware = false;
  bool requires_scale_hardware = false;

  bool any() const {
    return requires_accvgpr_support || requires_tensor_core ||
           requires_split_barrier_runtime || requires_fp8_hardware ||
           requires_scale_hardware;
  }
};

// Implicit register effects that are not captured in the operand list
// of DecodedInstruction but are semantically important for correctness.
struct ImplicitRegisterEffects {
  bool reads_exec = false;
  bool writes_exec = false;
  bool reads_vcc = false;
  bool writes_vcc = false;
  bool reads_scc = false;
  bool writes_scc = false;
  bool reads_m0 = false;
  bool writes_m0 = false;
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

  // Matrix instruction metadata.
  MatrixFragmentLayout fragment_layout = MatrixFragmentLayout::kNone;
  ElementType input_element_type = ElementType::kNone;
  ElementType output_element_type = ElementType::kNone;
  std::uint8_t matrix_m = 0;
  std::uint8_t matrix_n = 0;
  std::uint8_t matrix_k = 0;

  // Tensor memory metadata (gfx1250).
  TensorDescRole tensor_role = TensorDescRole::kNone;

  // Implicit register effects not in operand list.
  ImplicitRegisterEffects implicit_effects;

  // What runtime capabilities are needed for lowering.
  LoweringPrerequisites prerequisites;
};

using SemanticProgram = std::vector<SemanticInstruction>;

// Returns the human-readable name for a SemanticFamily value.
std::string_view SemanticFamilyName(SemanticFamily family);

}  // namespace mirage::sim::isa::jit

#endif  // MIRAGE_SIM_ISA_JIT_SEMANTIC_INSTRUCTION_H_
