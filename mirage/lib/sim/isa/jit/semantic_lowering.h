#ifndef MIRAGE_SIM_ISA_JIT_SEMANTIC_LOWERING_H_
#define MIRAGE_SIM_ISA_JIT_SEMANTIC_LOWERING_H_

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "lib/sim/isa/common/decoded_instruction.h"
#include "lib/sim/isa/jit/semantic_instruction.h"
#include "lib/sim/isa/jit/translation_diagnostics.h"

namespace mirage::sim::isa::jit {

class SemanticLowering {
 public:
  SemanticLowering() = default;

  TranslationStatus ClassifyForLowering(
      const SemanticInstruction& instruction) const;

  bool LowerToTarget(
      const SemanticInstruction& instruction,
      std::uint8_t target_arch,
      std::vector<DecodedInstruction>* output,
      std::string* error_message = nullptr) const;

  bool LiftFromDecoded(
      const DecodedInstruction& instruction,
      std::uint8_t source_arch,
      SemanticInstruction* output,
      std::string* error_message = nullptr) const;
};

}  // namespace mirage::sim::isa::jit

#endif  // MIRAGE_SIM_ISA_JIT_SEMANTIC_LOWERING_H_
