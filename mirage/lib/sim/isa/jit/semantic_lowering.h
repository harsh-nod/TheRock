#ifndef MIRAGE_SIM_ISA_JIT_SEMANTIC_LOWERING_H_
#define MIRAGE_SIM_ISA_JIT_SEMANTIC_LOWERING_H_

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "lib/sim/isa/common/decoded_instruction.h"
#include "lib/sim/isa/jit/semantic_instruction.h"
#include "lib/sim/isa/jit/translation_diagnostics.h"

namespace mirage::sim::isa::jit {

// Classifies a raw opcode string into a SemanticFamily based on its
// mnemonic prefix and known opcode patterns.  This is the first stage
// of the semantic IR pipeline and does not require a rule table lookup.
SemanticFamily ClassifyOpcodeFamily(std::string_view opcode);

class SemanticLowering {
 public:
  SemanticLowering() = default;

  // Determines the translation status for a semantic instruction based
  // on its family, architecture pair, and lowering prerequisites.
  TranslationStatus ClassifyForLowering(
      const SemanticInstruction& instruction) const;

  // Attempts to produce target-architecture DecodedInstructions from a
  // semantic instruction.  Returns false if lowering is not yet
  // implemented for the instruction's family or the prerequisites are
  // not met.
  bool LowerToTarget(
      const SemanticInstruction& instruction,
      std::uint8_t target_arch,
      std::vector<DecodedInstruction>* output,
      std::string* error_message = nullptr) const;

  // Lifts a DecodedInstruction into a SemanticInstruction by classifying
  // its opcode into a family and populating architecture-specific
  // semantic metadata (memory space, barrier kind, matrix layout, etc.).
  bool LiftFromDecoded(
      const DecodedInstruction& instruction,
      std::uint8_t source_arch,
      SemanticInstruction* output,
      std::string* error_message = nullptr) const;

  // Lifts an entire decoded program into a semantic program.
  bool LiftProgram(
      std::span<const DecodedInstruction> program,
      std::uint8_t source_arch,
      SemanticProgram* output,
      std::string* error_message = nullptr) const;
};

}  // namespace mirage::sim::isa::jit

#endif  // MIRAGE_SIM_ISA_JIT_SEMANTIC_LOWERING_H_
