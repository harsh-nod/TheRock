#ifndef MIRAGE_SIM_ISA_JIT_PEEPHOLE_PASS_H_
#define MIRAGE_SIM_ISA_JIT_PEEPHOLE_PASS_H_

#include <cstdint>
#include <vector>

#include "lib/sim/isa/common/decoded_instruction.h"
#include "lib/sim/isa/jit/translation_diagnostics.h"

namespace mirage::sim::isa::jit {

struct PeepholeStats {
  std::uint32_t redundant_nops_removed = 0;
  std::uint32_t redundant_accvgpr_transfers_removed = 0;
  std::uint32_t total_instructions_removed = 0;
};

class PeepholePass {
 public:
  static bool Apply(
      std::vector<DecodedInstruction>* program,
      std::vector<InstructionDiagnostic>* diagnostics,
      PeepholeStats* stats = nullptr);

 private:
  static std::uint32_t CompactProgram(
      std::vector<DecodedInstruction>* program,
      std::vector<InstructionDiagnostic>* diagnostics,
      const std::vector<bool>& remove_mask);

  static void RebuildDiagnosticIndices(
      std::vector<InstructionDiagnostic>* diagnostics);
};

}  // namespace mirage::sim::isa::jit

#endif  // MIRAGE_SIM_ISA_JIT_PEEPHOLE_PASS_H_
