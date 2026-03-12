#ifndef MIRAGE_SIM_ISA_JIT_HAZARD_PASS_H_
#define MIRAGE_SIM_ISA_JIT_HAZARD_PASS_H_

#include <cstdint>
#include <vector>

#include "lib/sim/isa/common/decoded_instruction.h"
#include "lib/sim/isa/jit/hazard_model.h"
#include "lib/sim/isa/jit/translation_diagnostics.h"

namespace mirage::sim::isa::jit {

struct HazardPassStats {
  std::uint32_t nops_stripped = 0;
  std::uint32_t waitcnts_stripped = 0;
  std::uint32_t nops_inserted = 0;
  std::uint32_t instructions_before = 0;
  std::uint32_t instructions_after = 0;
};

class HazardPass {
 public:
  static bool Apply(
      HazardPolicy policy,
      const HazardModel* source_model,
      const HazardModel* target_model,
      std::vector<DecodedInstruction>* program,
      std::vector<InstructionDiagnostic>* diagnostics,
      HazardPassStats* stats = nullptr);

  static bool StripSourceHazards(
      const HazardModel* source_model,
      std::vector<DecodedInstruction>* program,
      std::vector<InstructionDiagnostic>* diagnostics,
      HazardPassStats* stats);

  static bool InsertTargetHazards(
      const HazardModel* target_model,
      std::vector<DecodedInstruction>* program,
      std::vector<InstructionDiagnostic>* diagnostics,
      HazardPassStats* stats);

  static void RebuildDiagnosticIndices(
      std::vector<InstructionDiagnostic>* diagnostics);
};

}  // namespace mirage::sim::isa::jit

#endif  // MIRAGE_SIM_ISA_JIT_HAZARD_PASS_H_
