#ifndef MIRAGE_SIM_ISA_JIT_GFX1250_STUB_INGRESS_H_
#define MIRAGE_SIM_ISA_JIT_GFX1250_STUB_INGRESS_H_

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "lib/sim/isa/jit/semantic_instruction.h"
#include "lib/sim/isa/jit/translation_diagnostics.h"

namespace mirage::sim::isa::jit {

// Per-family coverage summary produced by the gfx1250 stub ingress.
struct Gfx1250FamilyCoverage {
  std::string_view family_name;
  SemanticFamily family = SemanticFamily::kOther;
  std::uint32_t total_count = 0;
  std::uint32_t coverage_only_count = 0;
  std::uint32_t blocked_on_runtime_count = 0;
  std::uint32_t unsupported_count = 0;
};

// Lifts the gfx1250 seed catalog into a vector of SemanticInstructions.
// Each seed instruction is classified by family and populated with
// architecture-specific metadata (prerequisites, memory space, etc.).
// All results are coverage-only; none are marked executable.
SemanticProgram LiftGfx1250SeedCatalog();

// Lifts the gfx1250 bringup focus instructions into semantic IR.
// This is a subset of the full seed catalog organized by focus area.
SemanticProgram LiftGfx1250FocusInstructions();

// Computes per-family coverage dashboards from a gfx1250 semantic
// program.  Groups instructions by SemanticFamily and reports the
// translation status distribution for each family.
std::vector<Gfx1250FamilyCoverage> ComputeGfx1250FamilyCoverage(
    const SemanticProgram& program);

// Computes coverage dashboards restricted to the 4 seeded families
// (vop3p, wmma, fp8_bf8, scale_paired).
std::vector<Gfx1250FamilyCoverage> ComputeGfx1250SeededFamilyCoverage();

}  // namespace mirage::sim::isa::jit

#endif  // MIRAGE_SIM_ISA_JIT_GFX1250_STUB_INGRESS_H_
