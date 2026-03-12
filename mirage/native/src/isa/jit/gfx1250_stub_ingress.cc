#include "lib/sim/isa/jit/gfx1250_stub_ingress.h"

#include <algorithm>
#include <unordered_map>

#include "lib/sim/isa/common/decoded_instruction.h"
#include "lib/sim/isa/gfx1250/bringup_profile.h"
#include "lib/sim/isa/gfx1250/decoder_seed_catalog.h"
#include "lib/sim/isa/jit/semantic_lowering.h"

namespace mirage::sim::isa::jit {

namespace {

constexpr std::uint8_t kGfx1250SourceArch =
    static_cast<std::uint8_t>(SourceArchitecture::kGfx1250);

}  // namespace

SemanticProgram LiftGfx1250SeedCatalog() {
  SemanticLowering lowering;
  auto seeds = gfx1250::GetDecoderSeedInfos();

  SemanticProgram program;
  program.reserve(seeds.size());

  for (const auto& seed : seeds) {
    auto decoded =
        mirage::sim::isa::DecodedInstruction::Nullary(seed.instruction_name);
    SemanticInstruction semantic;
    lowering.LiftFromDecoded(decoded, kGfx1250SourceArch, &semantic);
    program.push_back(std::move(semantic));
  }

  return program;
}

SemanticProgram LiftGfx1250FocusInstructions() {
  SemanticLowering lowering;
  SemanticProgram program;

  constexpr gfx1250::BringupFocusArea areas[] = {
      gfx1250::BringupFocusArea::kVop3p,
      gfx1250::BringupFocusArea::kWmma,
      gfx1250::BringupFocusArea::kFp8Bf8,
      gfx1250::BringupFocusArea::kScalePaired,
  };

  for (auto area : areas) {
    auto names = gfx1250::GetFocusInstructions(area);
    for (auto name : names) {
      auto decoded = mirage::sim::isa::DecodedInstruction::Nullary(name);
      SemanticInstruction semantic;
      lowering.LiftFromDecoded(decoded, kGfx1250SourceArch, &semantic);
      program.push_back(std::move(semantic));
    }
  }

  return program;
}

std::vector<Gfx1250FamilyCoverage> ComputeGfx1250FamilyCoverage(
    const SemanticProgram& program) {
  SemanticLowering lowering;

  // Accumulate per-family counts.
  std::unordered_map<std::uint8_t, Gfx1250FamilyCoverage> family_map;

  for (const auto& instr : program) {
    auto key = static_cast<std::uint8_t>(instr.family);
    auto& entry = family_map[key];
    entry.family = instr.family;
    entry.family_name = SemanticFamilyName(instr.family);
    entry.total_count++;

    TranslationStatus status = lowering.ClassifyForLowering(instr);
    switch (status) {
      case TranslationStatus::kCoverageOnly:
      case TranslationStatus::kRequiresSemanticLowering:
        entry.coverage_only_count++;
        break;
      case TranslationStatus::kBlockedOnRuntime:
        entry.blocked_on_runtime_count++;
        break;
      case TranslationStatus::kUnsupported:
      case TranslationStatus::kIdentity:
      case TranslationStatus::kRenamed:
      case TranslationStatus::kRewrittenWithFixup:
        entry.unsupported_count++;
        break;
    }
  }

  // Convert to sorted vector.
  std::vector<Gfx1250FamilyCoverage> result;
  result.reserve(family_map.size());
  for (auto& [_, entry] : family_map) {
    result.push_back(std::move(entry));
  }
  std::sort(result.begin(), result.end(),
            [](const Gfx1250FamilyCoverage& a,
               const Gfx1250FamilyCoverage& b) {
              return a.family < b.family;
            });

  return result;
}

std::vector<Gfx1250FamilyCoverage> ComputeGfx1250SeededFamilyCoverage() {
  SemanticLowering lowering;
  std::vector<Gfx1250FamilyCoverage> result;

  struct FamilyDef {
    gfx1250::SeedFamily seed_family;
    SemanticFamily semantic_family;
    std::string_view name;
  };

  constexpr FamilyDef families[] = {
      {gfx1250::SeedFamily::kVop3p, SemanticFamily::kVector, "vop3p"},
      {gfx1250::SeedFamily::kWmma, SemanticFamily::kWmma, "wmma"},
      {gfx1250::SeedFamily::kFp8Bf8, SemanticFamily::kFp8Bf8, "fp8_bf8"},
      {gfx1250::SeedFamily::kScalePaired, SemanticFamily::kScalePaired,
       "scale_paired"},
  };

  for (const auto& fam : families) {
    auto names = gfx1250::GetSeededInstructionNames(fam.seed_family);

    Gfx1250FamilyCoverage coverage;
    coverage.family_name = fam.name;
    coverage.total_count = static_cast<std::uint32_t>(names.size());

    for (auto name : names) {
      auto decoded = mirage::sim::isa::DecodedInstruction::Nullary(name);
      SemanticInstruction semantic;
      lowering.LiftFromDecoded(decoded, kGfx1250SourceArch, &semantic);

      TranslationStatus status = lowering.ClassifyForLowering(semantic);
      switch (status) {
        case TranslationStatus::kCoverageOnly:
        case TranslationStatus::kRequiresSemanticLowering:
          coverage.coverage_only_count++;
          break;
        case TranslationStatus::kBlockedOnRuntime:
          coverage.blocked_on_runtime_count++;
          break;
        default:
          coverage.unsupported_count++;
          break;
      }
    }

    // Determine the dominant semantic family from actual classification.
    if (!names.empty()) {
      auto decoded = mirage::sim::isa::DecodedInstruction::Nullary(names[0]);
      SemanticInstruction semantic;
      lowering.LiftFromDecoded(decoded, kGfx1250SourceArch, &semantic);
      coverage.family = semantic.family;
    } else {
      coverage.family = fam.semantic_family;
    }

    result.push_back(std::move(coverage));
  }

  return result;
}

}  // namespace mirage::sim::isa::jit
