#include <iostream>
#include <string>
#include <vector>

#include "lib/sim/isa/common/decoded_instruction.h"
#include "lib/sim/isa/gfx1250/bringup_profile.h"
#include "lib/sim/isa/gfx1250/decoder_seed_catalog.h"
#include "lib/sim/isa/jit/cross_arch_translator.h"
#include "lib/sim/isa/jit/gfx1250_stub_ingress.h"
#include "lib/sim/isa/jit/semantic_lowering.h"

namespace {

bool Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    return false;
  }
  return true;
}

using namespace mirage::sim::isa;
using namespace mirage::sim::isa::jit;
using namespace mirage::sim::isa::gfx1250;

// --- Seed catalog ingress tests ---

bool TestLiftSeedCatalogProducesNonEmpty() {
  SemanticProgram program = LiftGfx1250SeedCatalog();
  auto seeds = GetDecoderSeedInfos();

  return Expect(!program.empty(), "seed catalog lift should be non-empty") &&
         Expect(program.size() == seeds.size(),
                "should have one semantic instruction per seed");
}

bool TestLiftSeedCatalogAllGfx1250Arch() {
  SemanticProgram program = LiftGfx1250SeedCatalog();

  bool ok = true;
  for (std::size_t i = 0; i < program.size(); ++i) {
    if (program[i].source_arch !=
        static_cast<std::uint8_t>(SourceArchitecture::kGfx1250)) {
      std::cerr << "FAIL: instruction " << i << " (" << program[i].opcode
                << ") has wrong source_arch\n";
      ok = false;
      break;
    }
  }
  return ok;
}

bool TestLiftSeedCatalogHasWmmaInstructions() {
  SemanticProgram program = LiftGfx1250SeedCatalog();

  std::uint32_t wmma_count = 0;
  for (const auto& instr : program) {
    if (instr.family == SemanticFamily::kWmma ||
        instr.family == SemanticFamily::kSwmmac) {
      wmma_count++;
    }
  }

  std::cerr << "  gfx1250 seed catalog: " << wmma_count
            << " WMMA/SWMMAC instructions\n";

  return Expect(wmma_count > 0,
                "seed catalog should contain WMMA/SWMMAC instructions");
}

bool TestLiftSeedCatalogHasFp8Instructions() {
  SemanticProgram program = LiftGfx1250SeedCatalog();

  std::uint32_t fp8_count = 0;
  for (const auto& instr : program) {
    if (instr.family == SemanticFamily::kFp8Bf8) {
      fp8_count++;
    }
  }

  std::cerr << "  gfx1250 seed catalog: " << fp8_count
            << " FP8/BF8 instructions\n";

  return Expect(fp8_count > 0,
                "seed catalog should contain FP8/BF8 instructions");
}

// --- Focus instructions tests ---

bool TestLiftFocusInstructionsNonEmpty() {
  SemanticProgram program = LiftGfx1250FocusInstructions();

  std::cerr << "  gfx1250 focus instructions: " << program.size()
            << " total\n";

  return Expect(!program.empty(),
                "focus instructions lift should be non-empty") &&
         Expect(program.size() >= 20,
                "should have at least 20 focus instructions");
}

// --- Per-family coverage dashboard tests ---

bool TestSeededFamilyCoverageVop3p() {
  auto coverage = ComputeGfx1250SeededFamilyCoverage();

  const Gfx1250FamilyCoverage* vop3p = nullptr;
  for (const auto& c : coverage) {
    if (c.family_name == "vop3p") {
      vop3p = &c;
      break;
    }
  }

  if (vop3p == nullptr) {
    std::cerr << "FAIL: no vop3p family in coverage\n";
    return false;
  }

  std::cerr << "  vop3p coverage: total=" << vop3p->total_count
            << " coverage_only=" << vop3p->coverage_only_count
            << " blocked=" << vop3p->blocked_on_runtime_count
            << " unsupported=" << vop3p->unsupported_count << "\n";

  return Expect(vop3p->total_count > 0,
                "vop3p should have instructions") &&
         Expect(vop3p->total_count ==
                    vop3p->coverage_only_count +
                        vop3p->blocked_on_runtime_count +
                        vop3p->unsupported_count,
                "vop3p counts should sum to total");
}

bool TestSeededFamilyCoverageWmma() {
  auto coverage = ComputeGfx1250SeededFamilyCoverage();

  const Gfx1250FamilyCoverage* wmma = nullptr;
  for (const auto& c : coverage) {
    if (c.family_name == "wmma") {
      wmma = &c;
      break;
    }
  }

  if (wmma == nullptr) {
    std::cerr << "FAIL: no wmma family in coverage\n";
    return false;
  }

  std::cerr << "  wmma coverage: total=" << wmma->total_count
            << " coverage_only=" << wmma->coverage_only_count
            << " blocked=" << wmma->blocked_on_runtime_count
            << " unsupported=" << wmma->unsupported_count << "\n";

  return Expect(wmma->total_count > 0,
                "wmma should have instructions") &&
         Expect(wmma->total_count ==
                    wmma->coverage_only_count +
                        wmma->blocked_on_runtime_count +
                        wmma->unsupported_count,
                "wmma counts should sum to total");
}

bool TestSeededFamilyCoverageFp8Bf8() {
  auto coverage = ComputeGfx1250SeededFamilyCoverage();

  const Gfx1250FamilyCoverage* fp8 = nullptr;
  for (const auto& c : coverage) {
    if (c.family_name == "fp8_bf8") {
      fp8 = &c;
      break;
    }
  }

  if (fp8 == nullptr) {
    std::cerr << "FAIL: no fp8_bf8 family in coverage\n";
    return false;
  }

  std::cerr << "  fp8_bf8 coverage: total=" << fp8->total_count
            << " coverage_only=" << fp8->coverage_only_count
            << " blocked=" << fp8->blocked_on_runtime_count
            << " unsupported=" << fp8->unsupported_count << "\n";

  return Expect(fp8->total_count > 0,
                "fp8_bf8 should have instructions") &&
         Expect(fp8->total_count ==
                    fp8->coverage_only_count +
                        fp8->blocked_on_runtime_count +
                        fp8->unsupported_count,
                "fp8_bf8 counts should sum to total");
}

bool TestSeededFamilyCoverageScalePaired() {
  auto coverage = ComputeGfx1250SeededFamilyCoverage();

  const Gfx1250FamilyCoverage* scale = nullptr;
  for (const auto& c : coverage) {
    if (c.family_name == "scale_paired") {
      scale = &c;
      break;
    }
  }

  if (scale == nullptr) {
    std::cerr << "FAIL: no scale_paired family in coverage\n";
    return false;
  }

  std::cerr << "  scale_paired coverage: total=" << scale->total_count
            << " coverage_only=" << scale->coverage_only_count
            << " blocked=" << scale->blocked_on_runtime_count
            << " unsupported=" << scale->unsupported_count << "\n";

  return Expect(scale->total_count > 0,
                "scale_paired should have instructions") &&
         Expect(scale->total_count ==
                    scale->coverage_only_count +
                        scale->blocked_on_runtime_count +
                        scale->unsupported_count,
                "scale_paired counts should sum to total");
}

bool TestAllFourSeededFamiliesPresent() {
  auto coverage = ComputeGfx1250SeededFamilyCoverage();

  return Expect(coverage.size() == 4,
                "should have exactly 4 seeded families");
}

// --- Full seed catalog coverage ---

bool TestFullCatalogCoverageAllBucketed() {
  SemanticProgram program = LiftGfx1250SeedCatalog();
  auto coverage = ComputeGfx1250FamilyCoverage(program);

  std::uint32_t total = 0;
  std::cerr << "  gfx1250 full coverage by family:\n";
  for (const auto& c : coverage) {
    std::cerr << "    " << c.family_name << ": total=" << c.total_count
              << " cov=" << c.coverage_only_count
              << " blocked=" << c.blocked_on_runtime_count
              << " unsup=" << c.unsupported_count << "\n";
    total += c.total_count;
  }

  return Expect(total == program.size(),
                "all instructions should be bucketed") &&
         Expect(coverage.size() > 0,
                "should have at least one family");
}

// --- Translator integration tests ---

bool TestGfx1250ToGfx950NeverExecutable() {
  // gfx1250 -> gfx950 should NEVER be marked executable.
  // The rule table has no gfx1250->gfx950 entries.
  std::vector<DecodedInstruction> program = {
      DecodedInstruction::Unary("S_MOV_B32",
                                InstructionOperand::Sgpr(0),
                                InstructionOperand::Imm32(42)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  TranslationConfig config;
  config.source_arch = SourceArchitecture::kGfx1250;
  config.target_arch = TargetArchitecture::kGfx950;

  CrossArchTranslator translator(config);
  TranslationResult result = translator.Translate(program);

  return Expect(!result.is_executable,
                "gfx1250->gfx950 should not be executable") &&
         Expect(result.capability_summary.unsupported_count == 2,
                "all 2 instructions should be unsupported (no rules)");
}

bool TestGfx1250ToGfx950SemanticCoverageRicher() {
  // ComputeCoverageWithSemantic should produce richer results than
  // ComputeCoverage for gfx1250 instructions.
  std::vector<DecodedInstruction> program = {
      DecodedInstruction::Nullary("V_WMMA_F32_16X16X16_F16"),
      DecodedInstruction::Nullary("V_CVT_F32_FP8"),
      DecodedInstruction::Nullary("V_TRANSPOSE_B32"),
      DecodedInstruction::Nullary("TENSOR_LOAD_DWORD"),
  };

  TranslationConfig config;
  config.source_arch = SourceArchitecture::kGfx1250;
  config.target_arch = TargetArchitecture::kGfx950;

  CrossArchTranslator translator(config);

  CapabilitySummary old_cov = translator.ComputeCoverage(program);
  CapabilitySummary new_cov = translator.ComputeCoverageWithSemantic(program);

  // Old coverage: all unsupported (no rule table entries).
  bool ok = Expect(old_cov.unsupported_count == 4,
                   "old: all 4 should be unsupported");

  // New coverage should classify into richer buckets.
  ok = Expect(new_cov.unsupported_count < 4,
              "new: not all should be unsupported") &&
       Expect(new_cov.total() == 4,
              "new: total should still be 4") && ok;

  std::cerr << "  gfx1250 semantic coverage: exec="
            << new_cov.executable_count
            << " cov=" << new_cov.coverage_only_count
            << " blocked=" << new_cov.blocked_on_runtime_count
            << " unsup=" << new_cov.unsupported_count << "\n";

  return ok;
}

bool TestGfx1250CoverageOnlyModeTranslation() {
  // In coverage-only mode, the translator should not mark anything
  // as executable even if it would otherwise be translatable.
  std::vector<DecodedInstruction> program = {
      DecodedInstruction::Nullary("V_WMMA_F32_16X16X16_F16"),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  TranslationConfig config;
  config.source_arch = SourceArchitecture::kGfx1250;
  config.target_arch = TargetArchitecture::kGfx950;
  config.translation_mode = TranslationMode::kCoverageOnly;

  CrossArchTranslator translator(config);
  TranslationResult result = translator.Translate(program);

  return Expect(!result.is_executable,
                "coverage-only mode should not be executable") &&
         Expect(result.top_level_status == TranslationStatus::kCoverageOnly,
                "top-level status should be coverage_only");
}

// --- Diagnostic corpus tests ---

bool TestGfx1250DiagnosticsForUnsupportedFamilies() {
  std::vector<DecodedInstruction> program = {
      DecodedInstruction::Nullary("V_WMMA_F32_16X16X16_F16"),
      DecodedInstruction::Nullary("V_CVT_F32_FP8"),
      DecodedInstruction::Nullary("TENSOR_LOAD_DWORD"),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  TranslationConfig config;
  config.source_arch = SourceArchitecture::kGfx1250;
  config.target_arch = TargetArchitecture::kGfx950;

  CrossArchTranslator translator(config);
  TranslationResult result = translator.Translate(program);

  bool ok = Expect(!result.is_executable,
                   "program should not be executable") &&
            Expect(result.diagnostics.size() == 4,
                   "should have 4 diagnostics");

  // All instructions should have rejection reasons since there are no
  // gfx1250->gfx950 rules.
  for (std::size_t i = 0; i < result.diagnostics.size(); ++i) {
    const auto& diag = result.diagnostics[i];
    if (diag.status != TranslationStatus::kUnsupported) {
      std::string msg = "instruction " + std::to_string(i) + " (" +
                        std::string(diag.source_opcode) +
                        ") should be unsupported";
      ok = Expect(false, msg.c_str()) && ok;
    }
    if (diag.rejection_reason == RejectionReason::kNone) {
      std::string msg = "instruction " + std::to_string(i) + " (" +
                        std::string(diag.source_opcode) +
                        ") should have a rejection reason";
      ok = Expect(false, msg.c_str()) && ok;
    }
  }

  return ok;
}

bool TestBringupSummaryConsistentWithCoverage() {
  const auto& summary = GetBringupSummary();
  auto coverage = ComputeGfx1250SeededFamilyCoverage();

  bool ok = true;

  for (const auto& c : coverage) {
    if (c.family_name == "vop3p") {
      ok = Expect(c.total_count > 0,
                  "vop3p seed count should match bringup focus") && ok;
    } else if (c.family_name == "wmma") {
      ok = Expect(c.total_count > 0,
                  "wmma seed count should match bringup focus") && ok;
    } else if (c.family_name == "fp8_bf8") {
      ok = Expect(c.total_count > 0,
                  "fp8_bf8 seed count should match bringup focus") && ok;
    } else if (c.family_name == "scale_paired") {
      ok = Expect(c.total_count > 0,
                  "scale_paired seed count should match bringup focus") && ok;
    }
  }

  // Bringup summary should report non-zero for each focus area.
  ok = Expect(summary.vop3p_instruction_count > 0,
              "bringup summary vop3p > 0") &&
       Expect(summary.wmma_instruction_count > 0,
              "bringup summary wmma > 0") &&
       Expect(summary.fp8_bf8_instruction_count > 0,
              "bringup summary fp8_bf8 > 0") &&
       Expect(summary.scale_paired_instruction_count > 0,
              "bringup summary scale_paired > 0") && ok;

  return ok;
}

bool TestNoGfx1250InstructionMarkedExecutable() {
  // Verify the hard constraint: no gfx1250 instruction should ever
  // be classified as executable via the semantic path.
  SemanticProgram program = LiftGfx1250SeedCatalog();
  SemanticLowering lowering;

  bool ok = true;
  for (const auto& instr : program) {
    TranslationStatus status = lowering.ClassifyForLowering(instr);
    if (status == TranslationStatus::kIdentity ||
        status == TranslationStatus::kRenamed ||
        status == TranslationStatus::kRewrittenWithFixup) {
      std::string msg = "gfx1250 instruction " + std::string(instr.opcode) +
                        " was classified as executable (status=" +
                        std::string(CrossArchTranslator::TranslationStatusName(
                            status)) +
                        ")";
      ok = Expect(false, msg.c_str()) && ok;
    }
  }

  return ok;
}

}  // namespace

int main() {
  bool ok = true;

  std::cerr << "=== gfx1250 seed catalog ingress ===\n";
  ok = TestLiftSeedCatalogProducesNonEmpty() && ok;
  ok = TestLiftSeedCatalogAllGfx1250Arch() && ok;
  ok = TestLiftSeedCatalogHasWmmaInstructions() && ok;
  ok = TestLiftSeedCatalogHasFp8Instructions() && ok;

  std::cerr << "\n=== gfx1250 focus instructions ===\n";
  ok = TestLiftFocusInstructionsNonEmpty() && ok;

  std::cerr << "\n=== gfx1250 seeded family coverage dashboards ===\n";
  ok = TestAllFourSeededFamiliesPresent() && ok;
  ok = TestSeededFamilyCoverageVop3p() && ok;
  ok = TestSeededFamilyCoverageWmma() && ok;
  ok = TestSeededFamilyCoverageFp8Bf8() && ok;
  ok = TestSeededFamilyCoverageScalePaired() && ok;

  std::cerr << "\n=== gfx1250 full catalog coverage ===\n";
  ok = TestFullCatalogCoverageAllBucketed() && ok;

  std::cerr << "\n=== gfx1250 translator integration ===\n";
  ok = TestGfx1250ToGfx950NeverExecutable() && ok;
  ok = TestGfx1250ToGfx950SemanticCoverageRicher() && ok;
  ok = TestGfx1250CoverageOnlyModeTranslation() && ok;

  std::cerr << "\n=== gfx1250 diagnostic corpus ===\n";
  ok = TestGfx1250DiagnosticsForUnsupportedFamilies() && ok;
  ok = TestBringupSummaryConsistentWithCoverage() && ok;
  ok = TestNoGfx1250InstructionMarkedExecutable() && ok;

  if (ok) {
    std::cerr << "\nAll gfx1250_coverage tests passed.\n";
  }
  return ok ? 0 : 1;
}
