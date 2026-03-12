#include <iostream>
#include <vector>

#include "lib/sim/isa/common/decoded_instruction.h"
#include "lib/sim/isa/jit/cross_arch_translator.h"
#include "lib/sim/isa/jit/translation_rules.h"

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

bool TestGfx1201ToGfx950IdentityCoverage() {
  TranslationRuleTable table;
  table.BuildForDirection(
      static_cast<std::uint8_t>(SourceArchitecture::kGfx1201),
      static_cast<std::uint8_t>(TargetArchitecture::kGfx950));

  std::size_t identity = table.identity_count();
  std::size_t rename = table.rename_count();
  std::size_t total = table.rules().size();

  std::cerr << "gfx1201->gfx950 coverage:\n"
            << "  identity: " << identity << "\n"
            << "  rename:   " << rename << "\n"
            << "  total:    " << total << "\n";

  return Expect(identity > 90,
                "should have >90 identity translations") &&
         Expect(rename > 0,
                "should have >0 rename translations") &&
         Expect(total == identity + rename,
                "total should equal identity + rename (no fixup rules yet)");
}

bool TestFourBucketCoverage() {
  std::vector<DecodedInstruction> program = {
      DecodedInstruction::Unary("S_MOV_B32",
                                InstructionOperand::Sgpr(0),
                                InstructionOperand::Imm32(1)),
      DecodedInstruction::Binary("S_ADD_CO_U32",
                                 InstructionOperand::Sgpr(0),
                                 InstructionOperand::Sgpr(1),
                                 InstructionOperand::Sgpr(248)),
      DecodedInstruction::Nullary("UNSUPPORTED_OP"),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  TranslationConfig config;
  config.source_arch = SourceArchitecture::kGfx1201;
  config.target_arch = TargetArchitecture::kGfx950;

  CrossArchTranslator translator(config);
  CapabilitySummary summary = translator.ComputeCoverage(program);

  std::cerr << "four-bucket coverage:\n"
            << "  executable:        " << summary.executable_count << "\n"
            << "  coverage_only:     " << summary.coverage_only_count << "\n"
            << "  blocked_on_runtime:" << summary.blocked_on_runtime_count << "\n"
            << "  unsupported:       " << summary.unsupported_count << "\n"
            << "  total:             " << summary.total() << "\n";

  return Expect(summary.executable_count == 3,
                "3 instructions should be executable (MOV, ADD_CO renamed, ENDPGM)") &&
         Expect(summary.unsupported_count == 1,
                "1 instruction should be unsupported") &&
         Expect(summary.total() == 4,
                "total should be 4");
}

bool TestEmptyProgramCoverage() {
  std::vector<DecodedInstruction> program;

  TranslationConfig config;
  config.source_arch = SourceArchitecture::kGfx1201;
  config.target_arch = TargetArchitecture::kGfx950;

  CrossArchTranslator translator(config);
  CapabilitySummary summary = translator.ComputeCoverage(program);

  return Expect(summary.total() == 0,
                "empty program should have zero coverage") &&
         Expect(summary.executable_count == 0,
                "empty program should have zero executable");
}

bool TestAllIdentityProgramTranslatesSuccessfully() {
  std::vector<DecodedInstruction> program = {
      DecodedInstruction::Unary("S_MOV_B32",
                                InstructionOperand::Sgpr(0),
                                InstructionOperand::Imm32(1)),
      DecodedInstruction::Unary("S_MOV_B32",
                                InstructionOperand::Sgpr(1),
                                InstructionOperand::Imm32(2)),
      DecodedInstruction::Binary("S_ADD_U32",
                                 InstructionOperand::Sgpr(2),
                                 InstructionOperand::Sgpr(0),
                                 InstructionOperand::Sgpr(1)),
      DecodedInstruction::Binary("V_ADD_F32",
                                 InstructionOperand::Vgpr(0),
                                 InstructionOperand::Vgpr(1),
                                 InstructionOperand::Vgpr(2)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  TranslationConfig config;
  config.source_arch = SourceArchitecture::kGfx1201;
  config.target_arch = TargetArchitecture::kGfx950;

  CrossArchTranslator translator(config);
  TranslationResult result = translator.Translate(program);

  return Expect(result.is_executable,
                "all-identity program should be executable") &&
         Expect(result.capability_summary.executable_count == 5,
                "all 5 instructions should be executable") &&
         Expect(result.capability_summary.unsupported_count == 0,
                "no instructions should be unsupported") &&
         Expect(result.diagnostics.size() == 5,
                "should have 5 per-instruction diagnostics");
}

bool TestDiagnosticStatusNames() {
  return Expect(CrossArchTranslator::TranslationStatusName(
                    TranslationStatus::kIdentity) == "identity",
                "identity status name") &&
         Expect(CrossArchTranslator::TranslationStatusName(
                    TranslationStatus::kRenamed) == "renamed",
                "renamed status name") &&
         Expect(CrossArchTranslator::TranslationStatusName(
                    TranslationStatus::kUnsupported) == "unsupported",
                "unsupported status name") &&
         Expect(CrossArchTranslator::TranslationStatusName(
                    TranslationStatus::kBlockedOnRuntime) ==
                    "blocked_on_runtime",
                "blocked_on_runtime status name");
}

}  // namespace

int main() {
  bool ok = true;
  ok = TestGfx1201ToGfx950IdentityCoverage() && ok;
  ok = TestFourBucketCoverage() && ok;
  ok = TestEmptyProgramCoverage() && ok;
  ok = TestAllIdentityProgramTranslatesSuccessfully() && ok;
  ok = TestDiagnosticStatusNames() && ok;

  if (ok) {
    std::cerr << "All translation_coverage tests passed.\n";
  }
  return ok ? 0 : 1;
}
