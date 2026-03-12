#include <iostream>
#include <vector>

#include "lib/sim/isa/common/decoded_instruction.h"
#include "lib/sim/isa/jit/cross_arch_translator.h"

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

bool TestIdentityTranslationPreservesBranches() {
  // A simple program: mov, branch, endpgm.
  // With 1:1 identity translation, branch offsets should be preserved.
  std::vector<DecodedInstruction> program = {
      DecodedInstruction::Unary("S_MOV_B32",
                                InstructionOperand::Sgpr(0),
                                InstructionOperand::Imm32(42)),
      DecodedInstruction::OneOperand("S_BRANCH",
                                     InstructionOperand::Imm32(1)),
      DecodedInstruction::Nullary("S_NOP"),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  TranslationConfig config;
  config.source_arch = SourceArchitecture::kGfx1201;
  config.target_arch = TargetArchitecture::kGfx950;
  config.translation_mode = TranslationMode::kExecutableStrict;

  CrossArchTranslator translator(config);
  TranslationResult result = translator.Translate(program);

  return Expect(result.is_executable,
                "simple branch program should be executable") &&
         Expect(result.translated_program.size() == program.size(),
                "1:1 translation should preserve program size") &&
         Expect(result.translated_program[1].opcode == "S_BRANCH",
                "branch opcode should be preserved") &&
         Expect(result.translated_program[1].operands[0].imm32 == 1,
                "branch offset should be preserved in 1:1 translation");
}

bool TestConditionalBranchTranslation() {
  std::vector<DecodedInstruction> program = {
      DecodedInstruction::Binary("S_CMP_EQ_I32",
                                 InstructionOperand::Sgpr(0),
                                 InstructionOperand::Sgpr(1),
                                 InstructionOperand::Imm32(0)),
      DecodedInstruction::OneOperand("S_CBRANCH_SCC0",
                                     InstructionOperand::Imm32(2)),
      DecodedInstruction::Unary("S_MOV_B32",
                                InstructionOperand::Sgpr(2),
                                InstructionOperand::Imm32(1)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  TranslationConfig config;
  config.source_arch = SourceArchitecture::kGfx1201;
  config.target_arch = TargetArchitecture::kGfx950;

  CrossArchTranslator translator(config);
  TranslationResult result = translator.Translate(program);

  return Expect(result.is_executable,
                "conditional branch program should be executable") &&
         Expect(result.translated_program[1].opcode == "S_CBRANCH_SCC0",
                "conditional branch opcode should be preserved");
}

bool TestAllBranchVariantsSupported() {
  const std::string_view branch_opcodes[] = {
      "S_BRANCH", "S_CBRANCH_SCC0", "S_CBRANCH_SCC1",
      "S_CBRANCH_VCCZ", "S_CBRANCH_VCCNZ",
      "S_CBRANCH_EXECZ", "S_CBRANCH_EXECNZ",
  };

  TranslationConfig config;
  config.source_arch = SourceArchitecture::kGfx1201;
  config.target_arch = TargetArchitecture::kGfx950;

  CrossArchTranslator translator(config);

  for (const auto& opcode : branch_opcodes) {
    TranslationStatus status = translator.ClassifyInstruction(opcode);
    if (status != TranslationStatus::kIdentity) {
      std::cerr << "FAIL: " << opcode << " should classify as identity\n";
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  bool ok = true;
  ok = TestIdentityTranslationPreservesBranches() && ok;
  ok = TestConditionalBranchTranslation() && ok;
  ok = TestAllBranchVariantsSupported() && ok;

  if (ok) {
    std::cerr << "All branch_fixup tests passed.\n";
  }
  return ok ? 0 : 1;
}
