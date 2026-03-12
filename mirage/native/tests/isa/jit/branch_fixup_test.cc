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
  // A simple program: mov, branch, mov, endpgm.
  // With 1:1 identity translation, branch offsets should be preserved.
  std::vector<DecodedInstruction> program = {
      DecodedInstruction::Unary("S_MOV_B32",
                                InstructionOperand::Sgpr(0),
                                InstructionOperand::Imm32(42)),
      DecodedInstruction::OneOperand("S_BRANCH",
                                     InstructionOperand::Imm32(1)),
      DecodedInstruction::Unary("S_MOV_B32",
                                InstructionOperand::Sgpr(1),
                                InstructionOperand::Imm32(0)),
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

// Test: FixupBranchTargets with a simulated 1:N expansion.
//
// Source program (4 instructions):
//   [0] S_NOP            → output [0]        (1:1)
//   [1] S_BRANCH delta=1 → output [1]        (1:1, branch over [2])
//   [2] S_NOP            → output [2,3]      (1:2 expansion, 2 output instrs)
//   [3] S_ENDPGM         → output [4]        (1:1)
//
// Source branch at [1] targets source_target = 1 + 1 + 1 = 3 (S_ENDPGM).
// After expansion, S_ENDPGM moves to output index 4.
// New delta = output_target(4) - (output_branch(1) + 1) = 2.
bool TestFixupBranchWithExpansion() {
  // Construct the output program as if a 1:2 expansion occurred at source[2].
  std::vector<DecodedInstruction> program = {
      DecodedInstruction::Nullary("S_NOP"),                              // [0]
      DecodedInstruction::OneOperand("S_BRANCH",
                                     InstructionOperand::Imm32(1)),      // [1]
      DecodedInstruction::Nullary("S_NOP"),                              // [2]
      DecodedInstruction::Nullary("S_NOP"),                              // [3] (expansion)
      DecodedInstruction::Nullary("S_ENDPGM"),                           // [4]
  };

  // Diagnostics reflecting the source→output mapping.
  std::vector<InstructionDiagnostic> diagnostics(4);
  diagnostics[0].source_index = 0;
  diagnostics[0].source_opcode = "S_NOP";
  diagnostics[0].output_begin_index = 0;
  diagnostics[0].output_instruction_count = 1;

  diagnostics[1].source_index = 1;
  diagnostics[1].source_opcode = "S_BRANCH";
  diagnostics[1].output_begin_index = 1;
  diagnostics[1].output_instruction_count = 1;

  diagnostics[2].source_index = 2;
  diagnostics[2].source_opcode = "S_NOP";
  diagnostics[2].output_begin_index = 2;
  diagnostics[2].output_instruction_count = 2;  // 1:2 expansion

  diagnostics[3].source_index = 3;
  diagnostics[3].source_opcode = "S_ENDPGM";
  diagnostics[3].output_begin_index = 4;
  diagnostics[3].output_instruction_count = 1;

  bool ok = CrossArchTranslator::FixupBranchTargets(&program, diagnostics);

  // Branch delta should be adjusted from 1 to 2.
  return Expect(ok, "fixup should succeed") &&
         Expect(static_cast<std::int32_t>(program[1].operands[0].imm32) == 2,
                "branch delta should be adjusted from 1 to 2");
}

// Test: FixupBranchTargets with a backward branch across an expansion.
//
// Source program (4 instructions):
//   [0] S_NOP            → output [0,1]      (1:2 expansion)
//   [1] S_NOP            → output [2]        (1:1)
//   [2] S_NOP            → output [3]        (1:1)
//   [3] S_BRANCH delta=-3 → output [4]       (branch back to source[1])
//
// Source branch at [3] targets source_target = 3 + 1 + (-3) = 1.
// Source[1] maps to output index 2.
// New delta = output_target(2) - (output_branch(4) + 1) = -3.
// In this case, delta stays -3 because the expansion was before the loop body.
bool TestFixupBackwardBranchWithExpansion() {
  std::vector<DecodedInstruction> program = {
      DecodedInstruction::Nullary("S_NOP"),                              // [0]
      DecodedInstruction::Nullary("S_NOP"),                              // [1] (expansion)
      DecodedInstruction::Nullary("S_NOP"),                              // [2]
      DecodedInstruction::Nullary("S_NOP"),                              // [3]
      DecodedInstruction::OneOperand("S_BRANCH",
                                     InstructionOperand::Imm32(
                                         static_cast<std::uint32_t>(-3))), // [4]
  };

  std::vector<InstructionDiagnostic> diagnostics(4);
  diagnostics[0].source_index = 0;
  diagnostics[0].source_opcode = "S_NOP";
  diagnostics[0].output_begin_index = 0;
  diagnostics[0].output_instruction_count = 2;  // 1:2 expansion

  diagnostics[1].source_index = 1;
  diagnostics[1].source_opcode = "S_NOP";
  diagnostics[1].output_begin_index = 2;
  diagnostics[1].output_instruction_count = 1;

  diagnostics[2].source_index = 2;
  diagnostics[2].source_opcode = "S_NOP";
  diagnostics[2].output_begin_index = 3;
  diagnostics[2].output_instruction_count = 1;

  diagnostics[3].source_index = 3;
  diagnostics[3].source_opcode = "S_BRANCH";
  diagnostics[3].output_begin_index = 4;
  diagnostics[3].output_instruction_count = 1;

  bool ok = CrossArchTranslator::FixupBranchTargets(&program, diagnostics);

  // Branch should now target output[2] from output[4]: delta = 2 - (4+1) = -3
  return Expect(ok, "backward fixup should succeed") &&
         Expect(static_cast<std::int32_t>(program[4].operands[0].imm32) == -3,
                "backward branch delta should be -3 after fixup");
}

// Test: FixupBranchTargets with multiple expansions shifting a branch.
//
// Source program (5 instructions):
//   [0] S_NOP            → output [0,1]      (1:2)
//   [1] S_BRANCH delta=2 → output [2]        (branch to source[4])
//   [2] S_NOP            → output [3,4,5]    (1:3)
//   [3] S_NOP            → output [6]        (1:1)
//   [4] S_ENDPGM         → output [7]        (1:1)
//
// Source branch at [1]: source_target = 1 + 1 + 2 = 4.
// Source[4] maps to output index 7.
// New delta = 7 - (2 + 1) = 4.
bool TestFixupWithMultipleExpansions() {
  std::vector<DecodedInstruction> program = {
      DecodedInstruction::Nullary("S_NOP"),                              // [0]
      DecodedInstruction::Nullary("S_NOP"),                              // [1]
      DecodedInstruction::OneOperand("S_BRANCH",
                                     InstructionOperand::Imm32(2)),      // [2]
      DecodedInstruction::Nullary("S_NOP"),                              // [3]
      DecodedInstruction::Nullary("S_NOP"),                              // [4]
      DecodedInstruction::Nullary("S_NOP"),                              // [5]
      DecodedInstruction::Nullary("S_NOP"),                              // [6]
      DecodedInstruction::Nullary("S_ENDPGM"),                           // [7]
  };

  std::vector<InstructionDiagnostic> diagnostics(5);
  diagnostics[0].source_index = 0;
  diagnostics[0].source_opcode = "S_NOP";
  diagnostics[0].output_begin_index = 0;
  diagnostics[0].output_instruction_count = 2;

  diagnostics[1].source_index = 1;
  diagnostics[1].source_opcode = "S_BRANCH";
  diagnostics[1].output_begin_index = 2;
  diagnostics[1].output_instruction_count = 1;

  diagnostics[2].source_index = 2;
  diagnostics[2].source_opcode = "S_NOP";
  diagnostics[2].output_begin_index = 3;
  diagnostics[2].output_instruction_count = 3;

  diagnostics[3].source_index = 3;
  diagnostics[3].source_opcode = "S_NOP";
  diagnostics[3].output_begin_index = 6;
  diagnostics[3].output_instruction_count = 1;

  diagnostics[4].source_index = 4;
  diagnostics[4].source_opcode = "S_ENDPGM";
  diagnostics[4].output_begin_index = 7;
  diagnostics[4].output_instruction_count = 1;

  bool ok = CrossArchTranslator::FixupBranchTargets(&program, diagnostics);

  return Expect(ok, "multi-expansion fixup should succeed") &&
         Expect(static_cast<std::int32_t>(program[2].operands[0].imm32) == 4,
                "branch delta should be adjusted from 2 to 4");
}

// Test: 1:1 translation (no expansion) leaves branch offsets unchanged.
bool TestFixupNoExpansionNoop() {
  std::vector<DecodedInstruction> program = {
      DecodedInstruction::Nullary("S_NOP"),
      DecodedInstruction::OneOperand("S_CBRANCH_SCC1",
                                     InstructionOperand::Imm32(1)),
      DecodedInstruction::Nullary("S_NOP"),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  std::vector<InstructionDiagnostic> diagnostics(4);
  for (std::size_t i = 0; i < 4; ++i) {
    diagnostics[i].source_index = i;
    diagnostics[i].source_opcode = program[i].opcode;
    diagnostics[i].output_begin_index = i;
    diagnostics[i].output_instruction_count = 1;
  }

  bool ok = CrossArchTranslator::FixupBranchTargets(&program, diagnostics);

  return Expect(ok, "1:1 fixup should succeed") &&
         Expect(program[1].operands[0].imm32 == 1,
                "1:1 branch delta should remain 1");
}

}  // namespace

int main() {
  bool ok = true;
  ok = TestIdentityTranslationPreservesBranches() && ok;
  ok = TestConditionalBranchTranslation() && ok;
  ok = TestAllBranchVariantsSupported() && ok;
  ok = TestFixupBranchWithExpansion() && ok;
  ok = TestFixupBackwardBranchWithExpansion() && ok;
  ok = TestFixupWithMultipleExpansions() && ok;
  ok = TestFixupNoExpansionNoop() && ok;

  if (ok) {
    std::cerr << "All branch_fixup tests passed.\n";
  }
  return ok ? 0 : 1;
}
