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

bool TestTranslateSimpleScalarProgram() {
  std::vector<DecodedInstruction> program = {
      DecodedInstruction::Unary("S_MOV_B32",
                                InstructionOperand::Sgpr(0),
                                InstructionOperand::Imm32(42)),
      DecodedInstruction::Binary("S_ADD_U32",
                                 InstructionOperand::Sgpr(2),
                                 InstructionOperand::Sgpr(0),
                                 InstructionOperand::Sgpr(1)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  TranslationConfig config;
  config.source_arch = SourceArchitecture::kGfx1201;
  config.target_arch = TargetArchitecture::kGfx950;

  CrossArchTranslator translator(config);
  TranslationResult result = translator.Translate(program);

  return Expect(result.is_executable,
                "simple scalar program should be executable") &&
         Expect(result.translated_program.size() == 3,
                "should produce 3 translated instructions") &&
         Expect(result.translated_program[0].opcode == "S_MOV_B32",
                "first instruction opcode should be S_MOV_B32") &&
         Expect(result.translated_program[0].operands[1].imm32 == 42,
                "immediate value should be preserved") &&
         Expect(result.capability_summary.executable_count == 3,
                "all 3 instructions should be executable") &&
         Expect(result.capability_summary.unsupported_count == 0,
                "no instructions should be unsupported");
}

bool TestTranslateVectorProgram() {
  std::vector<DecodedInstruction> program = {
      DecodedInstruction::Unary("V_MOV_B32",
                                InstructionOperand::Vgpr(0),
                                InstructionOperand::Imm32(100)),
      DecodedInstruction::Binary("V_ADD_F32",
                                 InstructionOperand::Vgpr(2),
                                 InstructionOperand::Vgpr(0),
                                 InstructionOperand::Vgpr(1)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  TranslationConfig config;
  config.source_arch = SourceArchitecture::kGfx1201;
  config.target_arch = TargetArchitecture::kGfx950;

  CrossArchTranslator translator(config);
  TranslationResult result = translator.Translate(program);

  return Expect(result.is_executable,
                "vector program should be executable") &&
         Expect(result.translated_program.size() == 3,
                "should produce 3 instructions");
}

bool TestTranslateWithRename() {
  std::vector<DecodedInstruction> program = {
      DecodedInstruction::Binary("S_ADD_CO_U32",
                                 InstructionOperand::Sgpr(0),
                                 InstructionOperand::Sgpr(1),
                                 InstructionOperand::Sgpr(248)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  TranslationConfig config;
  config.source_arch = SourceArchitecture::kGfx1201;
  config.target_arch = TargetArchitecture::kGfx950;

  CrossArchTranslator translator(config);
  TranslationResult result = translator.Translate(program);

  return Expect(result.is_executable,
                "rename program should be executable") &&
         Expect(result.translated_program[0].opcode == "S_ADD_U32",
                "S_ADD_CO_U32 should be renamed to S_ADD_U32") &&
         Expect(result.translated_program[0].operands[2].index == 106,
                "VCC SGPR 248 should be remapped to 106");
}

bool TestUnsupportedOpcodeRejectsProgram() {
  std::vector<DecodedInstruction> program = {
      DecodedInstruction::Unary("S_MOV_B32",
                                InstructionOperand::Sgpr(0),
                                InstructionOperand::Imm32(1)),
      DecodedInstruction::Nullary("V_FAKE_UNSUPPORTED_OP"),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  TranslationConfig config;
  config.source_arch = SourceArchitecture::kGfx1201;
  config.target_arch = TargetArchitecture::kGfx950;
  config.translation_mode = TranslationMode::kExecutableStrict;

  CrossArchTranslator translator(config);
  TranslationResult result = translator.Translate(program);

  return Expect(!result.is_executable,
                "program with unsupported op should not be executable") &&
         Expect(result.capability_summary.unsupported_count > 0,
                "should report unsupported instructions");
}

bool TestWaveSensitiveOpcodeRejected() {
  TranslationConfig config;
  config.source_arch = SourceArchitecture::kGfx1201;
  config.target_arch = TargetArchitecture::kGfx950;
  config.translation_mode = TranslationMode::kExecutableStrict;

  CrossArchTranslator translator(config);
  TranslationStatus status =
      translator.ClassifyInstruction("V_READLANE_B32");

  return Expect(status == TranslationStatus::kUnsupported,
                "V_READLANE_B32 should be unsupported in executable mode");
}

bool TestCoverageReport() {
  std::vector<DecodedInstruction> program = {
      DecodedInstruction::Unary("S_MOV_B32",
                                InstructionOperand::Sgpr(0),
                                InstructionOperand::Imm32(1)),
      DecodedInstruction::Nullary("V_FAKE_OP"),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  TranslationConfig config;
  config.source_arch = SourceArchitecture::kGfx1201;
  config.target_arch = TargetArchitecture::kGfx950;

  CrossArchTranslator translator(config);
  CapabilitySummary summary = translator.ComputeCoverage(program);

  return Expect(summary.executable_count == 2,
                "S_MOV_B32 and S_ENDPGM should be executable") &&
         Expect(summary.unsupported_count == 1,
                "V_FAKE_OP should be unsupported") &&
         Expect(summary.total() == 3,
                "total should equal program size");
}

bool TestArchitectureNames() {
  return Expect(CrossArchTranslator::ArchitectureName(
                    SourceArchitecture::kGfx950) == "gfx950",
                "gfx950 source name") &&
         Expect(CrossArchTranslator::ArchitectureName(
                    SourceArchitecture::kGfx1201) == "gfx1201",
                "gfx1201 source name") &&
         Expect(CrossArchTranslator::ArchitectureName(
                    TargetArchitecture::kGfx950) == "gfx950",
                "gfx950 target name") &&
         Expect(CrossArchTranslator::TranslationModeName(
                    TranslationMode::kExecutableStrict) ==
                    "executable_strict",
                "executable_strict mode name");
}

bool TestCacheKeyOrdering() {
  TranslationCacheKey key1;
  key1.source_code_va = 100;
  key1.source_arch = SourceArchitecture::kGfx1201;
  key1.target_arch = TargetArchitecture::kGfx950;

  TranslationCacheKey key2;
  key2.source_code_va = 200;
  key2.source_arch = SourceArchitecture::kGfx1201;
  key2.target_arch = TargetArchitecture::kGfx950;

  TranslationCacheKey key3;
  key3.source_code_va = 100;
  key3.source_arch = SourceArchitecture::kGfx1201;
  key3.target_arch = TargetArchitecture::kGfx950;
  key3.translation_mode = TranslationMode::kCoverageOnly;

  return Expect(key1 < key2, "key with lower VA should sort first") &&
         Expect(!(key2 < key1), "key ordering should be consistent") &&
         Expect(key1 < key3, "different mode should produce different key") &&
         Expect(!(key1 < key1), "key should not be less than itself");
}

bool TestExecNarrowingFlagSet() {
  std::vector<DecodedInstruction> program = {
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  // gfx1201 (wave32) -> gfx950 (wave64): should require narrowing.
  TranslationConfig config;
  config.source_arch = SourceArchitecture::kGfx1201;
  config.target_arch = TargetArchitecture::kGfx950;

  CrossArchTranslator translator(config);
  TranslationResult result = translator.Translate(program);

  bool ok = Expect(result.is_executable, "program should be executable") &&
            Expect(result.requires_exec_narrowing,
                   "gfx1201->gfx950 should require exec narrowing");

  // Non-executable program should not have narrowing flag set.
  std::vector<DecodedInstruction> bad_program = {
      DecodedInstruction::Nullary("UNSUPPORTED_OP"),
  };

  TranslationResult bad_result = translator.Translate(bad_program);

  ok = Expect(!bad_result.is_executable,
              "unsupported program should not be executable") &&
       Expect(!bad_result.requires_exec_narrowing,
              "non-executable program should not require narrowing") && ok;

  return ok;
}

bool TestExecManipulatingFlagSet() {
  // Program with SAVEEXEC should set contains_exec_manipulating_instructions.
  std::vector<DecodedInstruction> program_with_saveexec = {
      DecodedInstruction::Unary("S_AND_SAVEEXEC_B64",
                                InstructionOperand::Sgpr(0),
                                InstructionOperand::Sgpr(2)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  TranslationConfig config;
  config.source_arch = SourceArchitecture::kGfx1201;
  config.target_arch = TargetArchitecture::kGfx950;

  CrossArchTranslator translator(config);
  TranslationResult result = translator.Translate(program_with_saveexec);

  bool ok = Expect(result.is_executable,
                   "SAVEEXEC program should be executable") &&
            Expect(result.contains_exec_manipulating_instructions,
                   "program with SAVEEXEC should flag exec-manipulating");

  // Program without EXEC-manipulating instructions should not set the flag.
  std::vector<DecodedInstruction> program_no_exec = {
      DecodedInstruction::Unary("S_MOV_B32",
                                InstructionOperand::Sgpr(0),
                                InstructionOperand::Imm32(1)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  TranslationResult result2 = translator.Translate(program_no_exec);

  ok = Expect(result2.is_executable,
              "simple program should be executable") &&
       Expect(!result2.contains_exec_manipulating_instructions,
              "program without EXEC ops should not flag exec-manipulating") &&
       ok;

  return ok;
}

bool TestLdsDetectionFlagSet() {
  // Program with LDS instruction should set contains_lds_instructions
  // and should NOT be executable (DS_READ_B32 has no translation rule).
  std::vector<DecodedInstruction> program_with_lds = {
      DecodedInstruction::Unary("S_MOV_B32",
                                InstructionOperand::Sgpr(0),
                                InstructionOperand::Imm32(0)),
      DecodedInstruction::Unary("DS_READ_B32",
                                InstructionOperand::Vgpr(0),
                                InstructionOperand::Vgpr(1)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  TranslationConfig config;
  config.source_arch = SourceArchitecture::kGfx1201;
  config.target_arch = TargetArchitecture::kGfx950;

  CrossArchTranslator translator(config);
  TranslationResult result = translator.Translate(program_with_lds);

  bool ok = Expect(!result.is_executable,
                   "program with LDS should not be executable") &&
            Expect(result.contains_lds_instructions,
                   "program with DS_READ_B32 should flag LDS usage");

  // Program without LDS should not set the flag.
  std::vector<DecodedInstruction> program_no_lds = {
      DecodedInstruction::Unary("S_MOV_B32",
                                InstructionOperand::Sgpr(0),
                                InstructionOperand::Imm32(1)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  TranslationResult result2 = translator.Translate(program_no_lds);

  ok = Expect(result2.is_executable,
              "program without LDS should be executable") &&
       Expect(!result2.contains_lds_instructions,
              "program without DS ops should not flag LDS") && ok;

  return ok;
}

bool TestStructuredRejectionReasons() {
  std::vector<DecodedInstruction> program = {
      DecodedInstruction::Unary("S_MOV_B32",
                                InstructionOperand::Sgpr(0),
                                InstructionOperand::Imm32(1)),
      DecodedInstruction::Binary("V_READLANE_B32",
                                 InstructionOperand::Sgpr(1),
                                 InstructionOperand::Vgpr(0),
                                 InstructionOperand::Sgpr(0)),
      DecodedInstruction::Unary("DS_READ_B32",
                                InstructionOperand::Vgpr(0),
                                InstructionOperand::Vgpr(1)),
      DecodedInstruction::Nullary("FAKE_NONEXISTENT_OP"),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  TranslationConfig config;
  config.source_arch = SourceArchitecture::kGfx1201;
  config.target_arch = TargetArchitecture::kGfx950;

  CrossArchTranslator translator(config);
  TranslationResult result = translator.Translate(program);

  return Expect(!result.is_executable,
                "mixed program should not be executable") &&
         Expect(result.diagnostics.size() == 5,
                "should have 5 diagnostics") &&
         Expect(result.diagnostics[0].rejection_reason ==
                    RejectionReason::kNone,
                "S_MOV_B32 should have no rejection") &&
         Expect(result.diagnostics[1].rejection_reason ==
                    RejectionReason::kWaveSensitive,
                "V_READLANE_B32 should be rejected as wave-sensitive") &&
         Expect(result.diagnostics[2].rejection_reason ==
                    RejectionReason::kLdsTouching,
                "DS_READ_B32 should be rejected as LDS-touching") &&
         Expect(result.diagnostics[3].rejection_reason ==
                    RejectionReason::kNoTranslationRule,
                "FAKE_OP should be rejected as no-rule");
}

bool TestReverseDirectionGfx950ToGfx1201WithRules() {
  // gfx950 -> gfx1201 now has identity rules for the shared opcode subset.
  // S_MOV_B32 and S_ENDPGM are both in the identity list.
  std::vector<DecodedInstruction> program = {
      DecodedInstruction::Unary("S_MOV_B32",
                                InstructionOperand::Sgpr(0),
                                InstructionOperand::Imm32(42)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  TranslationConfig config;
  config.source_arch = SourceArchitecture::kGfx950;
  config.target_arch = TargetArchitecture::kGfx1201;

  CrossArchTranslator translator(config);
  TranslationResult result = translator.Translate(program);

  bool ok = Expect(result.is_executable,
                   "gfx950->gfx1201 should be executable (identity rules)") &&
            Expect(result.capability_summary.executable_count == 2,
                   "both instructions should be executable") &&
            Expect(result.capability_summary.unsupported_count == 0,
                   "no instructions should be unsupported") &&
            Expect(!result.requires_exec_narrowing,
                   "gfx950->gfx1201 should not require exec narrowing");

  // Coverage report should show all executable.
  CapabilitySummary summary = translator.ComputeCoverage(program);
  ok = Expect(summary.executable_count == 2,
              "coverage should show 2 executable for gfx950->gfx1201") &&
       Expect(summary.unsupported_count == 0,
              "coverage should show 0 unsupported for gfx950->gfx1201") && ok;

  return ok;
}

bool TestReverseDirectionTranslateScalar() {
  // Translate a scalar program in the gfx950 -> gfx1201 direction.
  // All opcodes are in the shared identity set.
  std::vector<DecodedInstruction> program = {
      DecodedInstruction::Unary("S_MOV_B32",
                                InstructionOperand::Sgpr(0),
                                InstructionOperand::Imm32(42)),
      DecodedInstruction::Binary("S_ADD_U32",
                                 InstructionOperand::Sgpr(2),
                                 InstructionOperand::Sgpr(0),
                                 InstructionOperand::Sgpr(1)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  TranslationConfig config;
  config.source_arch = SourceArchitecture::kGfx950;
  config.target_arch = TargetArchitecture::kGfx1201;

  CrossArchTranslator translator(config);
  TranslationResult result = translator.Translate(program);

  return Expect(result.is_executable,
                "reverse scalar translation should be executable") &&
         Expect(result.capability_summary.executable_count == 3,
                "all 3 instructions should be executable") &&
         Expect(result.translated_program.size() == 3,
                "translated program should have 3 instructions") &&
         Expect(result.translated_program[0].opcode == "S_MOV_B32",
                "S_MOV_B32 should remain S_MOV_B32") &&
         Expect(result.translated_program[1].opcode == "S_ADD_U32",
                "S_ADD_U32 should remain S_ADD_U32") &&
         Expect(result.translated_program[2].opcode == "S_ENDPGM",
                "S_ENDPGM should remain S_ENDPGM");
}

bool TestReverseDirectionCoverage() {
  // Mixed program: identity opcodes + gfx950-only MFMA opcode.
  // MFMA should be unsupported in the reverse direction (no gfx1201 equivalent).
  std::vector<DecodedInstruction> program = {
      DecodedInstruction::Unary("S_MOV_B32",
                                InstructionOperand::Sgpr(0),
                                InstructionOperand::Imm32(1)),
      DecodedInstruction::Nullary("V_MFMA_F32_16X16X4_F32"),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  TranslationConfig config;
  config.source_arch = SourceArchitecture::kGfx950;
  config.target_arch = TargetArchitecture::kGfx1201;

  CrossArchTranslator translator(config);
  CapabilitySummary summary = translator.ComputeCoverage(program);

  return Expect(summary.executable_count == 2,
                "S_MOV_B32 + S_ENDPGM should be executable") &&
         Expect(summary.unsupported_count == 1,
                "V_MFMA should be unsupported in reverse direction");
}

bool TestReverseDirectionNoExecNarrowing() {
  // gfx950 (wave64) -> gfx1201 (wave32): no exec narrowing needed.
  std::vector<DecodedInstruction> program = {
      DecodedInstruction::Binary("V_ADD_F32",
                                 InstructionOperand::Vgpr(2),
                                 InstructionOperand::Vgpr(0),
                                 InstructionOperand::Vgpr(1)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  TranslationConfig config;
  config.source_arch = SourceArchitecture::kGfx950;
  config.target_arch = TargetArchitecture::kGfx1201;

  CrossArchTranslator translator(config);
  TranslationResult result = translator.Translate(program);

  return Expect(result.is_executable,
                "reverse vector translation should be executable") &&
         Expect(!result.requires_exec_narrowing,
                "gfx950->gfx1201 should not require exec narrowing");
}

bool TestReverseDirectionArchNames() {
  return Expect(CrossArchTranslator::ArchitectureName(
                    SourceArchitecture::kGfx950) == "gfx950",
                "source gfx950 name") &&
         Expect(CrossArchTranslator::ArchitectureName(
                    TargetArchitecture::kGfx1201) == "gfx1201",
                "target gfx1201 name") &&
         Expect(CrossArchTranslator::ArchitectureName(
                    TargetArchitecture::kGfx1250) == "gfx1250",
                "target gfx1250 name") &&
         Expect(CrossArchTranslator::ArchitectureName(
                    SourceArchitecture::kGfx1250) == "gfx1250",
                "source gfx1250 name");
}

}  // namespace

int main() {
  bool ok = true;
  ok = TestTranslateSimpleScalarProgram() && ok;
  ok = TestTranslateVectorProgram() && ok;
  ok = TestTranslateWithRename() && ok;
  ok = TestUnsupportedOpcodeRejectsProgram() && ok;
  ok = TestWaveSensitiveOpcodeRejected() && ok;
  ok = TestCoverageReport() && ok;
  ok = TestArchitectureNames() && ok;
  ok = TestCacheKeyOrdering() && ok;
  ok = TestExecNarrowingFlagSet() && ok;
  ok = TestExecManipulatingFlagSet() && ok;
  ok = TestLdsDetectionFlagSet() && ok;
  ok = TestStructuredRejectionReasons() && ok;
  ok = TestReverseDirectionGfx950ToGfx1201WithRules() && ok;
  ok = TestReverseDirectionTranslateScalar() && ok;
  ok = TestReverseDirectionCoverage() && ok;
  ok = TestReverseDirectionNoExecNarrowing() && ok;
  ok = TestReverseDirectionArchNames() && ok;

  if (ok) {
    std::cerr << "All cross_arch_translator tests passed.\n";
  }
  return ok ? 0 : 1;
}
