#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
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

using namespace mirage::sim::isa::jit;
using mirage::sim::isa::DecodedInstruction;
using mirage::sim::isa::OperandKind;

TranslationConfig DefaultConfig() {
  TranslationConfig config;
  config.source_arch = SourceArchitecture::kGfx1201;
  config.target_arch = TargetArchitecture::kGfx950;
  config.translation_mode = TranslationMode::kExecutableStrict;
  return config;
}

// --- Memory operation rename tests ---

bool TestScalarMemoryRenames() {
  TranslationRuleTable table;
  table.BuildForDirection(
      static_cast<std::uint8_t>(SourceArchitecture::kGfx1201),
      static_cast<std::uint8_t>(TargetArchitecture::kGfx950));

  // S_LOAD_B32 → S_LOAD_DWORD
  const auto* r1 = table.FindRule("S_LOAD_B32");
  if (!Expect(r1 != nullptr, "S_LOAD_B32 should have a rule")) return false;
  if (!Expect(r1->target_opcode == "S_LOAD_DWORD",
              "S_LOAD_B32 → S_LOAD_DWORD"))
    return false;
  if (!Expect(r1->tier == TranslationTier::kRename,
              "S_LOAD_B32 should be kRename"))
    return false;

  // S_LOAD_B64 → S_LOAD_DWORDX2
  const auto* r2 = table.FindRule("S_LOAD_B64");
  if (!Expect(r2 != nullptr && r2->target_opcode == "S_LOAD_DWORDX2",
              "S_LOAD_B64 → S_LOAD_DWORDX2"))
    return false;

  // S_LOAD_B128 → S_LOAD_DWORDX4
  const auto* r3 = table.FindRule("S_LOAD_B128");
  if (!Expect(r3 != nullptr && r3->target_opcode == "S_LOAD_DWORDX4",
              "S_LOAD_B128 → S_LOAD_DWORDX4"))
    return false;

  // S_LOAD_B256 → S_LOAD_DWORDX8
  const auto* r4 = table.FindRule("S_LOAD_B256");
  if (!Expect(r4 != nullptr && r4->target_opcode == "S_LOAD_DWORDX8",
              "S_LOAD_B256 → S_LOAD_DWORDX8"))
    return false;

  // S_BUFFER_LOAD renames
  const auto* r5 = table.FindRule("S_BUFFER_LOAD_B32");
  if (!Expect(r5 != nullptr && r5->target_opcode == "S_BUFFER_LOAD_DWORD",
              "S_BUFFER_LOAD_B32 → S_BUFFER_LOAD_DWORD"))
    return false;

  const auto* r6 = table.FindRule("S_BUFFER_LOAD_B128");
  if (!Expect(r6 != nullptr && r6->target_opcode == "S_BUFFER_LOAD_DWORDX4",
              "S_BUFFER_LOAD_B128 → S_BUFFER_LOAD_DWORDX4"))
    return false;

  return true;
}

bool TestGlobalMemoryRenames() {
  TranslationRuleTable table;
  table.BuildForDirection(
      static_cast<std::uint8_t>(SourceArchitecture::kGfx1201),
      static_cast<std::uint8_t>(TargetArchitecture::kGfx950));

  // GLOBAL_LOAD_B32 → GLOBAL_LOAD_DWORD
  const auto* r1 = table.FindRule("GLOBAL_LOAD_B32");
  if (!Expect(r1 != nullptr && r1->target_opcode == "GLOBAL_LOAD_DWORD",
              "GLOBAL_LOAD_B32 → GLOBAL_LOAD_DWORD"))
    return false;

  // GLOBAL_STORE_B32 → GLOBAL_STORE_DWORD
  const auto* r2 = table.FindRule("GLOBAL_STORE_B32");
  if (!Expect(r2 != nullptr && r2->target_opcode == "GLOBAL_STORE_DWORD",
              "GLOBAL_STORE_B32 → GLOBAL_STORE_DWORD"))
    return false;

  // GLOBAL_LOAD_B64 → GLOBAL_LOAD_DWORDX2
  const auto* r3 = table.FindRule("GLOBAL_LOAD_B64");
  if (!Expect(r3 != nullptr && r3->target_opcode == "GLOBAL_LOAD_DWORDX2",
              "GLOBAL_LOAD_B64 → GLOBAL_LOAD_DWORDX2"))
    return false;

  // GLOBAL_STORE_B128 → GLOBAL_STORE_DWORDX4
  const auto* r4 = table.FindRule("GLOBAL_STORE_B128");
  if (!Expect(r4 != nullptr && r4->target_opcode == "GLOBAL_STORE_DWORDX4",
              "GLOBAL_STORE_B128 → GLOBAL_STORE_DWORDX4"))
    return false;

  // Sub-dword renames
  const auto* r5 = table.FindRule("GLOBAL_LOAD_U8");
  if (!Expect(r5 != nullptr && r5->target_opcode == "GLOBAL_LOAD_UBYTE",
              "GLOBAL_LOAD_U8 → GLOBAL_LOAD_UBYTE"))
    return false;

  const auto* r6 = table.FindRule("GLOBAL_STORE_B8");
  if (!Expect(r6 != nullptr && r6->target_opcode == "GLOBAL_STORE_BYTE",
              "GLOBAL_STORE_B8 → GLOBAL_STORE_BYTE"))
    return false;

  return true;
}

bool TestMemoryIdentityOpcodes() {
  TranslationRuleTable table;
  table.BuildForDirection(
      static_cast<std::uint8_t>(SourceArchitecture::kGfx1201),
      static_cast<std::uint8_t>(TargetArchitecture::kGfx950));

  // DWORD-named memory ops should be identity (already in gfx950 naming).
  if (!Expect(table.Classify("S_LOAD_DWORD") == TranslationTier::kIdentity,
              "S_LOAD_DWORD should be identity"))
    return false;
  if (!Expect(table.Classify("S_LOAD_DWORDX4") == TranslationTier::kIdentity,
              "S_LOAD_DWORDX4 should be identity"))
    return false;
  if (!Expect(table.Classify("GLOBAL_LOAD_DWORD") == TranslationTier::kIdentity,
              "GLOBAL_LOAD_DWORD should be identity"))
    return false;
  if (!Expect(table.Classify("GLOBAL_STORE_DWORD") == TranslationTier::kIdentity,
              "GLOBAL_STORE_DWORD should be identity"))
    return false;
  if (!Expect(table.Classify("GLOBAL_LOAD_UBYTE") == TranslationTier::kIdentity,
              "GLOBAL_LOAD_UBYTE should be identity"))
    return false;

  return true;
}

// --- Scheduling hint strip tests ---

bool TestSchedulingStripRules() {
  TranslationRuleTable table;
  table.BuildForDirection(
      static_cast<std::uint8_t>(SourceArchitecture::kGfx1201),
      static_cast<std::uint8_t>(TargetArchitecture::kGfx950));

  // All gfx12-specific scheduling hints should translate to S_NOP.
  struct TestCase {
    std::string_view opcode;
    std::string_view description;
  };
  TestCase cases[] = {
      {"S_CLAUSE", "S_CLAUSE → S_NOP"},
      {"S_DELAY_ALU", "S_DELAY_ALU → S_NOP"},
      {"S_WAIT_KMCNT", "S_WAIT_KMCNT → S_NOP"},
      {"S_WAIT_ALU", "S_WAIT_ALU → S_NOP"},
      {"S_WAIT_LOADCNT", "S_WAIT_LOADCNT → S_NOP"},
      {"S_CODE_END", "S_CODE_END → S_NOP"},
  };

  for (const auto& tc : cases) {
    const auto* rule = table.FindRule(tc.opcode);
    if (!Expect(rule != nullptr,
                (std::string(tc.description) + " rule should exist").c_str()))
      return false;
    if (!Expect(rule->target_opcode == "S_NOP",
                (std::string(tc.description) + " target should be S_NOP")
                    .c_str()))
      return false;
    if (!Expect(rule->tier == TranslationTier::kRename,
                (std::string(tc.description) + " tier should be kRename")
                    .c_str()))
      return false;
  }

  return true;
}

bool TestSchedulingStripTranslation() {
  CrossArchTranslator translator(DefaultConfig());

  // A program with scheduling hints should translate to all-NOP substitutions.
  std::vector<DecodedInstruction> program;
  program.push_back(DecodedInstruction::Nullary("S_CLAUSE"));

  {
    DecodedInstruction sload;
    sload.opcode = "S_LOAD_B128";
    sload.operand_count = 2;
    sload.operands[0] = {OperandKind::kSgpr, 0, 0};
    sload.operands[1] = {OperandKind::kSgpr, 2, 0};
    program.push_back(sload);
  }

  program.push_back(DecodedInstruction::Nullary("S_WAIT_KMCNT"));
  program.push_back(DecodedInstruction::Nullary("S_ENDPGM"));

  auto result = translator.Translate(program);

  if (!Expect(result.is_executable, "program with scheduling hints should be executable"))
    return false;

  // S_CLAUSE → S_NOP, S_LOAD_B128 → S_LOAD_DWORDX4, S_WAIT_KMCNT → S_NOP,
  // S_ENDPGM → S_ENDPGM.  The peephole pass may remove some NOPs (e.g.,
  // NOP-before-ENDPGM pattern), so check the output is <= 4 instructions.
  if (!Expect(result.translated_program.size() <= 4,
              "should produce at most 4 output instructions"))
    return false;

  // The output should contain S_LOAD_DWORDX4 and S_ENDPGM.
  bool found_sload = false;
  bool found_endpgm = false;
  for (const auto& instr : result.translated_program) {
    if (instr.opcode == "S_LOAD_DWORDX4") found_sload = true;
    if (instr.opcode == "S_ENDPGM") found_endpgm = true;
  }
  if (!Expect(found_sload, "output should contain S_LOAD_DWORDX4"))
    return false;
  if (!Expect(found_endpgm, "output should contain S_ENDPGM"))
    return false;

  return true;
}

// --- Vector ALU identity tests ---

bool TestVectorAddAluIdentity() {
  TranslationRuleTable table;
  table.BuildForDirection(
      static_cast<std::uint8_t>(SourceArchitecture::kGfx1201),
      static_cast<std::uint8_t>(TargetArchitecture::kGfx950));

  // V_MAD_CO_U64_U32 - 64-bit multiply-add with carry
  if (!Expect(table.Classify("V_MAD_CO_U64_U32") == TranslationTier::kIdentity,
              "V_MAD_CO_U64_U32 should be identity"))
    return false;

  // V_ADD_CO_CI_U32 - carry-in add
  if (!Expect(table.Classify("V_ADD_CO_CI_U32") == TranslationTier::kIdentity,
              "V_ADD_CO_CI_U32 should be identity"))
    return false;

  // V_LSHLREV_B64 - 64-bit left shift
  if (!Expect(table.Classify("V_LSHLREV_B64") == TranslationTier::kIdentity,
              "V_LSHLREV_B64 should be identity"))
    return false;

  return true;
}

bool TestVcmpxIdentity() {
  TranslationRuleTable table;
  table.BuildForDirection(
      static_cast<std::uint8_t>(SourceArchitecture::kGfx1201),
      static_cast<std::uint8_t>(TargetArchitecture::kGfx950));

  // V_CMPX_GT_I32 - compare-and-write-exec (used in vector_add for bounds
  // check)
  if (!Expect(table.Classify("V_CMPX_GT_I32") == TranslationTier::kIdentity,
              "V_CMPX_GT_I32 should be identity"))
    return false;

  // V_CMPX opcodes should be flagged as EXEC-manipulating.
  const auto* rule = table.FindRule("V_CMPX_GT_I32");
  if (!Expect(rule != nullptr && rule->is_exec_manipulating,
              "V_CMPX_GT_I32 should be exec-manipulating"))
    return false;

  // Other V_CMPX variants should also be identity.
  if (!Expect(table.Classify("V_CMPX_LT_F32") == TranslationTier::kIdentity,
              "V_CMPX_LT_F32 should be identity"))
    return false;
  if (!Expect(table.Classify("V_CMPX_EQ_U32") == TranslationTier::kIdentity,
              "V_CMPX_EQ_U32 should be identity"))
    return false;
  if (!Expect(table.Classify("V_CMPX_GE_I32") == TranslationTier::kIdentity,
              "V_CMPX_GE_I32 should be identity"))
    return false;

  return true;
}

// --- vector_add opcode coverage test ---

bool TestVectorAddOpcodeCoverage() {
  // Verify that all 20 opcodes from the real vector_add binary are
  // translatable (identity, rename, or scheduling-strip).
  TranslationRuleTable table;
  table.BuildForDirection(
      static_cast<std::uint8_t>(SourceArchitecture::kGfx1201),
      static_cast<std::uint8_t>(TargetArchitecture::kGfx950));

  struct OpcodeExpectation {
    std::string_view opcode;
    TranslationTier expected_tier;
    const char* description;
  };

  OpcodeExpectation expectations[] = {
      // Scheduling hints → rename to S_NOP
      {"S_CLAUSE", TranslationTier::kRename, "S_CLAUSE"},
      {"S_DELAY_ALU", TranslationTier::kRename, "S_DELAY_ALU"},
      {"S_WAIT_KMCNT", TranslationTier::kRename, "S_WAIT_KMCNT"},
      {"S_WAIT_ALU", TranslationTier::kRename, "S_WAIT_ALU"},
      {"S_WAIT_LOADCNT", TranslationTier::kRename, "S_WAIT_LOADCNT"},
      {"S_CODE_END", TranslationTier::kRename, "S_CODE_END"},

      // Scalar memory renames (B→DWORD)
      {"S_LOAD_B32", TranslationTier::kRename, "S_LOAD_B32"},
      {"S_LOAD_B64", TranslationTier::kRename, "S_LOAD_B64"},
      {"S_LOAD_B128", TranslationTier::kRename, "S_LOAD_B128"},

      // Scalar ALU identity
      {"S_AND_B32", TranslationTier::kIdentity, "S_AND_B32"},
      {"S_MOV_B32", TranslationTier::kIdentity, "S_MOV_B32"},

      // Scalar branch identity
      {"S_CBRANCH_EXECZ", TranslationTier::kIdentity, "S_CBRANCH_EXECZ"},
      {"S_ENDPGM", TranslationTier::kIdentity, "S_ENDPGM"},

      // Vector ALU
      {"V_MAD_CO_U64_U32", TranslationTier::kIdentity, "V_MAD_CO_U64_U32"},
      {"V_ASHRREV_I32", TranslationTier::kIdentity, "V_ASHRREV_I32"},
      {"V_LSHLREV_B64", TranslationTier::kIdentity, "V_LSHLREV_B64"},
      {"V_ADD_CO_U32", TranslationTier::kRename, "V_ADD_CO_U32"},
      {"V_ADD_CO_CI_U32", TranslationTier::kIdentity, "V_ADD_CO_CI_U32"},
      {"V_CMPX_GT_I32", TranslationTier::kIdentity, "V_CMPX_GT_I32"},
      {"V_ADD_F32", TranslationTier::kIdentity, "V_ADD_F32"},

      // Global memory renames (B→DWORD)
      {"GLOBAL_LOAD_B32", TranslationTier::kRename, "GLOBAL_LOAD_B32"},
      {"GLOBAL_STORE_B32", TranslationTier::kRename, "GLOBAL_STORE_B32"},
  };

  bool ok = true;
  for (const auto& e : expectations) {
    TranslationTier actual = table.Classify(e.opcode);
    if (actual != e.expected_tier) {
      std::cerr << "FAIL: " << e.description << " expected tier "
                << static_cast<int>(e.expected_tier) << " got "
                << static_cast<int>(actual) << '\n';
      ok = false;
    }
  }

  return ok;
}

// --- Rule count test ---

bool TestRuleCounts() {
  TranslationRuleTable table;
  table.BuildForDirection(
      static_cast<std::uint8_t>(SourceArchitecture::kGfx1201),
      static_cast<std::uint8_t>(TargetArchitecture::kGfx950));

  // Phase 11 adds: 41 new identity opcodes (3 ALU + 18 V_CMPX + 8 SMEM + 12
  // GLOBAL) + 28 renames (6 ALU + 8 SMEM + 8 GLOBAL + 6 sub-dword) + 6
  // scheduling strips. Total: 173 identity + 34 renames = 207 rules.
  if (!Expect(table.identity_count() == 173,
              ("identity count should be 173, got " +
               std::to_string(table.identity_count()))
                  .c_str()))
    return false;

  // 28 memory/ALU renames + 6 scheduling strips = 34 rename rules.
  if (!Expect(table.rename_count() == 34,
              ("rename count should be 34, got " +
               std::to_string(table.rename_count()))
                  .c_str()))
    return false;

  return true;
}

// --- Memory operation e2e translation test ---

bool TestMemoryLoadStoreTranslation() {
  CrossArchTranslator translator(DefaultConfig());

  // A simple program that loads and stores via global memory.
  std::vector<DecodedInstruction> program;

  // GLOBAL_LOAD_B32 vdst, vaddr, sbase
  DecodedInstruction load;
  load.opcode = "GLOBAL_LOAD_B32";
  load.operand_count = 3;
  load.operands[0] = {OperandKind::kVgpr, 0, 0};  // vdst
  load.operands[1] = {OperandKind::kVgpr, 2, 0};  // vaddr
  load.operands[2] = {OperandKind::kSgpr, 0, 0};  // sbase
  program.push_back(load);

  // V_ADD_F32 v0, v0, v1
  DecodedInstruction add;
  add.opcode = "V_ADD_F32";
  add.operand_count = 3;
  add.operands[0] = {OperandKind::kVgpr, 0, 0};
  add.operands[1] = {OperandKind::kVgpr, 0, 0};
  add.operands[2] = {OperandKind::kVgpr, 1, 0};
  program.push_back(add);

  // GLOBAL_STORE_B32 vaddr, vdata, sbase
  DecodedInstruction store;
  store.opcode = "GLOBAL_STORE_B32";
  store.operand_count = 3;
  store.operands[0] = {OperandKind::kVgpr, 2, 0};  // vaddr
  store.operands[1] = {OperandKind::kVgpr, 0, 0};  // vdata
  store.operands[2] = {OperandKind::kSgpr, 0, 0};  // sbase
  program.push_back(store);

  // S_ENDPGM
  program.push_back(DecodedInstruction::Nullary("S_ENDPGM"));

  auto result = translator.Translate(program);

  if (!Expect(result.is_executable,
              "load-add-store program should be executable"))
    return false;

  if (!Expect(result.translated_program[0].opcode == "GLOBAL_LOAD_DWORD",
              "GLOBAL_LOAD_B32 should become GLOBAL_LOAD_DWORD"))
    return false;

  if (!Expect(result.translated_program[1].opcode == "V_ADD_F32",
              "V_ADD_F32 should stay V_ADD_F32"))
    return false;

  if (!Expect(result.translated_program[2].opcode == "GLOBAL_STORE_DWORD",
              "GLOBAL_STORE_B32 should become GLOBAL_STORE_DWORD"))
    return false;

  return true;
}

bool TestExecManipulatingFlagOnVcmpx() {
  CrossArchTranslator translator(DefaultConfig());

  std::vector<DecodedInstruction> program;

  // V_CMPX_GT_I32 src0, src1
  DecodedInstruction cmpx;
  cmpx.opcode = "V_CMPX_GT_I32";
  cmpx.operand_count = 2;
  cmpx.operands[0] = {OperandKind::kSgpr, 0, 0};
  cmpx.operands[1] = {OperandKind::kVgpr, 0, 0};
  program.push_back(cmpx);

  program.push_back(DecodedInstruction::Nullary("S_ENDPGM"));

  auto result = translator.Translate(program);

  if (!Expect(result.is_executable,
              "V_CMPX program should be executable"))
    return false;

  if (!Expect(result.contains_exec_manipulating_instructions,
              "V_CMPX should flag exec-manipulating"))
    return false;

  return true;
}

}  // namespace

int main() {
  bool ok = true;
  ok = TestScalarMemoryRenames() && ok;
  ok = TestGlobalMemoryRenames() && ok;
  ok = TestMemoryIdentityOpcodes() && ok;
  ok = TestSchedulingStripRules() && ok;
  ok = TestSchedulingStripTranslation() && ok;
  ok = TestVectorAddAluIdentity() && ok;
  ok = TestVcmpxIdentity() && ok;
  ok = TestVectorAddOpcodeCoverage() && ok;
  ok = TestRuleCounts() && ok;
  ok = TestMemoryLoadStoreTranslation() && ok;
  ok = TestExecManipulatingFlagOnVcmpx() && ok;

  if (ok) {
    std::cerr << "All memory_translation tests passed.\n";
  }
  return ok ? 0 : 1;
}
