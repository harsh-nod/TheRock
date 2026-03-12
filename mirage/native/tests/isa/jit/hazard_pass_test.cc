#include <cstdint>
#include <iostream>
#include <vector>

#include "lib/sim/isa/common/decoded_instruction.h"
#include "lib/sim/isa/jit/cross_arch_translator.h"
#include "lib/sim/isa/jit/hazard_model.h"
#include "lib/sim/isa/jit/hazard_pass.h"

namespace {

using namespace mirage::sim::isa;
using namespace mirage::sim::isa::jit;

bool Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    return false;
  }
  return true;
}

// Helper: build diagnostics as if each source instruction produced exactly
// one output instruction (1:1 mapping).
std::vector<InstructionDiagnostic> MakeDiagnostics(
    const std::vector<DecodedInstruction>& program) {
  std::vector<InstructionDiagnostic> diagnostics;
  for (std::size_t i = 0; i < program.size(); ++i) {
    InstructionDiagnostic d;
    d.source_index = i;
    d.source_opcode = program[i].opcode;
    d.output_begin_index = i;
    d.output_instruction_count = 1;
    d.status = TranslationStatus::kIdentity;
    diagnostics.push_back(d);
  }
  return diagnostics;
}

// --- Passthrough test ---

bool TestPassthroughPolicy() {
  std::vector<DecodedInstruction> program = {
      DecodedInstruction::OneOperand("S_NOP",
                                     InstructionOperand::Imm32(2)),
      DecodedInstruction::Nullary("V_ADD_F32"),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  auto diagnostics = MakeDiagnostics(program);

  const auto* source_model = GetHazardModel(TargetArchitecture::kGfx950);
  HazardPassStats stats;

  bool ok = HazardPass::Apply(HazardPolicy::kPassthrough, source_model,
                               nullptr, &program, &diagnostics, &stats);

  return Expect(ok, "passthrough should succeed") &&
         Expect(program.size() == 3, "passthrough should not change program size") &&
         Expect(stats.nops_stripped == 0, "passthrough should strip 0 nops") &&
         Expect(stats.instructions_before == 3, "instructions_before == 3") &&
         Expect(stats.instructions_after == 3, "instructions_after == 3");
}

// --- StripSource test ---

bool TestStripSourceNops() {
  std::vector<DecodedInstruction> program = {
      DecodedInstruction::OneOperand("S_NOP",
                                     InstructionOperand::Imm32(2)),
      DecodedInstruction::Nullary("V_ADD_F32"),
      DecodedInstruction::OneOperand("S_NOP",
                                     InstructionOperand::Imm32(1)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  auto diagnostics = MakeDiagnostics(program);

  const auto* source_model = GetHazardModel(TargetArchitecture::kGfx950);
  HazardPassStats stats;

  bool ok = HazardPass::Apply(HazardPolicy::kStripSource, source_model,
                               nullptr, &program, &diagnostics, &stats);

  return Expect(ok, "strip should succeed") &&
         Expect(program.size() == 2, "should have 2 instructions after strip") &&
         Expect(program[0].opcode == "V_ADD_F32",
                "first should be V_ADD_F32") &&
         Expect(program[1].opcode == "S_ENDPGM",
                "second should be S_ENDPGM") &&
         Expect(stats.nops_stripped == 2, "should strip 2 nops");
}

// --- StripSource preserves bare S_NOP ---

bool TestStripSourcePreservesBareNop() {
  std::vector<DecodedInstruction> program = {
      DecodedInstruction::Nullary("S_NOP"),  // bare NOP — preserve
      DecodedInstruction::OneOperand("S_NOP",
                                     InstructionOperand::Imm32(1)),  // strip
      DecodedInstruction::Nullary("V_ADD_F32"),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  auto diagnostics = MakeDiagnostics(program);

  const auto* source_model = GetHazardModel(TargetArchitecture::kGfx950);
  HazardPassStats stats;

  bool ok = HazardPass::Apply(HazardPolicy::kStripSource, source_model,
                               nullptr, &program, &diagnostics, &stats);

  return Expect(ok, "strip should succeed") &&
         Expect(program.size() == 3,
                "should have 3 instructions (bare NOP preserved)") &&
         Expect(program[0].opcode == "S_NOP", "bare S_NOP preserved") &&
         Expect(program[0].operand_count == 0, "bare S_NOP has 0 operands") &&
         Expect(stats.nops_stripped == 1, "should strip 1 nop (not bare)");
}

// --- StripSource strips S_WAITCNT ---

bool TestStripSourceWaitcnt() {
  std::vector<DecodedInstruction> program = {
      DecodedInstruction::Nullary("S_WAITCNT"),
      DecodedInstruction::Nullary("V_ADD_F32"),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  auto diagnostics = MakeDiagnostics(program);

  const auto* source_model = GetHazardModel(TargetArchitecture::kGfx950);
  HazardPassStats stats;

  bool ok = HazardPass::Apply(HazardPolicy::kStripSource, source_model,
                               nullptr, &program, &diagnostics, &stats);

  return Expect(ok, "strip should succeed") &&
         Expect(program.size() == 2,
                "should have 2 instructions after stripping waitcnt") &&
         Expect(stats.waitcnts_stripped == 1, "should strip 1 waitcnt");
}

// --- Retarget test ---

bool TestRetargetInsertsNops() {
  // After stripping, the program has S_LOAD_DWORD → S_ADD_U32.
  // On gfx1201, SMEM→SALU requires 3 NOPs (1 S_NOP instruction with count).
  std::vector<DecodedInstruction> program = {
      DecodedInstruction::Nullary("S_LOAD_DWORD"),
      DecodedInstruction::Nullary("S_ADD_U32"),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  auto diagnostics = MakeDiagnostics(program);

  const auto* source_model = GetHazardModel(TargetArchitecture::kGfx950);
  const auto* target_model = GetHazardModel(TargetArchitecture::kGfx1201);
  HazardPassStats stats;

  bool ok = HazardPass::Apply(HazardPolicy::kRetarget, source_model,
                               target_model, &program, &diagnostics, &stats);

  return Expect(ok, "retarget should succeed") &&
         Expect(program.size() == 4,
                "should have 4 instructions (1 NOP inserted)") &&
         Expect(program[0].opcode == "S_LOAD_DWORD", "first is S_LOAD") &&
         Expect(program[1].opcode == "S_NOP", "second is inserted S_NOP") &&
         Expect(program[2].opcode == "S_ADD_U32", "third is S_ADD") &&
         Expect(program[3].opcode == "S_ENDPGM", "fourth is S_ENDPGM") &&
         Expect(stats.nops_inserted == 1, "should insert 1 NOP");
}

// --- Diagnostic index integrity after strip ---

bool TestDiagnosticIndicesAfterStrip() {
  std::vector<DecodedInstruction> program = {
      DecodedInstruction::Nullary("V_ADD_F32"),
      DecodedInstruction::OneOperand("S_NOP",
                                     InstructionOperand::Imm32(2)),
      DecodedInstruction::Nullary("V_MUL_F32"),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  auto diagnostics = MakeDiagnostics(program);

  const auto* source_model = GetHazardModel(TargetArchitecture::kGfx950);
  HazardPassStats stats;

  bool ok = HazardPass::Apply(HazardPolicy::kStripSource, source_model,
                               nullptr, &program, &diagnostics, &stats);

  if (!Expect(ok, "strip should succeed")) return false;
  if (!Expect(program.size() == 3, "should have 3 instructions")) return false;

  // Verify diagnostic indices are contiguous.
  std::size_t expected_begin = 0;
  for (const auto& d : diagnostics) {
    if (!Expect(d.output_begin_index == expected_begin,
                "diagnostic output_begin_index mismatch")) {
      return false;
    }
    expected_begin += d.output_instruction_count;
  }

  return Expect(expected_begin == program.size(),
                "diagnostic indices should sum to program size");
}

// --- Null model returns false ---

bool TestNullModelFails() {
  std::vector<DecodedInstruction> program;
  std::vector<InstructionDiagnostic> diagnostics;

  bool ok = HazardPass::Apply(HazardPolicy::kStripSource, nullptr, nullptr,
                               &program, &diagnostics);

  return Expect(!ok, "strip with null model should fail");
}

}  // namespace

int main() {
  bool ok = true;

  auto run = [&ok](bool (*test_fn)(), const char* name) {
    std::cerr << name << "... ";
    bool passed = test_fn();
    ok = passed && ok;
    std::cerr << (passed ? "PASS" : "FAIL") << '\n';
  };

  run(TestPassthroughPolicy, "TestPassthroughPolicy");
  run(TestStripSourceNops, "TestStripSourceNops");
  run(TestStripSourcePreservesBareNop, "TestStripSourcePreservesBareNop");
  run(TestStripSourceWaitcnt, "TestStripSourceWaitcnt");
  run(TestRetargetInsertsNops, "TestRetargetInsertsNops");
  run(TestDiagnosticIndicesAfterStrip, "TestDiagnosticIndicesAfterStrip");
  run(TestNullModelFails, "TestNullModelFails");

  if (ok) {
    std::cerr << "All hazard_pass tests passed.\n";
  }
  return ok ? 0 : 1;
}
