#include <cstdint>
#include <iostream>
#include <vector>

#include "lib/sim/isa/common/decoded_instruction.h"
#include "lib/sim/isa/jit/peephole_pass.h"
#include "lib/sim/isa/jit/translation_diagnostics.h"

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

// --- Pattern 1: Consecutive bare S_NOP removal ---

bool TestConsecutiveBareNopRemoval() {
  std::vector<DecodedInstruction> program = {
      DecodedInstruction::Nullary("V_ADD_F32"),
      DecodedInstruction::Nullary("S_NOP"),
      DecodedInstruction::Nullary("S_NOP"),
      DecodedInstruction::Nullary("S_NOP"),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  auto diagnostics = MakeDiagnostics(program);
  PeepholeStats stats;

  bool ok = PeepholePass::Apply(&program, &diagnostics, &stats);

  // Three consecutive bare NOPs → first kept, second and third removed.
  return Expect(ok, "peephole should succeed") &&
         Expect(program.size() == 3,
                "should have 3 instructions (2 NOPs removed)") &&
         Expect(program[0].opcode == "V_ADD_F32", "first is V_ADD_F32") &&
         Expect(program[1].opcode == "S_NOP", "second is bare S_NOP") &&
         Expect(program[2].opcode == "S_ENDPGM", "third is S_ENDPGM") &&
         Expect(stats.redundant_nops_removed == 2,
                "should remove 2 redundant NOPs");
}

// --- Pattern 1: S_NOP with operands is NOT removed ---

bool TestNopWithOperandsNotRemoved() {
  std::vector<DecodedInstruction> program = {
      DecodedInstruction::OneOperand("S_NOP",
                                     InstructionOperand::Imm32(2)),
      DecodedInstruction::OneOperand("S_NOP",
                                     InstructionOperand::Imm32(1)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  auto diagnostics = MakeDiagnostics(program);
  PeepholeStats stats;

  bool ok = PeepholePass::Apply(&program, &diagnostics, &stats);

  // S_NOP with operands are not bare NOPs — not affected by pattern 1.
  return Expect(ok, "peephole should succeed") &&
         Expect(program.size() == 3,
                "should have 3 instructions (none removed)") &&
         Expect(stats.redundant_nops_removed == 0,
                "should remove 0 redundant NOPs");
}

// --- Pattern 2: Dead ACCVGPR transfer pair ---

bool TestDeadAccvgprTransferPair() {
  std::vector<DecodedInstruction> program = {
      DecodedInstruction::Nullary("V_ACCVGPR_WRITE_B32"),
      DecodedInstruction::Nullary("V_ACCVGPR_READ_B32"),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  auto diagnostics = MakeDiagnostics(program);
  PeepholeStats stats;

  bool ok = PeepholePass::Apply(&program, &diagnostics, &stats);

  return Expect(ok, "peephole should succeed") &&
         Expect(program.size() == 1, "should have 1 instruction") &&
         Expect(program[0].opcode == "S_ENDPGM", "only S_ENDPGM remains") &&
         Expect(stats.redundant_accvgpr_transfers_removed == 2,
                "should remove 2 ACCVGPR transfers");
}

// --- Pattern 2: MFMA between WRITE and READ prevents removal ---

bool TestAccvgprTransferWithMfma() {
  std::vector<DecodedInstruction> program = {
      DecodedInstruction::Nullary("V_ACCVGPR_WRITE_B32"),
      DecodedInstruction::Nullary("V_MFMA_F32_16X16X4_F32"),
      DecodedInstruction::Nullary("V_ACCVGPR_READ_B32"),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  auto diagnostics = MakeDiagnostics(program);
  PeepholeStats stats;

  bool ok = PeepholePass::Apply(&program, &diagnostics, &stats);

  return Expect(ok, "peephole should succeed") &&
         Expect(program.size() == 4,
                "should have 4 instructions (MFMA blocks removal)") &&
         Expect(stats.redundant_accvgpr_transfers_removed == 0,
                "should remove 0 ACCVGPR transfers");
}

// --- Pattern 3: NOP before ENDPGM ---

bool TestNopBeforeEndpgm() {
  std::vector<DecodedInstruction> program = {
      DecodedInstruction::Nullary("V_ADD_F32"),
      DecodedInstruction::Nullary("S_NOP"),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  auto diagnostics = MakeDiagnostics(program);
  PeepholeStats stats;

  bool ok = PeepholePass::Apply(&program, &diagnostics, &stats);

  return Expect(ok, "peephole should succeed") &&
         Expect(program.size() == 2,
                "should have 2 instructions (NOP before ENDPGM removed)") &&
         Expect(program[0].opcode == "V_ADD_F32", "first is V_ADD_F32") &&
         Expect(program[1].opcode == "S_ENDPGM", "second is S_ENDPGM") &&
         Expect(stats.redundant_nops_removed == 1,
                "should remove 1 redundant NOP");
}

// --- V_NOP before ENDPGM is also removed ---

bool TestVnopBeforeEndpgm() {
  std::vector<DecodedInstruction> program = {
      DecodedInstruction::Nullary("V_ADD_F32"),
      DecodedInstruction::Nullary("V_NOP"),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  auto diagnostics = MakeDiagnostics(program);
  PeepholeStats stats;

  bool ok = PeepholePass::Apply(&program, &diagnostics, &stats);

  return Expect(ok, "peephole should succeed") &&
         Expect(program.size() == 2,
                "should have 2 instructions (V_NOP before ENDPGM removed)") &&
         Expect(stats.redundant_nops_removed == 1,
                "should remove 1 redundant NOP");
}

// --- Diagnostic indices are valid after peephole ---

bool TestDiagnosticIndicesAfterPeephole() {
  std::vector<DecodedInstruction> program = {
      DecodedInstruction::Nullary("V_ADD_F32"),
      DecodedInstruction::Nullary("S_NOP"),
      DecodedInstruction::Nullary("S_NOP"),
      DecodedInstruction::Nullary("V_MUL_F32"),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  auto diagnostics = MakeDiagnostics(program);

  bool ok = PeepholePass::Apply(&program, &diagnostics);
  if (!Expect(ok, "peephole should succeed")) return false;

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

// --- Empty program is handled ---

bool TestEmptyProgram() {
  std::vector<DecodedInstruction> program;
  std::vector<InstructionDiagnostic> diagnostics;
  PeepholeStats stats;

  bool ok = PeepholePass::Apply(&program, &diagnostics, &stats);

  return Expect(ok, "empty program should succeed") &&
         Expect(program.empty(), "program should still be empty") &&
         Expect(stats.total_instructions_removed == 0,
                "should remove 0 instructions");
}

// --- No patterns match ---

bool TestNoPatterns() {
  std::vector<DecodedInstruction> program = {
      DecodedInstruction::Nullary("S_ADD_U32"),
      DecodedInstruction::Nullary("V_ADD_F32"),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  auto diagnostics = MakeDiagnostics(program);
  PeepholeStats stats;

  bool ok = PeepholePass::Apply(&program, &diagnostics, &stats);

  return Expect(ok, "peephole should succeed") &&
         Expect(program.size() == 3, "should have 3 instructions unchanged") &&
         Expect(stats.total_instructions_removed == 0,
                "should remove 0 instructions");
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

  run(TestConsecutiveBareNopRemoval, "TestConsecutiveBareNopRemoval");
  run(TestNopWithOperandsNotRemoved, "TestNopWithOperandsNotRemoved");
  run(TestDeadAccvgprTransferPair, "TestDeadAccvgprTransferPair");
  run(TestAccvgprTransferWithMfma, "TestAccvgprTransferWithMfma");
  run(TestNopBeforeEndpgm, "TestNopBeforeEndpgm");
  run(TestVnopBeforeEndpgm, "TestVnopBeforeEndpgm");
  run(TestDiagnosticIndicesAfterPeephole,
      "TestDiagnosticIndicesAfterPeephole");
  run(TestEmptyProgram, "TestEmptyProgram");
  run(TestNoPatterns, "TestNoPatterns");

  if (ok) {
    std::cerr << "All peephole_pass tests passed.\n";
  }
  return ok ? 0 : 1;
}
