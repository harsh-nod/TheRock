#include <cstdint>
#include <iostream>
#include <string_view>

#include "lib/sim/isa/common/decoded_instruction.h"
#include "lib/sim/isa/jit/hazard_model.h"

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

// --- Classification tests ---

bool TestSnopClassification() {
  const auto* model = GetHazardModel(TargetArchitecture::kGfx950);
  if (!Expect(model != nullptr, "expected gfx950 model")) return false;

  // S_NOP with imm32 operand (hazard wait).
  auto snop = DecodedInstruction::OneOperand(
      "S_NOP", InstructionOperand::Imm32(3));
  auto info = model->Classify(snop);

  return Expect(info.is_nop, "S_NOP should be nop") &&
         Expect(info.is_hazard_mitigation, "S_NOP should be hazard mitigation") &&
         Expect(info.nop_count == 3, "S_NOP nop_count should be 3") &&
         Expect(info.pipeline_stage == PipelineStage::kNone,
                "S_NOP pipeline stage should be kNone");
}

bool TestBareSnopClassification() {
  const auto* model = GetHazardModel(TargetArchitecture::kGfx950);
  if (!Expect(model != nullptr, "expected gfx950 model")) return false;

  // Bare S_NOP (0 operands) — from barrier signal lowering.
  auto snop = DecodedInstruction::Nullary("S_NOP");
  auto info = model->Classify(snop);

  return Expect(info.is_nop, "bare S_NOP should be nop") &&
         Expect(info.is_hazard_mitigation,
                "bare S_NOP should be hazard mitigation") &&
         Expect(info.nop_count == 0, "bare S_NOP nop_count should be 0");
}

bool TestVnopClassification() {
  const auto* model = GetHazardModel(TargetArchitecture::kGfx950);
  if (!Expect(model != nullptr, "expected gfx950 model")) return false;

  auto vnop = DecodedInstruction::Nullary("V_NOP");
  auto info = model->Classify(vnop);

  return Expect(info.is_nop, "V_NOP should be nop") &&
         Expect(info.is_hazard_mitigation, "V_NOP should be hazard mitigation");
}

bool TestWaitcntClassification() {
  const auto* model = GetHazardModel(TargetArchitecture::kGfx950);
  if (!Expect(model != nullptr, "expected gfx950 model")) return false;

  auto waitcnt = DecodedInstruction::Nullary("S_WAITCNT");
  auto info = model->Classify(waitcnt);

  return Expect(info.is_waitcnt, "S_WAITCNT should be waitcnt") &&
         Expect(info.is_hazard_mitigation,
                "S_WAITCNT should be hazard mitigation") &&
         Expect(!info.is_nop, "S_WAITCNT should not be nop");
}

// --- Pipeline stage tests ---

bool TestMfmaPipelineStage() {
  const auto* model = GetHazardModel(TargetArchitecture::kGfx950);
  if (!Expect(model != nullptr, "expected gfx950 model")) return false;

  auto mfma = DecodedInstruction::Nullary("V_MFMA_F32_16X16X4_F32");
  auto info = model->Classify(mfma);

  return Expect(info.pipeline_stage == PipelineStage::kMatrixAlu,
                "MFMA should be kMatrixAlu") &&
         Expect(info.result_latency == 8, "MFMA latency should be 8") &&
         Expect(!info.is_hazard_mitigation, "MFMA is not hazard mitigation");
}

bool TestVectorAluPipelineStage() {
  const auto* model = GetHazardModel(TargetArchitecture::kGfx950);
  if (!Expect(model != nullptr, "expected gfx950 model")) return false;

  auto vadd = DecodedInstruction::Nullary("V_ADD_F32");
  auto info = model->Classify(vadd);

  return Expect(info.pipeline_stage == PipelineStage::kVectorAlu,
                "V_ADD_F32 should be kVectorAlu") &&
         Expect(info.result_latency == 1, "VALU latency should be 1");
}

bool TestScalarMemPipelineStage() {
  const auto* model = GetHazardModel(TargetArchitecture::kGfx950);
  if (!Expect(model != nullptr, "expected gfx950 model")) return false;

  auto sload = DecodedInstruction::Nullary("S_LOAD_DWORD");
  auto info = model->Classify(sload);

  return Expect(info.pipeline_stage == PipelineStage::kScalarMemory,
                "S_LOAD_DWORD should be kScalarMemory") &&
         Expect(info.result_latency == 5, "SMEM latency should be 5");
}

bool TestLdsPipelineStage() {
  const auto* model = GetHazardModel(TargetArchitecture::kGfx950);
  if (!Expect(model != nullptr, "expected gfx950 model")) return false;

  auto ds = DecodedInstruction::Nullary("DS_READ_B32");
  auto info = model->Classify(ds);

  return Expect(info.pipeline_stage == PipelineStage::kLds,
                "DS_READ_B32 should be kLds") &&
         Expect(info.result_latency == 2, "LDS latency should be 2");
}

bool TestVectorMemPipelineStage() {
  const auto* model = GetHazardModel(TargetArchitecture::kGfx950);
  if (!Expect(model != nullptr, "expected gfx950 model")) return false;

  auto global = DecodedInstruction::Nullary("GLOBAL_LOAD_DWORD");
  auto info = model->Classify(global);

  return Expect(info.pipeline_stage == PipelineStage::kVectorMemory,
                "GLOBAL_LOAD_DWORD should be kVectorMemory") &&
         Expect(info.result_latency == 4, "VMEM latency should be 4");
}

bool TestBranchPipelineStage() {
  const auto* model = GetHazardModel(TargetArchitecture::kGfx950);
  if (!Expect(model != nullptr, "expected gfx950 model")) return false;

  auto br = DecodedInstruction::Nullary("S_BRANCH");
  auto info = model->Classify(br);

  return Expect(info.pipeline_stage == PipelineStage::kBranch,
                "S_BRANCH should be kBranch");
}

bool TestScalarAluPipelineStage() {
  const auto* model = GetHazardModel(TargetArchitecture::kGfx950);
  if (!Expect(model != nullptr, "expected gfx950 model")) return false;

  auto sadd = DecodedInstruction::Nullary("S_ADD_U32");
  auto info = model->Classify(sadd);

  return Expect(info.pipeline_stage == PipelineStage::kScalarAlu,
                "S_ADD_U32 should be kScalarAlu") &&
         Expect(info.result_latency == 1, "SALU latency should be 1");
}

bool TestBarrierPipelineStage() {
  const auto* model = GetHazardModel(TargetArchitecture::kGfx950);
  if (!Expect(model != nullptr, "expected gfx950 model")) return false;

  auto barrier = DecodedInstruction::Nullary("S_BARRIER");
  auto info = model->Classify(barrier);

  return Expect(info.pipeline_stage == PipelineStage::kBarrier,
                "S_BARRIER should be kBarrier");
}

bool TestAccvgprPipelineStage() {
  const auto* model = GetHazardModel(TargetArchitecture::kGfx950);
  if (!Expect(model != nullptr, "expected gfx950 model")) return false;

  auto accvgpr = DecodedInstruction::Nullary("V_ACCVGPR_WRITE_B32");
  auto info = model->Classify(accvgpr);

  return Expect(info.pipeline_stage == PipelineStage::kMatrixAlu,
                "V_ACCVGPR_WRITE_B32 should be kMatrixAlu");
}

// --- RequiredNopsBetween tests ---

bool TestRequiredNopsGfx950() {
  const auto* model = GetHazardModel(TargetArchitecture::kGfx950);
  if (!Expect(model != nullptr, "expected gfx950 model")) return false;

  bool ok = true;
  ok = Expect(model->RequiredNopsBetween(PipelineStage::kMatrixAlu,
                                          PipelineStage::kVectorAlu) == 2,
              "MFMA→VALU should need 2 NOPs") && ok;
  ok = Expect(model->RequiredNopsBetween(PipelineStage::kMatrixAlu,
                                          PipelineStage::kMatrixAlu) == 2,
              "MFMA→MFMA should need 2 NOPs") && ok;
  ok = Expect(model->RequiredNopsBetween(PipelineStage::kScalarMemory,
                                          PipelineStage::kScalarAlu) == 4,
              "SMEM→SALU should need 4 NOPs") && ok;
  ok = Expect(model->RequiredNopsBetween(PipelineStage::kVectorAlu,
                                          PipelineStage::kVectorAlu) == 0,
              "VALU→VALU should need 0 NOPs") && ok;
  ok = Expect(model->RequiredNopsBetween(PipelineStage::kScalarAlu,
                                          PipelineStage::kScalarAlu) == 0,
              "SALU→SALU should need 0 NOPs") && ok;
  return ok;
}

bool TestRequiredNopsGfx1201() {
  const auto* model = GetHazardModel(TargetArchitecture::kGfx1201);
  if (!Expect(model != nullptr, "expected gfx1201 model")) return false;

  bool ok = true;
  ok = Expect(model->RequiredNopsBetween(PipelineStage::kScalarMemory,
                                          PipelineStage::kScalarAlu) == 3,
              "SMEM→SALU should need 3 NOPs on gfx1201") && ok;
  ok = Expect(model->RequiredNopsBetween(PipelineStage::kMatrixAlu,
                                          PipelineStage::kVectorAlu) == 0,
              "MFMA→VALU should need 0 NOPs on gfx1201 (no MFMA)") && ok;
  return ok;
}

// --- Gfx1201 classification tests ---

bool TestGfx1201Classifications() {
  const auto* model = GetHazardModel(TargetArchitecture::kGfx1201);
  if (!Expect(model != nullptr, "expected gfx1201 model")) return false;

  bool ok = true;

  auto sload = DecodedInstruction::Nullary("S_LOAD_DWORD");
  auto sload_info = model->Classify(sload);
  ok = Expect(sload_info.pipeline_stage == PipelineStage::kScalarMemory,
              "S_LOAD_DWORD should be kScalarMemory on gfx1201") && ok;
  ok = Expect(sload_info.result_latency == 4,
              "SMEM latency should be 4 on gfx1201") && ok;

  auto vmem = DecodedInstruction::Nullary("GLOBAL_LOAD_DWORD");
  auto vmem_info = model->Classify(vmem);
  ok = Expect(vmem_info.result_latency == 3,
              "VMEM latency should be 3 on gfx1201") && ok;

  auto lds = DecodedInstruction::Nullary("DS_READ_B32");
  auto lds_info = model->Classify(lds);
  ok = Expect(lds_info.result_latency == 1,
              "LDS latency should be 1 on gfx1201") && ok;

  return ok;
}

// --- Factory tests ---

bool TestGetHazardModelFactory() {
  bool ok = true;

  const auto* gfx950 = GetHazardModel(TargetArchitecture::kGfx950);
  ok = Expect(gfx950 != nullptr, "gfx950 model should exist") && ok;
  ok = Expect(gfx950->ArchitectureName() == "gfx950",
              "gfx950 arch name") && ok;

  const auto* gfx1201 = GetHazardModel(TargetArchitecture::kGfx1201);
  ok = Expect(gfx1201 != nullptr, "gfx1201 model should exist") && ok;
  ok = Expect(gfx1201->ArchitectureName() == "gfx1201",
              "gfx1201 arch name") && ok;

  const auto* gfx1250 = GetHazardModel(TargetArchitecture::kGfx1250);
  ok = Expect(gfx1250 == nullptr, "gfx1250 model should be nullptr") && ok;

  return ok;
}

// --- PipelineStageName tests ---

bool TestPipelineStageName() {
  bool ok = true;
  ok = Expect(PipelineStageName(PipelineStage::kNone) == "none",
              "kNone name") && ok;
  ok = Expect(PipelineStageName(PipelineStage::kScalarAlu) == "scalar_alu",
              "kScalarAlu name") && ok;
  ok = Expect(PipelineStageName(PipelineStage::kVectorAlu) == "vector_alu",
              "kVectorAlu name") && ok;
  ok = Expect(PipelineStageName(PipelineStage::kMatrixAlu) == "matrix_alu",
              "kMatrixAlu name") && ok;
  ok = Expect(PipelineStageName(PipelineStage::kLds) == "lds",
              "kLds name") && ok;
  ok = Expect(PipelineStageName(PipelineStage::kBranch) == "branch",
              "kBranch name") && ok;
  ok = Expect(PipelineStageName(PipelineStage::kBarrier) == "barrier",
              "kBarrier name") && ok;
  return ok;
}

// --- IsHazardMitigation tests ---

bool TestIsHazardMitigation() {
  const auto* model = GetHazardModel(TargetArchitecture::kGfx950);
  if (!Expect(model != nullptr, "expected gfx950 model")) return false;

  bool ok = true;
  ok = Expect(model->IsHazardMitigation("S_NOP"), "S_NOP is mitigation") && ok;
  ok = Expect(model->IsHazardMitigation("V_NOP"), "V_NOP is mitigation") && ok;
  ok = Expect(model->IsHazardMitigation("S_WAITCNT"),
              "S_WAITCNT is mitigation") && ok;
  ok = Expect(!model->IsHazardMitigation("V_ADD_F32"),
              "V_ADD_F32 is not mitigation") && ok;
  ok = Expect(!model->IsHazardMitigation("S_ADD_U32"),
              "S_ADD_U32 is not mitigation") && ok;
  return ok;
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

  run(TestSnopClassification, "TestSnopClassification");
  run(TestBareSnopClassification, "TestBareSnopClassification");
  run(TestVnopClassification, "TestVnopClassification");
  run(TestWaitcntClassification, "TestWaitcntClassification");
  run(TestMfmaPipelineStage, "TestMfmaPipelineStage");
  run(TestVectorAluPipelineStage, "TestVectorAluPipelineStage");
  run(TestScalarMemPipelineStage, "TestScalarMemPipelineStage");
  run(TestLdsPipelineStage, "TestLdsPipelineStage");
  run(TestVectorMemPipelineStage, "TestVectorMemPipelineStage");
  run(TestBranchPipelineStage, "TestBranchPipelineStage");
  run(TestScalarAluPipelineStage, "TestScalarAluPipelineStage");
  run(TestBarrierPipelineStage, "TestBarrierPipelineStage");
  run(TestAccvgprPipelineStage, "TestAccvgprPipelineStage");
  run(TestRequiredNopsGfx950, "TestRequiredNopsGfx950");
  run(TestRequiredNopsGfx1201, "TestRequiredNopsGfx1201");
  run(TestGfx1201Classifications, "TestGfx1201Classifications");
  run(TestGetHazardModelFactory, "TestGetHazardModelFactory");
  run(TestPipelineStageName, "TestPipelineStageName");
  run(TestIsHazardMitigation, "TestIsHazardMitigation");

  if (ok) {
    std::cerr << "All hazard_model tests passed.\n";
  }
  return ok ? 0 : 1;
}
