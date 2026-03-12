#include <iostream>
#include <string>
#include <vector>

#include "lib/sim/isa/common/decoded_instruction.h"
#include "lib/sim/isa/jit/cross_arch_translator.h"
#include "lib/sim/isa/jit/semantic_instruction.h"
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

// --- ClassifyOpcodeFamily tests ---

bool TestClassifyScalarOpcodes() {
  return Expect(ClassifyOpcodeFamily("S_MOV_B32") == SemanticFamily::kScalar,
                "S_MOV_B32 should be scalar") &&
         Expect(ClassifyOpcodeFamily("S_ADD_U32") == SemanticFamily::kScalar,
                "S_ADD_U32 should be scalar") &&
         Expect(ClassifyOpcodeFamily("S_AND_B64") == SemanticFamily::kScalar,
                "S_AND_B64 should be scalar") &&
         Expect(ClassifyOpcodeFamily("S_NOP") == SemanticFamily::kScalar,
                "S_NOP should be scalar");
}

bool TestClassifyVectorOpcodes() {
  return Expect(ClassifyOpcodeFamily("V_MOV_B32") == SemanticFamily::kVector,
                "V_MOV_B32 should be vector") &&
         Expect(ClassifyOpcodeFamily("V_ADD_F32") == SemanticFamily::kVector,
                "V_ADD_F32 should be vector") &&
         Expect(ClassifyOpcodeFamily("V_READFIRSTLANE_B32") ==
                    SemanticFamily::kVector,
                "V_READFIRSTLANE_B32 should be vector");
}

bool TestClassifyBranchOpcodes() {
  return Expect(ClassifyOpcodeFamily("S_BRANCH") == SemanticFamily::kBranch,
                "S_BRANCH should be branch") &&
         Expect(ClassifyOpcodeFamily("S_CBRANCH_SCC0") ==
                    SemanticFamily::kBranch,
                "S_CBRANCH_SCC0 should be branch") &&
         Expect(ClassifyOpcodeFamily("S_ENDPGM") == SemanticFamily::kBranch,
                "S_ENDPGM should be branch") &&
         Expect(ClassifyOpcodeFamily("S_CBRANCH_EXECZ") ==
                    SemanticFamily::kBranch,
                "S_CBRANCH_EXECZ should be branch");
}

bool TestClassifyMemoryOpcodes() {
  return Expect(ClassifyOpcodeFamily("DS_READ_B32") == SemanticFamily::kMemory,
                "DS_READ_B32 should be memory") &&
         Expect(ClassifyOpcodeFamily("GLOBAL_LOAD_DWORD") ==
                    SemanticFamily::kMemory,
                "GLOBAL_LOAD_DWORD should be memory") &&
         Expect(ClassifyOpcodeFamily("BUFFER_LOAD_FORMAT_X") ==
                    SemanticFamily::kMemory,
                "BUFFER_LOAD_FORMAT_X should be memory") &&
         Expect(ClassifyOpcodeFamily("FLAT_LOAD_DWORD") ==
                    SemanticFamily::kMemory,
                "FLAT_LOAD_DWORD should be memory") &&
         Expect(ClassifyOpcodeFamily("SCRATCH_LOAD_DWORD") ==
                    SemanticFamily::kMemory,
                "SCRATCH_LOAD_DWORD should be memory") &&
         Expect(ClassifyOpcodeFamily("S_LOAD_DWORD") ==
                    SemanticFamily::kMemory,
                "S_LOAD_DWORD should be memory");
}

bool TestClassifyBarrierOpcodes() {
  return Expect(ClassifyOpcodeFamily("S_BARRIER") == SemanticFamily::kBarrier,
                "S_BARRIER should be barrier") &&
         Expect(ClassifyOpcodeFamily("S_BARRIER_SIGNAL_M0") ==
                    SemanticFamily::kBarrier,
                "S_BARRIER_SIGNAL_M0 should be barrier") &&
         Expect(ClassifyOpcodeFamily("S_BARRIER_WAIT") ==
                    SemanticFamily::kBarrier,
                "S_BARRIER_WAIT should be barrier") &&
         Expect(ClassifyOpcodeFamily("S_BARRIER_LEAVE") ==
                    SemanticFamily::kBarrier,
                "S_BARRIER_LEAVE should be barrier");
}

bool TestClassifyMatrixOpcodes() {
  return Expect(ClassifyOpcodeFamily("V_WMMA_F32_16X16X16_F16") ==
                    SemanticFamily::kWmma,
                "V_WMMA should be wmma") &&
         Expect(ClassifyOpcodeFamily("V_SWMMAC_F32_16X16X32_F16") ==
                    SemanticFamily::kSwmmac,
                "V_SWMMAC should be swmmac") &&
         Expect(ClassifyOpcodeFamily("V_MFMA_F32_16X16X4_F32") ==
                    SemanticFamily::kMfma,
                "V_MFMA should be mfma");
}

bool TestClassifyAdvancedOpcodes() {
  return Expect(ClassifyOpcodeFamily("TENSOR_LOAD_DWORD") ==
                    SemanticFamily::kTensorMemory,
                "TENSOR_LOAD should be tensor_memory") &&
         Expect(ClassifyOpcodeFamily("TENSOR_STORE_DWORD") ==
                    SemanticFamily::kTensorMemory,
                "TENSOR_STORE should be tensor_memory") &&
         Expect(ClassifyOpcodeFamily("V_TRANSPOSE_B32") ==
                    SemanticFamily::kTranspose,
                "V_TRANSPOSE should be transpose") &&
         Expect(ClassifyOpcodeFamily("V_CVT_F32_FP8") ==
                    SemanticFamily::kFp8Bf8,
                "FP8 opcode should be fp8_bf8") &&
         Expect(ClassifyOpcodeFamily("V_CVT_F32_BF8") ==
                    SemanticFamily::kFp8Bf8,
                "BF8 opcode should be fp8_bf8");
}

bool TestClassifyUnknownOpcode() {
  return Expect(ClassifyOpcodeFamily("COMPLETELY_UNKNOWN_OP") ==
                    SemanticFamily::kOther,
                "unknown opcode should be other");
}

// --- SemanticFamilyName tests ---

bool TestSemanticFamilyNames() {
  return Expect(SemanticFamilyName(SemanticFamily::kScalar) == "scalar",
                "scalar name") &&
         Expect(SemanticFamilyName(SemanticFamily::kVector) == "vector",
                "vector name") &&
         Expect(SemanticFamilyName(SemanticFamily::kMemory) == "memory",
                "memory name") &&
         Expect(SemanticFamilyName(SemanticFamily::kBranch) == "branch",
                "branch name") &&
         Expect(SemanticFamilyName(SemanticFamily::kBarrier) == "barrier",
                "barrier name") &&
         Expect(SemanticFamilyName(SemanticFamily::kTensorMemory) ==
                    "tensor_memory",
                "tensor_memory name") &&
         Expect(SemanticFamilyName(SemanticFamily::kTranspose) == "transpose",
                "transpose name") &&
         Expect(SemanticFamilyName(SemanticFamily::kWmma) == "wmma",
                "wmma name") &&
         Expect(SemanticFamilyName(SemanticFamily::kSwmmac) == "swmmac",
                "swmmac name") &&
         Expect(SemanticFamilyName(SemanticFamily::kMfma) == "mfma",
                "mfma name") &&
         Expect(SemanticFamilyName(SemanticFamily::kFp8Bf8) == "fp8_bf8",
                "fp8_bf8 name") &&
         Expect(SemanticFamilyName(SemanticFamily::kScalePaired) ==
                    "scale_paired",
                "scale_paired name") &&
         Expect(SemanticFamilyName(SemanticFamily::kOther) == "other",
                "other name");
}

// --- LiftFromDecoded tests ---

bool TestLiftScalarInstruction() {
  SemanticLowering lowering;
  auto decoded = DecodedInstruction::Unary(
      "S_MOV_B32", InstructionOperand::Sgpr(0), InstructionOperand::Imm32(42));

  SemanticInstruction semantic;
  bool ok = lowering.LiftFromDecoded(
      decoded, static_cast<std::uint8_t>(SourceArchitecture::kGfx1201),
      &semantic);

  return Expect(ok, "lift should succeed") &&
         Expect(semantic.family == SemanticFamily::kScalar,
                "family should be scalar") &&
         Expect(semantic.opcode == "S_MOV_B32",
                "opcode should be preserved") &&
         Expect(semantic.source_arch ==
                    static_cast<std::uint8_t>(SourceArchitecture::kGfx1201),
                "source arch should be set") &&
         Expect(!semantic.wave_size_sensitive,
                "S_MOV_B32 should not be wave-sensitive");
}

bool TestLiftBarrierInstruction() {
  SemanticLowering lowering;
  auto decoded = DecodedInstruction::Nullary("S_BARRIER");

  SemanticInstruction semantic;
  bool ok = lowering.LiftFromDecoded(
      decoded, static_cast<std::uint8_t>(SourceArchitecture::kGfx1201),
      &semantic);

  return Expect(ok, "lift should succeed") &&
         Expect(semantic.family == SemanticFamily::kBarrier,
                "family should be barrier") &&
         Expect(semantic.barrier_kind == BarrierKind::kMonolithic,
                "S_BARRIER should be monolithic");
}

bool TestLiftSplitBarrierInstruction() {
  SemanticLowering lowering;
  auto decoded = DecodedInstruction::Nullary("S_BARRIER_SIGNAL_M0");

  SemanticInstruction semantic;
  bool ok = lowering.LiftFromDecoded(
      decoded, static_cast<std::uint8_t>(SourceArchitecture::kGfx1250),
      &semantic);

  return Expect(ok, "lift should succeed") &&
         Expect(semantic.family == SemanticFamily::kBarrier,
                "family should be barrier") &&
         Expect(semantic.barrier_kind == BarrierKind::kSplitSignal,
                "should be split signal") &&
         Expect(semantic.prerequisites.requires_split_barrier_runtime,
                "split barrier should require runtime");
}

bool TestLiftMfmaInstruction() {
  SemanticLowering lowering;
  auto decoded = DecodedInstruction::Nullary("V_MFMA_F32_16X16X4_F32");

  SemanticInstruction semantic;
  bool ok = lowering.LiftFromDecoded(
      decoded, static_cast<std::uint8_t>(SourceArchitecture::kGfx950),
      &semantic);

  return Expect(ok, "lift should succeed") &&
         Expect(semantic.family == SemanticFamily::kMfma,
                "family should be mfma") &&
         Expect(semantic.prerequisites.requires_accvgpr_support,
                "MFMA should require ACCVGPR support") &&
         Expect(semantic.fragment_layout == MatrixFragmentLayout::kAccumulator,
                "MFMA should use accumulator layout");
}

bool TestLiftTensorMemoryInstruction() {
  SemanticLowering lowering;
  auto decoded = DecodedInstruction::Nullary("TENSOR_LOAD_DWORD");

  SemanticInstruction semantic;
  bool ok = lowering.LiftFromDecoded(
      decoded, static_cast<std::uint8_t>(SourceArchitecture::kGfx1250),
      &semantic);

  return Expect(ok, "lift should succeed") &&
         Expect(semantic.family == SemanticFamily::kTensorMemory,
                "family should be tensor_memory") &&
         Expect(semantic.memory_space == MemorySpace::kTensor,
                "memory space should be tensor") &&
         Expect(semantic.tensor_role == TensorDescRole::kLoad,
                "tensor role should be load") &&
         Expect(semantic.prerequisites.requires_tensor_core,
                "tensor memory should require tensor core");
}

bool TestLiftWaveSensitiveInstruction() {
  SemanticLowering lowering;
  auto decoded = DecodedInstruction::Nullary("V_READLANE_B32");

  SemanticInstruction semantic;
  bool ok = lowering.LiftFromDecoded(
      decoded, static_cast<std::uint8_t>(SourceArchitecture::kGfx1201),
      &semantic);

  return Expect(ok, "lift should succeed") &&
         Expect(semantic.wave_size_sensitive,
                "V_READLANE_B32 should be wave-sensitive");
}

bool TestLiftMemorySpaceClassification() {
  SemanticLowering lowering;

  auto ds = DecodedInstruction::Nullary("DS_READ_B32");
  SemanticInstruction sem_ds;
  lowering.LiftFromDecoded(
      ds, static_cast<std::uint8_t>(SourceArchitecture::kGfx1201), &sem_ds);

  auto global = DecodedInstruction::Nullary("GLOBAL_LOAD_DWORD");
  SemanticInstruction sem_global;
  lowering.LiftFromDecoded(
      global, static_cast<std::uint8_t>(SourceArchitecture::kGfx1201),
      &sem_global);

  auto flat = DecodedInstruction::Nullary("FLAT_LOAD_DWORD");
  SemanticInstruction sem_flat;
  lowering.LiftFromDecoded(
      flat, static_cast<std::uint8_t>(SourceArchitecture::kGfx1201),
      &sem_flat);

  auto scratch = DecodedInstruction::Nullary("SCRATCH_LOAD_DWORD");
  SemanticInstruction sem_scratch;
  lowering.LiftFromDecoded(
      scratch, static_cast<std::uint8_t>(SourceArchitecture::kGfx1201),
      &sem_scratch);

  return Expect(sem_ds.memory_space == MemorySpace::kLds,
                "DS_ should be LDS") &&
         Expect(sem_global.memory_space == MemorySpace::kGlobal,
                "GLOBAL_ should be global") &&
         Expect(sem_flat.memory_space == MemorySpace::kFlat,
                "FLAT_ should be flat") &&
         Expect(sem_scratch.memory_space == MemorySpace::kScratch,
                "SCRATCH_ should be scratch");
}

bool TestLiftImplicitEffects() {
  SemanticLowering lowering;

  auto saveexec = DecodedInstruction::Nullary("S_AND_SAVEEXEC_B64");
  SemanticInstruction sem;
  lowering.LiftFromDecoded(
      saveexec, static_cast<std::uint8_t>(SourceArchitecture::kGfx1201), &sem);

  bool ok = Expect(sem.implicit_effects.reads_exec,
                   "SAVEEXEC should read EXEC") &&
            Expect(sem.implicit_effects.writes_exec,
                   "SAVEEXEC should write EXEC");

  auto cmp = DecodedInstruction::Nullary("S_CMP_EQ_I32");
  SemanticInstruction sem_cmp;
  lowering.LiftFromDecoded(
      cmp, static_cast<std::uint8_t>(SourceArchitecture::kGfx1201), &sem_cmp);

  ok = Expect(sem_cmp.implicit_effects.writes_scc,
              "S_CMP should write SCC") && ok;

  auto cbranch_vcc = DecodedInstruction::Nullary("S_CBRANCH_VCCZ");
  SemanticInstruction sem_br;
  lowering.LiftFromDecoded(
      cbranch_vcc, static_cast<std::uint8_t>(SourceArchitecture::kGfx1201),
      &sem_br);

  ok = Expect(sem_br.implicit_effects.reads_vcc,
              "S_CBRANCH_VCCZ should read VCC") && ok;

  return ok;
}

// --- ClassifyForLowering tests ---

bool TestClassifyForLoweringBarrier() {
  SemanticLowering lowering;

  SemanticInstruction monolithic;
  monolithic.family = SemanticFamily::kBarrier;
  monolithic.barrier_kind = BarrierKind::kMonolithic;

  SemanticInstruction split;
  split.family = SemanticFamily::kBarrier;
  split.barrier_kind = BarrierKind::kSplitSignal;
  split.prerequisites.requires_split_barrier_runtime = true;

  return Expect(lowering.ClassifyForLowering(monolithic) ==
                    TranslationStatus::kRequiresSemanticLowering,
                "monolithic barrier should require semantic lowering") &&
         Expect(lowering.ClassifyForLowering(split) ==
                    TranslationStatus::kBlockedOnRuntime,
                "split barrier should be blocked on runtime");
}

bool TestClassifyForLoweringMatrix() {
  SemanticLowering lowering;

  SemanticInstruction mfma;
  mfma.family = SemanticFamily::kMfma;
  mfma.prerequisites.requires_accvgpr_support = true;

  SemanticInstruction wmma;
  wmma.family = SemanticFamily::kWmma;
  wmma.prerequisites.requires_accvgpr_support = true;

  SemanticInstruction swmmac;
  swmmac.family = SemanticFamily::kSwmmac;

  return Expect(lowering.ClassifyForLowering(mfma) ==
                    TranslationStatus::kBlockedOnRuntime,
                "MFMA with ACCVGPR prereq should be blocked") &&
         Expect(lowering.ClassifyForLowering(wmma) ==
                    TranslationStatus::kBlockedOnRuntime,
                "WMMA with ACCVGPR prereq should be blocked") &&
         Expect(lowering.ClassifyForLowering(swmmac) ==
                    TranslationStatus::kBlockedOnRuntime,
                "SWMMAC should be blocked");
}

bool TestClassifyForLoweringCoverageOnly() {
  SemanticLowering lowering;

  SemanticInstruction transpose;
  transpose.family = SemanticFamily::kTranspose;

  SemanticInstruction fp8;
  fp8.family = SemanticFamily::kFp8Bf8;

  SemanticInstruction scale;
  scale.family = SemanticFamily::kScalePaired;

  return Expect(lowering.ClassifyForLowering(transpose) ==
                    TranslationStatus::kCoverageOnly,
                "transpose should be coverage-only") &&
         Expect(lowering.ClassifyForLowering(fp8) ==
                    TranslationStatus::kCoverageOnly,
                "fp8 should be coverage-only") &&
         Expect(lowering.ClassifyForLowering(scale) ==
                    TranslationStatus::kCoverageOnly,
                "scale-paired should be coverage-only");
}

bool TestClassifyForLoweringUnsupported() {
  SemanticLowering lowering;

  SemanticInstruction scalar;
  scalar.family = SemanticFamily::kScalar;

  SemanticInstruction other;
  other.family = SemanticFamily::kOther;

  return Expect(lowering.ClassifyForLowering(scalar) ==
                    TranslationStatus::kUnsupported,
                "scalar should be unsupported via semantic path") &&
         Expect(lowering.ClassifyForLowering(other) ==
                    TranslationStatus::kUnsupported,
                "other should be unsupported");
}

// --- LiftProgram tests ---

bool TestLiftProgram() {
  SemanticLowering lowering;
  std::vector<DecodedInstruction> program = {
      DecodedInstruction::Unary("S_MOV_B32",
                                InstructionOperand::Sgpr(0),
                                InstructionOperand::Imm32(42)),
      DecodedInstruction::Nullary("S_BARRIER"),
      DecodedInstruction::Nullary("V_MFMA_F32_16X16X4_F32"),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  SemanticProgram semantic;
  bool ok = lowering.LiftProgram(
      program, static_cast<std::uint8_t>(SourceArchitecture::kGfx950),
      &semantic);

  return Expect(ok, "lift program should succeed") &&
         Expect(semantic.size() == 4, "should have 4 semantic instructions") &&
         Expect(semantic[0].family == SemanticFamily::kScalar,
                "first should be scalar") &&
         Expect(semantic[1].family == SemanticFamily::kBarrier,
                "second should be barrier") &&
         Expect(semantic[2].family == SemanticFamily::kMfma,
                "third should be mfma") &&
         Expect(semantic[3].family == SemanticFamily::kBranch,
                "fourth (S_ENDPGM) should be branch");
}

// --- Translator semantic integration tests ---

bool TestClassifyInstructionWithSemanticFallthrough() {
  TranslationConfig config;
  config.source_arch = SourceArchitecture::kGfx1201;
  config.target_arch = TargetArchitecture::kGfx950;

  CrossArchTranslator translator(config);

  // S_MOV_B32 has a direct rule -> should return identity via fast path.
  TranslationStatus mov_status =
      translator.ClassifyInstructionWithSemantic("S_MOV_B32");

  // V_MFMA_F32_16X16X4_F32 has no rule -> should fall through to
  // semantic classification and return blocked (ACCVGPR prereq).
  TranslationStatus mfma_status =
      translator.ClassifyInstructionWithSemantic("V_MFMA_F32_16X16X4_F32");

  // V_TRANSPOSE_B32 has no rule -> coverage-only.
  TranslationStatus transpose_status =
      translator.ClassifyInstructionWithSemantic("V_TRANSPOSE_B32");

  // TENSOR_LOAD_DWORD has no rule -> blocked on runtime.
  TranslationStatus tensor_status =
      translator.ClassifyInstructionWithSemantic("TENSOR_LOAD_DWORD");

  return Expect(mov_status == TranslationStatus::kIdentity,
                "S_MOV_B32 should be identity via fast path") &&
         Expect(mfma_status == TranslationStatus::kBlockedOnRuntime,
                "V_MFMA should be blocked on runtime via semantic") &&
         Expect(transpose_status == TranslationStatus::kCoverageOnly,
                "V_TRANSPOSE should be coverage-only via semantic") &&
         Expect(tensor_status == TranslationStatus::kBlockedOnRuntime,
                "TENSOR_LOAD should be blocked on runtime via semantic");
}

bool TestComputeCoverageWithSemantic() {
  TranslationConfig config;
  config.source_arch = SourceArchitecture::kGfx1201;
  config.target_arch = TargetArchitecture::kGfx950;

  CrossArchTranslator translator(config);

  std::vector<DecodedInstruction> program = {
      DecodedInstruction::Unary("S_MOV_B32",
                                InstructionOperand::Sgpr(0),
                                InstructionOperand::Imm32(1)),
      DecodedInstruction::Nullary("V_MFMA_F32_16X16X4_F32"),
      DecodedInstruction::Nullary("V_TRANSPOSE_B32"),
      DecodedInstruction::Nullary("TENSOR_LOAD_DWORD"),
      DecodedInstruction::Nullary("COMPLETELY_UNKNOWN_OP"),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  // Old coverage (without semantic) should classify MFMA/transpose/tensor
  // as unsupported.
  CapabilitySummary old_summary = translator.ComputeCoverage(program);

  // New coverage (with semantic) should produce richer buckets.
  CapabilitySummary new_summary =
      translator.ComputeCoverageWithSemantic(program);

  bool ok = true;

  // Old: S_MOV_B32 + S_ENDPGM executable, everything else unsupported.
  ok = Expect(old_summary.executable_count == 2,
              "old: 2 executable (S_MOV_B32, S_ENDPGM)") &&
       Expect(old_summary.unsupported_count == 4,
              "old: 4 unsupported (MFMA, transpose, tensor, unknown)") &&
       Expect(old_summary.coverage_only_count == 0,
              "old: 0 coverage-only") &&
       Expect(old_summary.blocked_on_runtime_count == 0,
              "old: 0 blocked") && ok;

  // New: S_MOV_B32 + S_ENDPGM executable, MFMA+tensor blocked,
  //      transpose coverage-only, unknown unsupported.
  ok = Expect(new_summary.executable_count == 2,
              "new: 2 executable") &&
       Expect(new_summary.blocked_on_runtime_count == 2,
              "new: 2 blocked (MFMA, tensor)") &&
       Expect(new_summary.coverage_only_count == 1,
              "new: 1 coverage-only (transpose)") &&
       Expect(new_summary.unsupported_count == 1,
              "new: 1 unsupported (unknown)") && ok;

  return ok;
}

bool TestLowerToTargetNotYetImplemented() {
  SemanticLowering lowering;
  SemanticInstruction instr;
  instr.family = SemanticFamily::kBarrier;
  instr.barrier_kind = BarrierKind::kMonolithic;

  std::vector<DecodedInstruction> output;
  std::string error;
  bool ok = lowering.LowerToTarget(
      instr, static_cast<std::uint8_t>(TargetArchitecture::kGfx950),
      &output, &error);

  return Expect(!ok, "LowerToTarget should return false (not yet implemented)")
      && Expect(!error.empty(), "should provide error message")
      && Expect(output.empty(), "should produce no output");
}

}  // namespace

int main() {
  bool ok = true;
  ok = TestClassifyScalarOpcodes() && ok;
  ok = TestClassifyVectorOpcodes() && ok;
  ok = TestClassifyBranchOpcodes() && ok;
  ok = TestClassifyMemoryOpcodes() && ok;
  ok = TestClassifyBarrierOpcodes() && ok;
  ok = TestClassifyMatrixOpcodes() && ok;
  ok = TestClassifyAdvancedOpcodes() && ok;
  ok = TestClassifyUnknownOpcode() && ok;
  ok = TestSemanticFamilyNames() && ok;
  ok = TestLiftScalarInstruction() && ok;
  ok = TestLiftBarrierInstruction() && ok;
  ok = TestLiftSplitBarrierInstruction() && ok;
  ok = TestLiftMfmaInstruction() && ok;
  ok = TestLiftTensorMemoryInstruction() && ok;
  ok = TestLiftWaveSensitiveInstruction() && ok;
  ok = TestLiftMemorySpaceClassification() && ok;
  ok = TestLiftImplicitEffects() && ok;
  ok = TestClassifyForLoweringBarrier() && ok;
  ok = TestClassifyForLoweringMatrix() && ok;
  ok = TestClassifyForLoweringCoverageOnly() && ok;
  ok = TestClassifyForLoweringUnsupported() && ok;
  ok = TestLiftProgram() && ok;
  ok = TestClassifyInstructionWithSemanticFallthrough() && ok;
  ok = TestComputeCoverageWithSemantic() && ok;
  ok = TestLowerToTargetNotYetImplemented() && ok;

  if (ok) {
    std::cerr << "All semantic_lowering tests passed.\n";
  }
  return ok ? 0 : 1;
}
