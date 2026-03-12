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
         Expect(!semantic.prerequisites.requires_split_barrier_runtime,
                "split barrier should no longer require runtime prereq") &&
         Expect(semantic.lowering_kind ==
                    LoweringKind::kBarrierSplitToMonolithic,
                "split barrier should have split-to-monolithic lowering");
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
         Expect(!semantic.prerequisites.requires_accvgpr_support,
                "MFMA should no longer require ACCVGPR prereq") &&
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
  split.lowering_kind = LoweringKind::kBarrierSplitToMonolithic;

  return Expect(lowering.ClassifyForLowering(monolithic) ==
                    TranslationStatus::kRequiresSemanticLowering,
                "monolithic barrier should require semantic lowering") &&
         Expect(lowering.ClassifyForLowering(split) ==
                    TranslationStatus::kRequiresSemanticLowering,
                "split barrier should require semantic lowering");
}

bool TestClassifyForLoweringMatrix() {
  SemanticLowering lowering;

  // MFMA is now directly lowerable (ACCVGPR support available).
  SemanticInstruction mfma;
  mfma.family = SemanticFamily::kMfma;

  // WMMA with a tile match is lowerable.
  SemanticInstruction wmma_match;
  wmma_match.family = SemanticFamily::kWmma;
  wmma_match.lowering_kind = LoweringKind::kWmmaToMfma;

  // WMMA without a tile match is coverage-only.
  SemanticInstruction wmma_no_match;
  wmma_no_match.family = SemanticFamily::kWmma;

  // SWMMAC remains coverage-only.
  SemanticInstruction swmmac;
  swmmac.family = SemanticFamily::kSwmmac;

  return Expect(lowering.ClassifyForLowering(mfma) ==
                    TranslationStatus::kRequiresSemanticLowering,
                "MFMA should require semantic lowering") &&
         Expect(lowering.ClassifyForLowering(wmma_match) ==
                    TranslationStatus::kRequiresSemanticLowering,
                "WMMA with tile match should require semantic lowering") &&
         Expect(lowering.ClassifyForLowering(wmma_no_match) ==
                    TranslationStatus::kCoverageOnly,
                "WMMA without tile match should be coverage-only") &&
         Expect(lowering.ClassifyForLowering(swmmac) ==
                    TranslationStatus::kCoverageOnly,
                "SWMMAC should be coverage-only");
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
  // semantic classification and return requires semantic lowering.
  TranslationStatus mfma_status =
      translator.ClassifyInstructionWithSemantic("V_MFMA_F32_16X16X4_F32");

  // V_TRANSPOSE_B32 has no rule -> requires semantic lowering (Phase 5).
  TranslationStatus transpose_status =
      translator.ClassifyInstructionWithSemantic("V_TRANSPOSE_B32");

  // TENSOR_LOAD_DWORD has no rule -> blocked on runtime.
  TranslationStatus tensor_status =
      translator.ClassifyInstructionWithSemantic("TENSOR_LOAD_DWORD");

  return Expect(mov_status == TranslationStatus::kIdentity,
                "S_MOV_B32 should be identity via fast path") &&
         Expect(mfma_status == TranslationStatus::kRequiresSemanticLowering,
                "V_MFMA should require semantic lowering") &&
         Expect(transpose_status ==
                    TranslationStatus::kRequiresSemanticLowering,
                "V_TRANSPOSE should require semantic lowering") &&
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

  // New: S_MOV_B32 + S_ENDPGM executable, tensor blocked,
  //      MFMA + transpose coverage-only, unknown unsupported.
  ok = Expect(new_summary.executable_count == 2,
              "new: 2 executable") &&
       Expect(new_summary.blocked_on_runtime_count == 1,
              "new: 1 blocked (tensor)") &&
       Expect(new_summary.coverage_only_count == 2,
              "new: 2 coverage-only (MFMA, transpose)") &&
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

// --- Phase 7: WMMA -> MFMA lowering tests ---

bool TestLiftWmmaInstruction() {
  SemanticLowering lowering;
  auto decoded = DecodedInstruction::Nullary("V_WMMA_F32_16X16X16_F16");

  SemanticInstruction semantic;
  bool ok = lowering.LiftFromDecoded(
      decoded, static_cast<std::uint8_t>(SourceArchitecture::kGfx1201),
      &semantic);

  return Expect(ok, "lift WMMA should succeed") &&
         Expect(semantic.family == SemanticFamily::kWmma,
                "family should be wmma") &&
         Expect(semantic.fragment_layout == MatrixFragmentLayout::kAccumulator,
                "WMMA should use accumulator layout") &&
         Expect(semantic.matrix_m == 16, "M should be 16") &&
         Expect(semantic.matrix_n == 16, "N should be 16") &&
         Expect(semantic.matrix_k == 16, "K should be 16") &&
         Expect(semantic.input_element_type == ElementType::kF16,
                "input should be F16") &&
         Expect(semantic.output_element_type == ElementType::kF32,
                "output should be F32") &&
         Expect(semantic.lowering_kind == LoweringKind::kWmmaToMfma,
                "lowering kind should be WmmaToMfma") &&
         Expect(semantic.lowered_opcode == "V_MFMA_F32_16X16X16_F16",
                "lowered opcode should be MFMA equivalent") &&
         Expect(!semantic.prerequisites.requires_accvgpr_support,
                "WMMA should not have ACCVGPR prereq");
}

bool TestLiftWmmaNoMatchInstruction() {
  SemanticLowering lowering;
  // V_WMMA_I32_16X16X16_IU8 has no direct MFMA tile match.
  auto decoded = DecodedInstruction::Nullary("V_WMMA_I32_16X16X16_IU8");

  SemanticInstruction semantic;
  bool ok = lowering.LiftFromDecoded(
      decoded, static_cast<std::uint8_t>(SourceArchitecture::kGfx1201),
      &semantic);

  return Expect(ok, "lift WMMA should succeed") &&
         Expect(semantic.family == SemanticFamily::kWmma,
                "family should be wmma") &&
         Expect(semantic.lowering_kind == LoweringKind::kNone,
                "no MFMA match, lowering kind should be none") &&
         Expect(semantic.lowered_opcode.empty(),
                "no lowered opcode for unmatched WMMA");
}

bool TestClassifyForLoweringWmmaWithMatch() {
  SemanticLowering lowering;

  SemanticInstruction wmma;
  wmma.family = SemanticFamily::kWmma;
  wmma.lowering_kind = LoweringKind::kWmmaToMfma;

  return Expect(lowering.ClassifyForLowering(wmma) ==
                    TranslationStatus::kRequiresSemanticLowering,
                "WMMA with tile match should require semantic lowering");
}

bool TestClassifyForLoweringWmmaNoMatch() {
  SemanticLowering lowering;

  SemanticInstruction wmma;
  wmma.family = SemanticFamily::kWmma;

  return Expect(lowering.ClassifyForLowering(wmma) ==
                    TranslationStatus::kCoverageOnly,
                "WMMA without tile match should be coverage-only");
}

bool TestLowerWmmaToMfma() {
  SemanticLowering lowering;

  SemanticInstruction instr;
  instr.opcode = "V_WMMA_F32_16X16X16_F16";
  instr.family = SemanticFamily::kWmma;
  instr.lowering_kind = LoweringKind::kWmmaToMfma;
  instr.lowered_opcode = "V_MFMA_F32_16X16X16_F16";

  std::vector<DecodedInstruction> output;
  std::string error;
  bool ok = lowering.LowerToTarget(
      instr, static_cast<std::uint8_t>(TargetArchitecture::kGfx950),
      &output, &error);

  return Expect(ok, "WMMA->MFMA lowering should succeed") &&
         Expect(output.size() == 3,
                "should produce 3 instructions (WRITE + MFMA + READ)") &&
         Expect(output[0].opcode == "V_ACCVGPR_WRITE",
                "first should be V_ACCVGPR_WRITE") &&
         Expect(output[1].opcode == "V_MFMA_F32_16X16X16_F16",
                "second should be V_MFMA_F32_16X16X16_F16") &&
         Expect(output[2].opcode == "V_ACCVGPR_READ",
                "third should be V_ACCVGPR_READ");
}

bool TestLowerWmmaToMfmaBf16() {
  SemanticLowering lowering;

  SemanticInstruction instr;
  instr.opcode = "V_WMMA_F32_16X16X16_BF16";
  instr.family = SemanticFamily::kWmma;
  instr.lowering_kind = LoweringKind::kWmmaToMfma;
  instr.lowered_opcode = "V_MFMA_F32_16X16X16_BF16";

  std::vector<DecodedInstruction> output;
  std::string error;
  bool ok = lowering.LowerToTarget(
      instr, static_cast<std::uint8_t>(TargetArchitecture::kGfx950),
      &output, &error);

  return Expect(ok, "WMMA BF16->MFMA lowering should succeed") &&
         Expect(output.size() == 3, "should produce 3 instructions") &&
         Expect(output[1].opcode == "V_MFMA_F32_16X16X16_BF16",
                "MFMA opcode should be BF16 variant");
}

// --- Phase 7: Barrier lowering tests ---

bool TestLowerSplitBarrierSignal() {
  SemanticLowering lowering;

  SemanticInstruction instr;
  instr.opcode = "S_BARRIER_SIGNAL_M0";
  instr.family = SemanticFamily::kBarrier;
  instr.barrier_kind = BarrierKind::kSplitSignal;
  instr.lowering_kind = LoweringKind::kBarrierSplitToMonolithic;

  std::vector<DecodedInstruction> output;
  std::string error;
  bool ok = lowering.LowerToTarget(
      instr, static_cast<std::uint8_t>(TargetArchitecture::kGfx950),
      &output, &error);

  return Expect(ok, "split signal lowering should succeed") &&
         Expect(output.size() == 1, "should produce 1 instruction") &&
         Expect(output[0].opcode == "S_NOP",
                "split signal should become S_NOP");
}

bool TestLowerSplitBarrierWait() {
  SemanticLowering lowering;

  SemanticInstruction instr;
  instr.opcode = "S_BARRIER_WAIT";
  instr.family = SemanticFamily::kBarrier;
  instr.barrier_kind = BarrierKind::kSplitWait;
  instr.lowering_kind = LoweringKind::kBarrierSplitToMonolithic;

  std::vector<DecodedInstruction> output;
  std::string error;
  bool ok = lowering.LowerToTarget(
      instr, static_cast<std::uint8_t>(TargetArchitecture::kGfx950),
      &output, &error);

  return Expect(ok, "split wait lowering should succeed") &&
         Expect(output.size() == 1, "should produce 1 instruction") &&
         Expect(output[0].opcode == "S_BARRIER",
                "split wait should become S_BARRIER");
}

bool TestLowerSplitBarrierLeave() {
  SemanticLowering lowering;

  SemanticInstruction instr;
  instr.opcode = "S_BARRIER_LEAVE";
  instr.family = SemanticFamily::kBarrier;
  instr.barrier_kind = BarrierKind::kSplitWait;
  instr.lowering_kind = LoweringKind::kBarrierSplitToMonolithic;

  std::vector<DecodedInstruction> output;
  std::string error;
  bool ok = lowering.LowerToTarget(
      instr, static_cast<std::uint8_t>(TargetArchitecture::kGfx950),
      &output, &error);

  return Expect(ok, "split leave lowering should succeed") &&
         Expect(output.size() == 1, "should produce 1 instruction") &&
         Expect(output[0].opcode == "S_BARRIER",
                "split leave should become S_BARRIER");
}

bool TestClassifyForLoweringSplitBarrier() {
  SemanticLowering lowering;

  SemanticInstruction signal;
  signal.family = SemanticFamily::kBarrier;
  signal.barrier_kind = BarrierKind::kSplitSignal;
  signal.lowering_kind = LoweringKind::kBarrierSplitToMonolithic;

  SemanticInstruction wait;
  wait.family = SemanticFamily::kBarrier;
  wait.barrier_kind = BarrierKind::kSplitWait;
  wait.lowering_kind = LoweringKind::kBarrierSplitToMonolithic;

  return Expect(lowering.ClassifyForLowering(signal) ==
                    TranslationStatus::kRequiresSemanticLowering,
                "split signal should require semantic lowering") &&
         Expect(lowering.ClassifyForLowering(wait) ==
                    TranslationStatus::kRequiresSemanticLowering,
                "split wait should require semantic lowering");
}

// --- Phase 7: End-to-end WMMA lift + classify + lower ---

bool TestWmmaEndToEndLiftClassifyLower() {
  SemanticLowering lowering;
  auto decoded = DecodedInstruction::Nullary("V_WMMA_F32_16X16X16_F16");

  // 1. Lift
  SemanticInstruction semantic;
  bool ok = lowering.LiftFromDecoded(
      decoded, static_cast<std::uint8_t>(SourceArchitecture::kGfx1201),
      &semantic);
  if (!Expect(ok, "e2e: lift should succeed")) return false;

  // 2. Classify
  TranslationStatus status = lowering.ClassifyForLowering(semantic);
  if (!Expect(status == TranslationStatus::kRequiresSemanticLowering,
              "e2e: should require semantic lowering"))
    return false;

  // 3. Lower
  std::vector<DecodedInstruction> output;
  std::string error;
  ok = lowering.LowerToTarget(
      semantic, static_cast<std::uint8_t>(TargetArchitecture::kGfx950),
      &output, &error);

  return Expect(ok, "e2e: lowering should succeed") &&
         Expect(output.size() == 3, "e2e: should produce 3 instructions") &&
         Expect(output[0].opcode == "V_ACCVGPR_WRITE",
                "e2e: first should be V_ACCVGPR_WRITE") &&
         Expect(output[1].opcode == "V_MFMA_F32_16X16X16_F16",
                "e2e: second should be MFMA") &&
         Expect(output[2].opcode == "V_ACCVGPR_READ",
                "e2e: third should be V_ACCVGPR_READ");
}

bool TestBarrierEndToEndLiftClassifyLower() {
  SemanticLowering lowering;
  auto decoded_signal = DecodedInstruction::Nullary("S_BARRIER_SIGNAL_M0");
  auto decoded_wait = DecodedInstruction::Nullary("S_BARRIER_WAIT");

  // Lift signal
  SemanticInstruction sem_signal;
  bool ok = lowering.LiftFromDecoded(
      decoded_signal,
      static_cast<std::uint8_t>(SourceArchitecture::kGfx1250), &sem_signal);
  if (!Expect(ok, "e2e barrier: lift signal should succeed")) return false;

  // Lift wait
  SemanticInstruction sem_wait;
  ok = lowering.LiftFromDecoded(
      decoded_wait,
      static_cast<std::uint8_t>(SourceArchitecture::kGfx1250), &sem_wait);
  if (!Expect(ok, "e2e barrier: lift wait should succeed")) return false;

  // Classify
  TranslationStatus signal_status = lowering.ClassifyForLowering(sem_signal);
  TranslationStatus wait_status = lowering.ClassifyForLowering(sem_wait);

  if (!Expect(signal_status == TranslationStatus::kRequiresSemanticLowering,
              "e2e barrier: signal should require lowering"))
    return false;
  if (!Expect(wait_status == TranslationStatus::kRequiresSemanticLowering,
              "e2e barrier: wait should require lowering"))
    return false;

  // Lower signal
  std::vector<DecodedInstruction> signal_out;
  std::string error;
  ok = lowering.LowerToTarget(
      sem_signal, static_cast<std::uint8_t>(TargetArchitecture::kGfx950),
      &signal_out, &error);
  if (!Expect(ok, "e2e barrier: signal lowering should succeed")) return false;

  // Lower wait
  std::vector<DecodedInstruction> wait_out;
  ok = lowering.LowerToTarget(
      sem_wait, static_cast<std::uint8_t>(TargetArchitecture::kGfx950),
      &wait_out, &error);
  if (!Expect(ok, "e2e barrier: wait lowering should succeed")) return false;

  return Expect(signal_out.size() == 1, "e2e barrier: signal -> 1 instr") &&
         Expect(signal_out[0].opcode == "S_NOP",
                "e2e barrier: signal -> S_NOP") &&
         Expect(wait_out.size() == 1, "e2e barrier: wait -> 1 instr") &&
         Expect(wait_out[0].opcode == "S_BARRIER",
                "e2e barrier: wait -> S_BARRIER");
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
  // Phase 7: WMMA -> MFMA lowering tests.
  ok = TestLiftWmmaInstruction() && ok;
  ok = TestLiftWmmaNoMatchInstruction() && ok;
  ok = TestClassifyForLoweringWmmaWithMatch() && ok;
  ok = TestClassifyForLoweringWmmaNoMatch() && ok;
  ok = TestLowerWmmaToMfma() && ok;
  ok = TestLowerWmmaToMfmaBf16() && ok;
  // Phase 7: Barrier lowering tests.
  ok = TestLowerSplitBarrierSignal() && ok;
  ok = TestLowerSplitBarrierWait() && ok;
  ok = TestLowerSplitBarrierLeave() && ok;
  ok = TestClassifyForLoweringSplitBarrier() && ok;
  // Phase 7: End-to-end tests.
  ok = TestWmmaEndToEndLiftClassifyLower() && ok;
  ok = TestBarrierEndToEndLiftClassifyLower() && ok;

  if (ok) {
    std::cerr << "All semantic_lowering tests passed.\n";
  }
  return ok ? 0 : 1;
}
