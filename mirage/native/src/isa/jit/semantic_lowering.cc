#include "lib/sim/isa/jit/semantic_lowering.h"

#include <algorithm>
#include <array>

#include "lib/sim/isa/jit/translation_rules.h"

namespace mirage::sim::isa::jit {

namespace {

// Barrier opcodes that use split signal/wait semantics.
bool IsSplitBarrierSignal(std::string_view opcode) {
  return opcode == "S_BARRIER_SIGNAL_M0" ||
         opcode == "S_BARRIER_SIGNAL_ISFIRST_M0" ||
         opcode == "S_BARRIER_SIGNAL_IMM" ||
         opcode == "S_BARRIER_SIGNAL_ISFIRST_IMM";
}

bool IsSplitBarrierWait(std::string_view opcode) {
  return opcode == "S_BARRIER_WAIT" || opcode == "S_BARRIER_LEAVE";
}

bool IsMonolithicBarrier(std::string_view opcode) {
  return opcode == "S_BARRIER";
}

// Transpose / permute opcodes (gfx1250-era).
bool IsTransposeOpcode(std::string_view opcode) {
  return opcode == "V_TRANSPOSE_B32" ||
         opcode == "V_TRANSPOSE_B16_B32" ||
         opcode == "V_TRANSPOSE_U4_B32" ||
         opcode == "V_TRANSPOSE_B32x4";
}

// WMMA family (gfx1201/gfx1250).
bool IsWmmaOpcode(std::string_view opcode) {
  return opcode.size() >= 5 && opcode.substr(0, 5) == "V_WMM" &&
         opcode.find("WMMA") != std::string_view::npos;
}

// SWMMAC family (gfx1250).
bool IsSwmmacOpcode(std::string_view opcode) {
  return opcode.find("SWMMAC") != std::string_view::npos;
}

// MFMA family (gfx950).
bool IsMfmaOpcode(std::string_view opcode) {
  return opcode.size() >= 7 && opcode.substr(0, 7) == "V_MFMA_";
}

// FP8/BF8/F4 composite opcodes.
bool IsFp8Bf8Opcode(std::string_view opcode) {
  return opcode.find("_FP8") != std::string_view::npos ||
         opcode.find("_BF8") != std::string_view::npos ||
         opcode.find("_F8F6F4") != std::string_view::npos ||
         opcode.find("_E4M3") != std::string_view::npos ||
         opcode.find("_E5M2") != std::string_view::npos;
}

// Scale-paired opcodes (gfx1250).
bool IsScalePairedOpcode(std::string_view opcode) {
  return opcode.find("_SCALE") != std::string_view::npos;
}

// --- VOP3P packed BF16 lowering tables ---

struct PackedBf16LoweringEntry {
  std::string_view source;
  std::string_view target;
};

// VOP3P packed BF16 instructions that can be approximately lowered to
// their F16 equivalents.  The lowering is approximate because BF16 has
// a different exponent/mantissa split than F16 (8/7 vs 5/10), so the
// precision characteristics differ.
constexpr PackedBf16LoweringEntry kPackedBf16Lowerings[] = {
    {"V_PK_ADD_BF16", "V_PK_ADD_F16"},
    {"V_PK_MUL_BF16", "V_PK_MUL_F16"},
    {"V_PK_FMA_BF16", "V_PK_FMA_F16"},
    {"V_PK_MAX_NUM_BF16", "V_PK_MAX_F16"},
    {"V_PK_MIN_NUM_BF16", "V_PK_MIN_F16"},
};

std::string_view FindPackedBf16Target(std::string_view opcode) {
  for (const auto& entry : kPackedBf16Lowerings) {
    if (entry.source == opcode) {
      return entry.target;
    }
  }
  return {};
}

bool IsPackedBf16Opcode(std::string_view opcode) {
  return opcode.find("V_PK_") != std::string_view::npos &&
         opcode.find("_BF16") != std::string_view::npos;
}

// --- FP8/BF8 conversion lowering classification ---

// Scalar FP8/BF8 conversions that produce a single F32/F16 result.
bool IsFp8ScalarConversion(std::string_view opcode) {
  return opcode == "V_CVT_F32_FP8" || opcode == "V_CVT_F32_BF8" ||
         opcode == "V_CVT_F16_FP8" || opcode == "V_CVT_F16_BF8";
}

// Packed FP8/BF8 conversions that produce packed F16 results.
bool IsFp8PackedConversion(std::string_view opcode) {
  return opcode == "V_CVT_PK_F16_FP8" || opcode == "V_CVT_PK_F16_BF8" ||
         opcode == "V_CVT_PK_FP8_F16" || opcode == "V_CVT_PK_BF8_F16" ||
         opcode == "V_CVT_PK_FP8_F32" ||
         opcode == "V_CVT_SR_FP8_F32" || opcode == "V_CVT_SR_BF8_F16" ||
         opcode == "V_CVT_SR_FP8_F16" ||
         opcode == "V_CVT_SR_FP8_F32_gfx12";
}

// Scaled conversions (V_CVT_SCALEF32_* and V_CVT_SCALE_*).
bool IsScaledConversion(std::string_view opcode) {
  return opcode.find("V_CVT_SCALEF32_") != std::string_view::npos ||
         opcode.find("V_CVT_SCALE_") != std::string_view::npos;
}

// Classify the LoweringKind for an instruction.
LoweringKind ClassifyLoweringKind(std::string_view opcode,
                                  SemanticFamily family) {
  if (family == SemanticFamily::kVector && IsPackedBf16Opcode(opcode)) {
    if (!FindPackedBf16Target(opcode).empty()) {
      return LoweringKind::kPackedBf16ToF16;
    }
  }

  if (family == SemanticFamily::kFp8Bf8) {
    if (IsFp8ScalarConversion(opcode)) {
      return LoweringKind::kFp8ScalarConversion;
    }
    if (IsFp8PackedConversion(opcode)) {
      return LoweringKind::kFp8PackedConversion;
    }
  }

  if (family == SemanticFamily::kScalePaired) {
    if (IsScaledConversion(opcode)) {
      return LoweringKind::kScaledConversion;
    }
  }

  if (family == SemanticFamily::kTranspose) {
    return LoweringKind::kApproximateTranspose;
  }

  return LoweringKind::kNone;
}

// Tensor memory opcodes (gfx1250 TENSOR_LOAD/STORE).
bool IsTensorMemoryOpcode(std::string_view opcode) {
  return opcode.find("TENSOR_LOAD") != std::string_view::npos ||
         opcode.find("TENSOR_STORE") != std::string_view::npos;
}

// Memory opcodes that access global/flat/scratch/LDS.
MemorySpace ClassifyMemorySpace(std::string_view opcode) {
  if (opcode.size() >= 3 && opcode.substr(0, 3) == "DS_") {
    return MemorySpace::kLds;
  }
  if (opcode.find("GLOBAL_") != std::string_view::npos ||
      opcode.find("BUFFER_") != std::string_view::npos) {
    return MemorySpace::kGlobal;
  }
  if (opcode.find("FLAT_") != std::string_view::npos) {
    return MemorySpace::kFlat;
  }
  if (opcode.find("SCRATCH_") != std::string_view::npos) {
    return MemorySpace::kScratch;
  }
  return MemorySpace::kNone;
}

// Populate implicit register effects based on opcode family.
void PopulateImplicitEffects(std::string_view opcode, SemanticFamily family,
                             ImplicitRegisterEffects* effects) {
  // Branch/compare instructions read/write SCC.
  if (opcode.find("S_CMP_") != std::string_view::npos ||
      opcode.find("S_BITCMP") != std::string_view::npos) {
    effects->writes_scc = true;
  }
  // Scalar ALU with carry read/write SCC.
  if (opcode.find("S_ADD_") != std::string_view::npos ||
      opcode.find("S_SUB_") != std::string_view::npos ||
      opcode.find("S_ADDC_") != std::string_view::npos ||
      opcode.find("S_SUBB_") != std::string_view::npos) {
    effects->writes_scc = true;
  }
  // SAVEEXEC reads and writes EXEC.
  if (opcode.find("SAVEEXEC") != std::string_view::npos) {
    effects->reads_exec = true;
    effects->writes_exec = true;
  }
  // CBRANCH_EXEC* reads EXEC.
  if (opcode == "S_CBRANCH_EXECZ" || opcode == "S_CBRANCH_EXECNZ") {
    effects->reads_exec = true;
  }
  // CBRANCH_VCCZ/VCCNZ reads VCC.
  if (opcode == "S_CBRANCH_VCCZ" || opcode == "S_CBRANCH_VCCNZ") {
    effects->reads_vcc = true;
  }
  // CBRANCH_SCC reads SCC.
  if (opcode == "S_CBRANCH_SCC0" || opcode == "S_CBRANCH_SCC1") {
    effects->reads_scc = true;
  }
  // VOPC instructions write VCC.
  if (opcode.find("V_CMP_") != std::string_view::npos ||
      opcode.find("V_CMPX_") != std::string_view::npos) {
    effects->writes_vcc = true;
  }
  // V_CMPX also writes EXEC.
  if (opcode.find("V_CMPX_") != std::string_view::npos) {
    effects->writes_exec = true;
  }
  // DS opcodes that touch LDS read M0 for base address.
  if (family == SemanticFamily::kMemory &&
      opcode.size() >= 3 && opcode.substr(0, 3) == "DS_") {
    effects->reads_m0 = true;
  }
}

}  // namespace

SemanticFamily ClassifyOpcodeFamily(std::string_view opcode) {
  // Barrier family (check before generic S_ scalar).
  if (IsMonolithicBarrier(opcode) || IsSplitBarrierSignal(opcode) ||
      IsSplitBarrierWait(opcode)) {
    return SemanticFamily::kBarrier;
  }

  // Tensor memory (gfx1250).
  if (IsTensorMemoryOpcode(opcode)) {
    return SemanticFamily::kTensorMemory;
  }

  // Transpose (gfx1250).
  if (IsTransposeOpcode(opcode)) {
    return SemanticFamily::kTranspose;
  }

  // SWMMAC must be checked before WMMA (both contain "WMMA" sometimes).
  if (IsSwmmacOpcode(opcode)) {
    return SemanticFamily::kSwmmac;
  }

  // WMMA family.
  if (IsWmmaOpcode(opcode)) {
    return SemanticFamily::kWmma;
  }

  // MFMA family.
  if (IsMfmaOpcode(opcode)) {
    return SemanticFamily::kMfma;
  }

  // Scale-paired (check before FP8 since some opcodes have both).
  if (IsScalePairedOpcode(opcode)) {
    return SemanticFamily::kScalePaired;
  }

  // FP8/BF8/F4.
  if (IsFp8Bf8Opcode(opcode)) {
    return SemanticFamily::kFp8Bf8;
  }

  // Branch family.
  if (opcode == "S_BRANCH" || opcode == "S_CBRANCH_SCC0" ||
      opcode == "S_CBRANCH_SCC1" || opcode == "S_CBRANCH_VCCZ" ||
      opcode == "S_CBRANCH_VCCNZ" || opcode == "S_CBRANCH_EXECZ" ||
      opcode == "S_CBRANCH_EXECNZ" || opcode == "S_ENDPGM" ||
      opcode == "S_SETPC_B64" || opcode == "S_SWAPPC_B64" ||
      opcode == "S_CALL_B64") {
    return SemanticFamily::kBranch;
  }

  // Memory family (global, flat, scratch, buffer, LDS).
  if (opcode.size() >= 3 && opcode.substr(0, 3) == "DS_") {
    return SemanticFamily::kMemory;
  }
  if (opcode.find("GLOBAL_") != std::string_view::npos ||
      opcode.find("BUFFER_") != std::string_view::npos ||
      opcode.find("FLAT_") != std::string_view::npos ||
      opcode.find("SCRATCH_") != std::string_view::npos) {
    return SemanticFamily::kMemory;
  }
  // SMEM instructions.
  if (opcode.find("S_LOAD_") != std::string_view::npos ||
      opcode.find("S_STORE_") != std::string_view::npos ||
      opcode.find("S_BUFFER_LOAD_") != std::string_view::npos) {
    return SemanticFamily::kMemory;
  }

  // Vector ALU (V_ prefix, not already classified).
  if (opcode.size() >= 2 && opcode.substr(0, 2) == "V_") {
    return SemanticFamily::kVector;
  }

  // Scalar (S_ prefix, not already classified).
  if (opcode.size() >= 2 && opcode.substr(0, 2) == "S_") {
    return SemanticFamily::kScalar;
  }

  return SemanticFamily::kOther;
}

std::string_view SemanticFamilyName(SemanticFamily family) {
  switch (family) {
    case SemanticFamily::kScalar:
      return "scalar";
    case SemanticFamily::kVector:
      return "vector";
    case SemanticFamily::kMemory:
      return "memory";
    case SemanticFamily::kBranch:
      return "branch";
    case SemanticFamily::kBarrier:
      return "barrier";
    case SemanticFamily::kTensorMemory:
      return "tensor_memory";
    case SemanticFamily::kTranspose:
      return "transpose";
    case SemanticFamily::kWmma:
      return "wmma";
    case SemanticFamily::kSwmmac:
      return "swmmac";
    case SemanticFamily::kMfma:
      return "mfma";
    case SemanticFamily::kFp8Bf8:
      return "fp8_bf8";
    case SemanticFamily::kScalePaired:
      return "scale_paired";
    case SemanticFamily::kOther:
      return "other";
  }
  return "unknown";
}

TranslationStatus SemanticLowering::ClassifyForLowering(
    const SemanticInstruction& instruction) const {
  // If prerequisites require runtime support that is not yet available,
  // the instruction is blocked.
  if (instruction.prerequisites.any()) {
    if (instruction.prerequisites.requires_tensor_core ||
        instruction.prerequisites.requires_split_barrier_runtime) {
      return TranslationStatus::kBlockedOnRuntime;
    }
  }

  switch (instruction.family) {
    case SemanticFamily::kScalar:
    case SemanticFamily::kVector:
    case SemanticFamily::kBranch:
      // These families go through the direct-rewrite fast path, not
      // semantic lowering.  If they reach this classifier, it means the
      // rule table had no entry and they are unsupported.
      return TranslationStatus::kUnsupported;

    case SemanticFamily::kMemory:
      // Memory instructions with LDS space are blocked until LDS
      // translation is implemented.
      if (instruction.memory_space == MemorySpace::kLds) {
        return TranslationStatus::kUnsupported;
      }
      return TranslationStatus::kUnsupported;

    case SemanticFamily::kBarrier:
      // Monolithic barriers can be lowered directly.
      if (instruction.barrier_kind == BarrierKind::kMonolithic) {
        return TranslationStatus::kRequiresSemanticLowering;
      }
      // Split barriers require runtime support.
      return TranslationStatus::kBlockedOnRuntime;

    case SemanticFamily::kTensorMemory:
      return TranslationStatus::kBlockedOnRuntime;

    case SemanticFamily::kTranspose:
      // Transpose lowering is only available in approximate mode.
      if (instruction.lowering_kind == LoweringKind::kApproximateTranspose) {
        return TranslationStatus::kRequiresSemanticLowering;
      }
      return TranslationStatus::kCoverageOnly;

    case SemanticFamily::kWmma:
      // WMMA->MFMA lowering requires ACCVGPR support.
      if (instruction.prerequisites.requires_accvgpr_support) {
        return TranslationStatus::kBlockedOnRuntime;
      }
      return TranslationStatus::kRequiresSemanticLowering;

    case SemanticFamily::kSwmmac:
      return TranslationStatus::kBlockedOnRuntime;

    case SemanticFamily::kMfma:
      if (instruction.prerequisites.requires_accvgpr_support) {
        return TranslationStatus::kBlockedOnRuntime;
      }
      return TranslationStatus::kRequiresSemanticLowering;

    case SemanticFamily::kFp8Bf8:
      if (instruction.lowering_kind == LoweringKind::kFp8ScalarConversion ||
          instruction.lowering_kind == LoweringKind::kFp8PackedConversion) {
        return TranslationStatus::kRequiresSemanticLowering;
      }
      return TranslationStatus::kCoverageOnly;

    case SemanticFamily::kScalePaired:
      if (instruction.lowering_kind == LoweringKind::kScaledConversion) {
        return TranslationStatus::kRequiresSemanticLowering;
      }
      return TranslationStatus::kCoverageOnly;

    case SemanticFamily::kOther:
      return TranslationStatus::kUnsupported;
  }
  return TranslationStatus::kUnsupported;
}

bool SemanticLowering::LowerToTarget(
    const SemanticInstruction& instruction,
    std::uint8_t /*target_arch*/,
    std::vector<DecodedInstruction>* output,
    std::string* error_message) const {
  switch (instruction.lowering_kind) {
    case LoweringKind::kPackedBf16ToF16: {
      // Approximate lowering: rewrite V_PK_*_BF16 -> V_PK_*_F16.
      // Operands are preserved; only the opcode changes.
      if (instruction.lowered_opcode.empty()) {
        if (error_message != nullptr) {
          *error_message = "No F16 target opcode for packed BF16 instruction.";
        }
        return false;
      }
      output->push_back(
          DecodedInstruction::Nullary(instruction.lowered_opcode));
      return true;
    }

    case LoweringKind::kFp8ScalarConversion: {
      // Scalar FP8/BF8 conversion: emit the same opcode as a stub.
      // The semantic contract is: one source register -> one destination.
      output->push_back(DecodedInstruction::Nullary(instruction.opcode));
      return true;
    }

    case LoweringKind::kFp8PackedConversion: {
      // Packed FP8/BF8 conversion: emit the same opcode as a stub.
      output->push_back(DecodedInstruction::Nullary(instruction.opcode));
      return true;
    }

    case LoweringKind::kScaledConversion: {
      // Scaled conversion: emit the same opcode as a stub.
      output->push_back(DecodedInstruction::Nullary(instruction.opcode));
      return true;
    }

    case LoweringKind::kApproximateTranspose: {
      // Approximate transpose lowering: emit a V_MOV_B32 as a
      // semantically-neutral placeholder.  The actual transpose effect
      // is dropped, making this an approximate lowering.
      output->push_back(DecodedInstruction::Nullary("V_MOV_B32"));
      return true;
    }

    case LoweringKind::kNone:
      break;
  }

  if (error_message != nullptr) {
    *error_message = "No semantic lowering available for this instruction.";
  }
  return false;
}

bool SemanticLowering::LiftFromDecoded(
    const DecodedInstruction& instruction,
    std::uint8_t source_arch,
    SemanticInstruction* output,
    std::string* /*error_message*/) const {
  output->opcode = instruction.opcode;
  output->source_arch = source_arch;
  output->family = ClassifyOpcodeFamily(instruction.opcode);

  // Memory space classification.
  output->memory_space = ClassifyMemorySpace(instruction.opcode);
  if (output->family == SemanticFamily::kTensorMemory) {
    output->memory_space = MemorySpace::kTensor;
  }

  // Barrier kind.
  if (output->family == SemanticFamily::kBarrier) {
    if (IsMonolithicBarrier(instruction.opcode)) {
      output->barrier_kind = BarrierKind::kMonolithic;
    } else if (IsSplitBarrierSignal(instruction.opcode)) {
      output->barrier_kind = BarrierKind::kSplitSignal;
    } else if (IsSplitBarrierWait(instruction.opcode)) {
      output->barrier_kind = BarrierKind::kSplitWait;
    }
  }

  // Wave-size sensitivity via the existing wave-sensitive opcode list.
  auto wave_sensitive = GetWaveSensitiveOpcodes();
  output->wave_size_sensitive =
      std::find(wave_sensitive.begin(), wave_sensitive.end(),
                instruction.opcode) != wave_sensitive.end();

  // Matrix instruction prerequisites.
  if (output->family == SemanticFamily::kMfma) {
    output->prerequisites.requires_accvgpr_support = true;
    output->fragment_layout = MatrixFragmentLayout::kAccumulator;
  }
  if (output->family == SemanticFamily::kWmma) {
    output->prerequisites.requires_accvgpr_support = true;
  }
  if (output->family == SemanticFamily::kSwmmac) {
    output->prerequisites.requires_accvgpr_support = true;
  }

  // Tensor memory prerequisites.
  if (output->family == SemanticFamily::kTensorMemory) {
    output->prerequisites.requires_tensor_core = true;
    if (instruction.opcode.find("LOAD") != std::string_view::npos) {
      output->tensor_role = TensorDescRole::kLoad;
    } else if (instruction.opcode.find("STORE") != std::string_view::npos) {
      output->tensor_role = TensorDescRole::kStore;
    }
  }

  // Split barrier prerequisites.
  if (output->barrier_kind == BarrierKind::kSplitSignal ||
      output->barrier_kind == BarrierKind::kSplitWait) {
    output->prerequisites.requires_split_barrier_runtime = true;
  }

  // FP8/BF8 prerequisites.
  if (output->family == SemanticFamily::kFp8Bf8) {
    output->prerequisites.requires_fp8_hardware = true;
  }

  // Scale-paired prerequisites.
  if (output->family == SemanticFamily::kScalePaired) {
    output->prerequisites.requires_scale_hardware = true;
  }

  // Lowering kind classification.
  output->lowering_kind =
      ClassifyLoweringKind(instruction.opcode, output->family);

  // For VOP3P BF16, populate the lowered opcode.
  if (output->lowering_kind == LoweringKind::kPackedBf16ToF16) {
    output->lowered_opcode = FindPackedBf16Target(instruction.opcode);
    output->input_element_type = ElementType::kBF16;
    output->output_element_type = ElementType::kF16;
    output->is_approximate = true;
  }

  // For FP8/BF8 conversions, populate element types.
  if (output->lowering_kind == LoweringKind::kFp8ScalarConversion ||
      output->lowering_kind == LoweringKind::kFp8PackedConversion) {
    if (instruction.opcode.find("_FP8") != std::string_view::npos) {
      output->input_element_type = ElementType::kFP8_E4M3;
    } else if (instruction.opcode.find("_BF8") != std::string_view::npos) {
      output->input_element_type = ElementType::kBF8;
    }
    if (instruction.opcode.find("_F32") != std::string_view::npos) {
      output->output_element_type = ElementType::kF32;
    } else if (instruction.opcode.find("_F16") != std::string_view::npos) {
      output->output_element_type = ElementType::kF16;
    }
  }

  // For transpose, mark as approximate.
  if (output->lowering_kind == LoweringKind::kApproximateTranspose) {
    output->is_approximate = true;
  }

  // Implicit register effects.
  PopulateImplicitEffects(instruction.opcode, output->family,
                          &output->implicit_effects);

  return true;
}

bool SemanticLowering::LiftProgram(
    std::span<const DecodedInstruction> program,
    std::uint8_t source_arch,
    SemanticProgram* output,
    std::string* error_message) const {
  output->clear();
  output->reserve(program.size());
  for (const auto& instr : program) {
    SemanticInstruction semantic;
    if (!LiftFromDecoded(instr, source_arch, &semantic, error_message)) {
      return false;
    }
    output->push_back(std::move(semantic));
  }
  return true;
}

}  // namespace mirage::sim::isa::jit
