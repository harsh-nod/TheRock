#include "lib/sim/isa/jit/hazard_model.h"

#include <string_view>

namespace mirage::sim::isa::jit {

namespace {

bool StartsWith(std::string_view str, std::string_view prefix) {
  return str.size() >= prefix.size() && str.substr(0, prefix.size()) == prefix;
}

bool IsBranchOpcode(std::string_view opcode) {
  return opcode == "S_BRANCH" || opcode == "S_CBRANCH_SCC0" ||
         opcode == "S_CBRANCH_SCC1" || opcode == "S_CBRANCH_VCCZ" ||
         opcode == "S_CBRANCH_VCCNZ" || opcode == "S_CBRANCH_EXECZ" ||
         opcode == "S_CBRANCH_EXECNZ" || opcode == "S_ENDPGM";
}

PipelineStage ClassifyPipelineStage(std::string_view opcode) {
  if (opcode == "S_NOP" || opcode == "V_NOP" || opcode == "DS_NOP" ||
      opcode == "S_WAITCNT") {
    return PipelineStage::kNone;
  }

  if (opcode == "S_BARRIER") {
    return PipelineStage::kBarrier;
  }

  if (IsBranchOpcode(opcode)) {
    return PipelineStage::kBranch;
  }

  if (StartsWith(opcode, "V_MFMA_")) {
    return PipelineStage::kMatrixAlu;
  }

  if (StartsWith(opcode, "V_ACCVGPR_")) {
    return PipelineStage::kMatrixAlu;
  }

  if (StartsWith(opcode, "DS_")) {
    return PipelineStage::kLds;
  }

  if (StartsWith(opcode, "GLOBAL_") || StartsWith(opcode, "BUFFER_") ||
      StartsWith(opcode, "FLAT_") || StartsWith(opcode, "SCRATCH_")) {
    return PipelineStage::kVectorMemory;
  }

  if (StartsWith(opcode, "S_LOAD_") || StartsWith(opcode, "S_BUFFER_LOAD_")) {
    return PipelineStage::kScalarMemory;
  }

  if (StartsWith(opcode, "V_")) {
    return PipelineStage::kVectorAlu;
  }

  if (StartsWith(opcode, "S_")) {
    return PipelineStage::kScalarAlu;
  }

  return PipelineStage::kNone;
}

}  // namespace

// --- Gfx950HazardModel ---

InstructionHazardInfo Gfx950HazardModel::Classify(
    const DecodedInstruction& instruction) const {
  InstructionHazardInfo info;

  if (instruction.opcode == "S_NOP") {
    info.is_nop = true;
    info.is_hazard_mitigation = true;
    if (instruction.operand_count > 0 &&
        instruction.operands[0].kind == OperandKind::kImm32) {
      info.nop_count =
          static_cast<std::uint8_t>(instruction.operands[0].imm32);
    }
    return info;
  }

  if (instruction.opcode == "V_NOP") {
    info.is_nop = true;
    info.is_hazard_mitigation = true;
    return info;
  }

  if (instruction.opcode == "S_WAITCNT") {
    info.is_waitcnt = true;
    info.is_hazard_mitigation = true;
    return info;
  }

  info.pipeline_stage = ClassifyPipelineStage(instruction.opcode);

  switch (info.pipeline_stage) {
    case PipelineStage::kMatrixAlu:
      info.result_latency = 8;
      break;
    case PipelineStage::kScalarMemory:
      info.result_latency = 5;
      break;
    case PipelineStage::kVectorMemory:
      info.result_latency = 4;
      break;
    case PipelineStage::kLds:
      info.result_latency = 2;
      break;
    case PipelineStage::kVectorAlu:
    case PipelineStage::kScalarAlu:
      info.result_latency = 1;
      break;
    default:
      break;
  }

  return info;
}

bool Gfx950HazardModel::IsHazardMitigation(std::string_view opcode) const {
  return opcode == "S_NOP" || opcode == "V_NOP" || opcode == "S_WAITCNT";
}

std::uint8_t Gfx950HazardModel::RequiredNopsBetween(
    PipelineStage producer_stage,
    PipelineStage consumer_stage) const {
  if (producer_stage == PipelineStage::kMatrixAlu) {
    if (consumer_stage == PipelineStage::kVectorAlu ||
        consumer_stage == PipelineStage::kMatrixAlu) {
      return 2;
    }
  }
  if (producer_stage == PipelineStage::kScalarMemory &&
      consumer_stage == PipelineStage::kScalarAlu) {
    return 4;
  }
  return 0;
}

std::string_view Gfx950HazardModel::ArchitectureName() const {
  return "gfx950";
}

// --- Gfx1201HazardModel ---

InstructionHazardInfo Gfx1201HazardModel::Classify(
    const DecodedInstruction& instruction) const {
  InstructionHazardInfo info;

  if (instruction.opcode == "S_NOP") {
    info.is_nop = true;
    info.is_hazard_mitigation = true;
    if (instruction.operand_count > 0 &&
        instruction.operands[0].kind == OperandKind::kImm32) {
      info.nop_count =
          static_cast<std::uint8_t>(instruction.operands[0].imm32);
    }
    return info;
  }

  if (instruction.opcode == "V_NOP") {
    info.is_nop = true;
    info.is_hazard_mitigation = true;
    return info;
  }

  if (instruction.opcode == "S_WAITCNT") {
    info.is_waitcnt = true;
    info.is_hazard_mitigation = true;
    return info;
  }

  info.pipeline_stage = ClassifyPipelineStage(instruction.opcode);

  // RDNA4 has shorter pipelines than CDNA3.
  switch (info.pipeline_stage) {
    case PipelineStage::kScalarMemory:
      info.result_latency = 4;
      break;
    case PipelineStage::kVectorMemory:
      info.result_latency = 3;
      break;
    case PipelineStage::kLds:
      info.result_latency = 1;
      break;
    case PipelineStage::kVectorAlu:
    case PipelineStage::kScalarAlu:
      info.result_latency = 1;
      break;
    default:
      break;
  }

  return info;
}

bool Gfx1201HazardModel::IsHazardMitigation(std::string_view opcode) const {
  return opcode == "S_NOP" || opcode == "V_NOP" || opcode == "S_WAITCNT";
}

std::uint8_t Gfx1201HazardModel::RequiredNopsBetween(
    PipelineStage producer_stage,
    PipelineStage consumer_stage) const {
  // RDNA4 has shorter pipelines; fewer explicit NOPs needed.
  if (producer_stage == PipelineStage::kScalarMemory &&
      consumer_stage == PipelineStage::kScalarAlu) {
    return 3;
  }
  return 0;
}

std::string_view Gfx1201HazardModel::ArchitectureName() const {
  return "gfx1201";
}

// --- Factory ---

const HazardModel* GetHazardModel(TargetArchitecture arch) {
  static const Gfx950HazardModel kGfx950Model;
  static const Gfx1201HazardModel kGfx1201Model;

  switch (arch) {
    case TargetArchitecture::kGfx950:
      return &kGfx950Model;
    case TargetArchitecture::kGfx1201:
      return &kGfx1201Model;
    case TargetArchitecture::kGfx1250:
      return nullptr;
  }
  return nullptr;
}

std::string_view PipelineStageName(PipelineStage stage) {
  switch (stage) {
    case PipelineStage::kNone:
      return "none";
    case PipelineStage::kScalarAlu:
      return "scalar_alu";
    case PipelineStage::kVectorAlu:
      return "vector_alu";
    case PipelineStage::kVectorMemory:
      return "vector_memory";
    case PipelineStage::kScalarMemory:
      return "scalar_memory";
    case PipelineStage::kLds:
      return "lds";
    case PipelineStage::kExport:
      return "export";
    case PipelineStage::kBranch:
      return "branch";
    case PipelineStage::kBarrier:
      return "barrier";
    case PipelineStage::kMatrixAlu:
      return "matrix_alu";
  }
  return "unknown";
}

}  // namespace mirage::sim::isa::jit
