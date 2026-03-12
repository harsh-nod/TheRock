#ifndef MIRAGE_SIM_ISA_JIT_HAZARD_MODEL_H_
#define MIRAGE_SIM_ISA_JIT_HAZARD_MODEL_H_

#include <cstdint>
#include <string_view>

#include "lib/sim/isa/common/decoded_instruction.h"
#include "lib/sim/isa/jit/translation_diagnostics.h"

namespace mirage::sim::isa::jit {

enum class PipelineStage : std::uint8_t {
  kNone,
  kScalarAlu,
  kVectorAlu,
  kVectorMemory,
  kScalarMemory,
  kLds,
  kExport,
  kBranch,
  kBarrier,
  kMatrixAlu,
};

struct InstructionHazardInfo {
  PipelineStage pipeline_stage = PipelineStage::kNone;
  bool is_hazard_mitigation = false;
  bool is_nop = false;
  bool is_waitcnt = false;
  std::uint8_t nop_count = 0;
  std::uint8_t result_latency = 0;
};

class HazardModel {
 public:
  virtual ~HazardModel() = default;

  virtual InstructionHazardInfo Classify(
      const DecodedInstruction& instruction) const = 0;

  virtual bool IsHazardMitigation(std::string_view opcode) const = 0;

  virtual std::uint8_t RequiredNopsBetween(
      PipelineStage producer_stage,
      PipelineStage consumer_stage) const = 0;

  virtual std::string_view ArchitectureName() const = 0;
};

const HazardModel* GetHazardModel(TargetArchitecture arch);

class Gfx950HazardModel final : public HazardModel {
 public:
  InstructionHazardInfo Classify(
      const DecodedInstruction& instruction) const override;
  bool IsHazardMitigation(std::string_view opcode) const override;
  std::uint8_t RequiredNopsBetween(
      PipelineStage producer_stage,
      PipelineStage consumer_stage) const override;
  std::string_view ArchitectureName() const override;
};

class Gfx1201HazardModel final : public HazardModel {
 public:
  InstructionHazardInfo Classify(
      const DecodedInstruction& instruction) const override;
  bool IsHazardMitigation(std::string_view opcode) const override;
  std::uint8_t RequiredNopsBetween(
      PipelineStage producer_stage,
      PipelineStage consumer_stage) const override;
  std::string_view ArchitectureName() const override;
};

std::string_view PipelineStageName(PipelineStage stage);

}  // namespace mirage::sim::isa::jit

#endif  // MIRAGE_SIM_ISA_JIT_HAZARD_MODEL_H_
