#include "lib/sim/isa/jit/wave_adapter.h"

#include <algorithm>

#include "lib/sim/isa/jit/translation_rules.h"

namespace mirage::sim::isa::jit {

WaveAdapter::WaveAdapter(std::uint16_t source_vcc_sgpr,
                         std::uint16_t target_vcc_sgpr,
                         WavePolicy wave_policy)
    : source_vcc_sgpr_(source_vcc_sgpr),
      target_vcc_sgpr_(target_vcc_sgpr),
      wave_policy_(wave_policy) {}

bool WaveAdapter::RemapVccOperands(DecodedInstruction* instruction) const {
  if (source_vcc_sgpr_ == target_vcc_sgpr_) {
    return true;
  }
  bool remapped = false;
  for (std::uint8_t i = 0; i < instruction->operand_count; ++i) {
    auto& operand = instruction->operands[i];
    if (operand.kind == OperandKind::kSgpr) {
      if (operand.index == source_vcc_sgpr_) {
        operand.index = target_vcc_sgpr_;
        remapped = true;
      } else if (operand.index == source_vcc_sgpr_ + 1) {
        operand.index = target_vcc_sgpr_ + 1;
        remapped = true;
      }
    }
  }
  return true;
}

bool WaveAdapter::IsWaveSensitive(std::string_view opcode) const {
  auto wave_sensitive = GetWaveSensitiveOpcodes();
  return std::find(wave_sensitive.begin(), wave_sensitive.end(), opcode) !=
         wave_sensitive.end();
}

bool WaveAdapter::ShouldReject(std::string_view opcode) const {
  if (!IsWaveSensitive(opcode)) {
    return false;
  }
  return wave_policy_ == WavePolicy::kRejectWaveSensitive;
}

}  // namespace mirage::sim::isa::jit
