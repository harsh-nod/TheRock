#ifndef MIRAGE_SIM_ISA_JIT_WAVE_ADAPTER_H_
#define MIRAGE_SIM_ISA_JIT_WAVE_ADAPTER_H_

#include <cstdint>

#include "lib/sim/isa/common/decoded_instruction.h"

namespace mirage::sim::isa::jit {

enum class WavePolicy : std::uint8_t {
  kWave32InWave64,
  kRejectWaveSensitive,
};

class WaveAdapter {
 public:
  WaveAdapter() = default;
  explicit WaveAdapter(std::uint16_t source_vcc_sgpr,
                       std::uint16_t target_vcc_sgpr,
                       WavePolicy wave_policy);

  bool RemapVccOperands(DecodedInstruction* instruction) const;

  bool IsWaveSensitive(std::string_view opcode) const;

  bool ShouldReject(std::string_view opcode) const;

  std::uint16_t source_vcc_sgpr() const { return source_vcc_sgpr_; }
  std::uint16_t target_vcc_sgpr() const { return target_vcc_sgpr_; }
  WavePolicy wave_policy() const { return wave_policy_; }

  static constexpr std::uint16_t kGfx950VccSgpr = 106;
  static constexpr std::uint16_t kGfx1201VccSgpr = 248;
  static constexpr std::uint16_t kGfx1250VccSgpr = 248;

 private:
  std::uint16_t source_vcc_sgpr_ = kGfx1201VccSgpr;
  std::uint16_t target_vcc_sgpr_ = kGfx950VccSgpr;
  WavePolicy wave_policy_ = WavePolicy::kWave32InWave64;
};

}  // namespace mirage::sim::isa::jit

#endif  // MIRAGE_SIM_ISA_JIT_WAVE_ADAPTER_H_
