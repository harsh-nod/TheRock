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

  // Returns true when the source architecture uses wave32 and the target
  // uses wave64, requiring the EXEC mask to be narrowed to 32 active lanes.
  bool RequiresExecNarrowing() const { return requires_exec_narrowing_; }

  // Narrow a 64-bit EXEC mask to wave32 by clearing lanes 32-63.
  static std::uint64_t NarrowExecMask(std::uint64_t mask) {
    return mask & kWave32ExecMask;
  }

  std::uint16_t source_vcc_sgpr() const { return source_vcc_sgpr_; }
  std::uint16_t target_vcc_sgpr() const { return target_vcc_sgpr_; }
  WavePolicy wave_policy() const { return wave_policy_; }

  static constexpr std::uint16_t kGfx950VccSgpr = 106;
  static constexpr std::uint16_t kGfx1201VccSgpr = 248;
  static constexpr std::uint16_t kGfx1250VccSgpr = 248;

  // EXEC mask with only the lower 32 lanes active (wave32-in-wave64).
  static constexpr std::uint64_t kWave32ExecMask = 0x00000000FFFFFFFFULL;

 private:
  std::uint16_t source_vcc_sgpr_ = kGfx1201VccSgpr;
  std::uint16_t target_vcc_sgpr_ = kGfx950VccSgpr;
  WavePolicy wave_policy_ = WavePolicy::kWave32InWave64;
  bool requires_exec_narrowing_ = false;
};

}  // namespace mirage::sim::isa::jit

#endif  // MIRAGE_SIM_ISA_JIT_WAVE_ADAPTER_H_
