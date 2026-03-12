#include <iostream>

#include "lib/sim/isa/common/decoded_instruction.h"
#include "lib/sim/isa/jit/wave_adapter.h"

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

bool TestVccRemapGfx1201ToGfx950() {
  WaveAdapter adapter(WaveAdapter::kGfx1201VccSgpr,
                      WaveAdapter::kGfx950VccSgpr,
                      WavePolicy::kWave32InWave64);

  auto instr = DecodedInstruction::Binary(
      "V_ADD_CO_U32",
      InstructionOperand::Vgpr(0),
      InstructionOperand::Vgpr(1),
      InstructionOperand::Sgpr(248));

  adapter.RemapVccOperands(&instr);

  return Expect(instr.operands[2].index == 106,
                "VCC SGPR 248 should remap to 106 for gfx1201->gfx950");
}

bool TestVccRemapGfx950ToGfx1201() {
  WaveAdapter adapter(WaveAdapter::kGfx950VccSgpr,
                      WaveAdapter::kGfx1201VccSgpr,
                      WavePolicy::kWave32InWave64);

  auto instr = DecodedInstruction::Binary(
      "V_ADD_U32",
      InstructionOperand::Vgpr(0),
      InstructionOperand::Vgpr(1),
      InstructionOperand::Sgpr(106));

  adapter.RemapVccOperands(&instr);

  return Expect(instr.operands[2].index == 248,
                "VCC SGPR 106 should remap to 248 for gfx950->gfx1201");
}

bool TestVccRemapPair() {
  WaveAdapter adapter(WaveAdapter::kGfx1201VccSgpr,
                      WaveAdapter::kGfx950VccSgpr,
                      WavePolicy::kWave32InWave64);

  auto instr = DecodedInstruction::ThreeOperand(
      "V_ADDC_CO_U32",
      InstructionOperand::Vgpr(0),
      InstructionOperand::Sgpr(248),
      InstructionOperand::Sgpr(249));

  adapter.RemapVccOperands(&instr);

  return Expect(instr.operands[1].index == 106,
                "VCC lo SGPR 248 should remap to 106") &&
         Expect(instr.operands[2].index == 107,
                "VCC hi SGPR 249 should remap to 107");
}

bool TestNoRemapWhenSameArch() {
  WaveAdapter adapter(WaveAdapter::kGfx950VccSgpr,
                      WaveAdapter::kGfx950VccSgpr,
                      WavePolicy::kWave32InWave64);

  auto instr = DecodedInstruction::Binary(
      "V_ADD_U32",
      InstructionOperand::Vgpr(0),
      InstructionOperand::Vgpr(1),
      InstructionOperand::Sgpr(106));

  adapter.RemapVccOperands(&instr);

  return Expect(instr.operands[2].index == 106,
                "VCC should remain 106 when source==target arch");
}

bool TestWaveSensitiveDetection() {
  WaveAdapter adapter(WaveAdapter::kGfx1201VccSgpr,
                      WaveAdapter::kGfx950VccSgpr,
                      WavePolicy::kWave32InWave64);

  return Expect(adapter.IsWaveSensitive("V_READLANE_B32"),
                "V_READLANE_B32 should be wave-sensitive") &&
         Expect(adapter.IsWaveSensitive("DS_PERMUTE_B32"),
                "DS_PERMUTE_B32 should be wave-sensitive") &&
         Expect(!adapter.IsWaveSensitive("V_ADD_F32"),
                "V_ADD_F32 should not be wave-sensitive") &&
         Expect(!adapter.IsWaveSensitive("S_MOV_B32"),
                "S_MOV_B32 should not be wave-sensitive");
}

bool TestRejectPolicy() {
  WaveAdapter reject_adapter(WaveAdapter::kGfx1201VccSgpr,
                             WaveAdapter::kGfx950VccSgpr,
                             WavePolicy::kRejectWaveSensitive);

  WaveAdapter allow_adapter(WaveAdapter::kGfx1201VccSgpr,
                            WaveAdapter::kGfx950VccSgpr,
                            WavePolicy::kWave32InWave64);

  return Expect(reject_adapter.ShouldReject("V_READLANE_B32"),
                "reject policy should reject V_READLANE_B32") &&
         Expect(!allow_adapter.ShouldReject("V_READLANE_B32"),
                "wave32-in-wave64 policy should not reject V_READLANE_B32") &&
         Expect(!reject_adapter.ShouldReject("V_ADD_F32"),
                "reject policy should not reject non-wave-sensitive ops");
}

bool TestExecNarrowingRequired() {
  WaveAdapter adapter_1201_950(WaveAdapter::kGfx1201VccSgpr,
                               WaveAdapter::kGfx950VccSgpr,
                               WavePolicy::kWave32InWave64);

  WaveAdapter adapter_1250_950(WaveAdapter::kGfx1250VccSgpr,
                               WaveAdapter::kGfx950VccSgpr,
                               WavePolicy::kWave32InWave64);

  WaveAdapter adapter_same(WaveAdapter::kGfx950VccSgpr,
                           WaveAdapter::kGfx950VccSgpr,
                           WavePolicy::kWave32InWave64);

  WaveAdapter adapter_reverse(WaveAdapter::kGfx950VccSgpr,
                              WaveAdapter::kGfx1201VccSgpr,
                              WavePolicy::kWave32InWave64);

  return Expect(adapter_1201_950.RequiresExecNarrowing(),
                "gfx1201->gfx950 should require exec narrowing") &&
         Expect(adapter_1250_950.RequiresExecNarrowing(),
                "gfx1250->gfx950 should require exec narrowing") &&
         Expect(!adapter_same.RequiresExecNarrowing(),
                "gfx950->gfx950 should not require exec narrowing") &&
         Expect(!adapter_reverse.RequiresExecNarrowing(),
                "gfx950->gfx1201 should not require exec narrowing");
}

bool TestNarrowExecMask() {
  // Full 64-bit mask should be narrowed to lower 32 bits.
  const std::uint64_t full = ~0ULL;
  const std::uint64_t narrowed = WaveAdapter::NarrowExecMask(full);
  bool ok = Expect(narrowed == 0x00000000FFFFFFFFULL,
                   "full mask narrowed to wave32");

  // Already-narrow mask should be unchanged.
  const std::uint64_t wave32_mask = 0x0000000000000FULL;
  ok = Expect(WaveAdapter::NarrowExecMask(wave32_mask) == wave32_mask,
              "wave32 mask unchanged after narrowing") && ok;

  // Mask with upper bits set should have them cleared.
  const std::uint64_t mixed = 0xFFFFFFFF0000000FULL;
  ok = Expect(WaveAdapter::NarrowExecMask(mixed) == 0x000000000000000FULL,
              "upper 32 bits cleared by narrowing") && ok;

  // Zero mask stays zero.
  ok = Expect(WaveAdapter::NarrowExecMask(0) == 0,
              "zero mask stays zero") && ok;

  return ok;
}

}  // namespace

int main() {
  bool ok = true;
  ok = TestVccRemapGfx1201ToGfx950() && ok;
  ok = TestVccRemapGfx950ToGfx1201() && ok;
  ok = TestVccRemapPair() && ok;
  ok = TestNoRemapWhenSameArch() && ok;
  ok = TestWaveSensitiveDetection() && ok;
  ok = TestRejectPolicy() && ok;
  ok = TestExecNarrowingRequired() && ok;
  ok = TestNarrowExecMask() && ok;

  if (ok) {
    std::cerr << "All wave_adapter tests passed.\n";
  }
  return ok ? 0 : 1;
}
