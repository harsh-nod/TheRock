#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <vector>

#include "lib/sim/isa/common/decoded_instruction.h"
#include "lib/sim/isa/common/wave_execution_state.h"
#include "lib/sim/isa/jit/cross_arch_translator.h"
#include "lib/sim/single_gpu_simulator.h"

namespace {

template <typename T>
std::span<const std::byte> AsBytes(const std::vector<T>& values) {
  return std::as_bytes(std::span<const T>(values.data(), values.size()));
}

template <typename T>
std::span<std::byte> AsWritableBytes(std::vector<T>& values) {
  return std::as_writable_bytes(std::span<T>(values.data(), values.size()));
}

bool Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    return false;
  }
  return true;
}

using namespace mirage::sim;
using namespace mirage::sim::isa;
using namespace mirage::sim::isa::jit;

SingleGpuSimulator MakeSimulator() {
  gpu::GpuProperties properties;
  properties.arch_name = "CDNA1";
  properties.gfx_target = "gfx908";
  properties.compute_units = 120;
  properties.hbm_bytes = 16ULL * 1024ULL * 1024ULL * 1024ULL;
  return SingleGpuSimulator(properties);
}

TranslationConfig DefaultTranslationConfig() {
  TranslationConfig config;
  config.source_arch = SourceArchitecture::kGfx1201;
  config.target_arch = TargetArchitecture::kGfx950;
  config.translation_mode = TranslationMode::kExecutableStrict;
  return config;
}

// Test: Translate a scalar add program from gfx1201 and execute it.
//   s0 = 10, s1 = 20
//   s2 = s0 + s1   (S_ADD_U32 is identity gfx1201→gfx950)
//   endpgm
// Expected: s2 = 30
bool TestTranslatedScalarAdd() {
  auto sim = MakeSimulator();
  const auto queue_id = sim.CreateComputeQueue();

  std::vector<DecodedInstruction> program = {
      DecodedInstruction::Binary("S_ADD_U32",
                                 InstructionOperand::Sgpr(2),
                                 InstructionOperand::Sgpr(0),
                                 InstructionOperand::Sgpr(1)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  // Allocate SGPR state: s0=10, s1=20, s2=0
  std::vector<std::uint32_t> sgpr_state = {10u, 20u, 0u};
  const auto sgpr_alloc = sim.AllocateMemory(memory::MemoryRegionKind::kHbm,
                                             sgpr_state.size() * sizeof(sgpr_state[0]));
  if (!Expect(sgpr_alloc.mapped_va != 0, "expected sgpr allocation") ||
      !Expect(sim.WriteMemory(sgpr_alloc.mapped_va, AsBytes(sgpr_state)),
              "expected sgpr write")) {
    return false;
  }

  exec::SyntheticDispatchPacket packet;
  packet.context.queue_id = queue_id;
  packet.opcode = exec::SyntheticKernelOpcode::kGfx1201TranslatedProgram;
  packet.args.sgpr_state_va = sgpr_alloc.mapped_va;
  packet.args.sgpr_state_count = sgpr_state.size();
  packet.args.exec_mask = 0x1ULL;

  const auto completion = sim.SubmitTranslatedProgram(
      queue_id, packet, program, DefaultTranslationConfig());

  if (!Expect(completion.completed, "expected dispatch to complete") ||
      !Expect(completion.success, "expected dispatch to succeed")) {
    return false;
  }

  std::vector<std::uint32_t> result(sgpr_state.size(), 0u);
  if (!Expect(sim.ReadMemory(sgpr_alloc.mapped_va, AsWritableBytes(result)),
              "expected sgpr readback")) {
    return false;
  }

  return Expect(result[0] == 10u, "expected s0 to persist") &&
         Expect(result[1] == 20u, "expected s1 to persist") &&
         Expect(result[2] == 30u, "expected s2 = s0 + s1 = 30");
}

// Test: Translate a vector add program and verify per-lane results.
//   v2 = v0 + v1   (V_ADD_F32 is identity gfx1201→gfx950)
//   endpgm
// v0 lanes: [1.0, 2.0, 3.0, ...]
// v1 lanes: [10.0, 20.0, 30.0, ...]
// Expected v2 lanes: [11.0, 22.0, 33.0, ...]
bool TestTranslatedVectorAdd() {
  auto sim = MakeSimulator();
  const auto queue_id = sim.CreateComputeQueue();

  std::vector<DecodedInstruction> program = {
      DecodedInstruction::Binary("V_ADD_F32",
                                 InstructionOperand::Vgpr(2),
                                 InstructionOperand::Vgpr(0),
                                 InstructionOperand::Vgpr(1)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  constexpr std::size_t kLanes = WaveExecutionState::kLaneCount;
  std::vector<std::uint32_t> vgpr_state(3 * kLanes, 0u);

  auto to_u32 = [](float f) -> std::uint32_t {
    std::uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
  };
  auto to_f32 = [](std::uint32_t u) -> float {
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
  };

  // Set v0 and v1 for lanes 0..3
  for (std::size_t lane = 0; lane < 4; ++lane) {
    vgpr_state[0 * kLanes + lane] = to_u32(static_cast<float>(lane + 1));
    vgpr_state[1 * kLanes + lane] = to_u32(static_cast<float>((lane + 1) * 10));
  }

  const auto vgpr_alloc = sim.AllocateMemory(memory::MemoryRegionKind::kHbm,
                                             vgpr_state.size() * sizeof(vgpr_state[0]));
  if (!Expect(vgpr_alloc.mapped_va != 0, "expected vgpr allocation") ||
      !Expect(sim.WriteMemory(vgpr_alloc.mapped_va, AsBytes(vgpr_state)),
              "expected vgpr write")) {
    return false;
  }

  exec::SyntheticDispatchPacket packet;
  packet.context.queue_id = queue_id;
  packet.opcode = exec::SyntheticKernelOpcode::kGfx1201TranslatedProgram;
  packet.args.vgpr_state_va = vgpr_alloc.mapped_va;
  packet.args.vgpr_state_count = 3;
  packet.args.exec_mask = 0xfULL;  // 4 active lanes

  const auto completion = sim.SubmitTranslatedProgram(
      queue_id, packet, program, DefaultTranslationConfig());

  if (!Expect(completion.completed, "expected dispatch to complete") ||
      !Expect(completion.success, "expected dispatch to succeed")) {
    return false;
  }

  std::vector<std::uint32_t> result(vgpr_state.size(), 0u);
  if (!Expect(sim.ReadMemory(vgpr_alloc.mapped_va, AsWritableBytes(result)),
              "expected vgpr readback")) {
    return false;
  }

  bool ok = true;
  for (std::size_t lane = 0; lane < 4; ++lane) {
    const float expected = static_cast<float>((lane + 1) + (lane + 1) * 10);
    const float actual = to_f32(result[2 * kLanes + lane]);
    if (actual != expected) {
      std::cerr << "FAIL: v2 lane " << lane << " expected " << expected
                << " got " << actual << '\n';
      ok = false;
    }
  }
  return ok;
}

// Test: Translate a program with a renamed opcode (S_ADD_CO_U32 → S_ADD_U32)
// and VCC remap (gfx1201 SGPR 248 → gfx950 SGPR 106).
//   s0 = 10, s1 = 20, s248 = VCC (gfx1201)
//   S_ADD_CO_U32 s2, s0, s1  (dst=s2, src0=s0, src1=s1, implicit vcc_lo=s248)
//   endpgm
// Expected: s2 = 30
bool TestTranslatedRenameWithVccRemap() {
  auto sim = MakeSimulator();
  const auto queue_id = sim.CreateComputeQueue();

  // S_ADD_CO_U32 in gfx1201: opcode renames to S_ADD_U32 on gfx950,
  // VCC operand at SGPR 248 remaps to SGPR 106.
  std::vector<DecodedInstruction> program = {
      DecodedInstruction::Binary("S_ADD_CO_U32",
                                 InstructionOperand::Sgpr(2),
                                 InstructionOperand::Sgpr(0),
                                 InstructionOperand::Sgpr(1)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  std::vector<std::uint32_t> sgpr_state = {10u, 20u, 0u};
  const auto sgpr_alloc = sim.AllocateMemory(memory::MemoryRegionKind::kHbm,
                                             sgpr_state.size() * sizeof(sgpr_state[0]));
  if (!Expect(sgpr_alloc.mapped_va != 0, "expected sgpr allocation") ||
      !Expect(sim.WriteMemory(sgpr_alloc.mapped_va, AsBytes(sgpr_state)),
              "expected sgpr write")) {
    return false;
  }

  exec::SyntheticDispatchPacket packet;
  packet.context.queue_id = queue_id;
  packet.opcode = exec::SyntheticKernelOpcode::kGfx1201TranslatedProgram;
  packet.args.sgpr_state_va = sgpr_alloc.mapped_va;
  packet.args.sgpr_state_count = sgpr_state.size();
  packet.args.exec_mask = 0x1ULL;

  const auto completion = sim.SubmitTranslatedProgram(
      queue_id, packet, program, DefaultTranslationConfig());

  if (!Expect(completion.completed, "expected dispatch to complete") ||
      !Expect(completion.success, "expected dispatch to succeed")) {
    return false;
  }

  std::vector<std::uint32_t> result(sgpr_state.size(), 0u);
  if (!Expect(sim.ReadMemory(sgpr_alloc.mapped_va, AsWritableBytes(result)),
              "expected sgpr readback")) {
    return false;
  }

  return Expect(result[2] == 30u, "expected s2 = s0 + s1 = 30 via renamed opcode");
}

// Test: Translated program with unsupported opcode fails cleanly.
bool TestUnsupportedOpcodeFailsExecution() {
  auto sim = MakeSimulator();
  const auto queue_id = sim.CreateComputeQueue();

  std::vector<DecodedInstruction> program = {
      DecodedInstruction::Nullary("UNSUPPORTED_OP"),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  exec::SyntheticDispatchPacket packet;
  packet.context.queue_id = queue_id;
  packet.opcode = exec::SyntheticKernelOpcode::kGfx1201TranslatedProgram;
  packet.args.exec_mask = 0x1ULL;

  const auto completion = sim.SubmitTranslatedProgram(
      queue_id, packet, program, DefaultTranslationConfig());

  return Expect(completion.completed,
                "expected dispatch to complete") &&
         Expect(!completion.success,
                "expected dispatch to fail for unsupported opcode");
}

// Test: Translated branch program executes correctly.
//   s0 = 0, s1 = 1
//   S_CMP_EQ_I32 s0, s1       // scc = (0 == 1) = 0
//   S_CBRANCH_SCC1 +1          // skip if scc=1 (not taken)
//   S_ADD_U32 s2, s0, s1       // s2 = 0 + 1 = 1
//   S_ENDPGM
// Expected: s2 = 1 (branch not taken because scc=0)
bool TestTranslatedBranchProgram() {
  auto sim = MakeSimulator();
  const auto queue_id = sim.CreateComputeQueue();

  // S_CMP_EQ_I32: 2 operands (ssrc0, ssrc1), sets SCC.
  // S_CBRANCH_SCC1: 1 operand (imm32 relative offset).
  //   target_pc = current_pc + 1 + delta.
  //   delta=+1 means skip the next instruction.
  std::vector<DecodedInstruction> program = {
      DecodedInstruction::TwoOperand("S_CMP_EQ_I32",
                                     InstructionOperand::Sgpr(0),
                                     InstructionOperand::Sgpr(1)),
      DecodedInstruction::OneOperand("S_CBRANCH_SCC1",
                                     InstructionOperand::Imm32(1)),
      DecodedInstruction::Binary("S_ADD_U32",
                                 InstructionOperand::Sgpr(2),
                                 InstructionOperand::Sgpr(0),
                                 InstructionOperand::Sgpr(1)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  std::vector<std::uint32_t> sgpr_state = {0u, 1u, 0u};
  const auto sgpr_alloc = sim.AllocateMemory(memory::MemoryRegionKind::kHbm,
                                             sgpr_state.size() * sizeof(sgpr_state[0]));
  if (!Expect(sgpr_alloc.mapped_va != 0, "expected sgpr allocation") ||
      !Expect(sim.WriteMemory(sgpr_alloc.mapped_va, AsBytes(sgpr_state)),
              "expected sgpr write")) {
    return false;
  }

  exec::SyntheticDispatchPacket packet;
  packet.context.queue_id = queue_id;
  packet.opcode = exec::SyntheticKernelOpcode::kGfx1201TranslatedProgram;
  packet.args.sgpr_state_va = sgpr_alloc.mapped_va;
  packet.args.sgpr_state_count = sgpr_state.size();
  packet.args.exec_mask = 0x1ULL;

  const auto completion = sim.SubmitTranslatedProgram(
      queue_id, packet, program, DefaultTranslationConfig());

  if (!Expect(completion.completed, "expected dispatch to complete") ||
      !Expect(completion.success, "expected dispatch to succeed")) {
    return false;
  }

  std::vector<std::uint32_t> result(sgpr_state.size(), 0u);
  if (!Expect(sim.ReadMemory(sgpr_alloc.mapped_va, AsWritableBytes(result)),
              "expected sgpr readback")) {
    return false;
  }

  return Expect(result[2] == 1u,
                "expected s2 = 1 (branch not taken, add executed)");
}

// Test: Multiple dispatches share the queue correctly.
bool TestMultipleTranslatedDispatches() {
  auto sim = MakeSimulator();
  const auto queue_id = sim.CreateComputeQueue();

  std::vector<DecodedInstruction> program = {
      DecodedInstruction::Binary("S_ADD_U32",
                                 InstructionOperand::Sgpr(0),
                                 InstructionOperand::Sgpr(0),
                                 InstructionOperand::Sgpr(1)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  // s0 = 5, s1 = 3
  std::vector<std::uint32_t> sgpr_state = {5u, 3u};
  const auto sgpr_alloc = sim.AllocateMemory(memory::MemoryRegionKind::kHbm,
                                             sgpr_state.size() * sizeof(sgpr_state[0]));
  if (!Expect(sgpr_alloc.mapped_va != 0, "expected sgpr allocation") ||
      !Expect(sim.WriteMemory(sgpr_alloc.mapped_va, AsBytes(sgpr_state)),
              "expected sgpr write")) {
    return false;
  }

  exec::SyntheticDispatchPacket packet;
  packet.context.queue_id = queue_id;
  packet.opcode = exec::SyntheticKernelOpcode::kGfx1201TranslatedProgram;
  packet.args.sgpr_state_va = sgpr_alloc.mapped_va;
  packet.args.sgpr_state_count = sgpr_state.size();
  packet.args.exec_mask = 0x1ULL;

  auto config = DefaultTranslationConfig();

  // First dispatch: s0 = 5 + 3 = 8
  auto c1 = sim.SubmitTranslatedProgram(queue_id, packet, program, config);
  if (!Expect(c1.completed && c1.success, "expected first dispatch success")) {
    return false;
  }

  // Second dispatch: s0 = 8 + 3 = 11
  auto c2 = sim.SubmitTranslatedProgram(queue_id, packet, program, config);
  if (!Expect(c2.completed && c2.success, "expected second dispatch success")) {
    return false;
  }

  std::vector<std::uint32_t> result(sgpr_state.size(), 0u);
  if (!Expect(sim.ReadMemory(sgpr_alloc.mapped_va, AsWritableBytes(result)),
              "expected sgpr readback")) {
    return false;
  }

  return Expect(result[0] == 11u,
                "expected s0 = 11 after two dispatches (5+3+3)");
}

// Test: EXEC mask narrowing for wave32-in-wave64.
//   Caller passes a full 64-bit exec mask (~0ULL), but because the
//   source arch is gfx1201 (wave32), the translator should narrow the
//   EXEC mask to 32 lanes.  Lanes 32+ should be untouched.
//   v1 = v0 + v0  (V_ADD_F32)
//   v0 lanes 0..63: lane index as float
//   Expected: v1 lanes 0..31 = 2*lane, v1 lanes 32..63 = 0 (inactive)
bool TestExecMaskNarrowingWave32() {
  auto sim = MakeSimulator();
  const auto queue_id = sim.CreateComputeQueue();

  std::vector<DecodedInstruction> program = {
      DecodedInstruction::Binary("V_ADD_F32",
                                 InstructionOperand::Vgpr(1),
                                 InstructionOperand::Vgpr(0),
                                 InstructionOperand::Vgpr(0)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  constexpr std::size_t kLanes = WaveExecutionState::kLaneCount;
  // 2 VGPRs: v0 (input) and v1 (output)
  std::vector<std::uint32_t> vgpr_state(2 * kLanes, 0u);

  auto to_u32 = [](float f) -> std::uint32_t {
    std::uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
  };
  auto to_f32 = [](std::uint32_t u) -> float {
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
  };

  // Fill v0 for all 64 lanes with lane index as float.
  for (std::size_t lane = 0; lane < kLanes; ++lane) {
    vgpr_state[0 * kLanes + lane] = to_u32(static_cast<float>(lane));
  }

  const auto vgpr_alloc = sim.AllocateMemory(memory::MemoryRegionKind::kHbm,
                                             vgpr_state.size() * sizeof(vgpr_state[0]));
  if (!Expect(vgpr_alloc.mapped_va != 0, "expected vgpr allocation") ||
      !Expect(sim.WriteMemory(vgpr_alloc.mapped_va, AsBytes(vgpr_state)),
              "expected vgpr write")) {
    return false;
  }

  exec::SyntheticDispatchPacket packet;
  packet.context.queue_id = queue_id;
  packet.opcode = exec::SyntheticKernelOpcode::kGfx1201TranslatedProgram;
  packet.args.vgpr_state_va = vgpr_alloc.mapped_va;
  packet.args.vgpr_state_count = 2;
  // Intentionally pass full 64-bit mask; narrowing should restrict to 32.
  packet.args.exec_mask = ~0ULL;

  const auto completion = sim.SubmitTranslatedProgram(
      queue_id, packet, program, DefaultTranslationConfig());

  if (!Expect(completion.completed, "expected dispatch to complete") ||
      !Expect(completion.success, "expected dispatch to succeed")) {
    return false;
  }

  std::vector<std::uint32_t> result(vgpr_state.size(), 0u);
  if (!Expect(sim.ReadMemory(vgpr_alloc.mapped_va, AsWritableBytes(result)),
              "expected vgpr readback")) {
    return false;
  }

  bool ok = true;
  // Lanes 0..31 should have v1 = v0 + v0 = 2 * lane.
  for (std::size_t lane = 0; lane < 32; ++lane) {
    const float expected = static_cast<float>(lane * 2);
    const float actual = to_f32(result[1 * kLanes + lane]);
    if (actual != expected) {
      std::cerr << "FAIL: v1 lane " << lane << " expected " << expected
                << " got " << actual << '\n';
      ok = false;
    }
  }
  // Lanes 32..63 should still be 0 (untouched by narrowed EXEC).
  for (std::size_t lane = 32; lane < kLanes; ++lane) {
    const float actual = to_f32(result[1 * kLanes + lane]);
    if (actual != 0.0f) {
      std::cerr << "FAIL: v1 lane " << lane << " expected 0 got "
                << actual << " (should be inactive)\n";
      ok = false;
    }
  }
  return ok;
}

}  // namespace

int main() {
  bool ok = true;

  std::cerr << "TestTranslatedScalarAdd... ";
  ok = TestTranslatedScalarAdd() && ok;
  std::cerr << (ok ? "PASS" : "FAIL") << '\n';

  bool v = true;
  std::cerr << "TestTranslatedVectorAdd... ";
  v = TestTranslatedVectorAdd();
  ok = v && ok;
  std::cerr << (v ? "PASS" : "FAIL") << '\n';

  bool r = true;
  std::cerr << "TestTranslatedRenameWithVccRemap... ";
  r = TestTranslatedRenameWithVccRemap();
  ok = r && ok;
  std::cerr << (r ? "PASS" : "FAIL") << '\n';

  bool u = true;
  std::cerr << "TestUnsupportedOpcodeFailsExecution... ";
  u = TestUnsupportedOpcodeFailsExecution();
  ok = u && ok;
  std::cerr << (u ? "PASS" : "FAIL") << '\n';

  bool b = true;
  std::cerr << "TestTranslatedBranchProgram... ";
  b = TestTranslatedBranchProgram();
  ok = b && ok;
  std::cerr << (b ? "PASS" : "FAIL") << '\n';

  bool m = true;
  std::cerr << "TestMultipleTranslatedDispatches... ";
  m = TestMultipleTranslatedDispatches();
  ok = m && ok;
  std::cerr << (m ? "PASS" : "FAIL") << '\n';

  bool e = true;
  std::cerr << "TestExecMaskNarrowingWave32... ";
  e = TestExecMaskNarrowingWave32();
  ok = e && ok;
  std::cerr << (e ? "PASS" : "FAIL") << '\n';

  if (ok) {
    std::cerr << "All translated_execution tests passed.\n";
  }
  return ok ? 0 : 1;
}
