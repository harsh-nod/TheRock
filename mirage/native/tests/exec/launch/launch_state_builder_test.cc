#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>

#include "lib/sim/exec/launch/code_object_loader.h"
#include "lib/sim/exec/launch/kernel_launch_abi.h"

namespace {

bool Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    return false;
  }
  return true;
}

using namespace mirage::sim::exec::launch;

std::string FixturePath() {
  std::string candidates[] = {
      "native/tests/data/code_objects/gfx1201/vector_add/vector_add.co",
      "../native/tests/data/code_objects/gfx1201/vector_add/vector_add.co",
      "tests/data/code_objects/gfx1201/vector_add/vector_add.co",
      "../tests/data/code_objects/gfx1201/vector_add/vector_add.co",
  };
  for (const auto& path : candidates) {
    if (std::filesystem::exists(path)) return path;
  }
  return std::string(MIRAGE_TEST_DATA_DIR) +
         "/code_objects/gfx1201/vector_add/vector_add.co";
}

bool TestBuildLaunchState() {
  CodeObjectLoader loader;
  auto co = loader.LoadFromFile(FixturePath());
  if (!Expect(co.ok(), ("load failed: " + co.error_message).c_str()))
    return false;

  LaunchStateBuilderInput input;
  input.kernel = &co.kernels[0];
  input.grid_x = 1;
  input.grid_y = 1;
  input.grid_z = 1;
  input.block_x = 64;
  input.block_y = 1;
  input.block_z = 1;
  input.workgroup_id_x = 0;
  input.wave_index = 0;

  // Set explicit args: 3 pointers + n.
  // out = VA 0x1000, a = VA 0x2000, b = VA 0x3000, n = 64
  auto MakeU64Arg = [](std::uint32_t offset, std::uint64_t val) {
    KernelArgValue arg;
    arg.offset = offset;
    arg.data.resize(8);
    std::memcpy(arg.data.data(), &val, 8);
    return arg;
  };
  auto MakeU32Arg = [](std::uint32_t offset, std::uint32_t val) {
    KernelArgValue arg;
    arg.offset = offset;
    arg.data.resize(4);
    std::memcpy(arg.data.data(), &val, 4);
    return arg;
  };

  input.explicit_args.push_back(MakeU64Arg(0, 0x1000));   // out
  input.explicit_args.push_back(MakeU64Arg(8, 0x2000));   // a
  input.explicit_args.push_back(MakeU64Arg(16, 0x3000));  // b
  input.explicit_args.push_back(MakeU32Arg(24, 64));       // n

  LaunchStateBuilder builder;
  auto state = builder.Build(input);

  if (!Expect(state.ok(), ("build failed: " + state.error_message).c_str()))
    return false;

  // Check kernarg buffer size.
  if (!Expect(state.kernarg_buffer.size() == 288,
              "kernarg_buffer should be 288 bytes"))
    return false;

  // Check explicit args are in the buffer.
  std::uint64_t out_ptr;
  std::memcpy(&out_ptr, state.kernarg_buffer.data() + 0, 8);
  if (!Expect(out_ptr == 0x1000, "out pointer should be 0x1000"))
    return false;

  std::uint32_t n_val;
  std::memcpy(&n_val, state.kernarg_buffer.data() + 24, 4);
  if (!Expect(n_val == 64, "n should be 64"))
    return false;

  // Check hidden args: block_count_x = grid_x = 1.
  std::uint32_t block_count_x;
  std::memcpy(&block_count_x, state.kernarg_buffer.data() + 32, 4);
  if (!Expect(block_count_x == 1, "hidden block_count_x should be 1"))
    return false;

  // Check hidden group_size_x = block_x = 64.
  std::uint16_t group_size_x;
  std::memcpy(&group_size_x, state.kernarg_buffer.data() + 44, 2);
  if (!Expect(group_size_x == 64, "hidden group_size_x should be 64"))
    return false;

  return true;
}

bool TestSgprSeed() {
  CodeObjectLoader loader;
  auto co = loader.LoadFromFile(FixturePath());
  if (!Expect(co.ok(), ("load failed: " + co.error_message).c_str()))
    return false;

  LaunchStateBuilderInput input;
  input.kernel = &co.kernels[0];
  input.block_x = 32;
  input.grid_x = 1;

  LaunchStateBuilder builder;
  auto state = builder.Build(input);

  if (!Expect(state.ok(), ("build failed: " + state.error_message).c_str()))
    return false;

  // SGPR seed should have at least kernel's sgpr_count entries.
  if (!Expect(state.sgpr_seed.size() >= co.kernels[0].sgpr_count,
              "sgpr_seed should have >= sgpr_count entries"))
    return false;

  return true;
}

bool TestVgprSeed() {
  CodeObjectLoader loader;
  auto co = loader.LoadFromFile(FixturePath());
  if (!Expect(co.ok(), ("load failed: " + co.error_message).c_str()))
    return false;

  LaunchStateBuilderInput input;
  input.kernel = &co.kernels[0];
  input.block_x = 64;
  input.grid_x = 1;
  input.wave_index = 0;

  LaunchStateBuilder builder;
  auto state = builder.Build(input);

  if (!Expect(state.ok(), ("build failed: " + state.error_message).c_str()))
    return false;

  // v0 should be workitem IDs: 0, 1, 2, ..., 63.
  for (std::uint32_t lane = 0; lane < 64; ++lane) {
    std::uint32_t v0 = state.vgpr_seed[0 * LaunchState::kLaneCount + lane];
    if (!Expect(v0 == lane,
                ("v0[" + std::to_string(lane) + "] should be " +
                 std::to_string(lane) + " but got " + std::to_string(v0))
                    .c_str()))
      return false;
  }

  return true;
}

bool TestVgprSeedWave1() {
  CodeObjectLoader loader;
  auto co = loader.LoadFromFile(FixturePath());
  if (!Expect(co.ok(), ("load failed: " + co.error_message).c_str()))
    return false;

  LaunchStateBuilderInput input;
  input.kernel = &co.kernels[0];
  input.block_x = 128;
  input.grid_x = 1;
  input.wave_index = 1;  // Second wave.

  LaunchStateBuilder builder;
  auto state = builder.Build(input);

  if (!Expect(state.ok(), ("build failed: " + state.error_message).c_str()))
    return false;

  // v0 for wave 1 should be 32, 33, ..., 95 (wavefront_size=32).
  // base_workitem = 1 * 32 = 32
  std::uint32_t v0_lane0 = state.vgpr_seed[0 * LaunchState::kLaneCount + 0];
  if (!Expect(v0_lane0 == 32,
              ("v0[0] for wave 1 should be 32, got " +
               std::to_string(v0_lane0)).c_str()))
    return false;

  return true;
}

bool TestExecMask() {
  CodeObjectLoader loader;
  auto co = loader.LoadFromFile(FixturePath());
  if (!Expect(co.ok(), ("load failed: " + co.error_message).c_str()))
    return false;

  LaunchStateBuilderInput input;
  input.kernel = &co.kernels[0];
  input.block_x = 48;  // Not a multiple of wavefront_size.
  input.grid_x = 1;
  input.wave_index = 1;  // Second wave: workitems 32..47.

  LaunchStateBuilder builder;
  auto state = builder.Build(input);

  if (!Expect(state.ok(), ("build failed: " + state.error_message).c_str()))
    return false;

  // Wavefront_size = 32, wave 1 base = 32, total = 48.
  // Active lanes: 32..47, that's 16 lanes (0..15 in wave-local indexing).
  // Exec mask should have bits 0..15 set.
  std::uint64_t expected = (1ULL << 16) - 1;
  if (!Expect(state.exec_mask == expected,
              ("exec_mask should be 0x" + std::to_string(expected) +
               " got 0x" + std::to_string(state.exec_mask)).c_str()))
    return false;

  return true;
}

bool TestNullKernelFails() {
  LaunchStateBuilderInput input;
  input.kernel = nullptr;

  LaunchStateBuilder builder;
  auto state = builder.Build(input);

  if (!Expect(!state.ok(), "should fail with null kernel"))
    return false;

  return true;
}

}  // namespace

int main() {
  bool ok = true;
  ok = TestBuildLaunchState() && ok;
  ok = TestSgprSeed() && ok;
  ok = TestVgprSeed() && ok;
  ok = TestVgprSeedWave1() && ok;
  ok = TestExecMask() && ok;
  ok = TestNullKernelFails() && ok;

  if (ok) {
    std::cerr << "All launch_state_builder tests passed.\n";
  }
  return ok ? 0 : 1;
}
