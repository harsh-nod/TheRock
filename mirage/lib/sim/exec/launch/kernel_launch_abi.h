#ifndef MIRAGE_SIM_EXEC_LAUNCH_KERNEL_LAUNCH_ABI_H_
#define MIRAGE_SIM_EXEC_LAUNCH_KERNEL_LAUNCH_ABI_H_

#include <cstdint>
#include <span>
#include <vector>

#include "lib/sim/exec/launch/code_object_loader.h"

namespace mirage::sim::exec::launch {

// Concrete kernel argument values to be written into the kernarg buffer.
struct KernelArgValue {
  std::uint32_t offset = 0;
  std::vector<std::byte> data;
};

// Input to the launch state builder.
struct LaunchStateBuilderInput {
  const KernelMetadata* kernel = nullptr;
  std::vector<KernelArgValue> explicit_args;
  // Grid dimensions (number of workgroups in each dimension).
  std::uint32_t grid_x = 1;
  std::uint32_t grid_y = 1;
  std::uint32_t grid_z = 1;
  // Workgroup dimensions (threads per workgroup in each dimension).
  std::uint32_t block_x = 1;
  std::uint32_t block_y = 1;
  std::uint32_t block_z = 1;
  // Which workgroup this wave belongs to.
  std::uint32_t workgroup_id_x = 0;
  std::uint32_t workgroup_id_y = 0;
  std::uint32_t workgroup_id_z = 0;
  // Wave index within the workgroup.
  std::uint32_t wave_index = 0;
};

// Output from the launch state builder — the seeded state Mirage consumes.
struct LaunchState {
  // Kernarg buffer (byte-addressable, laid out per AMDGPU ABI).
  std::vector<std::byte> kernarg_buffer;
  // Virtual address where kernarg buffer will be placed.
  std::uint64_t kernarg_va = 0;

  // SGPR seed values.
  // AMDGPU ABI: s[0:1] = kernarg segment pointer
  //             s[2:3] = dispatch pointer (AQL packet)
  //             ttmp9  = workgroup_id_x (implicit, via TTMP)
  std::vector<std::uint32_t> sgpr_seed;

  // VGPR seed values (per-lane).
  // v0 = threadIdx.x (within workgroup)
  // v[0][lane] = wave_index * wavefront_size + lane
  static constexpr std::uint32_t kLaneCount = 64;
  std::vector<std::uint32_t> vgpr_seed;  // [vgpr_count * kLaneCount]
  std::uint32_t vgpr_count = 0;

  // Exec mask for this wave.
  std::uint64_t exec_mask = ~0ULL;

  // Code words and entry offset.
  std::span<const std::uint32_t> code_words;
  std::uint32_t entry_offset_words = 0;

  std::string error_message;
  bool ok() const { return error_message.empty(); }
};

// Builds the seeded launch state from kernel metadata and arguments.
class LaunchStateBuilder {
 public:
  LaunchState Build(const LaunchStateBuilderInput& input);

 private:
  void PopulateKernargBuffer(const LaunchStateBuilderInput& input,
                             LaunchState* state);
  void PopulateSgprSeed(const LaunchStateBuilderInput& input,
                        LaunchState* state);
  void PopulateVgprSeed(const LaunchStateBuilderInput& input,
                        LaunchState* state);
  void ComputeExecMask(const LaunchStateBuilderInput& input,
                       LaunchState* state);
};

}  // namespace mirage::sim::exec::launch

#endif  // MIRAGE_SIM_EXEC_LAUNCH_KERNEL_LAUNCH_ABI_H_
