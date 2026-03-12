#include "lib/sim/exec/launch/kernel_launch_abi.h"

#include <cstring>

namespace mirage::sim::exec::launch {

LaunchState LaunchStateBuilder::Build(const LaunchStateBuilderInput& input) {
  LaunchState state;

  if (input.kernel == nullptr) {
    state.error_message = "No kernel metadata provided";
    return state;
  }

  PopulateKernargBuffer(input, &state);
  PopulateSgprSeed(input, &state);
  PopulateVgprSeed(input, &state);
  ComputeExecMask(input, &state);

  return state;
}

void LaunchStateBuilder::PopulateKernargBuffer(
    const LaunchStateBuilderInput& input, LaunchState* state) {
  const auto& kernel = *input.kernel;
  state->kernarg_buffer.resize(kernel.kernarg_segment_size, std::byte{0});

  // Write explicit kernel arguments.
  for (const auto& arg : input.explicit_args) {
    if (arg.offset + arg.data.size() <= state->kernarg_buffer.size()) {
      std::memcpy(state->kernarg_buffer.data() + arg.offset,
                  arg.data.data(), arg.data.size());
    }
  }

  // Write hidden arguments based on metadata descriptors.
  for (const auto& desc : kernel.args) {
    switch (desc.value_kind) {
      case KernelArgDescriptor::ValueKind::kHiddenBlockCountX: {
        std::uint32_t val = input.grid_x;
        if (desc.offset + sizeof(val) <= state->kernarg_buffer.size())
          std::memcpy(state->kernarg_buffer.data() + desc.offset, &val, sizeof(val));
        break;
      }
      case KernelArgDescriptor::ValueKind::kHiddenBlockCountY: {
        std::uint32_t val = input.grid_y;
        if (desc.offset + sizeof(val) <= state->kernarg_buffer.size())
          std::memcpy(state->kernarg_buffer.data() + desc.offset, &val, sizeof(val));
        break;
      }
      case KernelArgDescriptor::ValueKind::kHiddenBlockCountZ: {
        std::uint32_t val = input.grid_z;
        if (desc.offset + sizeof(val) <= state->kernarg_buffer.size())
          std::memcpy(state->kernarg_buffer.data() + desc.offset, &val, sizeof(val));
        break;
      }
      case KernelArgDescriptor::ValueKind::kHiddenGroupSizeX: {
        std::uint16_t val = static_cast<std::uint16_t>(input.block_x);
        if (desc.offset + sizeof(val) <= state->kernarg_buffer.size())
          std::memcpy(state->kernarg_buffer.data() + desc.offset, &val, sizeof(val));
        break;
      }
      case KernelArgDescriptor::ValueKind::kHiddenGroupSizeY: {
        std::uint16_t val = static_cast<std::uint16_t>(input.block_y);
        if (desc.offset + sizeof(val) <= state->kernarg_buffer.size())
          std::memcpy(state->kernarg_buffer.data() + desc.offset, &val, sizeof(val));
        break;
      }
      case KernelArgDescriptor::ValueKind::kHiddenGroupSizeZ: {
        std::uint16_t val = static_cast<std::uint16_t>(input.block_z);
        if (desc.offset + sizeof(val) <= state->kernarg_buffer.size())
          std::memcpy(state->kernarg_buffer.data() + desc.offset, &val, sizeof(val));
        break;
      }
      case KernelArgDescriptor::ValueKind::kHiddenGridDims: {
        std::uint16_t dims = 1;
        if (input.grid_y > 1 || input.block_y > 1) dims = 2;
        if (input.grid_z > 1 || input.block_z > 1) dims = 3;
        if (desc.offset + sizeof(dims) <= state->kernarg_buffer.size())
          std::memcpy(state->kernarg_buffer.data() + desc.offset, &dims, sizeof(dims));
        break;
      }
      case KernelArgDescriptor::ValueKind::kHiddenRemainderX: {
        std::uint32_t total_x = input.grid_x * input.block_x;
        std::uint16_t rem = static_cast<std::uint16_t>(total_x % input.block_x);
        if (desc.offset + sizeof(rem) <= state->kernarg_buffer.size())
          std::memcpy(state->kernarg_buffer.data() + desc.offset, &rem, sizeof(rem));
        break;
      }
      case KernelArgDescriptor::ValueKind::kHiddenRemainderY: {
        std::uint32_t total_y = input.grid_y * input.block_y;
        std::uint16_t rem = static_cast<std::uint16_t>(total_y % input.block_y);
        if (desc.offset + sizeof(rem) <= state->kernarg_buffer.size())
          std::memcpy(state->kernarg_buffer.data() + desc.offset, &rem, sizeof(rem));
        break;
      }
      case KernelArgDescriptor::ValueKind::kHiddenRemainderZ: {
        std::uint32_t total_z = input.grid_z * input.block_z;
        std::uint16_t rem = static_cast<std::uint16_t>(total_z % input.block_z);
        if (desc.offset + sizeof(rem) <= state->kernarg_buffer.size())
          std::memcpy(state->kernarg_buffer.data() + desc.offset, &rem, sizeof(rem));
        break;
      }
      case KernelArgDescriptor::ValueKind::kHiddenGlobalOffsetX:
      case KernelArgDescriptor::ValueKind::kHiddenGlobalOffsetY:
      case KernelArgDescriptor::ValueKind::kHiddenGlobalOffsetZ: {
        // Global offsets default to 0.
        std::uint64_t val = 0;
        if (desc.offset + sizeof(val) <= state->kernarg_buffer.size())
          std::memcpy(state->kernarg_buffer.data() + desc.offset, &val, sizeof(val));
        break;
      }
      default:
        break;
    }
  }
}

void LaunchStateBuilder::PopulateSgprSeed(
    const LaunchStateBuilderInput& input, LaunchState* state) {
  // AMDGPU compute ABI for SGPRs (gfx9/gfx10/gfx11/gfx12):
  // s[0:1] = kernarg segment pointer (64-bit VA)
  // Beyond s[0:1], the kernel descriptor controls which SGPRs are
  // initialized.  For simplicity, we populate the most common ones:
  // s[0:1] = kernarg_va (will be set later when kernarg is allocated)
  //
  // The workgroup ID is passed via TTMP registers on gfx12 (not SGPRs).
  // For the interpreter, we pass workgroup_id_x in the kernarg buffer.
  // The kernel reads it via S_LOAD from kernarg.

  // Allocate enough SGPRs for the kernel's stated count.
  std::uint32_t sgpr_count = std::max(input.kernel->sgpr_count, 4u);
  state->sgpr_seed.resize(sgpr_count, 0u);

  // s[0:1] will be set to kernarg_va by the caller after allocation.
  // Leave as 0 for now.
}

void LaunchStateBuilder::PopulateVgprSeed(
    const LaunchStateBuilderInput& input, LaunchState* state) {
  // AMDGPU compute ABI for VGPRs:
  // v0 = threadIdx.x (workitem ID within the workgroup)
  // On gfx12: ttmp9 has workgroup_id_x, but the v0 is still workitem ID.

  const auto& kernel = *input.kernel;
  std::uint32_t vgpr_count = std::max(kernel.vgpr_count, 1u);
  state->vgpr_count = vgpr_count;
  state->vgpr_seed.resize(
      static_cast<std::size_t>(vgpr_count) * LaunchState::kLaneCount, 0u);

  // Compute v0 = workitem ID within the workgroup.
  // workitem_id = wave_index * wavefront_size + lane_id
  std::uint32_t wavefront_size = kernel.wavefront_size;
  if (wavefront_size == 0) wavefront_size = 32;

  std::uint32_t base_workitem = input.wave_index * wavefront_size;

  for (std::uint32_t lane = 0; lane < LaunchState::kLaneCount; ++lane) {
    // v0 for each lane.
    state->vgpr_seed[0 * LaunchState::kLaneCount + lane] =
        base_workitem + lane;
  }
}

void LaunchStateBuilder::ComputeExecMask(
    const LaunchStateBuilderInput& input, LaunchState* state) {
  const auto& kernel = *input.kernel;
  std::uint32_t wavefront_size = kernel.wavefront_size;
  if (wavefront_size == 0) wavefront_size = 32;

  // Total workitems in this workgroup.
  std::uint32_t total_workitems = input.block_x * input.block_y * input.block_z;
  std::uint32_t base_workitem = input.wave_index * wavefront_size;

  // All lanes that have a valid workitem are active.
  std::uint64_t mask = 0;
  for (std::uint32_t lane = 0; lane < LaunchState::kLaneCount; ++lane) {
    if (base_workitem + lane < total_workitems) {
      mask |= (1ULL << lane);
    }
  }

  state->exec_mask = mask;
}

}  // namespace mirage::sim::exec::launch
