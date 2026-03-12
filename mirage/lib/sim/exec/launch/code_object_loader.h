#ifndef MIRAGE_SIM_EXEC_LAUNCH_CODE_OBJECT_LOADER_H_
#define MIRAGE_SIM_EXEC_LAUNCH_CODE_OBJECT_LOADER_H_

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace mirage::sim::exec::launch {

// Describes a single kernel argument from AMDGPU metadata.
struct KernelArgDescriptor {
  enum class ValueKind : std::uint8_t {
    kGlobalBuffer,
    kByValue,
    kHiddenBlockCountX,
    kHiddenBlockCountY,
    kHiddenBlockCountZ,
    kHiddenGroupSizeX,
    kHiddenGroupSizeY,
    kHiddenGroupSizeZ,
    kHiddenRemainderX,
    kHiddenRemainderY,
    kHiddenRemainderZ,
    kHiddenGlobalOffsetX,
    kHiddenGlobalOffsetY,
    kHiddenGlobalOffsetZ,
    kHiddenGridDims,
    kOther,
  };

  std::string name;
  std::uint32_t offset = 0;
  std::uint32_t size = 0;
  ValueKind value_kind = ValueKind::kOther;
};

// Kernel metadata extracted from the AMDGPU code object.
struct KernelMetadata {
  std::string name;
  std::string symbol;
  std::uint32_t wavefront_size = 0;
  std::uint32_t max_flat_workgroup_size = 0;
  std::uint32_t sgpr_count = 0;
  std::uint32_t vgpr_count = 0;
  std::uint32_t group_segment_fixed_size = 0;
  std::uint32_t private_segment_fixed_size = 0;
  std::uint32_t kernarg_segment_size = 0;
  std::uint32_t kernarg_segment_align = 0;
  std::vector<KernelArgDescriptor> args;

  // Code location within the .text section.
  std::uint64_t entry_vaddr = 0;
  std::uint64_t entry_size = 0;
  // Offset of the entry point within the code words array (byte offset
  // from the start of .text).
  std::uint64_t entry_offset_in_text = 0;
};

// Result of loading an AMDGPU code object.
struct LoadedCodeObject {
  std::string target;  // e.g. "amdgcn-amd-amdhsa--gfx1201"
  std::vector<std::uint32_t> code_words;  // .text section as 32-bit words
  std::uint64_t text_vaddr = 0;           // Virtual address of .text start
  std::vector<KernelMetadata> kernels;
  std::string error_message;

  bool ok() const { return error_message.empty() && !kernels.empty(); }
};

// Minimal ELF parser for AMDGPU code objects.
// Extracts .text section (as code words), kernel metadata from .note,
// and kernel entry symbols.
class CodeObjectLoader {
 public:
  LoadedCodeObject Load(std::span<const std::byte> elf_bytes);
  LoadedCodeObject LoadFromFile(std::string_view path);

 private:
  bool ParseElf64Header(std::span<const std::byte> elf,
                        LoadedCodeObject* result);
  bool ExtractTextSection(std::span<const std::byte> elf,
                          LoadedCodeObject* result);
  bool ExtractMetadataNote(std::span<const std::byte> elf,
                           LoadedCodeObject* result);
  bool ResolveKernelSymbols(std::span<const std::byte> elf,
                            LoadedCodeObject* result);

  // ELF parsing state.
  std::uint64_t shoff_ = 0;
  std::uint16_t shentsize_ = 0;
  std::uint16_t shnum_ = 0;
  std::uint16_t shstrndx_ = 0;
};

}  // namespace mirage::sim::exec::launch

#endif  // MIRAGE_SIM_EXEC_LAUNCH_CODE_OBJECT_LOADER_H_
