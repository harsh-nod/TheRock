#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>

#include "lib/sim/exec/launch/code_object_loader.h"

namespace {

bool Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    return false;
  }
  return true;
}

using namespace mirage::sim::exec::launch;

// Path to the checked-in vector_add fixture.
std::string FixturePath() {
  // Try paths relative to common build locations.
  std::string candidates[] = {
      "native/tests/data/code_objects/gfx1201/vector_add/vector_add.co",
      "../native/tests/data/code_objects/gfx1201/vector_add/vector_add.co",
      "tests/data/code_objects/gfx1201/vector_add/vector_add.co",
      "../tests/data/code_objects/gfx1201/vector_add/vector_add.co",
  };
  for (const auto& path : candidates) {
    if (std::filesystem::exists(path)) return path;
  }
  // Fall back to absolute path from source tree.
  return std::string(MIRAGE_TEST_DATA_DIR) +
         "/code_objects/gfx1201/vector_add/vector_add.co";
}

bool TestLoadVectorAdd() {
  CodeObjectLoader loader;
  auto result = loader.LoadFromFile(FixturePath());

  if (!Expect(result.ok(), ("load failed: " + result.error_message).c_str()))
    return false;

  if (!Expect(!result.code_words.empty(), "code_words should not be empty"))
    return false;

  if (!Expect(result.target.find("gfx1201") != std::string::npos,
              ("target should contain gfx1201, got: " + result.target).c_str()))
    return false;

  if (!Expect(result.kernels.size() == 1, "should have exactly 1 kernel"))
    return false;

  return true;
}

bool TestKernelMetadata() {
  CodeObjectLoader loader;
  auto result = loader.LoadFromFile(FixturePath());
  if (!Expect(result.ok(), ("load failed: " + result.error_message).c_str()))
    return false;

  const auto& k = result.kernels[0];

  if (!Expect(k.name == "vector_add",
              ("kernel name should be vector_add, got: " + k.name).c_str()))
    return false;

  if (!Expect(k.wavefront_size == 32,
              "wavefront_size should be 32"))
    return false;

  if (!Expect(k.sgpr_count == 10, "sgpr_count should be 10"))
    return false;

  if (!Expect(k.vgpr_count == 6, "vgpr_count should be 6"))
    return false;

  if (!Expect(k.kernarg_segment_size == 288,
              "kernarg_segment_size should be 288"))
    return false;

  if (!Expect(k.max_flat_workgroup_size == 1024,
              "max_flat_workgroup_size should be 1024"))
    return false;

  return true;
}

bool TestKernelArgs() {
  CodeObjectLoader loader;
  auto result = loader.LoadFromFile(FixturePath());
  if (!Expect(result.ok(), ("load failed: " + result.error_message).c_str()))
    return false;

  const auto& args = result.kernels[0].args;

  // Should have at least the 4 explicit args + several hidden args.
  if (!Expect(args.size() >= 4, "should have at least 4 args"))
    return false;

  // First 3 args should be global buffers.
  if (!Expect(args[0].value_kind == KernelArgDescriptor::ValueKind::kGlobalBuffer,
              "arg 0 should be global_buffer"))
    return false;
  if (!Expect(args[0].offset == 0, "arg 0 offset should be 0"))
    return false;
  if (!Expect(args[0].size == 8, "arg 0 size should be 8"))
    return false;

  if (!Expect(args[1].value_kind == KernelArgDescriptor::ValueKind::kGlobalBuffer,
              "arg 1 should be global_buffer"))
    return false;
  if (!Expect(args[1].offset == 8, "arg 1 offset should be 8"))
    return false;

  if (!Expect(args[2].value_kind == KernelArgDescriptor::ValueKind::kGlobalBuffer,
              "arg 2 should be global_buffer"))
    return false;
  if (!Expect(args[2].offset == 16, "arg 2 offset should be 16"))
    return false;

  // Fourth arg is 'n' (by_value).
  if (!Expect(args[3].value_kind == KernelArgDescriptor::ValueKind::kByValue,
              "arg 3 should be by_value"))
    return false;
  if (!Expect(args[3].offset == 24, "arg 3 offset should be 24"))
    return false;
  if (!Expect(args[3].size == 4, "arg 3 size should be 4"))
    return false;

  return true;
}

bool TestKernelEntry() {
  CodeObjectLoader loader;
  auto result = loader.LoadFromFile(FixturePath());
  if (!Expect(result.ok(), ("load failed: " + result.error_message).c_str()))
    return false;

  const auto& k = result.kernels[0];

  if (!Expect(k.entry_vaddr == 0x1900,
              "entry_vaddr should be 0x1900"))
    return false;

  if (!Expect(k.entry_size > 0, "entry_size should be > 0"))
    return false;

  if (!Expect(k.entry_offset_in_text == 0x1900 - result.text_vaddr,
              "entry_offset_in_text should be offset from text base"))
    return false;

  return true;
}

bool TestCodeWordsNonEmpty() {
  CodeObjectLoader loader;
  auto result = loader.LoadFromFile(FixturePath());
  if (!Expect(result.ok(), ("load failed: " + result.error_message).c_str()))
    return false;

  // The .text section should have 0x280 bytes = 160 dwords.
  if (!Expect(result.code_words.size() == 160,
              "code_words should have 160 dwords (0x280 bytes)"))
    return false;

  // First instruction at the kernel entry should be S_CLAUSE 0x1
  // which encodes as 0xBF850001.
  std::size_t entry_word_offset = result.kernels[0].entry_offset_in_text / 4;
  if (!Expect(result.code_words[entry_word_offset] == 0xBF850001,
              "first code word at entry should be 0xBF850001 (S_CLAUSE 0x1)"))
    return false;

  return true;
}

bool TestInvalidFile() {
  CodeObjectLoader loader;
  auto result = loader.LoadFromFile("/nonexistent/path/file.co");

  if (!Expect(!result.ok(), "should fail for nonexistent file"))
    return false;

  return true;
}

bool TestNotElfFile() {
  // Create a fake non-ELF buffer.
  std::vector<std::byte> fake(64, std::byte{0});
  CodeObjectLoader loader;
  auto result = loader.Load(fake);

  if (!Expect(!result.ok(), "should fail for non-ELF data"))
    return false;
  if (!Expect(result.error_message.find("ELF") != std::string::npos,
              "error should mention ELF"))
    return false;

  return true;
}

}  // namespace

int main() {
  bool ok = true;
  ok = TestLoadVectorAdd() && ok;
  ok = TestKernelMetadata() && ok;
  ok = TestKernelArgs() && ok;
  ok = TestKernelEntry() && ok;
  ok = TestCodeWordsNonEmpty() && ok;
  ok = TestInvalidFile() && ok;
  ok = TestNotElfFile() && ok;

  if (ok) {
    std::cerr << "All code_object_loader tests passed.\n";
  }
  return ok ? 0 : 1;
}
