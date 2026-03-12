#include "lib/sim/exec/launch/code_object_loader.h"

#include <algorithm>
#include <cstring>
#include <fstream>

namespace mirage::sim::exec::launch {

namespace {

// ELF64 header layout.
struct Elf64_Ehdr {
  std::uint8_t e_ident[16];
  std::uint16_t e_type;
  std::uint16_t e_machine;
  std::uint32_t e_version;
  std::uint64_t e_entry;
  std::uint64_t e_phoff;
  std::uint64_t e_shoff;
  std::uint32_t e_flags;
  std::uint16_t e_ehsize;
  std::uint16_t e_phentsize;
  std::uint16_t e_phnum;
  std::uint16_t e_shentsize;
  std::uint16_t e_shnum;
  std::uint16_t e_shstrndx;
};

struct Elf64_Shdr {
  std::uint32_t sh_name;
  std::uint32_t sh_type;
  std::uint64_t sh_flags;
  std::uint64_t sh_addr;
  std::uint64_t sh_offset;
  std::uint64_t sh_size;
  std::uint32_t sh_link;
  std::uint32_t sh_info;
  std::uint64_t sh_addralign;
  std::uint64_t sh_entsize;
};

struct Elf64_Sym {
  std::uint32_t st_name;
  std::uint8_t st_info;
  std::uint8_t st_other;
  std::uint16_t st_shndx;
  std::uint64_t st_value;
  std::uint64_t st_size;
};

struct Elf64_Nhdr {
  std::uint32_t n_namesz;
  std::uint32_t n_descsz;
  std::uint32_t n_type;
};

constexpr std::uint32_t SHT_NOTE = 7;
constexpr std::uint32_t SHT_SYMTAB = 2;
constexpr std::uint32_t SHT_DYNSYM = 11;
constexpr std::uint32_t SHT_STRTAB = 3;
constexpr std::uint32_t SHT_PROGBITS = 1;
constexpr std::uint32_t NT_AMDGPU_METADATA = 32;

constexpr std::uint16_t EM_AMDGPU = 0xE0;

template <typename T>
bool ReadAt(std::span<const std::byte> data, std::size_t offset, T* out) {
  if (offset + sizeof(T) > data.size()) return false;
  std::memcpy(out, data.data() + offset, sizeof(T));
  return true;
}

std::string_view GetSectionName(std::span<const std::byte> elf,
                                const Elf64_Shdr& shstrtab,
                                std::uint32_t name_offset) {
  std::size_t str_offset = shstrtab.sh_offset + name_offset;
  if (str_offset >= elf.size()) return "";
  const char* s = reinterpret_cast<const char*>(elf.data() + str_offset);
  std::size_t max_len = elf.size() - str_offset;
  std::size_t len = strnlen(s, max_len);
  return {s, len};
}

std::string_view GetString(std::span<const std::byte> elf,
                           std::uint64_t strtab_offset,
                           std::uint32_t name_offset) {
  std::size_t str_offset = strtab_offset + name_offset;
  if (str_offset >= elf.size()) return "";
  const char* s = reinterpret_cast<const char*>(elf.data() + str_offset);
  std::size_t max_len = elf.size() - str_offset;
  std::size_t len = strnlen(s, max_len);
  return {s, len};
}

KernelArgDescriptor::ValueKind ParseValueKind(std::string_view kind) {
  if (kind == "global_buffer") return KernelArgDescriptor::ValueKind::kGlobalBuffer;
  if (kind == "by_value") return KernelArgDescriptor::ValueKind::kByValue;
  if (kind == "hidden_block_count_x") return KernelArgDescriptor::ValueKind::kHiddenBlockCountX;
  if (kind == "hidden_block_count_y") return KernelArgDescriptor::ValueKind::kHiddenBlockCountY;
  if (kind == "hidden_block_count_z") return KernelArgDescriptor::ValueKind::kHiddenBlockCountZ;
  if (kind == "hidden_group_size_x") return KernelArgDescriptor::ValueKind::kHiddenGroupSizeX;
  if (kind == "hidden_group_size_y") return KernelArgDescriptor::ValueKind::kHiddenGroupSizeY;
  if (kind == "hidden_group_size_z") return KernelArgDescriptor::ValueKind::kHiddenGroupSizeZ;
  if (kind == "hidden_remainder_x") return KernelArgDescriptor::ValueKind::kHiddenRemainderX;
  if (kind == "hidden_remainder_y") return KernelArgDescriptor::ValueKind::kHiddenRemainderY;
  if (kind == "hidden_remainder_z") return KernelArgDescriptor::ValueKind::kHiddenRemainderZ;
  if (kind == "hidden_global_offset_x") return KernelArgDescriptor::ValueKind::kHiddenGlobalOffsetX;
  if (kind == "hidden_global_offset_y") return KernelArgDescriptor::ValueKind::kHiddenGlobalOffsetY;
  if (kind == "hidden_global_offset_z") return KernelArgDescriptor::ValueKind::kHiddenGlobalOffsetZ;
  if (kind == "hidden_grid_dims") return KernelArgDescriptor::ValueKind::kHiddenGridDims;
  return KernelArgDescriptor::ValueKind::kOther;
}

// ---- Minimal msgpack decoder for AMDGPU metadata. ----
// AMDGPU code objects encode metadata as msgpack in the .note section.

struct MsgpackValue {
  enum Type { kNil, kBool, kUint, kInt, kStr, kArray, kMap };
  Type type = kNil;
  std::uint64_t uint_val = 0;
  std::int64_t int_val = 0;
  std::string str_val;
  std::vector<MsgpackValue> array_items;
  std::vector<std::pair<MsgpackValue, MsgpackValue>> map_items;

  std::string_view as_str() const { return str_val; }
  std::uint64_t as_uint() const { return type == kInt ? static_cast<std::uint64_t>(int_val) : uint_val; }

  const MsgpackValue* find(std::string_view key) const {
    if (type != kMap) return nullptr;
    for (const auto& [k, v] : map_items) {
      if (k.type == kStr && k.str_val == key) return &v;
    }
    return nullptr;
  }
};

class MsgpackReader {
 public:
  MsgpackReader(const std::uint8_t* data, std::size_t size)
      : data_(data), size_(size), pos_(0) {}

  bool Read(MsgpackValue* out) {
    if (pos_ >= size_) return false;
    std::uint8_t b = data_[pos_++];

    // Positive fixint (0x00 - 0x7f).
    if (b <= 0x7f) {
      out->type = MsgpackValue::kUint;
      out->uint_val = b;
      return true;
    }
    // Fixmap (0x80 - 0x8f).
    if ((b & 0xf0) == 0x80) {
      return ReadMap(b & 0x0f, out);
    }
    // Fixarray (0x90 - 0x9f).
    if ((b & 0xf0) == 0x90) {
      return ReadArray(b & 0x0f, out);
    }
    // Fixstr (0xa0 - 0xbf).
    if ((b & 0xe0) == 0xa0) {
      return ReadStr(b & 0x1f, out);
    }
    // Negative fixint (0xe0 - 0xff).
    if (b >= 0xe0) {
      out->type = MsgpackValue::kInt;
      out->int_val = static_cast<std::int8_t>(b);
      return true;
    }

    switch (b) {
      case 0xc0: out->type = MsgpackValue::kNil; return true;
      case 0xc2: out->type = MsgpackValue::kBool; out->uint_val = 0; return true;
      case 0xc3: out->type = MsgpackValue::kBool; out->uint_val = 1; return true;
      case 0xcc: return ReadUint(1, out);  // uint8
      case 0xcd: return ReadUint(2, out);  // uint16
      case 0xce: return ReadUint(4, out);  // uint32
      case 0xcf: return ReadUint(8, out);  // uint64
      case 0xd0: return ReadSint(1, out);  // int8
      case 0xd1: return ReadSint(2, out);  // int16
      case 0xd2: return ReadSint(4, out);  // int32
      case 0xd3: return ReadSint(8, out);  // int64
      case 0xd9: { // str8
        if (pos_ >= size_) return false;
        std::uint8_t len = data_[pos_++];
        return ReadStr(len, out);
      }
      case 0xda: { // str16
        if (pos_ + 2 > size_) return false;
        std::uint16_t len = (static_cast<std::uint16_t>(data_[pos_]) << 8) | data_[pos_+1];
        pos_ += 2;
        return ReadStr(len, out);
      }
      case 0xdc: { // array16
        if (pos_ + 2 > size_) return false;
        std::uint16_t count = (static_cast<std::uint16_t>(data_[pos_]) << 8) | data_[pos_+1];
        pos_ += 2;
        return ReadArray(count, out);
      }
      case 0xdd: { // array32
        if (pos_ + 4 > size_) return false;
        std::uint32_t count = ReadBE32();
        return ReadArray(count, out);
      }
      case 0xde: { // map16
        if (pos_ + 2 > size_) return false;
        std::uint16_t count = (static_cast<std::uint16_t>(data_[pos_]) << 8) | data_[pos_+1];
        pos_ += 2;
        return ReadMap(count, out);
      }
      case 0xdf: { // map32
        if (pos_ + 4 > size_) return false;
        std::uint32_t count = ReadBE32();
        return ReadMap(count, out);
      }
      default:
        return false;  // Unsupported msgpack type.
    }
  }

 private:
  bool ReadStr(std::size_t len, MsgpackValue* out) {
    if (pos_ + len > size_) return false;
    out->type = MsgpackValue::kStr;
    out->str_val.assign(reinterpret_cast<const char*>(data_ + pos_), len);
    pos_ += len;
    return true;
  }

  bool ReadUint(int bytes, MsgpackValue* out) {
    if (pos_ + bytes > size_) return false;
    out->type = MsgpackValue::kUint;
    out->uint_val = 0;
    for (int i = 0; i < bytes; ++i) {
      out->uint_val = (out->uint_val << 8) | data_[pos_++];
    }
    return true;
  }

  bool ReadSint(int bytes, MsgpackValue* out) {
    MsgpackValue tmp;
    if (!ReadUint(bytes, &tmp)) return false;
    out->type = MsgpackValue::kInt;
    switch (bytes) {
      case 1: out->int_val = static_cast<std::int8_t>(tmp.uint_val); break;
      case 2: out->int_val = static_cast<std::int16_t>(tmp.uint_val); break;
      case 4: out->int_val = static_cast<std::int32_t>(tmp.uint_val); break;
      case 8: out->int_val = static_cast<std::int64_t>(tmp.uint_val); break;
      default: return false;
    }
    return true;
  }

  bool ReadArray(std::size_t count, MsgpackValue* out) {
    out->type = MsgpackValue::kArray;
    out->array_items.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
      if (!Read(&out->array_items[i])) return false;
    }
    return true;
  }

  bool ReadMap(std::size_t count, MsgpackValue* out) {
    out->type = MsgpackValue::kMap;
    out->map_items.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
      if (!Read(&out->map_items[i].first)) return false;
      if (!Read(&out->map_items[i].second)) return false;
    }
    return true;
  }

  std::uint32_t ReadBE32() {
    std::uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v = (v << 8) | data_[pos_++];
    return v;
  }

  const std::uint8_t* data_;
  std::size_t size_;
  std::size_t pos_;
};

bool ParseMsgpackMetadata(const std::uint8_t* data, std::size_t size,
                          LoadedCodeObject* result) {
  MsgpackReader reader(data, size);
  MsgpackValue root;
  if (!reader.Read(&root) || root.type != MsgpackValue::kMap) {
    result->error_message = "Failed to parse msgpack metadata root";
    return false;
  }

  // Extract target.
  if (auto* target = root.find("amdhsa.target")) {
    result->target = std::string(target->as_str());
  }

  // Extract kernels.
  auto* kernels = root.find("amdhsa.kernels");
  if (!kernels || kernels->type != MsgpackValue::kArray) {
    result->error_message = "No amdhsa.kernels array in metadata";
    return false;
  }

  for (const auto& kv : kernels->array_items) {
    if (kv.type != MsgpackValue::kMap) continue;
    KernelMetadata km;

    if (auto* v = kv.find(".name")) km.name = std::string(v->as_str());
    if (auto* v = kv.find(".symbol")) km.symbol = std::string(v->as_str());
    if (auto* v = kv.find(".wavefront_size")) km.wavefront_size = static_cast<std::uint32_t>(v->as_uint());
    if (auto* v = kv.find(".max_flat_workgroup_size")) km.max_flat_workgroup_size = static_cast<std::uint32_t>(v->as_uint());
    if (auto* v = kv.find(".sgpr_count")) km.sgpr_count = static_cast<std::uint32_t>(v->as_uint());
    if (auto* v = kv.find(".vgpr_count")) km.vgpr_count = static_cast<std::uint32_t>(v->as_uint());
    if (auto* v = kv.find(".group_segment_fixed_size")) km.group_segment_fixed_size = static_cast<std::uint32_t>(v->as_uint());
    if (auto* v = kv.find(".private_segment_fixed_size")) km.private_segment_fixed_size = static_cast<std::uint32_t>(v->as_uint());
    if (auto* v = kv.find(".kernarg_segment_size")) km.kernarg_segment_size = static_cast<std::uint32_t>(v->as_uint());
    if (auto* v = kv.find(".kernarg_segment_align")) km.kernarg_segment_align = static_cast<std::uint32_t>(v->as_uint());

    // Parse args.
    if (auto* args = kv.find(".args")) {
      if (args->type == MsgpackValue::kArray) {
        for (const auto& arg : args->array_items) {
          if (arg.type != MsgpackValue::kMap) continue;
          KernelArgDescriptor desc;
          if (auto* v = arg.find(".name")) desc.name = std::string(v->as_str());
          if (auto* v = arg.find(".offset")) desc.offset = static_cast<std::uint32_t>(v->as_uint());
          if (auto* v = arg.find(".size")) desc.size = static_cast<std::uint32_t>(v->as_uint());
          if (auto* v = arg.find(".value_kind")) desc.value_kind = ParseValueKind(v->as_str());
          km.args.push_back(std::move(desc));
        }
      }
    }

    result->kernels.push_back(std::move(km));
  }

  return !result->kernels.empty();
}

}  // namespace

LoadedCodeObject CodeObjectLoader::Load(std::span<const std::byte> elf_bytes) {
  LoadedCodeObject result;

  if (!ParseElf64Header(elf_bytes, &result)) return result;
  if (!ExtractTextSection(elf_bytes, &result)) return result;
  if (!ExtractMetadataNote(elf_bytes, &result)) return result;
  if (!ResolveKernelSymbols(elf_bytes, &result)) return result;

  return result;
}

LoadedCodeObject CodeObjectLoader::LoadFromFile(std::string_view path) {
  LoadedCodeObject result;

  std::ifstream file(std::string(path), std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    result.error_message = "Failed to open file: " + std::string(path);
    return result;
  }

  auto size = file.tellg();
  file.seekg(0);
  std::vector<std::byte> bytes(static_cast<std::size_t>(size));
  file.read(reinterpret_cast<char*>(bytes.data()),
            static_cast<std::streamsize>(size));

  return Load(bytes);
}

bool CodeObjectLoader::ParseElf64Header(std::span<const std::byte> elf,
                                         LoadedCodeObject* result) {
  if (elf.size() < sizeof(Elf64_Ehdr)) {
    result->error_message = "File too small for ELF header";
    return false;
  }

  Elf64_Ehdr ehdr;
  std::memcpy(&ehdr, elf.data(), sizeof(ehdr));

  // Check ELF magic.
  if (ehdr.e_ident[0] != 0x7f || ehdr.e_ident[1] != 'E' ||
      ehdr.e_ident[2] != 'L' || ehdr.e_ident[3] != 'F') {
    result->error_message = "Not an ELF file (bad magic)";
    return false;
  }

  // Check 64-bit.
  if (ehdr.e_ident[4] != 2) {
    result->error_message = "Not a 64-bit ELF file";
    return false;
  }

  // Check AMDGPU machine.
  if (ehdr.e_machine != EM_AMDGPU) {
    result->error_message = "Not an AMDGPU ELF (e_machine=" +
                            std::to_string(ehdr.e_machine) + ")";
    return false;
  }

  shoff_ = ehdr.e_shoff;
  shentsize_ = ehdr.e_shentsize;
  shnum_ = ehdr.e_shnum;
  shstrndx_ = ehdr.e_shstrndx;

  return true;
}

bool CodeObjectLoader::ExtractTextSection(std::span<const std::byte> elf,
                                           LoadedCodeObject* result) {
  // Read section header string table.
  if (shstrndx_ >= shnum_) {
    result->error_message = "Invalid section header string table index";
    return false;
  }

  Elf64_Shdr shstrtab;
  if (!ReadAt(elf, shoff_ + shstrndx_ * shentsize_, &shstrtab)) {
    result->error_message = "Failed to read shstrtab header";
    return false;
  }

  // Find .text section.
  for (std::uint16_t i = 0; i < shnum_; ++i) {
    Elf64_Shdr shdr;
    if (!ReadAt(elf, shoff_ + i * shentsize_, &shdr)) continue;

    auto name = GetSectionName(elf, shstrtab, shdr.sh_name);
    if (name == ".text" && shdr.sh_type == SHT_PROGBITS) {
      if (shdr.sh_offset + shdr.sh_size > elf.size()) {
        result->error_message = ".text section extends past end of file";
        return false;
      }
      if (shdr.sh_size % 4 != 0) {
        result->error_message = ".text section size is not 4-byte aligned";
        return false;
      }

      result->text_vaddr = shdr.sh_addr;
      result->code_words.resize(shdr.sh_size / 4);
      std::memcpy(result->code_words.data(),
                  elf.data() + shdr.sh_offset,
                  shdr.sh_size);
      return true;
    }
  }

  result->error_message = "No .text section found";
  return false;
}

bool CodeObjectLoader::ExtractMetadataNote(std::span<const std::byte> elf,
                                            LoadedCodeObject* result) {
  Elf64_Shdr shstrtab;
  ReadAt(elf, shoff_ + shstrndx_ * shentsize_, &shstrtab);

  for (std::uint16_t i = 0; i < shnum_; ++i) {
    Elf64_Shdr shdr;
    if (!ReadAt(elf, shoff_ + i * shentsize_, &shdr)) continue;

    if (shdr.sh_type != SHT_NOTE) continue;

    // Walk through note entries.
    std::size_t note_offset = shdr.sh_offset;
    std::size_t note_end = shdr.sh_offset + shdr.sh_size;

    while (note_offset + sizeof(Elf64_Nhdr) <= note_end) {
      Elf64_Nhdr nhdr;
      if (!ReadAt(elf, note_offset, &nhdr)) break;
      note_offset += sizeof(Elf64_Nhdr);

      // Align name and desc to 4 bytes.
      std::uint32_t namesz_aligned = (nhdr.n_namesz + 3) & ~3u;
      std::uint32_t descsz_aligned = (nhdr.n_descsz + 3) & ~3u;

      if (nhdr.n_type == NT_AMDGPU_METADATA) {
        // The name should be "AMDGPU\0".
        std::size_t desc_offset = note_offset + namesz_aligned;
        if (desc_offset + nhdr.n_descsz > elf.size()) {
          result->error_message = "AMDGPU metadata note extends past EOF";
          return false;
        }

        const auto* desc_data =
            reinterpret_cast<const std::uint8_t*>(elf.data() + desc_offset);
        return ParseMsgpackMetadata(desc_data, nhdr.n_descsz, result);
      }

      note_offset += namesz_aligned + descsz_aligned;
    }
  }

  result->error_message = "No AMDGPU metadata note found";
  return false;
}

bool CodeObjectLoader::ResolveKernelSymbols(std::span<const std::byte> elf,
                                             LoadedCodeObject* result) {
  Elf64_Shdr shstrtab;
  ReadAt(elf, shoff_ + shstrndx_ * shentsize_, &shstrtab);

  // Find .dynsym and .dynstr, or .symtab and .strtab.
  Elf64_Shdr symtab_hdr{};
  std::uint64_t strtab_offset = 0;
  bool found_symtab = false;

  for (std::uint16_t i = 0; i < shnum_; ++i) {
    Elf64_Shdr shdr;
    if (!ReadAt(elf, shoff_ + i * shentsize_, &shdr)) continue;

    if (shdr.sh_type == SHT_DYNSYM) {
      symtab_hdr = shdr;
      // Get linked string table.
      Elf64_Shdr strtab;
      if (ReadAt(elf, shoff_ + shdr.sh_link * shentsize_, &strtab)) {
        strtab_offset = strtab.sh_offset;
        found_symtab = true;
      }
    }
  }

  if (!found_symtab) {
    // Try .symtab.
    for (std::uint16_t i = 0; i < shnum_; ++i) {
      Elf64_Shdr shdr;
      if (!ReadAt(elf, shoff_ + i * shentsize_, &shdr)) continue;

      if (shdr.sh_type == SHT_SYMTAB) {
        symtab_hdr = shdr;
        Elf64_Shdr strtab;
        if (ReadAt(elf, shoff_ + shdr.sh_link * shentsize_, &strtab)) {
          strtab_offset = strtab.sh_offset;
          found_symtab = true;
        }
        break;
      }
    }
  }

  if (!found_symtab) {
    // No symbol table — leave kernels without entry info.
    return true;
  }

  // Iterate symbols and match to kernels.
  std::size_t sym_count = 0;
  if (symtab_hdr.sh_entsize > 0) {
    sym_count = symtab_hdr.sh_size / symtab_hdr.sh_entsize;
  }

  for (std::size_t s = 0; s < sym_count; ++s) {
    Elf64_Sym sym;
    if (!ReadAt(elf, symtab_hdr.sh_offset + s * symtab_hdr.sh_entsize, &sym))
      continue;

    auto sym_name = GetString(elf, strtab_offset, sym.st_name);
    std::uint8_t sym_type = sym.st_info & 0xf;

    // Match FUNC symbols to kernel names.
    if (sym_type == 2 /* STT_FUNC */ && sym.st_size > 0) {
      for (auto& kernel : result->kernels) {
        if (sym_name == kernel.name) {
          kernel.entry_vaddr = sym.st_value;
          kernel.entry_size = sym.st_size;
          // Compute offset within code_words.
          if (sym.st_value >= result->text_vaddr) {
            kernel.entry_offset_in_text =
                sym.st_value - result->text_vaddr;
          }
        }
      }
    }
  }

  return true;
}

}  // namespace mirage::sim::exec::launch
