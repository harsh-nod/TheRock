#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "lib/sim/exec/launch/code_object_loader.h"
#include "lib/sim/isa/gfx1201/binary_decoder.h"

namespace {

bool Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    return false;
  }
  return true;
}

using mirage::sim::isa::DecodedInstruction;
using mirage::sim::isa::Gfx1201BinaryDecoder;
using mirage::sim::isa::OperandKind;

// Helper to build an instruction word with specific bit fields.
constexpr std::uint32_t MakeBits(std::uint32_t value, std::uint32_t offset,
                                  std::uint32_t width) {
  return (value & ((1u << width) - 1u)) << offset;
}

// --- SOPP encoding helpers ---
// SOPP: bits[31:23]=0x17f, bits[22:16]=opcode, bits[15:0]=SIMM16.
constexpr std::uint32_t kEncSopp = 0x17f;
constexpr std::uint32_t MakeSopp(std::uint32_t opcode, std::uint16_t simm16) {
  return MakeBits(kEncSopp, 23, 9) | MakeBits(opcode, 16, 7) |
         static_cast<std::uint32_t>(simm16);
}

// --- VOP2 encoding helpers ---
// VOP2: bit[31]=0, bits[30:25]=opcode, bits[24:17]=VDST, bits[16:9]=VSRC1,
//       bits[8:0]=SRC0.
constexpr std::uint32_t MakeVop2(std::uint32_t opcode, std::uint32_t vdst,
                                  std::uint32_t vsrc1, std::uint32_t src0) {
  return MakeBits(opcode, 25, 6) | MakeBits(vdst, 17, 8) |
         MakeBits(vsrc1, 9, 8) | MakeBits(src0, 0, 9);
}

// --- SMEM encoding helpers ---
// SMEM: bits[31:26]=0x3d, bits[25:18]=opcode, bits[13:7]=SDST, bits[6:0]=SBASE.
constexpr std::uint32_t kEncSmem = 0x3d;
constexpr std::uint32_t MakeSmemWord0(std::uint32_t opcode, std::uint32_t sdst,
                                       std::uint32_t sbase) {
  return MakeBits(kEncSmem, 26, 6) | MakeBits(opcode, 18, 8) |
         MakeBits(sdst, 7, 7) | MakeBits(sbase, 0, 7);
}

// --- VGLOBAL encoding helpers (gfx12 3-dword format) ---
// Word 0: bits[31:26]=0x3b, bits[24:14]=opcode (11 bits),
//         bits[13:0]=OFFSET (14-bit signed).
// Word 1: bits[7:0]=VDST (loads) or bits[31:24]=VDATA (stores).
// Word 2: bits[7:0]=VADDR, bits[14:8]=SADDR (0x7f=off).
constexpr std::uint32_t kEncVglobal = 59;
constexpr std::uint32_t MakeVglobalWord0(std::uint32_t opcode,
                                          std::uint32_t offset) {
  return MakeBits(kEncVglobal, 26, 6) | MakeBits(opcode, 14, 11) |
         MakeBits(offset, 0, 14);
}

// Word 1 for loads: VDST at bits[7:0].
constexpr std::uint32_t MakeVglobalLoadWord1(std::uint32_t vdst) {
  return MakeBits(vdst, 0, 8);
}

// Word 1 for stores: VDATA at bits[31:24].
constexpr std::uint32_t MakeVglobalStoreWord1(std::uint32_t vdata) {
  return MakeBits(vdata, 24, 8);
}

// Word 2: VADDR + SADDR.
constexpr std::uint32_t MakeVglobalWord2(std::uint32_t saddr,
                                          std::uint32_t vaddr) {
  return MakeBits(saddr, 8, 7) | MakeBits(vaddr, 0, 8);
}

// --- Tests ---

bool TestDecodeSoppSchedulingHints() {
  Gfx1201BinaryDecoder decoder;

  // S_CLAUSE (opcode 5): 1 dword.
  {
    std::uint32_t word = MakeSopp(5, 3);  // clause count = 3
    DecodedInstruction instr;
    std::size_t consumed = 0;
    std::string error;
    if (!Expect(decoder.DecodeInstruction({&word, 1}, &instr, &consumed, &error),
                ("S_CLAUSE decode failed: " + error).c_str()))
      return false;
    if (!Expect(instr.opcode == "S_CLAUSE", "opcode should be S_CLAUSE"))
      return false;
    if (!Expect(consumed == 1, "S_CLAUSE should consume 1 word"))
      return false;
    if (!Expect(instr.operand_count == 1, "S_CLAUSE should have 1 operand"))
      return false;
  }

  // S_DELAY_ALU (opcode 7).
  {
    std::uint32_t word = MakeSopp(7, 0);
    DecodedInstruction instr;
    std::size_t consumed = 0;
    std::string error;
    if (!Expect(decoder.DecodeInstruction({&word, 1}, &instr, &consumed, &error),
                ("S_DELAY_ALU decode failed: " + error).c_str()))
      return false;
    if (!Expect(instr.opcode == "S_DELAY_ALU", "opcode should be S_DELAY_ALU"))
      return false;
  }

  // S_WAIT_KMCNT (opcode 71).
  {
    std::uint32_t word = MakeSopp(71, 0);
    DecodedInstruction instr;
    std::size_t consumed = 0;
    std::string error;
    if (!Expect(decoder.DecodeInstruction({&word, 1}, &instr, &consumed, &error),
                ("S_WAIT_KMCNT decode failed: " + error).c_str()))
      return false;
    if (!Expect(instr.opcode == "S_WAIT_KMCNT", "opcode should be S_WAIT_KMCNT"))
      return false;
  }

  // S_CODE_END (opcode 31).
  {
    std::uint32_t word = MakeSopp(31, 0);
    DecodedInstruction instr;
    std::size_t consumed = 0;
    std::string error;
    if (!Expect(decoder.DecodeInstruction({&word, 1}, &instr, &consumed, &error),
                ("S_CODE_END decode failed: " + error).c_str()))
      return false;
    if (!Expect(instr.opcode == "S_CODE_END", "opcode should be S_CODE_END"))
      return false;
    if (!Expect(instr.operand_count == 0, "S_CODE_END should have 0 operands"))
      return false;
  }

  return true;
}

bool TestDecodeSmem() {
  Gfx1201BinaryDecoder decoder;

  // S_LOAD_B32 (opcode 0): SDST=s4, SBASE=s[0:1] (pair 0), offset=16.
  {
    std::uint32_t words[2] = {
        MakeSmemWord0(0, 4, 0),  // S_LOAD_B32, sdst=4, sbase=0
        16u,                      // offset=16
    };
    DecodedInstruction instr;
    std::size_t consumed = 0;
    std::string error;
    if (!Expect(decoder.DecodeInstruction(words, &instr, &consumed, &error),
                ("S_LOAD_B32 decode failed: " + error).c_str()))
      return false;
    if (!Expect(instr.opcode == "S_LOAD_B32", "opcode should be S_LOAD_B32"))
      return false;
    if (!Expect(consumed == 2, "S_LOAD_B32 should consume 2 words"))
      return false;
    if (!Expect(instr.operand_count == 3,
                "S_LOAD_B32 should have 3 operands (dst, sbase, offset)"))
      return false;
    // Check destination register.
    if (!Expect(instr.operands[0].kind == OperandKind::kSgpr &&
                    instr.operands[0].index == 4,
                "S_LOAD_B32 dst should be s4"))
      return false;
    // Check source base register (sbase * 2 = 0).
    if (!Expect(instr.operands[1].kind == OperandKind::kSgpr &&
                    instr.operands[1].index == 0,
                "S_LOAD_B32 sbase should be s0"))
      return false;
    // Check offset.
    if (!Expect(instr.operands[2].kind == OperandKind::kImm32 &&
                    instr.operands[2].imm32 == 16,
                "S_LOAD_B32 offset should be 16"))
      return false;
  }

  // S_LOAD_B128 (opcode 2): SDST=s[8:11], SBASE=s[2:3] (pair 1), offset=0.
  {
    std::uint32_t words[2] = {
        MakeSmemWord0(2, 8, 1),  // S_LOAD_B128, sdst=8, sbase=1 (pair → s2)
        0u,                       // offset=0
    };
    DecodedInstruction instr;
    std::size_t consumed = 0;
    std::string error;
    if (!Expect(decoder.DecodeInstruction(words, &instr, &consumed, &error),
                ("S_LOAD_B128 decode failed: " + error).c_str()))
      return false;
    if (!Expect(instr.opcode == "S_LOAD_B128", "opcode should be S_LOAD_B128"))
      return false;
    if (!Expect(instr.operands[0].index == 8,
                "S_LOAD_B128 dst should be s8"))
      return false;
    if (!Expect(instr.operands[1].index == 2,
                "S_LOAD_B128 sbase should be s2 (pair 1)"))
      return false;
  }

  return true;
}

bool TestDecodeVglobalLoad() {
  Gfx1201BinaryDecoder decoder;

  // GLOBAL_LOAD_B32 (opcode 20): VDST=v5, VADDR=v[2:3], SADDR=off (0x7f).
  {
    std::uint32_t words[3] = {
        MakeVglobalWord0(20, 0),           // GLOBAL_LOAD_B32, offset=0
        MakeVglobalLoadWord1(5),           // vdst=5
        MakeVglobalWord2(0x7f, 2),         // saddr=off, vaddr=2
    };
    DecodedInstruction instr;
    std::size_t consumed = 0;
    std::string error;
    if (!Expect(decoder.DecodeInstruction(words, &instr, &consumed, &error),
                ("GLOBAL_LOAD_B32 decode failed: " + error).c_str()))
      return false;
    if (!Expect(instr.opcode == "GLOBAL_LOAD_B32",
                "opcode should be GLOBAL_LOAD_B32"))
      return false;
    if (!Expect(consumed == 3, "GLOBAL_LOAD_B32 should consume 3 words"))
      return false;
    // Without SADDR, operand_count = 3 (dst, vaddr, offset).
    if (!Expect(instr.operand_count == 3,
                "GLOBAL_LOAD_B32 without saddr should have 3 operands"))
      return false;
    if (!Expect(instr.operands[0].kind == OperandKind::kVgpr &&
                    instr.operands[0].index == 5,
                "GLOBAL_LOAD_B32 dst should be v5"))
      return false;
    if (!Expect(instr.operands[1].kind == OperandKind::kVgpr &&
                    instr.operands[1].index == 2,
                "GLOBAL_LOAD_B32 vaddr should be v2"))
      return false;
  }

  // GLOBAL_LOAD_B32 with SADDR: VDST=v5, VADDR=v2, SADDR=s[4:5].
  {
    std::uint32_t words[3] = {
        MakeVglobalWord0(20, 8),           // GLOBAL_LOAD_B32, offset=8
        MakeVglobalLoadWord1(5),           // vdst=5
        MakeVglobalWord2(2, 2),            // saddr=2 (pair->s4), vaddr=2
    };
    DecodedInstruction instr;
    std::size_t consumed = 0;
    std::string error;
    if (!Expect(decoder.DecodeInstruction(words, &instr, &consumed, &error),
                ("GLOBAL_LOAD_B32 saddr decode failed: " + error).c_str()))
      return false;
    if (!Expect(instr.operand_count == 4,
                "GLOBAL_LOAD_B32 with saddr should have 4 operands"))
      return false;
    // Check saddr register.
    if (!Expect(instr.operands[2].kind == OperandKind::kSgpr &&
                    instr.operands[2].index == 4,
                "GLOBAL_LOAD_B32 saddr should be s4 (pair 2)"))
      return false;
    // Check offset.
    if (!Expect(instr.operands[3].kind == OperandKind::kImm32 &&
                    instr.operands[3].imm32 == 8,
                "GLOBAL_LOAD_B32 offset should be 8"))
      return false;
  }

  return true;
}

bool TestDecodeVglobalStore() {
  Gfx1201BinaryDecoder decoder;

  // GLOBAL_STORE_B32 (opcode 26): VDATA=v0, VADDR=v[2:3], SADDR=off.
  {
    std::uint32_t words[3] = {
        MakeVglobalWord0(26, 0),           // GLOBAL_STORE_B32, offset=0
        MakeVglobalStoreWord1(0),          // vdata=v0
        MakeVglobalWord2(0x7f, 2),         // saddr=off, vaddr=2
    };
    DecodedInstruction instr;
    std::size_t consumed = 0;
    std::string error;
    if (!Expect(decoder.DecodeInstruction(words, &instr, &consumed, &error),
                ("GLOBAL_STORE_B32 decode failed: " + error).c_str()))
      return false;
    if (!Expect(instr.opcode == "GLOBAL_STORE_B32",
                "opcode should be GLOBAL_STORE_B32"))
      return false;
    if (!Expect(consumed == 3, "GLOBAL_STORE_B32 should consume 3 words"))
      return false;
    if (!Expect(instr.operand_count == 3,
                "GLOBAL_STORE_B32 without saddr should have 3 operands"))
      return false;
  }

  return true;
}

bool TestDecodeVop2Additions() {
  Gfx1201BinaryDecoder decoder;

  // V_ADD_F32 (VOP2 opcode 3): VDST=v0, SRC0=v0, VSRC1=v1.
  // SRC0=v0 → encoded as 256+0=256 in VOP2 SRC0 field.
  {
    std::uint32_t word = MakeVop2(3, 0, 1, 256);  // V_ADD_F32 v0, v0, v1
    DecodedInstruction instr;
    std::size_t consumed = 0;
    std::string error;
    if (!Expect(decoder.DecodeInstruction({&word, 1}, &instr, &consumed, &error),
                ("V_ADD_F32 decode failed: " + error).c_str()))
      return false;
    if (!Expect(instr.opcode == "V_ADD_F32", "opcode should be V_ADD_F32"))
      return false;
    if (!Expect(consumed == 1, "V_ADD_F32 should consume 1 word"))
      return false;
    if (!Expect(instr.operand_count == 3,
                "V_ADD_F32 should have 3 operands (dst, src0, src1)"))
      return false;
    if (!Expect(instr.operands[0].kind == OperandKind::kVgpr &&
                    instr.operands[0].index == 0,
                "V_ADD_F32 dst should be v0"))
      return false;
    if (!Expect(instr.operands[1].kind == OperandKind::kVgpr &&
                    instr.operands[1].index == 0,
                "V_ADD_F32 src0 should be v0"))
      return false;
    if (!Expect(instr.operands[2].kind == OperandKind::kVgpr &&
                    instr.operands[2].index == 1,
                "V_ADD_F32 src1 should be v1"))
      return false;
  }

  // V_LSHLREV_B64 (VOP2 opcode 31): VDST=v[6:7], SRC0=2 (inline), VSRC1=v4.
  // SRC0=2 → inline integer encoded as 128+2=130.
  {
    std::uint32_t word = MakeVop2(31, 6, 4, 130);
    DecodedInstruction instr;
    std::size_t consumed = 0;
    std::string error;
    if (!Expect(decoder.DecodeInstruction({&word, 1}, &instr, &consumed, &error),
                ("V_LSHLREV_B64 decode failed: " + error).c_str()))
      return false;
    if (!Expect(instr.opcode == "V_LSHLREV_B64",
                "opcode should be V_LSHLREV_B64"))
      return false;
    if (!Expect(consumed == 1, "V_LSHLREV_B64 should consume 1 word"))
      return false;
    if (!Expect(instr.operands[0].index == 6,
                "V_LSHLREV_B64 dst should be v6"))
      return false;
    // src0 should be immediate value 2.
    if (!Expect(instr.operands[1].kind == OperandKind::kImm32 &&
                    instr.operands[1].imm32 == 2,
                "V_LSHLREV_B64 src0 should be 2"))
      return false;
  }

  // V_ADD_CO_CI_U32 (VOP2 opcode 32): VDST=v5, SRC0=v3, VSRC1=v4.
  {
    std::uint32_t word = MakeVop2(32, 5, 4, 256 + 3);
    DecodedInstruction instr;
    std::size_t consumed = 0;
    std::string error;
    if (!Expect(decoder.DecodeInstruction({&word, 1}, &instr, &consumed, &error),
                ("V_ADD_CO_CI_U32 decode failed: " + error).c_str()))
      return false;
    if (!Expect(instr.opcode == "V_ADD_CO_CI_U32",
                "opcode should be V_ADD_CO_CI_U32"))
      return false;
  }

  return true;
}

bool TestDecodeMultipleInstructions() {
  Gfx1201BinaryDecoder decoder;

  // A short program: S_CLAUSE, S_LOAD_B128, S_WAIT_KMCNT, S_ENDPGM.
  std::vector<std::uint32_t> program;
  program.push_back(MakeSopp(5, 1));             // S_CLAUSE count=1
  program.push_back(MakeSmemWord0(2, 8, 1));     // S_LOAD_B128 s[8:11], s[2:3]
  program.push_back(0);                           // offset=0
  program.push_back(MakeSopp(71, 0));             // S_WAIT_KMCNT 0
  program.push_back(MakeSopp(48, 0));             // S_ENDPGM (opcode 48)

  std::vector<DecodedInstruction> decoded;
  std::string error;
  if (!Expect(decoder.DecodeProgram(program, &decoded, &error),
              ("DecodeProgram failed: " + error).c_str()))
    return false;

  if (!Expect(decoded.size() == 4,
              ("expected 4 instructions, got " +
               std::to_string(decoded.size())).c_str()))
    return false;

  if (!Expect(decoded[0].opcode == "S_CLAUSE", "first should be S_CLAUSE"))
    return false;
  if (!Expect(decoded[1].opcode == "S_LOAD_B128", "second should be S_LOAD_B128"))
    return false;
  if (!Expect(decoded[2].opcode == "S_WAIT_KMCNT", "third should be S_WAIT_KMCNT"))
    return false;
  if (!Expect(decoded[3].opcode == "S_ENDPGM", "fourth should be S_ENDPGM"))
    return false;

  return true;
}

bool TestDecodeRealVectorAddPartial() {
  // Test that we can at least identify known opcodes from a code object.
  // We use the code_object_loader to extract the .text section, then try
  // to decode individual instructions we know should be present.
  using mirage::sim::exec::launch::CodeObjectLoader;

  std::string candidates[] = {
      "mirage/native/tests/data/code_objects/gfx1201/vector_add/vector_add.co",
      "native/tests/data/code_objects/gfx1201/vector_add/vector_add.co",
      "../native/tests/data/code_objects/gfx1201/vector_add/vector_add.co",
      "tests/data/code_objects/gfx1201/vector_add/vector_add.co",
      "../tests/data/code_objects/gfx1201/vector_add/vector_add.co",
  };

  std::string fixture_path;
  for (const auto& path : candidates) {
    if (std::filesystem::exists(path)) {
      fixture_path = path;
      break;
    }
  }

  if (fixture_path.empty()) {
    // Cannot find fixture - this test is optional.
    std::cerr << "INFO: Skipping real binary decode (fixture not found)\n";
    return true;
  }

  CodeObjectLoader loader;
  auto co = loader.LoadFromFile(fixture_path);
  if (!Expect(co.ok(), ("load failed: " + co.error_message).c_str()))
    return false;

  if (!Expect(!co.code_words.empty(), "code_words should not be empty"))
    return false;

  // Try to decode individual instructions from the beginning of the text.
  Gfx1201BinaryDecoder decoder;

  // Count how many instructions we can decode before hitting an unsupported one.
  std::size_t offset = 0;
  std::size_t decoded_count = 0;
  std::vector<std::string> decoded_opcodes;

  while (offset < co.code_words.size()) {
    DecodedInstruction instr;
    std::size_t consumed = 0;
    std::string error;
    bool ok = decoder.DecodeInstruction(
        std::span<const std::uint32_t>(co.code_words.data() + offset,
                                        co.code_words.size() - offset),
        &instr, &consumed, &error);
    if (!ok) {
      // Record where we stopped.
      std::cerr << "INFO: Stopped decoding at word offset " << offset
                << " (" << decoded_count << " instructions decoded): "
                << error << '\n';
      break;
    }
    if (consumed == 0) break;

    decoded_opcodes.push_back(std::string(instr.opcode));
    offset += consumed;
    ++decoded_count;

    // Stop after S_ENDPGM.
    if (instr.opcode == "S_ENDPGM") break;
  }

  // We should decode at least a few instructions.
  if (!Expect(decoded_count >= 5,
              ("expected >= 5 decoded instructions, got " +
               std::to_string(decoded_count)).c_str()))
    return false;

  // Print all decoded opcodes for diagnostic purposes.
  std::cerr << "Decoded " << decoded_count << " instructions from vector_add:\n";
  for (std::size_t i = 0; i < decoded_opcodes.size(); ++i) {
    std::cerr << "  [" << i << "] " << decoded_opcodes[i] << '\n';
  }

  return true;
}

bool TestSupportedOpcodeCheck() {
  Gfx1201BinaryDecoder decoder;

  // New opcodes should be supported.
  if (!Expect(decoder.SupportsPhase0ExecutableOpcode("S_CLAUSE"),
              "S_CLAUSE should be supported"))
    return false;
  if (!Expect(decoder.SupportsPhase0ExecutableOpcode("S_LOAD_B32"),
              "S_LOAD_B32 should be supported"))
    return false;
  if (!Expect(decoder.SupportsPhase0ExecutableOpcode("GLOBAL_LOAD_B32"),
              "GLOBAL_LOAD_B32 should be supported"))
    return false;
  if (!Expect(decoder.SupportsPhase0ExecutableOpcode("V_ADD_F32"),
              "V_ADD_F32 should be supported"))
    return false;
  if (!Expect(decoder.SupportsPhase0ExecutableOpcode("V_ADD_CO_CI_U32"),
              "V_ADD_CO_CI_U32 should be supported"))
    return false;

  // Existing opcodes should still work.
  if (!Expect(decoder.SupportsPhase0ExecutableOpcode("S_ENDPGM"),
              "S_ENDPGM should be supported"))
    return false;
  if (!Expect(decoder.SupportsPhase0ExecutableOpcode("V_MOV_B32"),
              "V_MOV_B32 should be supported"))
    return false;

  // V_ADD_NC_U32 should normalize to V_ADD_U32 and be supported.
  if (!Expect(decoder.SupportsPhase0ExecutableOpcode("V_ADD_NC_U32"),
              "V_ADD_NC_U32 should normalize and be supported"))
    return false;

  return true;
}

}  // namespace

int main() {
  bool ok = true;
  ok = TestDecodeSoppSchedulingHints() && ok;
  ok = TestDecodeSmem() && ok;
  ok = TestDecodeVglobalLoad() && ok;
  ok = TestDecodeVglobalStore() && ok;
  ok = TestDecodeVop2Additions() && ok;
  ok = TestDecodeMultipleInstructions() && ok;
  ok = TestDecodeRealVectorAddPartial() && ok;
  ok = TestSupportedOpcodeCheck() && ok;

  if (ok) {
    std::cerr << "All binary_decoder_expansion tests passed.\n";
  }
  return ok ? 0 : 1;
}
