#include "lib/sim/isa/gfx1201/binary_decoder.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>

#include "lib/sim/isa/common/wave_execution_state.h"

namespace mirage::sim::isa {
namespace {

constexpr std::uint16_t kImplicitVccPairSgprIndex = 248;
constexpr std::uint16_t kSrcVcczSgprIndex = 251;
constexpr std::uint16_t kSrcExeczSgprIndex = 252;
constexpr std::uint16_t kSrcSccSgprIndex = 253;

constexpr std::array<std::string_view, 214> kPhase0ExecutableOpcodes{{
    "S_ENDPGM",
    "S_NOP",
    "S_ADD_U32",
    "S_ADD_I32",
    "S_SUB_U32",
    "S_CMP_EQ_I32",
    "S_CMP_LG_I32",
    "S_CMP_GT_I32",
    "S_CMP_EQ_U32",
    "S_CMP_LG_U32",
    "S_CMP_GE_I32",
    "S_CMP_LT_I32",
    "S_CMP_LE_I32",
    "S_CMP_GT_U32",
    "S_CMP_GE_U32",
    "S_CMP_LT_U32",
    "S_CMP_LE_U32",
    "S_BRANCH",
    "S_CBRANCH_SCC0",
    "S_CBRANCH_SCC1",
    "S_CBRANCH_VCCZ",
    "S_CBRANCH_VCCNZ",
    "S_CBRANCH_EXECZ",
    "S_CBRANCH_EXECNZ",
    "S_MOV_B32",
    "S_MOVK_I32",
    "V_MOV_B32",
    "V_CMP_EQ_I32",
    "V_CMP_NE_I32",
    "V_CMP_LT_I32",
    "V_CMP_LE_I32",
    "V_CMP_GT_I32",
    "V_CMP_GE_I32",
    "V_CMP_EQ_U32",
    "V_CMP_NE_U32",
    "V_CMP_LT_U32",
    "V_CMP_LE_U32",
    "V_CMP_GT_U32",
    "V_CMP_GE_U32",
    "V_CMPX_EQ_I32",
    "V_CMPX_NE_I32",
    "V_CMPX_LT_I32",
    "V_CMPX_LE_I32",
    "V_CMPX_GT_I32",
    "V_CMPX_GE_I32",
    "V_CMPX_EQ_U32",
    "V_CMPX_NE_U32",
    "V_CMPX_LT_U32",
    "V_CMPX_LE_U32",
    "V_CMPX_GT_U32",
    "V_CMPX_GE_U32",
    "V_CMP_EQ_I64",
    "V_CMP_NE_I64",
    "V_CMP_LT_I64",
    "V_CMP_LE_I64",
    "V_CMP_GT_I64",
    "V_CMP_GE_I64",
    "V_CMP_EQ_U64",
    "V_CMP_NE_U64",
    "V_CMP_LT_U64",
    "V_CMP_LE_U64",
    "V_CMP_GT_U64",
    "V_CMP_GE_U64",
    "V_CMPX_EQ_I64",
    "V_CMPX_NE_I64",
    "V_CMPX_LT_I64",
    "V_CMPX_LE_I64",
    "V_CMPX_GT_I64",
    "V_CMPX_GE_I64",
    "V_CMPX_EQ_U64",
    "V_CMPX_NE_U64",
    "V_CMPX_LT_U64",
    "V_CMPX_LE_U64",
    "V_CMPX_GT_U64",
    "V_CMPX_GE_U64",
    "V_CMP_CLASS_F32",
    "V_CMP_EQ_F32",
    "V_CMP_GE_F32",
    "V_CMP_GT_F32",
    "V_CMP_LE_F32",
    "V_CMP_LG_F32",
    "V_CMP_LT_F32",
    "V_CMP_NEQ_F32",
    "V_CMP_O_F32",
    "V_CMP_U_F32",
    "V_CMP_NGE_F32",
    "V_CMP_NLG_F32",
    "V_CMP_NGT_F32",
    "V_CMP_NLE_F32",
    "V_CMP_NLT_F32",
    "V_CMPX_CLASS_F32",
    "V_CMPX_EQ_F32",
    "V_CMPX_GE_F32",
    "V_CMPX_GT_F32",
    "V_CMPX_LE_F32",
    "V_CMPX_LG_F32",
    "V_CMPX_LT_F32",
    "V_CMPX_NEQ_F32",
    "V_CMPX_O_F32",
    "V_CMPX_U_F32",
    "V_CMPX_NGE_F32",
    "V_CMPX_NLG_F32",
    "V_CMPX_NGT_F32",
    "V_CMPX_NLE_F32",
    "V_CMPX_NLT_F32",
    "V_CMP_CLASS_F64",
    "V_CMP_EQ_F64",
    "V_CMP_GE_F64",
    "V_CMP_GT_F64",
    "V_CMP_LE_F64",
    "V_CMP_LG_F64",
    "V_CMP_LT_F64",
    "V_CMP_NEQ_F64",
    "V_CMP_O_F64",
    "V_CMP_U_F64",
    "V_CMP_NGE_F64",
    "V_CMP_NLG_F64",
    "V_CMP_NGT_F64",
    "V_CMP_NLE_F64",
    "V_CMP_NLT_F64",
    "V_CMPX_CLASS_F64",
    "V_CMPX_EQ_F64",
    "V_CMPX_GE_F64",
    "V_CMPX_GT_F64",
    "V_CMPX_LE_F64",
    "V_CMPX_LG_F64",
    "V_CMPX_LT_F64",
    "V_CMPX_NEQ_F64",
    "V_CMPX_O_F64",
    "V_CMPX_U_F64",
    "V_CMPX_NGE_F64",
    "V_CMPX_NLG_F64",
    "V_CMPX_NGT_F64",
    "V_CMPX_NLE_F64",
    "V_CMPX_NLT_F64",
    "V_NOT_B32",
    "V_BFREV_B32",
    "V_CVT_F32_UBYTE0",
    "V_CVT_F32_UBYTE1",
    "V_CVT_F32_UBYTE2",
    "V_CVT_F32_UBYTE3",
    "V_CVT_F32_I32",
    "V_CVT_F32_U32",
    "V_CVT_F32_F64",
    "V_CVT_F64_F32",
    "V_CVT_F64_I32",
    "V_CVT_F64_U32",
    "V_CVT_U32_F32",
    "V_CVT_U32_F64",
    "V_CVT_I32_F32",
    "V_CVT_I32_F64",
    "V_ADD_U32",
    "V_SUB_U32",
    "V_SUBREV_U32",
    "V_MIN_I32",
    "V_MAX_I32",
    "V_MIN_U32",
    "V_MAX_U32",
    "V_CNDMASK_B32",
    "V_LSHRREV_B32",
    "V_ASHRREV_I32",
    "V_LSHLREV_B32",
    "V_AND_B32",
    "V_OR_B32",
    "V_XOR_B32",
    // VOP2 additions: vector ALU used by vector_add kernel.
    "V_ADD_F32",
    "V_SUB_F32",
    "V_MUL_F32",
    "V_LSHLREV_B64",
    "V_ADD_CO_CI_U32",
    // SOPP scheduling hints: gfx12-specific, no gfx950 equivalent.
    "S_CLAUSE",
    "S_DELAY_ALU",
    "S_WAIT_KMCNT",
    "S_WAIT_ALU",
    "S_WAIT_LOADCNT",
    "S_CODE_END",
    // SMEM: scalar memory loads.
    "S_LOAD_B32",
    "S_LOAD_B64",
    "S_LOAD_B128",
    "S_LOAD_B256",
    "S_BUFFER_LOAD_B32",
    "S_BUFFER_LOAD_B64",
    "S_BUFFER_LOAD_B128",
    "S_BUFFER_LOAD_B256",
    // VGLOBAL: global memory loads and stores.
    "GLOBAL_LOAD_B32",
    "GLOBAL_LOAD_B64",
    "GLOBAL_LOAD_B128",
    "GLOBAL_STORE_B32",
    "GLOBAL_STORE_B64",
    "GLOBAL_STORE_B128",
    "GLOBAL_LOAD_U8",
    "GLOBAL_LOAD_I8",
    "GLOBAL_LOAD_U16",
    "GLOBAL_LOAD_I16",
    "GLOBAL_STORE_B8",
    "GLOBAL_STORE_B16",
    // SOP2: scalar ALU binary operations.
    "S_AND_B32",
    "S_AND_B64",
    "S_OR_B32",
    "S_OR_B64",
    "S_XOR_B32",
    "S_XOR_B64",
    "S_ADD_U32",
    "S_SUB_U32",
    "S_ADD_I32",
    "S_SUB_I32",
    "S_LSHL_B32",
    "S_LSHR_B32",
    "S_ASHR_I32",
    "S_MUL_I32",
    "S_CSELECT_B32",
    "S_CSELECT_B64",
    // VOP3-only: gfx12-specific multiply-add and add-with-carry.
    "V_MAD_CO_U64_U32",
    "V_ADD_CO_U32",
}};

constexpr std::uint32_t ExtractBits(std::uint32_t value,
                                    std::uint32_t bit_offset,
                                    std::uint32_t bit_count) {
  if (bit_count == 32) {
    return value;
  }
  return (value >> bit_offset) & ((1u << bit_count) - 1u);
}

constexpr bool IsInlineInteger(std::uint32_t raw_value) {
  return raw_value >= 128u && raw_value <= 208u;
}

constexpr std::uint32_t DecodeInlineInteger(std::uint32_t raw_value) {
  if (raw_value <= 192u) {
    return raw_value - 128u;
  }
  const std::int32_t signed_value =
      -static_cast<std::int32_t>(raw_value - 192u);
  return static_cast<std::uint32_t>(signed_value);
}

constexpr std::int32_t SignExtend16(std::uint32_t value) {
  return static_cast<std::int32_t>(static_cast<std::int16_t>(value & 0xffffu));
}

bool IsPhase0ExecutableOpcode(std::string_view opcode) {
  for (std::string_view executable_opcode : kPhase0ExecutableOpcodes) {
    if (executable_opcode == opcode) {
      return true;
    }
  }
  return false;
}

std::string_view NormalizeExecutableInstructionName(std::string_view opcode) {
  if (opcode == "S_ADD_CO_U32") {
    return "S_ADD_U32";
  }
  if (opcode == "S_ADD_CO_I32") {
    return "S_ADD_I32";
  }
  if (opcode == "S_SUB_CO_U32") {
    return "S_SUB_U32";
  }
  if (opcode == "V_ADD_NC_U32") {
    return "V_ADD_U32";
  }
  if (opcode == "V_SUB_NC_U32") {
    return "V_SUB_U32";
  }
  if (opcode == "V_SUBREV_NC_U32") {
    return "V_SUBREV_U32";
  }
  return opcode;
}

std::string BuildExecutableOpcodeList() {
  std::string message;
  bool first = true;
  for (std::string_view opcode : kPhase0ExecutableOpcodes) {
    message.append(first ? "" : ", ");
    message.append(opcode);
    first = false;
  }
  return message;
}

std::string BuildDecoderBringupMessage() {
  std::string message = "gfx1201 decoder seed slice supports ";
  message.append(BuildExecutableOpcodeList());
  message.append("; remaining phase-0 compute seed encodings:");

  bool first = true;
  for (const Gfx1201DecoderSeedEncoding& encoding :
       GetGfx1201Phase0ComputeDecoderSeeds()) {
    message.append(first ? " " : ", ");
    message.append(encoding.encoding_name);
    first = false;
  }
  return message;
}

OperandDescriptor MakeScalarRegisterDescriptor(OperandRole role,
                                               OperandSlotKind slot_kind,
                                               OperandAccess access,
                                               std::uint8_t element_bit_width = 32,
                                               std::uint8_t component_count = 1,
                                               bool is_implicit = false) {
  OperandDescriptor descriptor;
  descriptor.role = role;
  descriptor.slot_kind = slot_kind;
  descriptor.value_class = OperandValueClass::kScalarRegister;
  descriptor.access = access;
  descriptor.fragment_shape = MakeScalarFragmentShape(element_bit_width);
  descriptor.component_count = component_count;
  descriptor.is_implicit = is_implicit;
  return descriptor;
}

OperandDescriptor MakeVectorRegisterDescriptor(OperandRole role,
                                               OperandSlotKind slot_kind,
                                               OperandAccess access,
                                               std::uint8_t element_bit_width = 32,
                                               std::uint8_t component_count = 1) {
  OperandDescriptor descriptor;
  descriptor.role = role;
  descriptor.slot_kind = slot_kind;
  descriptor.value_class = OperandValueClass::kVectorRegister;
  descriptor.access = access;
  descriptor.fragment_shape = MakeVectorFragmentShape(1u, element_bit_width);
  descriptor.component_count = component_count;
  return descriptor;
}

OperandDescriptor MakeImmediateDescriptor(OperandRole role,
                                          OperandSlotKind slot_kind,
                                          std::uint8_t element_bit_width = 32) {
  OperandDescriptor descriptor;
  descriptor.role = role;
  descriptor.slot_kind = slot_kind;
  descriptor.access = OperandAccess::kRead;
  descriptor.fragment_shape = MakeScalarFragmentShape(element_bit_width);
  descriptor.component_count = 1;
  return descriptor;
}

InstructionOperand DescribeSourceOperand(InstructionOperand operand,
                                         OperandRole role,
                                         OperandSlotKind slot_kind) {
  if (operand.kind == OperandKind::kSgpr) {
    return operand.WithDescriptor(
        MakeScalarRegisterDescriptor(role, slot_kind, OperandAccess::kRead));
  }
  if (operand.kind == OperandKind::kVgpr) {
    return operand.WithDescriptor(
        MakeVectorRegisterDescriptor(role, slot_kind, OperandAccess::kRead));
  }
  return operand.WithDescriptor(MakeImmediateDescriptor(role, slot_kind));
}

InstructionOperand DescribeWideSourceOperand(InstructionOperand operand,
                                             OperandRole role,
                                             OperandSlotKind slot_kind) {
  if (operand.kind == OperandKind::kSgpr) {
    return operand.WithDescriptor(MakeScalarRegisterDescriptor(
        role, slot_kind, OperandAccess::kRead, 64, 2));
  }
  if (operand.kind == OperandKind::kVgpr) {
    return operand.WithDescriptor(
        MakeVectorRegisterDescriptor(role, slot_kind, OperandAccess::kRead, 64, 2));
  }
  return operand.WithDescriptor(MakeImmediateDescriptor(role, slot_kind, 64));
}

InstructionOperand DescribeScalarDestinationOperand(InstructionOperand operand,
                                                    bool is_implicit = false,
                                                    std::uint8_t element_bit_width = 32,
                                                    std::uint8_t component_count = 1) {
  return operand.WithDescriptor(MakeScalarRegisterDescriptor(
      OperandRole::kDestination, OperandSlotKind::kScalarDestination,
      OperandAccess::kWrite, element_bit_width, component_count, is_implicit));
}

InstructionOperand DescribeVectorDestinationOperand(InstructionOperand operand) {
  return operand.WithDescriptor(MakeVectorRegisterDescriptor(
      OperandRole::kDestination, OperandSlotKind::kDestination,
      OperandAccess::kWrite));
}

InstructionOperand DescribeWideVectorDestinationOperand(InstructionOperand operand) {
  return operand.WithDescriptor(MakeVectorRegisterDescriptor(
      OperandRole::kDestination, OperandSlotKind::kDestination,
      OperandAccess::kWrite, 64, 2));
}

InstructionOperand MakeImplicitVccDestinationOperand() {
  return InstructionOperand::Sgpr(
      kImplicitVccPairSgprIndex,
      MakeScalarRegisterDescriptor(OperandRole::kDestination,
                                   OperandSlotKind::kScalarDestination,
                                   OperandAccess::kWrite, 64, 2, true));
}

std::string BuildRouteMessage(const Gfx1201OpcodeRoute& route) {
  std::ostringstream stream;
  stream << "gfx1201 decoder stub routed phase-0 compute opcode to "
         << route.selector_rule->encoding_name << " opcode " << route.opcode;
  if (route.seed_entry != nullptr) {
    stream << " (" << route.seed_entry->instruction_name << ")";
  }
  if (route.status == Gfx1201OpcodeRouteStatus::kNeedsMoreWords) {
    stream << " but needs " << route.words_required << " dwords";
  } else if (route.status == Gfx1201OpcodeRouteStatus::kMatchedEncodingOnly) {
    stream << " with no matching seed entry";
  }
  return stream.str();
}

bool DecodeScalarDestination(std::uint32_t raw_value,
                             InstructionOperand* operand,
                             std::string* error_message) {
  if (operand == nullptr) {
    if (error_message != nullptr) {
      *error_message = "scalar destination output must not be null";
    }
    return false;
  }
  if (raw_value >= WaveExecutionState::kScalarRegisterCount) {
    if (error_message != nullptr) {
      *error_message = "scalar destination out of range";
    }
    return false;
  }

  *operand = InstructionOperand::Sgpr(static_cast<std::uint16_t>(raw_value));
  if (error_message != nullptr) {
    error_message->clear();
  }
  return true;
}

bool DecodeVectorDestination(std::uint32_t raw_value,
                             InstructionOperand* operand,
                             std::string* error_message) {
  if (operand == nullptr) {
    if (error_message != nullptr) {
      *error_message = "vector destination output must not be null";
    }
    return false;
  }
  if (raw_value >= WaveExecutionState::kVectorRegisterCount) {
    if (error_message != nullptr) {
      *error_message = "vector destination out of range";
    }
    return false;
  }

  *operand = InstructionOperand::Vgpr(static_cast<std::uint16_t>(raw_value));
  if (error_message != nullptr) {
    error_message->clear();
  }
  return true;
}

bool DecodeScalarSource(std::uint32_t raw_value,
                        std::span<const std::uint32_t> literal_words,
                        std::size_t* literal_words_consumed,
                        InstructionOperand* operand,
                        std::string* error_message) {
  if (literal_words_consumed == nullptr || operand == nullptr) {
    if (error_message != nullptr) {
      *error_message = "scalar source outputs must not be null";
    }
    return false;
  }
  *literal_words_consumed = 0;

  if (raw_value < WaveExecutionState::kScalarRegisterCount) {
    *operand = InstructionOperand::Sgpr(static_cast<std::uint16_t>(raw_value));
  } else if (raw_value == kSrcVcczSgprIndex || raw_value == kSrcExeczSgprIndex ||
             raw_value == kSrcSccSgprIndex) {
    *operand = InstructionOperand::Sgpr(static_cast<std::uint16_t>(raw_value));
  } else if (IsInlineInteger(raw_value)) {
    *operand = InstructionOperand::Imm32(DecodeInlineInteger(raw_value));
  } else if (raw_value == 255u) {
    if (literal_words.empty()) {
      if (error_message != nullptr) {
        *error_message = "missing literal dword";
      }
      return false;
    }
    *operand = InstructionOperand::Imm32(literal_words.front());
    *literal_words_consumed = 1;
  } else {
    if (error_message != nullptr) {
      *error_message = "unsupported scalar source operand encoding";
    }
    return false;
  }

  if (error_message != nullptr) {
    error_message->clear();
  }
  return true;
}

bool DecodeVectorSource(std::uint32_t raw_value,
                        std::span<const std::uint32_t> literal_words,
                        std::size_t* literal_words_consumed,
                        InstructionOperand* operand,
                        std::string* error_message) {
  if (literal_words_consumed == nullptr || operand == nullptr) {
    if (error_message != nullptr) {
      *error_message = "vector source outputs must not be null";
    }
    return false;
  }
  *literal_words_consumed = 0;

  if (raw_value < WaveExecutionState::kScalarRegisterCount) {
    *operand = InstructionOperand::Sgpr(static_cast<std::uint16_t>(raw_value));
  } else if (raw_value == kSrcVcczSgprIndex || raw_value == kSrcExeczSgprIndex ||
             raw_value == kSrcSccSgprIndex) {
    *operand = InstructionOperand::Sgpr(static_cast<std::uint16_t>(raw_value));
  } else if (IsInlineInteger(raw_value)) {
    *operand = InstructionOperand::Imm32(DecodeInlineInteger(raw_value));
  } else if (raw_value == 255u) {
    if (literal_words.empty()) {
      if (error_message != nullptr) {
        *error_message = "missing literal dword";
      }
      return false;
    }
    *operand = InstructionOperand::Imm32(literal_words.front());
    *literal_words_consumed = 1;
  } else if (raw_value >= 256u &&
             raw_value < 256u + WaveExecutionState::kVectorRegisterCount) {
    *operand =
        InstructionOperand::Vgpr(static_cast<std::uint16_t>(raw_value - 256u));
  } else {
    if (error_message != nullptr) {
      *error_message = "unsupported vector source operand encoding";
    }
    return false;
  }

  if (error_message != nullptr) {
    error_message->clear();
  }
  return true;
}

bool DecodeVectorRegisterSource(std::uint32_t raw_value,
                                InstructionOperand* operand,
                                std::string* error_message) {
  if (operand == nullptr) {
    if (error_message != nullptr) {
      *error_message = "vector register source output must not be null";
    }
    return false;
  }
  if (raw_value >= WaveExecutionState::kVectorRegisterCount) {
    if (error_message != nullptr) {
      *error_message = "unsupported VSRC1 register";
    }
    return false;
  }

  *operand = InstructionOperand::Vgpr(static_cast<std::uint16_t>(raw_value));
  if (error_message != nullptr) {
    error_message->clear();
  }
  return true;
}

bool TryDecodeExecutableSeedInstruction(const Gfx1201OpcodeRoute& route,
                                        std::span<const std::uint32_t> words,
                                        DecodedInstruction* instruction,
                                        std::size_t* words_consumed,
                                        std::string* error_message) {
  if (instruction == nullptr || words_consumed == nullptr) {
    if (error_message != nullptr) {
      *error_message = "decode outputs must not be null";
    }
    return false;
  }
  if (route.seed_entry == nullptr) {
    return false;
  }

  const std::string_view instruction_name =
      NormalizeExecutableInstructionName(route.seed_entry->instruction_name);
  if (!IsPhase0ExecutableOpcode(instruction_name)) {
    return false;
  }

  const std::uint32_t word = words.front();
  *instruction = DecodedInstruction{};
  *words_consumed = 0;

  if (instruction_name == "S_ENDPGM") {
    *instruction = DecodedInstruction::Nullary(instruction_name);
    *words_consumed = 1;
  } else if (instruction_name == "S_NOP") {
    *instruction = DecodedInstruction::OneOperand(
        instruction_name,
        InstructionOperand::Imm32(ExtractBits(word, 0, 16))
            .WithDescriptor(MakeImmediateDescriptor(OperandRole::kSource0,
                                                   OperandSlotKind::kSource0)));
    *words_consumed = 1;
  } else if (instruction_name == "S_BRANCH" ||
             instruction_name == "S_CBRANCH_SCC0" ||
             instruction_name == "S_CBRANCH_SCC1" ||
             instruction_name == "S_CBRANCH_VCCZ" ||
             instruction_name == "S_CBRANCH_VCCNZ" ||
             instruction_name == "S_CBRANCH_EXECZ" ||
             instruction_name == "S_CBRANCH_EXECNZ") {
    *instruction = DecodedInstruction::OneOperand(
        instruction_name,
        InstructionOperand::Imm32(static_cast<std::uint32_t>(
            SignExtend16(ExtractBits(word, 0, 16))))
            .WithDescriptor(MakeImmediateDescriptor(OperandRole::kSource0,
                                                   OperandSlotKind::kSource0)));
    *words_consumed = 1;
  } else if (instruction_name == "S_MOV_B32") {
    InstructionOperand dst;
    if (!DecodeScalarDestination(ExtractBits(word, 16, 7), &dst, error_message)) {
      return false;
    }

    std::size_t literal_words_consumed = 0;
    InstructionOperand src0;
    if (!DecodeScalarSource(ExtractBits(word, 0, 8), words.subspan(1),
                            &literal_words_consumed, &src0, error_message)) {
      return false;
    }

    *instruction = DecodedInstruction::Unary(
        instruction_name, DescribeScalarDestinationOperand(dst),
        DescribeSourceOperand(src0, OperandRole::kSource0,
                              OperandSlotKind::kSource0));
    *words_consumed = 1 + literal_words_consumed;
  } else if (instruction_name == "S_MOVK_I32") {
    InstructionOperand dst;
    if (!DecodeScalarDestination(ExtractBits(word, 16, 7), &dst, error_message)) {
      return false;
    }

    *instruction = DecodedInstruction::Unary(
        instruction_name, DescribeScalarDestinationOperand(dst),
        InstructionOperand::Imm32(static_cast<std::uint32_t>(
            SignExtend16(ExtractBits(word, 0, 16))))
            .WithDescriptor(MakeImmediateDescriptor(OperandRole::kSource0,
                                                   OperandSlotKind::kSource0)));
    *words_consumed = 1;
  } else if (instruction_name == "S_CLAUSE" ||
             instruction_name == "S_DELAY_ALU" ||
             instruction_name == "S_WAIT_KMCNT" ||
             instruction_name == "S_WAIT_ALU" ||
             instruction_name == "S_WAIT_LOADCNT") {
    // SOPP scheduling hints: 1 dword, SIMM16 at bits[15:0].
    *instruction = DecodedInstruction::OneOperand(
        instruction_name,
        InstructionOperand::Imm32(ExtractBits(word, 0, 16))
            .WithDescriptor(MakeImmediateDescriptor(OperandRole::kSource0,
                                                   OperandSlotKind::kSource0)));
    *words_consumed = 1;
  } else if (instruction_name == "S_CODE_END") {
    // SOPP with no meaningful operand (post-endpgm padding).
    *instruction = DecodedInstruction::Nullary(instruction_name);
    *words_consumed = 1;
  } else if (instruction_name == "S_ADD_U32" || instruction_name == "S_ADD_I32" ||
             instruction_name == "S_SUB_U32" || instruction_name == "S_SUB_I32" ||
             instruction_name == "S_AND_B32" || instruction_name == "S_AND_B64" ||
             instruction_name == "S_OR_B32" || instruction_name == "S_OR_B64" ||
             instruction_name == "S_XOR_B32" || instruction_name == "S_XOR_B64" ||
             instruction_name == "S_LSHL_B32" || instruction_name == "S_LSHR_B32" ||
             instruction_name == "S_ASHR_I32" || instruction_name == "S_MUL_I32" ||
             instruction_name == "S_CSELECT_B32" || instruction_name == "S_CSELECT_B64") {
    InstructionOperand dst;
    if (!DecodeScalarDestination(ExtractBits(word, 16, 7), &dst, error_message)) {
      return false;
    }

    std::size_t src0_literal_words_consumed = 0;
    InstructionOperand src0;
    if (!DecodeScalarSource(ExtractBits(word, 0, 8), words.subspan(1),
                            &src0_literal_words_consumed, &src0,
                            error_message)) {
      return false;
    }

    std::size_t src1_literal_words_consumed = 0;
    InstructionOperand src1;
    if (!DecodeScalarSource(ExtractBits(word, 8, 8),
                            words.subspan(1 + src0_literal_words_consumed),
                            &src1_literal_words_consumed, &src1,
                            error_message)) {
      return false;
    }

    *instruction = DecodedInstruction::Binary(
        instruction_name, DescribeScalarDestinationOperand(dst),
        DescribeSourceOperand(src0, OperandRole::kSource0,
                              OperandSlotKind::kSource0),
        DescribeSourceOperand(src1, OperandRole::kSource1,
                              OperandSlotKind::kSource1));
    *words_consumed =
        1 + src0_literal_words_consumed + src1_literal_words_consumed;
  } else if (instruction_name == "S_CMP_EQ_I32" ||
             instruction_name == "S_CMP_LG_I32" ||
             instruction_name == "S_CMP_GT_I32" ||
             instruction_name == "S_CMP_EQ_U32" ||
             instruction_name == "S_CMP_LG_U32" ||
             instruction_name == "S_CMP_GE_I32" ||
             instruction_name == "S_CMP_LT_I32" ||
             instruction_name == "S_CMP_LE_I32" ||
             instruction_name == "S_CMP_GT_U32" ||
             instruction_name == "S_CMP_GE_U32" ||
             instruction_name == "S_CMP_LT_U32" ||
             instruction_name == "S_CMP_LE_U32") {
    std::size_t src0_literal_words_consumed = 0;
    InstructionOperand src0;
    if (!DecodeScalarSource(ExtractBits(word, 0, 8), words.subspan(1),
                            &src0_literal_words_consumed, &src0,
                            error_message)) {
      return false;
    }

    std::size_t src1_literal_words_consumed = 0;
    InstructionOperand src1;
    if (!DecodeScalarSource(ExtractBits(word, 8, 8),
                            words.subspan(1 + src0_literal_words_consumed),
                            &src1_literal_words_consumed, &src1,
                            error_message)) {
      return false;
    }

    *instruction = DecodedInstruction::TwoOperand(
        instruction_name,
        DescribeSourceOperand(src0, OperandRole::kSource0,
                              OperandSlotKind::kSource0),
        DescribeSourceOperand(src1, OperandRole::kSource1,
                              OperandSlotKind::kSource1));
    *words_consumed =
        1 + src0_literal_words_consumed + src1_literal_words_consumed;
  } else if (instruction_name == "V_MOV_B32" ||
             instruction_name == "V_NOT_B32" ||
             instruction_name == "V_BFREV_B32" ||
             instruction_name == "V_CVT_F32_UBYTE0" ||
             instruction_name == "V_CVT_F32_UBYTE1" ||
             instruction_name == "V_CVT_F32_UBYTE2" ||
             instruction_name == "V_CVT_F32_UBYTE3" ||
             instruction_name == "V_CVT_F32_I32" ||
             instruction_name == "V_CVT_F32_U32" ||
             instruction_name == "V_CVT_U32_F32" ||
             instruction_name == "V_CVT_I32_F32") {
    InstructionOperand dst;
    if (!DecodeVectorDestination(ExtractBits(word, 17, 8), &dst, error_message)) {
      return false;
    }

    std::size_t literal_words_consumed = 0;
    InstructionOperand src0;
    if (!DecodeVectorSource(ExtractBits(word, 0, 9), words.subspan(1),
                            &literal_words_consumed, &src0, error_message)) {
      return false;
    }

    *instruction = DecodedInstruction::Unary(
        instruction_name, DescribeVectorDestinationOperand(dst),
        DescribeSourceOperand(src0, OperandRole::kSource0,
                              OperandSlotKind::kSource0));
    *words_consumed = 1 + literal_words_consumed;
  } else if (instruction_name == "V_CVT_F64_F32" ||
             instruction_name == "V_CVT_F64_I32" ||
             instruction_name == "V_CVT_F64_U32") {
    InstructionOperand dst;
    if (!DecodeVectorDestination(ExtractBits(word, 17, 8), &dst, error_message)) {
      return false;
    }

    std::size_t literal_words_consumed = 0;
    InstructionOperand src0;
    if (!DecodeVectorSource(ExtractBits(word, 0, 9), words.subspan(1),
                            &literal_words_consumed, &src0, error_message)) {
      return false;
    }

    *instruction = DecodedInstruction::Unary(
        instruction_name, DescribeWideVectorDestinationOperand(dst),
        DescribeSourceOperand(src0, OperandRole::kSource0,
                              OperandSlotKind::kSource0));
    *words_consumed = 1 + literal_words_consumed;
  } else if (instruction_name == "V_CVT_F32_F64" ||
             instruction_name == "V_CVT_I32_F64" ||
             instruction_name == "V_CVT_U32_F64") {
    InstructionOperand dst;
    if (!DecodeVectorDestination(ExtractBits(word, 17, 8), &dst, error_message)) {
      return false;
    }

    std::size_t literal_words_consumed = 0;
    InstructionOperand src0;
    if (!DecodeVectorSource(ExtractBits(word, 0, 9), words.subspan(1),
                            &literal_words_consumed, &src0, error_message)) {
      return false;
    }

    *instruction = DecodedInstruction::Unary(
        instruction_name, DescribeVectorDestinationOperand(dst),
        DescribeWideSourceOperand(src0, OperandRole::kSource0,
                                  OperandSlotKind::kSource0));
    *words_consumed = 1 + literal_words_consumed;
  } else if (instruction_name == "V_CMP_EQ_I32" ||
             instruction_name == "V_CMP_NE_I32" ||
             instruction_name == "V_CMP_LT_I32" ||
             instruction_name == "V_CMP_LE_I32" ||
             instruction_name == "V_CMP_GT_I32" ||
             instruction_name == "V_CMP_GE_I32" ||
             instruction_name == "V_CMP_EQ_U32" ||
             instruction_name == "V_CMP_NE_U32" ||
             instruction_name == "V_CMP_LT_U32" ||
             instruction_name == "V_CMP_LE_U32" ||
             instruction_name == "V_CMP_GT_U32" ||
             instruction_name == "V_CMP_GE_U32" ||
             instruction_name == "V_CMPX_EQ_I32" ||
             instruction_name == "V_CMPX_NE_I32" ||
             instruction_name == "V_CMPX_LT_I32" ||
             instruction_name == "V_CMPX_LE_I32" ||
             instruction_name == "V_CMPX_GT_I32" ||
             instruction_name == "V_CMPX_GE_I32" ||
             instruction_name == "V_CMPX_EQ_U32" ||
             instruction_name == "V_CMPX_NE_U32" ||
             instruction_name == "V_CMPX_LT_U32" ||
             instruction_name == "V_CMPX_LE_U32" ||
             instruction_name == "V_CMPX_GT_U32" ||
             instruction_name == "V_CMPX_GE_U32" ||
             instruction_name == "V_CMP_CLASS_F32" ||
             instruction_name == "V_CMP_EQ_F32" ||
             instruction_name == "V_CMP_GE_F32" ||
             instruction_name == "V_CMP_GT_F32" ||
             instruction_name == "V_CMP_LE_F32" ||
             instruction_name == "V_CMP_LG_F32" ||
             instruction_name == "V_CMP_LT_F32" ||
             instruction_name == "V_CMP_NEQ_F32" ||
             instruction_name == "V_CMP_O_F32" ||
             instruction_name == "V_CMP_U_F32" ||
             instruction_name == "V_CMP_NGE_F32" ||
             instruction_name == "V_CMP_NLG_F32" ||
             instruction_name == "V_CMP_NGT_F32" ||
             instruction_name == "V_CMP_NLE_F32" ||
             instruction_name == "V_CMP_NLT_F32" ||
             instruction_name == "V_CMPX_CLASS_F32" ||
             instruction_name == "V_CMPX_EQ_F32" ||
             instruction_name == "V_CMPX_GE_F32" ||
             instruction_name == "V_CMPX_GT_F32" ||
             instruction_name == "V_CMPX_LE_F32" ||
             instruction_name == "V_CMPX_LG_F32" ||
             instruction_name == "V_CMPX_LT_F32" ||
             instruction_name == "V_CMPX_NEQ_F32" ||
             instruction_name == "V_CMPX_O_F32" ||
             instruction_name == "V_CMPX_U_F32" ||
             instruction_name == "V_CMPX_NGE_F32" ||
             instruction_name == "V_CMPX_NLG_F32" ||
             instruction_name == "V_CMPX_NGT_F32" ||
             instruction_name == "V_CMPX_NLE_F32" ||
             instruction_name == "V_CMPX_NLT_F32") {
    const std::string_view enc_name = route.selector_rule->encoding_name;
    if (enc_name == "ENC_VOP3") {
      // VOP3 encoding: 2 dwords.
      if (words.size() < 2) return false;
      const std::uint32_t word1 = words[1];

      std::size_t src0_literal = 0;
      InstructionOperand src0;
      if (!DecodeVectorSource(ExtractBits(word1, 0, 9), words.subspan(2),
                              &src0_literal, &src0, error_message)) {
        return false;
      }

      std::size_t src1_literal = 0;
      InstructionOperand src1;
      if (!DecodeVectorSource(ExtractBits(word1, 9, 9),
                              words.subspan(2 + src0_literal),
                              &src1_literal, &src1, error_message)) {
        return false;
      }

      *instruction = DecodedInstruction::Binary(
          instruction_name, MakeImplicitVccDestinationOperand(),
          DescribeSourceOperand(src0, OperandRole::kSource0,
                                OperandSlotKind::kSource0),
          DescribeSourceOperand(src1, OperandRole::kSource1,
                                OperandSlotKind::kSource1));
      *words_consumed = 2 + src0_literal + src1_literal;
    } else {
      // VOPC encoding: 1 dword.
      std::size_t literal_words_consumed = 0;
      InstructionOperand src0;
      if (!DecodeVectorSource(ExtractBits(word, 0, 9), words.subspan(1),
                              &literal_words_consumed, &src0, error_message)) {
        return false;
      }

      InstructionOperand src1;
      if (!DecodeVectorRegisterSource(ExtractBits(word, 9, 8), &src1,
                                      error_message)) {
        return false;
      }

      *instruction = DecodedInstruction::Binary(
          instruction_name, MakeImplicitVccDestinationOperand(),
          DescribeSourceOperand(src0, OperandRole::kSource0,
                                OperandSlotKind::kSource0),
          DescribeSourceOperand(src1, OperandRole::kSource1,
                                OperandSlotKind::kSource1));
      *words_consumed = 1 + literal_words_consumed;
    }
  } else if (instruction_name == "V_CMP_CLASS_F64" ||
             instruction_name == "V_CMPX_CLASS_F64") {
    std::size_t literal_words_consumed = 0;
    InstructionOperand src0;
    if (!DecodeVectorSource(ExtractBits(word, 0, 9), words.subspan(1),
                            &literal_words_consumed, &src0, error_message)) {
      return false;
    }

    InstructionOperand src1;
    if (!DecodeVectorRegisterSource(ExtractBits(word, 9, 8), &src1,
                                    error_message)) {
      return false;
    }

    *instruction = DecodedInstruction::Binary(
        instruction_name, MakeImplicitVccDestinationOperand(),
        DescribeWideSourceOperand(src0, OperandRole::kSource0,
                                  OperandSlotKind::kSource0),
        DescribeSourceOperand(src1, OperandRole::kSource1,
                              OperandSlotKind::kSource1));
    *words_consumed = 1 + literal_words_consumed;
  } else if (instruction_name == "V_CMP_EQ_F64" ||
             instruction_name == "V_CMP_NE_I64" ||
             instruction_name == "V_CMP_LT_I64" ||
             instruction_name == "V_CMP_LE_I64" ||
             instruction_name == "V_CMP_GT_I64" ||
             instruction_name == "V_CMP_GE_I64" ||
             instruction_name == "V_CMP_EQ_I64" ||
             instruction_name == "V_CMP_NE_U64" ||
             instruction_name == "V_CMP_LT_U64" ||
             instruction_name == "V_CMP_LE_U64" ||
             instruction_name == "V_CMP_GT_U64" ||
             instruction_name == "V_CMP_GE_U64" ||
             instruction_name == "V_CMP_EQ_U64" ||
             instruction_name == "V_CMP_GE_F64" ||
             instruction_name == "V_CMP_GT_F64" ||
             instruction_name == "V_CMP_LE_F64" ||
             instruction_name == "V_CMP_LG_F64" ||
             instruction_name == "V_CMP_LT_F64" ||
             instruction_name == "V_CMP_NEQ_F64" ||
             instruction_name == "V_CMP_O_F64" ||
             instruction_name == "V_CMP_U_F64" ||
             instruction_name == "V_CMP_NGE_F64" ||
             instruction_name == "V_CMP_NLG_F64" ||
             instruction_name == "V_CMP_NGT_F64" ||
             instruction_name == "V_CMP_NLE_F64" ||
             instruction_name == "V_CMP_NLT_F64" ||
             instruction_name == "V_CMPX_NE_I64" ||
             instruction_name == "V_CMPX_LT_I64" ||
             instruction_name == "V_CMPX_LE_I64" ||
             instruction_name == "V_CMPX_GT_I64" ||
             instruction_name == "V_CMPX_GE_I64" ||
             instruction_name == "V_CMPX_EQ_I64" ||
             instruction_name == "V_CMPX_NE_U64" ||
             instruction_name == "V_CMPX_LT_U64" ||
             instruction_name == "V_CMPX_LE_U64" ||
             instruction_name == "V_CMPX_GT_U64" ||
             instruction_name == "V_CMPX_GE_U64" ||
             instruction_name == "V_CMPX_EQ_U64" ||
             instruction_name == "V_CMPX_EQ_F64" ||
             instruction_name == "V_CMPX_GE_F64" ||
             instruction_name == "V_CMPX_GT_F64" ||
             instruction_name == "V_CMPX_LE_F64" ||
             instruction_name == "V_CMPX_LG_F64" ||
             instruction_name == "V_CMPX_LT_F64" ||
             instruction_name == "V_CMPX_NEQ_F64" ||
             instruction_name == "V_CMPX_O_F64" ||
             instruction_name == "V_CMPX_U_F64" ||
             instruction_name == "V_CMPX_NGE_F64" ||
             instruction_name == "V_CMPX_NLG_F64" ||
             instruction_name == "V_CMPX_NGT_F64" ||
             instruction_name == "V_CMPX_NLE_F64" ||
             instruction_name == "V_CMPX_NLT_F64") {
    std::size_t literal_words_consumed = 0;
    InstructionOperand src0;
    if (!DecodeVectorSource(ExtractBits(word, 0, 9), words.subspan(1),
                            &literal_words_consumed, &src0, error_message)) {
      return false;
    }

    InstructionOperand src1;
    if (!DecodeVectorRegisterSource(ExtractBits(word, 9, 8), &src1,
                                    error_message)) {
      return false;
    }

    *instruction = DecodedInstruction::Binary(
        instruction_name, MakeImplicitVccDestinationOperand(),
        DescribeWideSourceOperand(src0, OperandRole::kSource0,
                                  OperandSlotKind::kSource0),
        DescribeWideSourceOperand(src1, OperandRole::kSource1,
                                  OperandSlotKind::kSource1));
    *words_consumed = 1 + literal_words_consumed;
  } else if (instruction_name == "V_ADD_U32" ||
             instruction_name == "V_SUBREV_U32" ||
             instruction_name == "V_MIN_I32" ||
             instruction_name == "V_MAX_I32" ||
             instruction_name == "V_MIN_U32" ||
             instruction_name == "V_MAX_U32" ||
             instruction_name == "V_CNDMASK_B32" ||
             instruction_name == "V_LSHRREV_B32" ||
             instruction_name == "V_ASHRREV_I32" ||
             instruction_name == "V_LSHLREV_B32" ||
             instruction_name == "V_AND_B32" ||
             instruction_name == "V_OR_B32" ||
             instruction_name == "V_XOR_B32") {
    InstructionOperand dst;
    if (!DecodeVectorDestination(ExtractBits(word, 17, 8), &dst, error_message)) {
      return false;
    }

    std::size_t literal_words_consumed = 0;
    InstructionOperand src0;
    if (!DecodeVectorSource(ExtractBits(word, 0, 9), words.subspan(1),
                            &literal_words_consumed, &src0, error_message)) {
      return false;
    }

    InstructionOperand src1;
    if (!DecodeVectorRegisterSource(ExtractBits(word, 9, 8), &src1,
                                    error_message)) {
      return false;
    }

    *instruction = DecodedInstruction::Binary(
        instruction_name, DescribeVectorDestinationOperand(dst),
        DescribeSourceOperand(src0, OperandRole::kSource0,
                              OperandSlotKind::kSource0),
        DescribeSourceOperand(src1, OperandRole::kSource1,
                              OperandSlotKind::kSource1));
    *words_consumed = 1 + literal_words_consumed;
  } else if (instruction_name == "V_SUB_U32") {
    InstructionOperand dst;
    if (!DecodeVectorDestination(ExtractBits(word, 17, 8), &dst, error_message)) {
      return false;
    }

    std::size_t literal_words_consumed = 0;
    InstructionOperand src0;
    if (!DecodeVectorSource(ExtractBits(word, 0, 9), words.subspan(1),
                            &literal_words_consumed, &src0, error_message)) {
      return false;
    }

    InstructionOperand src1;
    if (!DecodeVectorRegisterSource(ExtractBits(word, 9, 8), &src1,
                                    error_message)) {
      return false;
    }

    *instruction = DecodedInstruction::Binary(
        instruction_name, DescribeVectorDestinationOperand(dst),
        DescribeSourceOperand(src0, OperandRole::kSource0,
                              OperandSlotKind::kSource0),
        DescribeSourceOperand(src1, OperandRole::kSource1,
                              OperandSlotKind::kSource1));
    *words_consumed = 1 + literal_words_consumed;
  } else if (instruction_name == "V_ADD_F32" ||
             instruction_name == "V_SUB_F32" ||
             instruction_name == "V_MUL_F32") {
    // VOP2 3-operand vector ALU: dst=VDST[24:17], src0=SRC0[8:0], src1=VSRC1[16:9].
    const std::string_view enc_name = route.selector_rule->encoding_name;
    if (enc_name == "ENC_VOP3") {
      // VOP3 2-dword encoding: word0 bits[7:0]=VDST, word1 bits[8:0]=SRC0,
      // word1 bits[17:9]=SRC1, word1 bits[26:18]=SRC2.
      if (words.size() < 2) return false;
      const std::uint32_t word1 = words[1];

      InstructionOperand dst;
      if (!DecodeVectorDestination(ExtractBits(word, 0, 8), &dst, error_message)) {
        return false;
      }

      std::size_t src0_literal = 0;
      InstructionOperand src0;
      if (!DecodeVectorSource(ExtractBits(word1, 0, 9), words.subspan(2),
                              &src0_literal, &src0, error_message)) {
        return false;
      }

      std::size_t src1_literal = 0;
      InstructionOperand src1;
      if (!DecodeVectorSource(ExtractBits(word1, 9, 9),
                              words.subspan(2 + src0_literal),
                              &src1_literal, &src1, error_message)) {
        return false;
      }

      *instruction = DecodedInstruction::Binary(
          instruction_name, DescribeVectorDestinationOperand(dst),
          DescribeSourceOperand(src0, OperandRole::kSource0,
                                OperandSlotKind::kSource0),
          DescribeSourceOperand(src1, OperandRole::kSource1,
                                OperandSlotKind::kSource1));
      *words_consumed = 2 + src0_literal + src1_literal;
    } else {
      // VOP2 1-dword encoding.
      InstructionOperand dst;
      if (!DecodeVectorDestination(ExtractBits(word, 17, 8), &dst, error_message)) {
        return false;
      }

      std::size_t literal_words_consumed = 0;
      InstructionOperand src0;
      if (!DecodeVectorSource(ExtractBits(word, 0, 9), words.subspan(1),
                              &literal_words_consumed, &src0, error_message)) {
        return false;
      }

      InstructionOperand src1;
      if (!DecodeVectorRegisterSource(ExtractBits(word, 9, 8), &src1,
                                      error_message)) {
        return false;
      }

      *instruction = DecodedInstruction::Binary(
          instruction_name, DescribeVectorDestinationOperand(dst),
          DescribeSourceOperand(src0, OperandRole::kSource0,
                                OperandSlotKind::kSource0),
          DescribeSourceOperand(src1, OperandRole::kSource1,
                                OperandSlotKind::kSource1));
      *words_consumed = 1 + literal_words_consumed;
    }
  } else if (instruction_name == "V_LSHLREV_B64") {
    // VOP2 encoding: 64-bit left shift. dst is a register pair.
    const std::string_view enc_name = route.selector_rule->encoding_name;
    if (enc_name == "ENC_VOP3") {
      if (words.size() < 2) return false;
      const std::uint32_t word1 = words[1];

      InstructionOperand dst;
      if (!DecodeVectorDestination(ExtractBits(word, 0, 8), &dst, error_message)) {
        return false;
      }

      std::size_t src0_literal = 0;
      InstructionOperand src0;
      if (!DecodeVectorSource(ExtractBits(word1, 0, 9), words.subspan(2),
                              &src0_literal, &src0, error_message)) {
        return false;
      }

      std::size_t src1_literal = 0;
      InstructionOperand src1;
      if (!DecodeVectorSource(ExtractBits(word1, 9, 9),
                              words.subspan(2 + src0_literal),
                              &src1_literal, &src1, error_message)) {
        return false;
      }

      *instruction = DecodedInstruction::Binary(
          instruction_name, DescribeWideVectorDestinationOperand(dst),
          DescribeSourceOperand(src0, OperandRole::kSource0,
                                OperandSlotKind::kSource0),
          DescribeWideSourceOperand(src1, OperandRole::kSource1,
                                    OperandSlotKind::kSource1));
      *words_consumed = 2 + src0_literal + src1_literal;
    } else {
      InstructionOperand dst;
      if (!DecodeVectorDestination(ExtractBits(word, 17, 8), &dst, error_message)) {
        return false;
      }

      std::size_t literal_words_consumed = 0;
      InstructionOperand src0;
      if (!DecodeVectorSource(ExtractBits(word, 0, 9), words.subspan(1),
                              &literal_words_consumed, &src0, error_message)) {
        return false;
      }

      InstructionOperand src1;
      if (!DecodeVectorRegisterSource(ExtractBits(word, 9, 8), &src1,
                                      error_message)) {
        return false;
      }

      *instruction = DecodedInstruction::Binary(
          instruction_name, DescribeWideVectorDestinationOperand(dst),
          DescribeSourceOperand(src0, OperandRole::kSource0,
                                OperandSlotKind::kSource0),
          DescribeWideSourceOperand(src1, OperandRole::kSource1,
                                    OperandSlotKind::kSource1));
      *words_consumed = 1 + literal_words_consumed;
    }
  } else if (instruction_name == "V_ADD_CO_CI_U32") {
    // VOP2/VOP3: carry-in add. Reads and writes VCC implicitly.
    const std::string_view enc_name = route.selector_rule->encoding_name;
    if (enc_name == "ENC_VOP3") {
      if (words.size() < 2) return false;
      const std::uint32_t word1 = words[1];

      InstructionOperand dst;
      if (!DecodeVectorDestination(ExtractBits(word, 0, 8), &dst, error_message)) {
        return false;
      }

      std::size_t src0_literal = 0;
      InstructionOperand src0;
      if (!DecodeVectorSource(ExtractBits(word1, 0, 9), words.subspan(2),
                              &src0_literal, &src0, error_message)) {
        return false;
      }

      std::size_t src1_literal = 0;
      InstructionOperand src1;
      if (!DecodeVectorSource(ExtractBits(word1, 9, 9),
                              words.subspan(2 + src0_literal),
                              &src1_literal, &src1, error_message)) {
        return false;
      }

      *instruction = DecodedInstruction::Binary(
          instruction_name, DescribeVectorDestinationOperand(dst),
          DescribeSourceOperand(src0, OperandRole::kSource0,
                                OperandSlotKind::kSource0),
          DescribeSourceOperand(src1, OperandRole::kSource1,
                                OperandSlotKind::kSource1));
      *words_consumed = 2 + src0_literal + src1_literal;
    } else {
      InstructionOperand dst;
      if (!DecodeVectorDestination(ExtractBits(word, 17, 8), &dst, error_message)) {
        return false;
      }

      std::size_t literal_words_consumed = 0;
      InstructionOperand src0;
      if (!DecodeVectorSource(ExtractBits(word, 0, 9), words.subspan(1),
                              &literal_words_consumed, &src0, error_message)) {
        return false;
      }

      InstructionOperand src1;
      if (!DecodeVectorRegisterSource(ExtractBits(word, 9, 8), &src1,
                                      error_message)) {
        return false;
      }

      *instruction = DecodedInstruction::Binary(
          instruction_name, DescribeVectorDestinationOperand(dst),
          DescribeSourceOperand(src0, OperandRole::kSource0,
                                OperandSlotKind::kSource0),
          DescribeSourceOperand(src1, OperandRole::kSource1,
                                OperandSlotKind::kSource1));
      *words_consumed = 1 + literal_words_consumed;
    }
  } else if (instruction_name == "V_ADD_CO_U32") {
    // VOP3-only: 32-bit add with carry-out to VCC.
    // Word 0: bits[7:0]=VDST.
    // Word 1: bits[8:0]=SRC0, bits[17:9]=SRC1.
    if (words.size() < 2) return false;
    const std::uint32_t word1 = words[1];

    InstructionOperand dst;
    if (!DecodeVectorDestination(ExtractBits(word, 0, 8), &dst, error_message)) {
      return false;
    }

    std::size_t src0_literal = 0;
    InstructionOperand src0;
    if (!DecodeVectorSource(ExtractBits(word1, 0, 9), words.subspan(2),
                            &src0_literal, &src0, error_message)) {
      return false;
    }

    std::size_t src1_literal = 0;
    InstructionOperand src1;
    if (!DecodeVectorSource(ExtractBits(word1, 9, 9),
                            words.subspan(2 + src0_literal),
                            &src1_literal, &src1, error_message)) {
      return false;
    }

    *instruction = DecodedInstruction::Binary(
        instruction_name, DescribeVectorDestinationOperand(dst),
        DescribeSourceOperand(src0, OperandRole::kSource0,
                              OperandSlotKind::kSource0),
        DescribeSourceOperand(src1, OperandRole::kSource1,
                              OperandSlotKind::kSource1));
    *words_consumed = 2 + src0_literal + src1_literal;
  } else if (instruction_name == "V_MAD_CO_U64_U32") {
    // VOP3-only: 64-bit multiply-add with carry.
    // Word 0: bits[7:0]=VDST (register pair for 64-bit result).
    // Word 1: bits[8:0]=SRC0, bits[17:9]=SRC1, bits[26:18]=SRC2.
    if (words.size() < 2) return false;
    const std::uint32_t word1 = words[1];

    InstructionOperand dst;
    if (!DecodeVectorDestination(ExtractBits(word, 0, 8), &dst, error_message)) {
      return false;
    }

    std::size_t src0_literal = 0;
    InstructionOperand src0;
    if (!DecodeVectorSource(ExtractBits(word1, 0, 9), words.subspan(2),
                            &src0_literal, &src0, error_message)) {
      return false;
    }

    std::size_t src1_literal = 0;
    InstructionOperand src1;
    if (!DecodeVectorSource(ExtractBits(word1, 9, 9),
                            words.subspan(2 + src0_literal),
                            &src1_literal, &src1, error_message)) {
      return false;
    }

    std::size_t src2_literal = 0;
    InstructionOperand src2;
    if (!DecodeVectorSource(ExtractBits(word1, 18, 9),
                            words.subspan(2 + src0_literal + src1_literal),
                            &src2_literal, &src2, error_message)) {
      return false;
    }

    *instruction = DecodedInstruction{};
    instruction->opcode = instruction_name;
    instruction->operand_count = 4;
    instruction->operands[0] = DescribeWideVectorDestinationOperand(dst);
    instruction->operands[1] =
        DescribeSourceOperand(src0, OperandRole::kSource0,
                              OperandSlotKind::kSource0);
    instruction->operands[2] =
        DescribeSourceOperand(src1, OperandRole::kSource1,
                              OperandSlotKind::kSource1);
    instruction->operands[3] =
        DescribeWideSourceOperand(src2, OperandRole::kSource2,
                                  OperandSlotKind::kSource2);
    *words_consumed = 2 + src0_literal + src1_literal + src2_literal;
  } else if (instruction_name == "S_LOAD_B32" ||
             instruction_name == "S_LOAD_B64" ||
             instruction_name == "S_LOAD_B128" ||
             instruction_name == "S_LOAD_B256" ||
             instruction_name == "S_BUFFER_LOAD_B32" ||
             instruction_name == "S_BUFFER_LOAD_B64" ||
             instruction_name == "S_BUFFER_LOAD_B128" ||
             instruction_name == "S_BUFFER_LOAD_B256") {
    // SMEM 2-dword encoding.
    // Word 0: bits[13:7]=SDATA/SDST (7 bits), bits[6:0]=SBASE (7 bits,
    //         register pair index, actual register = sbase * 2).
    // Word 1: OFFSET (21-bit unsigned immediate).
    if (words.size() < 2) return false;
    const std::uint32_t word1 = words[1];

    const std::uint32_t sdst_raw = ExtractBits(word, 7, 7);
    const std::uint32_t sbase_raw = ExtractBits(word, 0, 7);
    const std::uint32_t offset = ExtractBits(word1, 0, 21);

    // Determine destination width from instruction name.
    std::uint8_t dst_width = 1;  // B32 = 1 dword
    if (instruction_name == "S_LOAD_B64" ||
        instruction_name == "S_BUFFER_LOAD_B64") {
      dst_width = 2;
    } else if (instruction_name == "S_LOAD_B128" ||
               instruction_name == "S_BUFFER_LOAD_B128") {
      dst_width = 4;
    } else if (instruction_name == "S_LOAD_B256" ||
               instruction_name == "S_BUFFER_LOAD_B256") {
      dst_width = 8;
    }

    InstructionOperand dst =
        InstructionOperand::Sgpr(static_cast<std::uint16_t>(sdst_raw))
            .WithDescriptor(MakeScalarRegisterDescriptor(
                OperandRole::kDestination, OperandSlotKind::kScalarDestination,
                OperandAccess::kWrite, static_cast<std::uint8_t>(dst_width * 32),
                dst_width));

    // SBASE is a register pair (base address pointer).
    InstructionOperand sbase =
        InstructionOperand::Sgpr(static_cast<std::uint16_t>(sbase_raw * 2))
            .WithDescriptor(MakeScalarRegisterDescriptor(
                OperandRole::kSource0, OperandSlotKind::kSource0,
                OperandAccess::kRead, 64, 2));

    InstructionOperand imm_offset =
        InstructionOperand::Imm32(offset)
            .WithDescriptor(MakeImmediateDescriptor(OperandRole::kSource1,
                                                   OperandSlotKind::kSource1));

    *instruction = DecodedInstruction{};
    instruction->opcode = instruction_name;
    instruction->operand_count = 3;
    instruction->operands[0] = dst;
    instruction->operands[1] = sbase;
    instruction->operands[2] = imm_offset;
    *words_consumed = 2;
  } else if (instruction_name == "GLOBAL_LOAD_B32" ||
             instruction_name == "GLOBAL_LOAD_B64" ||
             instruction_name == "GLOBAL_LOAD_B128" ||
             instruction_name == "GLOBAL_LOAD_U8" ||
             instruction_name == "GLOBAL_LOAD_I8" ||
             instruction_name == "GLOBAL_LOAD_U16" ||
             instruction_name == "GLOBAL_LOAD_I16") {
    // VGLOBAL 3-dword encoding for loads (gfx12).
    // Word 0: bits[24:14]=opcode (11 bits), bits[13:0]=OFFSET (signed 14-bit).
    // Word 1: bits[7:0]=VDST (8 bits).
    // Word 2: bits[7:0]=VADDR (8 bits), bits[14:8]=SADDR (7 bits, 0x7f=off).
    if (words.size() < 3) return false;
    const std::uint32_t word1 = words[1];
    const std::uint32_t word2 = words[2];

    const std::uint32_t vdst_raw = ExtractBits(word1, 0, 8);
    const std::uint32_t vaddr_raw = ExtractBits(word2, 0, 8);
    const std::uint32_t saddr_raw = ExtractBits(word2, 8, 7);
    const std::uint32_t offset = ExtractBits(word, 0, 14);

    // Determine destination width.
    std::uint8_t dst_components = 1;
    if (instruction_name == "GLOBAL_LOAD_B64") {
      dst_components = 2;
    } else if (instruction_name == "GLOBAL_LOAD_B128") {
      dst_components = 4;
    }

    InstructionOperand dst =
        InstructionOperand::Vgpr(static_cast<std::uint16_t>(vdst_raw))
            .WithDescriptor(MakeVectorRegisterDescriptor(
                OperandRole::kDestination, OperandSlotKind::kDestination,
                OperandAccess::kWrite,
                static_cast<std::uint8_t>(dst_components * 32), dst_components));

    InstructionOperand vaddr =
        InstructionOperand::Vgpr(static_cast<std::uint16_t>(vaddr_raw))
            .WithDescriptor(MakeVectorRegisterDescriptor(
                OperandRole::kSource0, OperandSlotKind::kSource0,
                OperandAccess::kRead, 64, 2));

    *instruction = DecodedInstruction{};
    instruction->opcode = instruction_name;

    if (saddr_raw != 0x7f) {
      InstructionOperand saddr =
          InstructionOperand::Sgpr(static_cast<std::uint16_t>(saddr_raw * 2))
              .WithDescriptor(MakeScalarRegisterDescriptor(
                  OperandRole::kSource1, OperandSlotKind::kSource1,
                  OperandAccess::kRead, 64, 2));
      instruction->operand_count = 4;
      instruction->operands[0] = dst;
      instruction->operands[1] = vaddr;
      instruction->operands[2] = saddr;
      instruction->operands[3] =
          InstructionOperand::Imm32(offset)
              .WithDescriptor(MakeImmediateDescriptor(
                  OperandRole::kSource2, OperandSlotKind::kSource2));
    } else {
      instruction->operand_count = 3;
      instruction->operands[0] = dst;
      instruction->operands[1] = vaddr;
      instruction->operands[2] =
          InstructionOperand::Imm32(offset)
              .WithDescriptor(MakeImmediateDescriptor(
                  OperandRole::kSource1, OperandSlotKind::kSource1));
    }
    *words_consumed = 3;
  } else if (instruction_name == "GLOBAL_STORE_B32" ||
             instruction_name == "GLOBAL_STORE_B64" ||
             instruction_name == "GLOBAL_STORE_B128" ||
             instruction_name == "GLOBAL_STORE_B8" ||
             instruction_name == "GLOBAL_STORE_B16") {
    // VGLOBAL 3-dword encoding for stores (gfx12).
    // Word 0: bits[24:14]=opcode (11 bits), bits[13:0]=OFFSET (signed 14-bit).
    // Word 1: bits[31:24]=VDATA (8 bits).
    // Word 2: bits[7:0]=VADDR (8 bits), bits[14:8]=SADDR (7 bits, 0x7f=off).
    if (words.size() < 3) return false;
    const std::uint32_t word1 = words[1];
    const std::uint32_t word2 = words[2];

    const std::uint32_t vdata_raw = ExtractBits(word1, 24, 8);
    const std::uint32_t vaddr_raw = ExtractBits(word2, 0, 8);
    const std::uint32_t saddr_raw = ExtractBits(word2, 8, 7);
    const std::uint32_t offset = ExtractBits(word, 0, 14);

    // Determine source data width.
    std::uint8_t data_components = 1;
    if (instruction_name == "GLOBAL_STORE_B64") {
      data_components = 2;
    } else if (instruction_name == "GLOBAL_STORE_B128") {
      data_components = 4;
    }

    InstructionOperand vaddr =
        InstructionOperand::Vgpr(static_cast<std::uint16_t>(vaddr_raw))
            .WithDescriptor(MakeVectorRegisterDescriptor(
                OperandRole::kDestination, OperandSlotKind::kDestination,
                OperandAccess::kRead, 64, 2));

    InstructionOperand vdata =
        InstructionOperand::Vgpr(static_cast<std::uint16_t>(vdata_raw))
            .WithDescriptor(MakeVectorRegisterDescriptor(
                OperandRole::kSource0, OperandSlotKind::kSource0,
                OperandAccess::kRead,
                static_cast<std::uint8_t>(data_components * 32),
                data_components));

    *instruction = DecodedInstruction{};
    instruction->opcode = instruction_name;

    if (saddr_raw != 0x7f) {
      InstructionOperand saddr =
          InstructionOperand::Sgpr(static_cast<std::uint16_t>(saddr_raw * 2))
              .WithDescriptor(MakeScalarRegisterDescriptor(
                  OperandRole::kSource1, OperandSlotKind::kSource1,
                  OperandAccess::kRead, 64, 2));
      instruction->operand_count = 4;
      instruction->operands[0] = vaddr;
      instruction->operands[1] = vdata;
      instruction->operands[2] = saddr;
      instruction->operands[3] =
          InstructionOperand::Imm32(offset)
              .WithDescriptor(MakeImmediateDescriptor(
                  OperandRole::kSource2, OperandSlotKind::kSource2));
    } else {
      instruction->operand_count = 3;
      instruction->operands[0] = vaddr;
      instruction->operands[1] = vdata;
      instruction->operands[2] =
          InstructionOperand::Imm32(offset)
              .WithDescriptor(MakeImmediateDescriptor(
                  OperandRole::kSource1, OperandSlotKind::kSource1));
    }
    *words_consumed = 3;
  } else {
    return false;
  }

  if (error_message != nullptr) {
    error_message->clear();
  }
  return true;
}

}  // namespace

bool Gfx1201BinaryDecoder::DecodeInstruction(
    std::span<const std::uint32_t> words,
    DecodedInstruction* instruction,
    std::size_t* words_consumed,
    std::string* error_message) const {
  if (instruction == nullptr || words_consumed == nullptr) {
    if (error_message != nullptr) {
      *error_message = "decode outputs must not be null";
    }
    return false;
  }

  *instruction = DecodedInstruction{};
  *words_consumed = 0;

  if (words.empty()) {
    if (error_message != nullptr) {
      *error_message = "instruction stream is empty";
    }
    return false;
  }

  Gfx1201OpcodeRoute route;
  if (SelectPhase0ComputeRoute(words, &route, error_message)) {
    std::string decode_error;
    if (route.status == Gfx1201OpcodeRouteStatus::kMatchedSeedEntry &&
        TryDecodeExecutableSeedInstruction(route, words, instruction,
                                           words_consumed, &decode_error)) {
      if (error_message != nullptr) {
        error_message->clear();
      }
      return true;
    }
    if (error_message != nullptr) {
      *error_message =
          decode_error.empty() ? BuildRouteMessage(route) : decode_error;
    }
    return false;
  }

  if (error_message != nullptr) {
    *error_message = BuildDecoderBringupMessage();
  }
  return false;
}

bool Gfx1201BinaryDecoder::DecodeProgram(std::span<const std::uint32_t> words,
                                         std::vector<DecodedInstruction>* program,
                                         std::string* error_message) const {
  if (program == nullptr) {
    if (error_message != nullptr) {
      *error_message = "decoded program output must not be null";
    }
    return false;
  }

  program->clear();
  std::size_t word_offset = 0;
  while (word_offset < words.size()) {
    DecodedInstruction instruction;
    std::size_t words_consumed = 0;
    std::string decode_error;
    if (!DecodeInstruction(words.subspan(word_offset), &instruction, &words_consumed,
                           &decode_error)) {
      if (error_message != nullptr) {
        *error_message = decode_error;
      }
      return false;
    }
    if (words_consumed == 0) {
      if (error_message != nullptr) {
        *error_message = "gfx1201 decoder made no progress";
      }
      return false;
    }
    program->push_back(instruction);
    word_offset += words_consumed;
  }

  if (error_message != nullptr) {
    error_message->clear();
  }
  return true;
}

bool Gfx1201BinaryDecoder::SelectPhase0ComputeRoute(
    std::span<const std::uint32_t> words,
    Gfx1201OpcodeRoute* route,
    std::string* error_message) const {
  return SelectGfx1201Phase0ComputeOpcodeRoute(words, route, error_message);
}

bool Gfx1201BinaryDecoder::SupportsPhase0ExecutableOpcode(
    std::string_view opcode) const {
  return IsPhase0ExecutableOpcode(NormalizeExecutableInstructionName(opcode));
}

std::span<const Gfx1201DecoderSeedEncoding>
Gfx1201BinaryDecoder::Phase0ComputeSeeds() const {
  return GetGfx1201Phase0ComputeDecoderSeeds();
}

std::span<const std::string_view> Gfx1201BinaryDecoder::Phase0ExecutableOpcodes()
    const {
  return kPhase0ExecutableOpcodes;
}

std::span<const Gfx1201OpcodeSelectorRule>
Gfx1201BinaryDecoder::Phase0ComputeSelectorRules() const {
  return GetGfx1201Phase0ComputeOpcodeSelectorRules();
}

const Gfx1201DecoderSeedEncoding* Gfx1201BinaryDecoder::FindPhase0ComputeSeed(
    std::string_view encoding_name) const {
  return FindGfx1201Phase0ComputeDecoderSeed(encoding_name);
}

std::span<const Gfx1201EncodingFocus> Gfx1201BinaryDecoder::Phase0EncodingFocus()
    const {
  return GetGfx1201Phase0DecoderFocus();
}

std::span<const Gfx1201EncodingFocus> Gfx1201BinaryDecoder::Phase1EncodingFocus()
    const {
  return GetGfx1201Phase1DecoderFocus();
}

std::string_view Gfx1201BinaryDecoder::BringupStatus() const {
  return DescribeGfx1201BringupPhase();
}

}  // namespace mirage::sim::isa
