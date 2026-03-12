#include "lib/sim/isa/jit/translation_rules.h"

#include <algorithm>
#include <array>

#include "lib/sim/isa/jit/translation_diagnostics.h"

namespace mirage::sim::isa::jit {

namespace {

// Wave-topology-sensitive opcodes that cannot safely execute in
// wave32-in-wave64 mode because their semantics depend on lanes 32..63
// or on wave-topology-sensitive cross-lane behavior.
constexpr std::array<std::string_view, 6> kWaveSensitiveOpcodes = {
    "V_READLANE_B32",
    "V_WRITELANE_B32",
    "DS_PERMUTE_B32",
    "DS_BPERMUTE_B32",
    "V_MBCNT_LO_U32_B32",
    "V_MBCNT_HI_U32_B32",
};

// Opcodes that directly read or modify the EXEC mask.  These are safe in
// wave32-in-wave64 when the initial EXEC mask is narrowed, but the
// translator must flag their presence so callers can verify the EXEC
// invariant is maintained throughout program execution.
constexpr std::array<std::string_view, 8> kExecManipulatingOpcodes = {
    "S_AND_SAVEEXEC_B64",
    "S_OR_SAVEEXEC_B64",
    "S_XOR_SAVEEXEC_B64",
    "S_NAND_SAVEEXEC_B64",
    "S_NOR_SAVEEXEC_B64",
    "S_XNOR_SAVEEXEC_B64",
    "S_CBRANCH_EXECZ",
    "S_CBRANCH_EXECNZ",
};

// gfx1201 -> gfx950 identity opcodes: instructions whose opcode name,
// operand layout, and semantics are identical across both architectures.
// Sourced from the kTransferableAsIs bucket in the gfx1201 support catalog.
constexpr std::array<std::string_view, 132> kGfx1201ToGfx950IdentityOpcodes = {
    // Scalar move and control
    "S_MOV_B32", "S_MOV_B64", "S_CMOV_B32", "S_CMOV_B64",
    "S_NOT_B32", "S_NOT_B64", "S_BREV_B32", "S_BREV_B64",
    "S_ABS_I32", "S_ABSDIFF_I32",
    "S_BCNT0_I32_B32", "S_BCNT0_I32_B64", "S_BCNT1_I32_B32", "S_BCNT1_I32_B64",
    "S_SEXT_I32_I8", "S_SEXT_I32_I16",
    "S_BITSET0_B32", "S_BITSET0_B64", "S_BITSET1_B32", "S_BITSET1_B64",
    "S_BITREPLICATE_B64_B32",
    "S_QUADMASK_B32", "S_QUADMASK_B64",
    "S_PACK_LL_B32_B16", "S_PACK_LH_B32_B16", "S_PACK_HH_B32_B16",

    // Scalar ALU
    "S_ADD_U32", "S_SUB_U32", "S_ADD_I32", "S_SUB_I32",
    "S_MUL_I32", "S_MUL_HI_I32", "S_MUL_HI_U32",
    "S_MIN_I32", "S_MIN_U32", "S_MAX_I32", "S_MAX_U32",
    "S_CSELECT_B32", "S_CSELECT_B64",
    "S_AND_B32", "S_AND_B64", "S_OR_B32", "S_OR_B64",
    "S_XOR_B32", "S_XOR_B64", "S_NAND_B32", "S_NAND_B64",
    "S_NOR_B32", "S_NOR_B64", "S_XNOR_B32", "S_XNOR_B64",
    "S_LSHL_B32", "S_LSHL_B64", "S_LSHR_B32", "S_LSHR_B64",
    "S_ASHR_I32", "S_ASHR_I64",
    "S_LSHL1_ADD_U32", "S_LSHL2_ADD_U32", "S_LSHL3_ADD_U32", "S_LSHL4_ADD_U32",
    "S_BFE_U32", "S_BFE_I32", "S_BFE_U64", "S_BFE_I64",
    "S_BFM_B32", "S_BFM_B64",

    // Scalar compare
    "S_CMP_EQ_I32", "S_CMP_EQ_U32", "S_CMP_EQ_U64",
    "S_CMP_LG_I32", "S_CMP_LG_U32", "S_CMP_LG_U64",
    "S_CMP_GT_I32", "S_CMP_GT_U32", "S_CMP_GE_I32", "S_CMP_GE_U32",
    "S_CMP_LT_I32", "S_CMP_LT_U32", "S_CMP_LE_I32", "S_CMP_LE_U32",
    "S_BITCMP0_B32", "S_BITCMP0_B64", "S_BITCMP1_B32", "S_BITCMP1_B64",

    // Scalar SAVEEXEC
    "S_AND_SAVEEXEC_B64", "S_OR_SAVEEXEC_B64",
    "S_XOR_SAVEEXEC_B64", "S_NAND_SAVEEXEC_B64",
    "S_NOR_SAVEEXEC_B64", "S_XNOR_SAVEEXEC_B64",

    // Scalar branch and control
    "S_BRANCH", "S_CBRANCH_SCC0", "S_CBRANCH_SCC1",
    "S_CBRANCH_VCCZ", "S_CBRANCH_VCCNZ",
    "S_CBRANCH_EXECZ", "S_CBRANCH_EXECNZ",
    "S_ENDPGM", "S_NOP", "S_BARRIER",

    // Vector move
    "V_MOV_B32",
    "V_READFIRSTLANE_B32",

    // Vector ALU (subset)
    "V_ADD_F32", "V_SUB_F32", "V_MUL_F32",
    "V_ADD_U32", "V_SUB_U32",
    "V_AND_B32", "V_OR_B32", "V_XOR_B32",
    "V_LSHLREV_B32", "V_LSHRREV_B32", "V_ASHRREV_I32",
    "V_MAX_F32", "V_MIN_F32", "V_MAX_I32", "V_MIN_I32",
    "V_MAX_U32", "V_MIN_U32",
    "V_CVT_F32_I32", "V_CVT_F32_U32", "V_CVT_U32_F32", "V_CVT_I32_F32",
    "V_CEIL_F32", "V_FLOOR_F32", "V_TRUNC_F32", "V_RNDNE_F32",
    "V_BFREV_B32",
    "V_BFE_U32", "V_BFE_I32", "V_BFI_B32",
};

// gfx1201 -> gfx950 rename opcodes: instructions that have a different
// mnemonic but identical semantics and operand layout.
struct OpcodeRename {
  std::string_view source;
  std::string_view target;
};

constexpr std::array<OpcodeRename, 6> kGfx1201ToGfx950Renames = {{
    {"S_ADD_CO_U32", "S_ADD_U32"},
    {"S_SUB_CO_U32", "S_SUB_U32"},
    {"S_ADDC_CO_U32", "S_ADDC_U32"},
    {"S_SUBB_CO_U32", "S_SUBB_U32"},
    {"V_ADD_CO_U32", "V_ADD_U32"},
    {"V_SUB_CO_U32", "V_SUB_U32"},
}};

}  // namespace

void TranslationRuleTable::BuildForDirection(std::uint8_t source_arch,
                                              std::uint8_t target_arch) {
  rules_.clear();

  const bool is_gfx1201_to_gfx950 =
      source_arch == static_cast<std::uint8_t>(SourceArchitecture::kGfx1201) &&
      target_arch == static_cast<std::uint8_t>(TargetArchitecture::kGfx950);

  if (is_gfx1201_to_gfx950) {
    rules_.reserve(kGfx1201ToGfx950IdentityOpcodes.size() +
                   kGfx1201ToGfx950Renames.size());

    for (const auto& opcode : kGfx1201ToGfx950IdentityOpcodes) {
      TranslationRule rule;
      rule.source_opcode = opcode;
      rule.target_opcode = opcode;
      rule.tier = TranslationTier::kIdentity;
      rule.is_branch =
          (opcode == "S_BRANCH" || opcode == "S_CBRANCH_SCC0" ||
           opcode == "S_CBRANCH_SCC1" || opcode == "S_CBRANCH_VCCZ" ||
           opcode == "S_CBRANCH_VCCNZ" || opcode == "S_CBRANCH_EXECZ" ||
           opcode == "S_CBRANCH_EXECNZ");
      rule.is_exec_manipulating =
          std::find(kExecManipulatingOpcodes.begin(),
                    kExecManipulatingOpcodes.end(),
                    opcode) != kExecManipulatingOpcodes.end();
      rules_.push_back(rule);
    }

    for (const auto& rename : kGfx1201ToGfx950Renames) {
      TranslationRule rule;
      rule.source_opcode = rename.source;
      rule.target_opcode = rename.target;
      rule.tier = TranslationTier::kRename;
      rule.requires_vcc_remap = true;
      rules_.push_back(rule);
    }
  }

  std::sort(rules_.begin(), rules_.end(),
            [](const TranslationRule& a, const TranslationRule& b) {
              return a.source_opcode < b.source_opcode;
            });
}

const TranslationRule* TranslationRuleTable::FindRule(
    std::string_view source_opcode) const {
  auto it = std::lower_bound(
      rules_.begin(), rules_.end(), source_opcode,
      [](const TranslationRule& rule, std::string_view opcode) {
        return rule.source_opcode < opcode;
      });
  if (it != rules_.end() && it->source_opcode == source_opcode) {
    return &(*it);
  }
  return nullptr;
}

TranslationTier TranslationRuleTable::Classify(
    std::string_view source_opcode) const {
  const TranslationRule* rule = FindRule(source_opcode);
  if (rule != nullptr) {
    return rule->tier;
  }
  return TranslationTier::kUnsupported;
}

std::size_t TranslationRuleTable::identity_count() const {
  return static_cast<std::size_t>(std::count_if(
      rules_.begin(), rules_.end(), [](const TranslationRule& r) {
        return r.tier == TranslationTier::kIdentity;
      }));
}

std::size_t TranslationRuleTable::rename_count() const {
  return static_cast<std::size_t>(std::count_if(
      rules_.begin(), rules_.end(), [](const TranslationRule& r) {
        return r.tier == TranslationTier::kRename;
      }));
}

std::size_t TranslationRuleTable::fixup_count() const {
  return static_cast<std::size_t>(std::count_if(
      rules_.begin(), rules_.end(), [](const TranslationRule& r) {
        return r.tier == TranslationTier::kOperandFixup;
      }));
}

std::size_t TranslationRuleTable::unsupported_count() const {
  return static_cast<std::size_t>(std::count_if(
      rules_.begin(), rules_.end(), [](const TranslationRule& r) {
        return r.tier == TranslationTier::kUnsupported;
      }));
}

std::span<const std::string_view> GetWaveSensitiveOpcodes() {
  return kWaveSensitiveOpcodes;
}

std::span<const std::string_view> GetExecManipulatingOpcodes() {
  return kExecManipulatingOpcodes;
}

}  // namespace mirage::sim::isa::jit
