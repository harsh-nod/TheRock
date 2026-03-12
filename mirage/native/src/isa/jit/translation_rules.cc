#include "lib/sim/isa/jit/translation_rules.h"

#include <algorithm>
#include <array>

#include "lib/sim/isa/jit/translation_diagnostics.h"

namespace mirage::sim::isa::jit {

namespace {

// Wave-topology-sensitive opcodes that cannot safely execute in
// wave32-in-wave64 mode because their semantics depend on lanes 32..63,
// on wave-topology-sensitive cross-lane behavior, or on arbitrary lane
// index addressing that may reference inactive lanes 32..63.
//
// V_READFIRSTLANE_B32 is intentionally NOT in this list: it reads from
// the first active lane which is always within lanes 0..31 when the
// EXEC mask is narrowed to wave32.
//
// S_QUADMASK_B32/B64 are intentionally NOT in this list: quad masks
// are computed from the EXEC mask and produce correct results when
// the upper 32 lanes are inactive.
constexpr std::array<std::string_view, 8> kWaveSensitiveOpcodes = {
    "V_READLANE_B32",
    "V_WRITELANE_B32",
    "DS_PERMUTE_B32",
    "DS_BPERMUTE_B32",
    "DS_SWIZZLE_B32",
    "DS_BPERMUTE_FI_B32",
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

// gfx1201 scheduling hint opcodes that have no gfx950 equivalent.
// These are converted to identity S_NOP instructions during translation
// since they are purely scheduling/timing hints with no functional effect.
constexpr std::array<std::string_view, 6> kGfx1201SchedulingStripOpcodes = {
    "S_CLAUSE",       // Software clause grouping hint.
    "S_DELAY_ALU",    // VALU/SALU scheduling hint.
    "S_WAIT_KMCNT",   // Scalar memory wait counter (replaces S_WAITCNT for SMEM).
    "S_WAIT_ALU",     // ALU pipeline wait (gfx12 specific).
    "S_WAIT_LOADCNT", // Vector memory load wait counter.
    "S_CODE_END",     // Post-endpgm padding (NOP fill).
};

// gfx1201 -> gfx950 identity opcodes: instructions whose opcode name,
// operand layout, and semantics are identical across both architectures.
// Sourced from the kTransferableAsIs bucket in the gfx1201 support catalog.
constexpr std::array<std::string_view, 173> kGfx1201ToGfx950IdentityOpcodes = {
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

    // 64-bit vector ALU (shared between gfx950 and gfx1201)
    "V_MAD_CO_U64_U32",
    "V_ADD_CO_CI_U32",
    "V_LSHLREV_B64",

    // Vector compare-and-write-exec (VOPC): these compare two operands and
    // write the result directly to the EXEC mask.  Present on both gfx9 and
    // gfx12 with identical semantics (though gfx12 is wave32 and gfx9 is
    // wave64, the translator handles EXEC width via WaveAdapter).
    "V_CMPX_LT_F32", "V_CMPX_EQ_F32", "V_CMPX_LE_F32",
    "V_CMPX_GT_F32", "V_CMPX_LG_F32", "V_CMPX_GE_F32",
    "V_CMPX_LT_I32", "V_CMPX_EQ_I32", "V_CMPX_LE_I32",
    "V_CMPX_GT_I32", "V_CMPX_NE_I32", "V_CMPX_GE_I32",
    "V_CMPX_LT_U32", "V_CMPX_EQ_U32", "V_CMPX_LE_U32",
    "V_CMPX_GT_U32", "V_CMPX_NE_U32", "V_CMPX_GE_U32",

    // Scalar memory: identical on gfx950 and gfx1201 when using DWORD naming.
    "S_LOAD_DWORD", "S_LOAD_DWORDX2", "S_LOAD_DWORDX4", "S_LOAD_DWORDX8",
    "S_BUFFER_LOAD_DWORD", "S_BUFFER_LOAD_DWORDX2",
    "S_BUFFER_LOAD_DWORDX4", "S_BUFFER_LOAD_DWORDX8",

    // Global memory: identical addressing model on gfx950 and gfx1201
    // when using DWORD naming.
    "GLOBAL_LOAD_DWORD", "GLOBAL_LOAD_DWORDX2", "GLOBAL_LOAD_DWORDX4",
    "GLOBAL_STORE_DWORD", "GLOBAL_STORE_DWORDX2", "GLOBAL_STORE_DWORDX4",
    "GLOBAL_LOAD_UBYTE", "GLOBAL_LOAD_SBYTE",
    "GLOBAL_LOAD_USHORT", "GLOBAL_LOAD_SSHORT",
    "GLOBAL_STORE_BYTE", "GLOBAL_STORE_SHORT",
};

// gfx1201 -> gfx950 rename opcodes: instructions that have a different
// mnemonic but identical semantics and operand layout.
struct OpcodeRename {
  std::string_view source;
  std::string_view target;
};

constexpr std::array<OpcodeRename, 28> kGfx1201ToGfx950Renames = {{
    // Scalar ALU carry-out renames.
    {"S_ADD_CO_U32", "S_ADD_U32"},
    {"S_SUB_CO_U32", "S_SUB_U32"},
    {"S_ADDC_CO_U32", "S_ADDC_U32"},
    {"S_SUBB_CO_U32", "S_SUBB_U32"},
    {"V_ADD_CO_U32", "V_ADD_U32"},
    {"V_SUB_CO_U32", "V_SUB_U32"},

    // Scalar memory: gfx1201 uses B32/B64/B128/B256 naming,
    // gfx950 uses DWORD/DWORDX2/DWORDX4/DWORDX8 naming.
    // Operand layout and semantics are identical.
    {"S_LOAD_B32", "S_LOAD_DWORD"},
    {"S_LOAD_B64", "S_LOAD_DWORDX2"},
    {"S_LOAD_B128", "S_LOAD_DWORDX4"},
    {"S_LOAD_B256", "S_LOAD_DWORDX8"},
    {"S_BUFFER_LOAD_B32", "S_BUFFER_LOAD_DWORD"},
    {"S_BUFFER_LOAD_B64", "S_BUFFER_LOAD_DWORDX2"},
    {"S_BUFFER_LOAD_B128", "S_BUFFER_LOAD_DWORDX4"},
    {"S_BUFFER_LOAD_B256", "S_BUFFER_LOAD_DWORDX8"},

    // Global memory: gfx1201 uses B32/B64/B128 naming,
    // gfx950 uses DWORD/DWORDX2/DWORDX4 naming.
    // Operand layout and addressing model are identical.
    {"GLOBAL_LOAD_B32", "GLOBAL_LOAD_DWORD"},
    {"GLOBAL_LOAD_B64", "GLOBAL_LOAD_DWORDX2"},
    {"GLOBAL_LOAD_B128", "GLOBAL_LOAD_DWORDX4"},
    {"GLOBAL_STORE_B32", "GLOBAL_STORE_DWORD"},
    {"GLOBAL_STORE_B64", "GLOBAL_STORE_DWORDX2"},
    {"GLOBAL_STORE_B128", "GLOBAL_STORE_DWORDX4"},

    // Vector memory sub-dword: gfx1201 uses B8/B16 naming,
    // gfx950 uses UBYTE/USHORT naming.
    {"GLOBAL_LOAD_U8", "GLOBAL_LOAD_UBYTE"},
    {"GLOBAL_LOAD_I8", "GLOBAL_LOAD_SBYTE"},
    {"GLOBAL_LOAD_U16", "GLOBAL_LOAD_USHORT"},
    {"GLOBAL_LOAD_I16", "GLOBAL_LOAD_SSHORT"},
    {"GLOBAL_STORE_B8", "GLOBAL_STORE_BYTE"},
    {"GLOBAL_STORE_B16", "GLOBAL_STORE_SHORT"},
}};

bool IsBranchOpcodeForRule(std::string_view opcode) {
  return opcode == "S_BRANCH" || opcode == "S_CBRANCH_SCC0" ||
         opcode == "S_CBRANCH_SCC1" || opcode == "S_CBRANCH_VCCZ" ||
         opcode == "S_CBRANCH_VCCNZ" || opcode == "S_CBRANCH_EXECZ" ||
         opcode == "S_CBRANCH_EXECNZ";
}

bool IsExecManipulatingForRule(std::string_view opcode) {
  if (std::find(kExecManipulatingOpcodes.begin(),
                kExecManipulatingOpcodes.end(),
                opcode) != kExecManipulatingOpcodes.end()) {
    return true;
  }
  // V_CMPX_* opcodes write the comparison result directly to EXEC.
  if (opcode.size() >= 7 && opcode.substr(0, 7) == "V_CMPX_") {
    return true;
  }
  return false;
}

}  // namespace

void TranslationRuleTable::BuildForDirection(std::uint8_t source_arch,
                                              std::uint8_t target_arch) {
  rules_.clear();

  const bool is_gfx1201_to_gfx950 =
      source_arch == static_cast<std::uint8_t>(SourceArchitecture::kGfx1201) &&
      target_arch == static_cast<std::uint8_t>(TargetArchitecture::kGfx950);

  const bool is_gfx950_to_gfx1201 =
      source_arch == static_cast<std::uint8_t>(SourceArchitecture::kGfx950) &&
      target_arch == static_cast<std::uint8_t>(TargetArchitecture::kGfx1201);

  if (is_gfx1201_to_gfx950) {
    rules_.reserve(kGfx1201ToGfx950IdentityOpcodes.size() +
                   kGfx1201ToGfx950Renames.size() +
                   kGfx1201SchedulingStripOpcodes.size());

    for (const auto& opcode : kGfx1201ToGfx950IdentityOpcodes) {
      TranslationRule rule;
      rule.source_opcode = opcode;
      rule.target_opcode = opcode;
      rule.tier = TranslationTier::kIdentity;
      rule.is_branch = IsBranchOpcodeForRule(opcode);
      rule.is_exec_manipulating = IsExecManipulatingForRule(opcode);
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

    // gfx1201 scheduling hints → S_NOP on gfx950.  These are purely
    // performance hints with no functional effect; dropping them is safe.
    for (const auto& opcode : kGfx1201SchedulingStripOpcodes) {
      TranslationRule rule;
      rule.source_opcode = opcode;
      rule.target_opcode = "S_NOP";
      rule.tier = TranslationTier::kRename;
      rules_.push_back(rule);
    }
  }

  if (is_gfx950_to_gfx1201) {
    // The same 132 identity opcodes are valid in both directions since they
    // share identical names, operand layouts, and semantics.  No renames are
    // needed: the _CO_ variants (S_ADD_CO_U32, etc.) exist only on gfx1201
    // and never appear in gfx950 source programs.  gfx950-only opcodes
    // (ACCVGPR, MFMA) are not in this list and remain unsupported.
    rules_.reserve(kGfx1201ToGfx950IdentityOpcodes.size());

    for (const auto& opcode : kGfx1201ToGfx950IdentityOpcodes) {
      TranslationRule rule;
      rule.source_opcode = opcode;
      rule.target_opcode = opcode;
      rule.tier = TranslationTier::kIdentity;
      rule.is_branch = IsBranchOpcodeForRule(opcode);
      rule.is_exec_manipulating = IsExecManipulatingForRule(opcode);
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

bool IsLdsTouchingOpcode(std::string_view opcode) {
  // Must start with "DS_" prefix.
  if (opcode.size() < 3 || opcode.substr(0, 3) != "DS_") {
    return false;
  }
  // Lane routing opcodes use the DS hardware but do NOT access LDS memory.
  if (opcode == "DS_PERMUTE_B32" || opcode == "DS_BPERMUTE_B32" ||
      opcode == "DS_SWIZZLE_B32" || opcode == "DS_BPERMUTE_FI_B32" ||
      opcode == "DS_NOP") {
    return false;
  }
  return true;
}

}  // namespace mirage::sim::isa::jit
