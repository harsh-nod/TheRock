#include <cstddef>
#include <iostream>

#include "lib/sim/isa/jit/translation_rules.h"

namespace {

bool Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    return false;
  }
  return true;
}

using namespace mirage::sim::isa::jit;

bool TestBuildGfx1201ToGfx950() {
  TranslationRuleTable table;
  table.BuildForDirection(
      static_cast<std::uint8_t>(SourceArchitecture::kGfx1201),
      static_cast<std::uint8_t>(TargetArchitecture::kGfx950));

  return Expect(table.rules().size() > 100,
                "gfx1201->gfx950 rule table should have >100 rules") &&
         Expect(table.identity_count() > 90,
                "should have >90 identity rules") &&
         Expect(table.rename_count() > 0,
                "should have >0 rename rules") &&
         Expect(table.version() > 0,
                "version should be positive");
}

bool TestIdentityClassification() {
  TranslationRuleTable table;
  table.BuildForDirection(
      static_cast<std::uint8_t>(SourceArchitecture::kGfx1201),
      static_cast<std::uint8_t>(TargetArchitecture::kGfx950));

  return Expect(table.Classify("S_MOV_B32") == TranslationTier::kIdentity,
                "S_MOV_B32 should be identity") &&
         Expect(table.Classify("S_ENDPGM") == TranslationTier::kIdentity,
                "S_ENDPGM should be identity") &&
         Expect(table.Classify("V_ADD_F32") == TranslationTier::kIdentity,
                "V_ADD_F32 should be identity") &&
         Expect(table.Classify("S_BRANCH") == TranslationTier::kIdentity,
                "S_BRANCH should be identity");
}

bool TestRenameClassification() {
  TranslationRuleTable table;
  table.BuildForDirection(
      static_cast<std::uint8_t>(SourceArchitecture::kGfx1201),
      static_cast<std::uint8_t>(TargetArchitecture::kGfx950));

  return Expect(table.Classify("S_ADD_CO_U32") == TranslationTier::kRename,
                "S_ADD_CO_U32 should be rename") &&
         Expect(table.Classify("V_ADD_CO_U32") == TranslationTier::kRename,
                "V_ADD_CO_U32 should be rename");
}

bool TestRenameTarget() {
  TranslationRuleTable table;
  table.BuildForDirection(
      static_cast<std::uint8_t>(SourceArchitecture::kGfx1201),
      static_cast<std::uint8_t>(TargetArchitecture::kGfx950));

  const TranslationRule* rule = table.FindRule("S_ADD_CO_U32");
  return Expect(rule != nullptr, "S_ADD_CO_U32 rule should exist") &&
         Expect(rule->target_opcode == "S_ADD_U32",
                "S_ADD_CO_U32 should rename to S_ADD_U32") &&
         Expect(rule->requires_vcc_remap,
                "S_ADD_CO_U32 should require VCC remap");
}

bool TestUnknownOpcodeReturnsUnsupported() {
  TranslationRuleTable table;
  table.BuildForDirection(
      static_cast<std::uint8_t>(SourceArchitecture::kGfx1201),
      static_cast<std::uint8_t>(TargetArchitecture::kGfx950));

  return Expect(table.Classify("FAKE_OPCODE_XYZ") ==
                    TranslationTier::kUnsupported,
                "unknown opcode should be unsupported") &&
         Expect(table.FindRule("FAKE_OPCODE_XYZ") == nullptr,
                "unknown opcode should return nullptr rule");
}

bool TestBranchFlagSet() {
  TranslationRuleTable table;
  table.BuildForDirection(
      static_cast<std::uint8_t>(SourceArchitecture::kGfx1201),
      static_cast<std::uint8_t>(TargetArchitecture::kGfx950));

  const TranslationRule* branch_rule = table.FindRule("S_BRANCH");
  const TranslationRule* mov_rule = table.FindRule("S_MOV_B32");
  return Expect(branch_rule != nullptr && branch_rule->is_branch,
                "S_BRANCH should have is_branch=true") &&
         Expect(mov_rule != nullptr && !mov_rule->is_branch,
                "S_MOV_B32 should have is_branch=false");
}

bool TestWaveSensitiveOpcodes() {
  auto opcodes = GetWaveSensitiveOpcodes();
  bool found_readlane = false;
  bool found_permute = false;
  for (const auto& op : opcodes) {
    if (op == "V_READLANE_B32") found_readlane = true;
    if (op == "DS_PERMUTE_B32") found_permute = true;
  }
  return Expect(opcodes.size() >= 6,
                "should have at least 6 wave-sensitive opcodes") &&
         Expect(found_readlane, "V_READLANE_B32 should be wave-sensitive") &&
         Expect(found_permute, "DS_PERMUTE_B32 should be wave-sensitive");
}

}  // namespace

int main() {
  bool ok = true;
  ok = TestBuildGfx1201ToGfx950() && ok;
  ok = TestIdentityClassification() && ok;
  ok = TestRenameClassification() && ok;
  ok = TestRenameTarget() && ok;
  ok = TestUnknownOpcodeReturnsUnsupported() && ok;
  ok = TestBranchFlagSet() && ok;
  ok = TestWaveSensitiveOpcodes() && ok;

  if (ok) {
    std::cerr << "All translation_rules tests passed.\n";
  }
  return ok ? 0 : 1;
}
