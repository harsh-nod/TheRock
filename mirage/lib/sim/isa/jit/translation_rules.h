#ifndef MIRAGE_SIM_ISA_JIT_TRANSLATION_RULES_H_
#define MIRAGE_SIM_ISA_JIT_TRANSLATION_RULES_H_

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "lib/sim/isa/common/decoded_instruction.h"
#include "lib/sim/isa/jit/translation_diagnostics.h"

namespace mirage::sim::isa::jit {

enum class TranslationTier : std::uint8_t {
  kIdentity,
  kRename,
  kOperandFixup,
  kExpansion,
  kSemanticLowering,
  kUnsupported,
};

struct TranslationRule {
  std::string_view source_opcode;
  std::string_view target_opcode;
  TranslationTier tier = TranslationTier::kUnsupported;
  bool requires_vcc_remap = false;
  bool is_wave_sensitive = false;
  bool is_branch = false;
  // True for instructions that read or modify the EXEC mask
  // (SAVEEXEC, WREXEC, CBRANCH_EXEC*, S_MOV_B64 to EXEC, etc.).
  bool is_exec_manipulating = false;
};

// Returns the list of opcodes that directly manipulate the EXEC mask
// (SAVEEXEC and WREXEC variants, EXEC-conditional branches).
std::span<const std::string_view> GetExecManipulatingOpcodes();

class TranslationRuleTable {
 public:
  TranslationRuleTable() = default;

  void BuildForDirection(std::uint8_t source_arch, std::uint8_t target_arch);

  const TranslationRule* FindRule(std::string_view source_opcode) const;

  TranslationTier Classify(std::string_view source_opcode) const;

  std::span<const TranslationRule> rules() const { return rules_; }

  std::uint32_t version() const { return version_; }

  std::size_t identity_count() const;
  std::size_t rename_count() const;
  std::size_t fixup_count() const;
  std::size_t unsupported_count() const;

 private:
  std::vector<TranslationRule> rules_;
  std::uint32_t version_ = 1;
};

std::span<const std::string_view> GetWaveSensitiveOpcodes();

}  // namespace mirage::sim::isa::jit

#endif  // MIRAGE_SIM_ISA_JIT_TRANSLATION_RULES_H_
