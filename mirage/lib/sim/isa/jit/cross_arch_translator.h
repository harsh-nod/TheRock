#ifndef MIRAGE_SIM_ISA_JIT_CROSS_ARCH_TRANSLATOR_H_
#define MIRAGE_SIM_ISA_JIT_CROSS_ARCH_TRANSLATOR_H_

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "lib/sim/isa/common/decoded_instruction.h"
#include "lib/sim/isa/jit/translation_diagnostics.h"
#include "lib/sim/isa/jit/translation_rules.h"
#include "lib/sim/isa/jit/wave_adapter.h"

namespace mirage::sim::isa::jit {

enum class TranslationMode : std::uint8_t {
  kExecutableStrict,
  kCoverageOnly,
  kApproximateExperimental,
};

enum class HazardPolicy : std::uint8_t {
  kPassthrough,
  kStripSource,
  kRetarget,
};

struct TranslationConfig {
  SourceArchitecture source_arch = SourceArchitecture::kGfx1201;
  TargetArchitecture target_arch = TargetArchitecture::kGfx950;
  TranslationMode translation_mode = TranslationMode::kExecutableStrict;
  bool allow_approximate_lowering = false;
  WavePolicy wave_policy = WavePolicy::kWave32InWave64;
  HazardPolicy hazard_policy = HazardPolicy::kPassthrough;
  DiagnosticDetail diagnostic_detail = DiagnosticDetail::kSummary;

  bool operator==(const TranslationConfig&) const = default;
};

struct TranslationCacheKey {
  std::uint64_t source_code_va = 0;
  std::uint32_t code_word_count = 0;
  SourceArchitecture source_arch = SourceArchitecture::kGfx1201;
  TargetArchitecture target_arch = TargetArchitecture::kGfx950;
  TranslationMode translation_mode = TranslationMode::kExecutableStrict;
  bool allow_approximate_lowering = false;
  WavePolicy wave_policy = WavePolicy::kWave32InWave64;
  HazardPolicy hazard_policy = HazardPolicy::kPassthrough;
  std::uint32_t rule_table_version = 0;
  std::uint64_t source_allocation_write_version = 0;

  bool operator<(const TranslationCacheKey& other) const;
};

struct CapabilitySummary {
  std::uint32_t executable_count = 0;
  std::uint32_t coverage_only_count = 0;
  std::uint32_t blocked_on_runtime_count = 0;
  std::uint32_t unsupported_count = 0;

  std::uint32_t total() const {
    return executable_count + coverage_only_count +
           blocked_on_runtime_count + unsupported_count;
  }
};

struct TranslationResult {
  TranslationStatus top_level_status = TranslationStatus::kUnsupported;
  std::vector<DecodedInstruction> translated_program;
  std::vector<InstructionDiagnostic> diagnostics;
  CapabilitySummary capability_summary;
  bool is_executable = false;
  // True when the source program was wave32 and the target is wave64.
  // The caller must narrow the EXEC mask to 32 lanes before execution.
  bool requires_exec_narrowing = false;
  // True when the translated program contains instructions that read or
  // modify the EXEC mask (SAVEEXEC, CBRANCH_EXEC*, etc.).  When combined
  // with requires_exec_narrowing, the caller should be aware that the
  // program may dynamically alter the EXEC mask width.
  bool contains_exec_manipulating_instructions = false;
};

class CrossArchTranslator {
 public:
  explicit CrossArchTranslator(TranslationConfig config);

  const TranslationConfig& config() const { return config_; }

  TranslationResult Translate(
      std::span<const DecodedInstruction> source_program) const;

  TranslationStatus ClassifyInstruction(
      std::string_view opcode) const;

  CapabilitySummary ComputeCoverage(
      std::span<const DecodedInstruction> source_program) const;

  static std::string_view ArchitectureName(SourceArchitecture arch);
  static std::string_view ArchitectureName(TargetArchitecture arch);
  static std::string_view TranslationModeName(TranslationMode mode);
  static std::string_view TranslationStatusName(TranslationStatus status);

  // Adjust branch target offsets in a translated program to account for
  // index shifts caused by 1:N instruction expansions.  Uses diagnostics
  // to map source instruction indices to output instruction indices.
  // Exposed for direct unit testing.
  static bool FixupBranchTargets(
      std::vector<DecodedInstruction>* program,
      const std::vector<InstructionDiagnostic>& diagnostics);

 private:
  bool TranslateInstruction(
      const DecodedInstruction& source,
      std::vector<DecodedInstruction>* output,
      InstructionDiagnostic* diagnostic) const;

  bool ApplyBranchFixup(
      std::vector<DecodedInstruction>* program,
      const std::vector<InstructionDiagnostic>& diagnostics) const;

  TranslationConfig config_;
  TranslationRuleTable rule_table_;
  WaveAdapter wave_adapter_;
};

}  // namespace mirage::sim::isa::jit

#endif  // MIRAGE_SIM_ISA_JIT_CROSS_ARCH_TRANSLATOR_H_
