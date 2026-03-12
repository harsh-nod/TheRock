#ifndef MIRAGE_SIM_ISA_JIT_TRANSLATION_DIAGNOSTICS_H_
#define MIRAGE_SIM_ISA_JIT_TRANSLATION_DIAGNOSTICS_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace mirage::sim::isa::jit {

enum class SourceArchitecture : std::uint8_t {
  kGfx950,
  kGfx1201,
  kGfx1250,
};

enum class TargetArchitecture : std::uint8_t {
  kGfx950,
  kGfx1201,
  kGfx1250,
};

enum class TranslationStatus : std::uint8_t {
  kIdentity,
  kRenamed,
  kRewrittenWithFixup,
  kRequiresSemanticLowering,
  kCoverageOnly,
  kBlockedOnRuntime,
  kUnsupported,
};

enum class DiagnosticDetail : std::uint8_t {
  kNone,
  kSummary,
  kPerInstruction,
  kVerbose,
};

struct InstructionDiagnostic {
  std::size_t source_index = 0;
  std::string_view source_opcode;
  TranslationStatus status = TranslationStatus::kUnsupported;
  std::string message;
  std::uint32_t output_instruction_count = 0;
  std::size_t output_begin_index = 0;
};

inline bool IsExecutable(TranslationStatus status) {
  switch (status) {
    case TranslationStatus::kIdentity:
    case TranslationStatus::kRenamed:
    case TranslationStatus::kRewrittenWithFixup:
      return true;
    case TranslationStatus::kRequiresSemanticLowering:
    case TranslationStatus::kCoverageOnly:
    case TranslationStatus::kBlockedOnRuntime:
    case TranslationStatus::kUnsupported:
      return false;
  }
  return false;
}

}  // namespace mirage::sim::isa::jit

#endif  // MIRAGE_SIM_ISA_JIT_TRANSLATION_DIAGNOSTICS_H_
