#include "lib/sim/isa/jit/semantic_lowering.h"

namespace mirage::sim::isa::jit {

TranslationStatus SemanticLowering::ClassifyForLowering(
    const SemanticInstruction& instruction) const {
  switch (instruction.family) {
    case SemanticFamily::kScalar:
    case SemanticFamily::kVector:
    case SemanticFamily::kMemory:
    case SemanticFamily::kBranch:
      return TranslationStatus::kUnsupported;

    case SemanticFamily::kBarrier:
      return TranslationStatus::kRequiresSemanticLowering;

    case SemanticFamily::kTensorMemory:
      return TranslationStatus::kBlockedOnRuntime;

    case SemanticFamily::kTranspose:
      return TranslationStatus::kCoverageOnly;

    case SemanticFamily::kWmma:
    case SemanticFamily::kSwmmac:
    case SemanticFamily::kMfma:
      return TranslationStatus::kBlockedOnRuntime;

    case SemanticFamily::kFp8Bf8:
    case SemanticFamily::kScalePaired:
      return TranslationStatus::kCoverageOnly;

    case SemanticFamily::kOther:
      return TranslationStatus::kUnsupported;
  }
  return TranslationStatus::kUnsupported;
}

bool SemanticLowering::LowerToTarget(
    const SemanticInstruction& /*instruction*/,
    std::uint8_t /*target_arch*/,
    std::vector<DecodedInstruction>* /*output*/,
    std::string* error_message) const {
  if (error_message != nullptr) {
    *error_message =
        "Semantic lowering is not yet implemented. This is a Phase 3+ feature.";
  }
  return false;
}

bool SemanticLowering::LiftFromDecoded(
    const DecodedInstruction& /*instruction*/,
    std::uint8_t /*source_arch*/,
    SemanticInstruction* /*output*/,
    std::string* error_message) const {
  if (error_message != nullptr) {
    *error_message =
        "Semantic lifting is not yet implemented. This is a Phase 3+ feature.";
  }
  return false;
}

}  // namespace mirage::sim::isa::jit
