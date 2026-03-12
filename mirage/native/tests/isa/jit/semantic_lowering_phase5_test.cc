#include <iostream>
#include <string>
#include <vector>

#include "lib/sim/isa/common/decoded_instruction.h"
#include "lib/sim/isa/jit/cross_arch_translator.h"
#include "lib/sim/isa/jit/semantic_instruction.h"
#include "lib/sim/isa/jit/semantic_lowering.h"

namespace {

bool Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    return false;
  }
  return true;
}

using namespace mirage::sim::isa;
using namespace mirage::sim::isa::jit;

// ============================================================
// VOP3P Packed BF16 -> F16 lowering tests
// ============================================================

bool TestVop3pBf16LoweringKindClassification() {
  SemanticLowering lowering;

  auto pk_add = DecodedInstruction::Nullary("V_PK_ADD_BF16");
  SemanticInstruction sem;
  lowering.LiftFromDecoded(
      pk_add, static_cast<std::uint8_t>(SourceArchitecture::kGfx1250), &sem);

  bool ok = Expect(sem.family == SemanticFamily::kVector,
                   "V_PK_ADD_BF16 should be vector family") &&
            Expect(sem.lowering_kind == LoweringKind::kPackedBf16ToF16,
                   "V_PK_ADD_BF16 should have BF16->F16 lowering kind") &&
            Expect(sem.lowered_opcode == "V_PK_ADD_F16",
                   "lowered opcode should be V_PK_ADD_F16") &&
            Expect(sem.is_approximate,
                   "BF16->F16 lowering should be marked approximate") &&
            Expect(sem.input_element_type == ElementType::kBF16,
                   "input element type should be BF16") &&
            Expect(sem.output_element_type == ElementType::kF16,
                   "output element type should be F16");

  return ok;
}

bool TestVop3pBf16AllMappings() {
  SemanticLowering lowering;

  struct TestCase {
    std::string_view source;
    std::string_view expected_target;
  };

  constexpr TestCase cases[] = {
      {"V_PK_ADD_BF16", "V_PK_ADD_F16"},
      {"V_PK_MUL_BF16", "V_PK_MUL_F16"},
      {"V_PK_FMA_BF16", "V_PK_FMA_F16"},
      {"V_PK_MAX_NUM_BF16", "V_PK_MAX_F16"},
      {"V_PK_MIN_NUM_BF16", "V_PK_MIN_F16"},
  };

  bool ok = true;
  for (const auto& tc : cases) {
    auto decoded = DecodedInstruction::Nullary(tc.source);
    SemanticInstruction sem;
    lowering.LiftFromDecoded(
        decoded, static_cast<std::uint8_t>(SourceArchitecture::kGfx1250),
        &sem);

    if (sem.lowered_opcode != tc.expected_target) {
      std::cerr << "FAIL: " << tc.source << " -> expected "
                << tc.expected_target << ", got " << sem.lowered_opcode
                << "\n";
      ok = false;
    }
  }

  return ok;
}

bool TestVop3pBf16LowerToTarget() {
  SemanticLowering lowering;

  auto decoded = DecodedInstruction::Nullary("V_PK_ADD_BF16");
  SemanticInstruction sem;
  lowering.LiftFromDecoded(
      decoded, static_cast<std::uint8_t>(SourceArchitecture::kGfx1250), &sem);

  std::vector<DecodedInstruction> output;
  std::string error;
  bool ok = lowering.LowerToTarget(
      sem, static_cast<std::uint8_t>(TargetArchitecture::kGfx950),
      &output, &error);

  return Expect(ok, "LowerToTarget should succeed for V_PK_ADD_BF16") &&
         Expect(output.size() == 1, "should produce exactly 1 instruction") &&
         Expect(output[0].opcode == "V_PK_ADD_F16",
                "lowered opcode should be V_PK_ADD_F16");
}

bool TestVop3pBf16UnmappedOpcodeHasNoLowering() {
  SemanticLowering lowering;

  // V_PK_ADD_MAX_I16 is a VOP3P opcode but NOT a BF16 opcode.
  auto decoded = DecodedInstruction::Nullary("V_PK_ADD_MAX_I16");
  SemanticInstruction sem;
  lowering.LiftFromDecoded(
      decoded, static_cast<std::uint8_t>(SourceArchitecture::kGfx1250), &sem);

  return Expect(sem.lowering_kind == LoweringKind::kNone,
                "non-BF16 VOP3P should have no lowering kind");
}

// ============================================================
// FP8/BF8 conversion lowering tests
// ============================================================

bool TestFp8ScalarConversionLowering() {
  SemanticLowering lowering;

  auto decoded = DecodedInstruction::Nullary("V_CVT_F32_FP8");
  SemanticInstruction sem;
  lowering.LiftFromDecoded(
      decoded, static_cast<std::uint8_t>(SourceArchitecture::kGfx1250), &sem);

  bool ok = Expect(sem.family == SemanticFamily::kFp8Bf8,
                   "V_CVT_F32_FP8 should be fp8_bf8 family") &&
            Expect(sem.lowering_kind == LoweringKind::kFp8ScalarConversion,
                   "V_CVT_F32_FP8 should have scalar conversion kind") &&
            Expect(sem.input_element_type == ElementType::kFP8_E4M3,
                   "input should be FP8") &&
            Expect(sem.output_element_type == ElementType::kF32,
                   "output should be F32");

  // Classification should be requires_semantic_lowering.
  TranslationStatus status = lowering.ClassifyForLowering(sem);
  ok = Expect(status == TranslationStatus::kRequiresSemanticLowering,
              "scalar FP8 conversion should require semantic lowering") && ok;

  // LowerToTarget should produce output.
  std::vector<DecodedInstruction> output;
  bool lower_ok = lowering.LowerToTarget(
      sem, static_cast<std::uint8_t>(TargetArchitecture::kGfx950),
      &output);
  ok = Expect(lower_ok, "LowerToTarget should succeed") &&
       Expect(output.size() == 1, "should produce 1 instruction") &&
       Expect(output[0].opcode == "V_CVT_F32_FP8",
              "should preserve opcode as stub") && ok;

  return ok;
}

bool TestFp8PackedConversionLowering() {
  SemanticLowering lowering;

  auto decoded = DecodedInstruction::Nullary("V_CVT_PK_F16_FP8");
  SemanticInstruction sem;
  lowering.LiftFromDecoded(
      decoded, static_cast<std::uint8_t>(SourceArchitecture::kGfx1250), &sem);

  bool ok = Expect(sem.lowering_kind == LoweringKind::kFp8PackedConversion,
                   "V_CVT_PK_F16_FP8 should have packed conversion kind");

  TranslationStatus status = lowering.ClassifyForLowering(sem);
  ok = Expect(status == TranslationStatus::kRequiresSemanticLowering,
              "packed FP8 conversion should require semantic lowering") && ok;

  std::vector<DecodedInstruction> output;
  bool lower_ok = lowering.LowerToTarget(
      sem, static_cast<std::uint8_t>(TargetArchitecture::kGfx950),
      &output);
  ok = Expect(lower_ok, "LowerToTarget should succeed") &&
       Expect(output.size() == 1, "should produce 1 instruction") && ok;

  return ok;
}

bool TestBf8ConversionLowering() {
  SemanticLowering lowering;

  auto decoded = DecodedInstruction::Nullary("V_CVT_F16_BF8");
  SemanticInstruction sem;
  lowering.LiftFromDecoded(
      decoded, static_cast<std::uint8_t>(SourceArchitecture::kGfx1250), &sem);

  return Expect(sem.lowering_kind == LoweringKind::kFp8ScalarConversion,
                "V_CVT_F16_BF8 should have scalar conversion kind") &&
         Expect(sem.input_element_type == ElementType::kBF8,
                "input should be BF8") &&
         Expect(sem.output_element_type == ElementType::kF16,
                "output should be F16");
}

bool TestFp8SrConversionLowering() {
  SemanticLowering lowering;

  auto decoded = DecodedInstruction::Nullary("V_CVT_SR_FP8_F32");
  SemanticInstruction sem;
  lowering.LiftFromDecoded(
      decoded, static_cast<std::uint8_t>(SourceArchitecture::kGfx1250), &sem);

  bool ok = Expect(sem.lowering_kind == LoweringKind::kFp8PackedConversion,
                   "V_CVT_SR_FP8_F32 should have packed conversion kind");

  TranslationStatus status = lowering.ClassifyForLowering(sem);
  ok = Expect(status == TranslationStatus::kRequiresSemanticLowering,
              "SR FP8 conversion should require semantic lowering") && ok;

  return ok;
}

bool TestFp8WithoutKnownLoweringStaysCoverageOnly() {
  SemanticLowering lowering;

  // A WMMA instruction with FP8 in its name is classified as FP8 by the
  // seed catalog but the WMMA-specific FP8 instructions don't have a
  // direct scalar/packed conversion lowering — they stay coverage-only
  // because their lowering_kind will remain kNone (they don't match the
  // scalar/packed conversion opcode patterns).
  SemanticInstruction sem;
  sem.family = SemanticFamily::kFp8Bf8;
  sem.lowering_kind = LoweringKind::kNone;

  TranslationStatus status = lowering.ClassifyForLowering(sem);
  return Expect(status == TranslationStatus::kCoverageOnly,
                "FP8 without known lowering kind should be coverage-only");
}

// ============================================================
// Scale and paired-scale lowering tests
// ============================================================

bool TestScaledConversionLowering() {
  SemanticLowering lowering;

  auto decoded = DecodedInstruction::Nullary("V_CVT_SCALEF32_PK8_FP8_F32");
  SemanticInstruction sem;
  lowering.LiftFromDecoded(
      decoded, static_cast<std::uint8_t>(SourceArchitecture::kGfx1250), &sem);

  bool ok = Expect(sem.family == SemanticFamily::kScalePaired,
                   "V_CVT_SCALEF32_PK8_FP8_F32 should be scale_paired") &&
            Expect(sem.lowering_kind == LoweringKind::kScaledConversion,
                   "should have scaled conversion kind");

  TranslationStatus status = lowering.ClassifyForLowering(sem);
  ok = Expect(status == TranslationStatus::kRequiresSemanticLowering,
              "scaled conversion should require semantic lowering") && ok;

  std::vector<DecodedInstruction> output;
  bool lower_ok = lowering.LowerToTarget(
      sem, static_cast<std::uint8_t>(TargetArchitecture::kGfx950),
      &output);
  ok = Expect(lower_ok, "LowerToTarget should succeed") &&
       Expect(output.size() == 1, "should produce 1 instruction") &&
       Expect(output[0].opcode == "V_CVT_SCALEF32_PK8_FP8_F32",
              "should preserve opcode as stub") && ok;

  return ok;
}

bool TestScaleConversionInverse() {
  SemanticLowering lowering;

  auto decoded = DecodedInstruction::Nullary("V_CVT_SCALE_PK8_F32_FP8");
  SemanticInstruction sem;
  lowering.LiftFromDecoded(
      decoded, static_cast<std::uint8_t>(SourceArchitecture::kGfx1250), &sem);

  return Expect(sem.lowering_kind == LoweringKind::kScaledConversion,
                "V_CVT_SCALE_PK8_F32_FP8 should have scaled conversion kind");
}

bool TestScalePairedWithoutConversionStaysCoverageOnly() {
  SemanticLowering lowering;

  // V_DIV_SCALE_F64 contains _SCALE but is not a scaled conversion
  // instruction (it's a legacy scalar division scale).
  // However, it does match IsScaledConversion because V_CVT_SCALE_ is
  // not present. Let's test an opcode that genuinely has no lowering.
  SemanticInstruction sem;
  sem.family = SemanticFamily::kScalePaired;
  sem.lowering_kind = LoweringKind::kNone;

  TranslationStatus status = lowering.ClassifyForLowering(sem);
  return Expect(status == TranslationStatus::kCoverageOnly,
                "scale-paired without conversion kind should be coverage-only");
}

// ============================================================
// Approximate transpose lowering tests
// ============================================================

bool TestTransposeLoweringKind() {
  SemanticLowering lowering;

  auto decoded = DecodedInstruction::Nullary("V_TRANSPOSE_B32");
  SemanticInstruction sem;
  lowering.LiftFromDecoded(
      decoded, static_cast<std::uint8_t>(SourceArchitecture::kGfx1250), &sem);

  return Expect(sem.family == SemanticFamily::kTranspose,
                "V_TRANSPOSE_B32 should be transpose family") &&
         Expect(sem.lowering_kind == LoweringKind::kApproximateTranspose,
                "should have approximate transpose kind") &&
         Expect(sem.is_approximate,
                "transpose should be marked approximate");
}

bool TestTransposeClassifiesAsSemanticLowering() {
  SemanticLowering lowering;

  auto decoded = DecodedInstruction::Nullary("V_TRANSPOSE_B32");
  SemanticInstruction sem;
  lowering.LiftFromDecoded(
      decoded, static_cast<std::uint8_t>(SourceArchitecture::kGfx1250), &sem);

  TranslationStatus status = lowering.ClassifyForLowering(sem);
  return Expect(status == TranslationStatus::kRequiresSemanticLowering,
                "transpose with lowering kind should require semantic "
                "lowering");
}

bool TestTransposeLowerToTargetProducesMovPlaceholder() {
  SemanticLowering lowering;

  auto decoded = DecodedInstruction::Nullary("V_TRANSPOSE_B32");
  SemanticInstruction sem;
  lowering.LiftFromDecoded(
      decoded, static_cast<std::uint8_t>(SourceArchitecture::kGfx1250), &sem);

  std::vector<DecodedInstruction> output;
  bool ok = lowering.LowerToTarget(
      sem, static_cast<std::uint8_t>(TargetArchitecture::kGfx950),
      &output);

  return Expect(ok, "LowerToTarget should succeed for transpose") &&
         Expect(output.size() == 1, "should produce 1 instruction") &&
         Expect(output[0].opcode == "V_MOV_B32",
                "approximate transpose should lower to V_MOV_B32");
}

bool TestTransposeWithoutLoweringKindIsCoverageOnly() {
  SemanticLowering lowering;

  SemanticInstruction sem;
  sem.family = SemanticFamily::kTranspose;
  sem.lowering_kind = LoweringKind::kNone;

  TranslationStatus status = lowering.ClassifyForLowering(sem);
  return Expect(status == TranslationStatus::kCoverageOnly,
                "transpose without lowering kind should be coverage-only");
}

// ============================================================
// LowerToTarget failure for unknown kind
// ============================================================

bool TestLowerToTargetFailsForNoLoweringKind() {
  SemanticLowering lowering;

  SemanticInstruction sem;
  sem.family = SemanticFamily::kOther;
  sem.lowering_kind = LoweringKind::kNone;

  std::vector<DecodedInstruction> output;
  std::string error;
  bool ok = lowering.LowerToTarget(
      sem, static_cast<std::uint8_t>(TargetArchitecture::kGfx950),
      &output, &error);

  return Expect(!ok, "LowerToTarget should fail for kNone lowering kind") &&
         Expect(!error.empty(), "should provide error message") &&
         Expect(output.empty(), "should produce no output");
}

// ============================================================
// Integration: gfx1250 seed catalog with Phase 5 lowerings
// ============================================================

bool TestGfx1250SeedCatalogLoweringKindsPopulated() {
  SemanticLowering lowering;

  struct TestCase {
    std::string_view opcode;
    LoweringKind expected_kind;
  };

  constexpr TestCase cases[] = {
      {"V_PK_ADD_BF16", LoweringKind::kPackedBf16ToF16},
      {"V_PK_FMA_BF16", LoweringKind::kPackedBf16ToF16},
      {"V_CVT_F32_FP8", LoweringKind::kFp8ScalarConversion},
      {"V_CVT_F16_BF8", LoweringKind::kFp8ScalarConversion},
      {"V_CVT_PK_F16_FP8", LoweringKind::kFp8PackedConversion},
      {"V_CVT_PK_BF8_F16", LoweringKind::kFp8PackedConversion},
      {"V_CVT_SR_FP8_F32", LoweringKind::kFp8PackedConversion},
      {"V_CVT_SCALEF32_PK8_FP8_F32", LoweringKind::kScaledConversion},
      {"V_CVT_SCALE_PK8_F32_FP8", LoweringKind::kScaledConversion},
      {"V_TRANSPOSE_B32", LoweringKind::kApproximateTranspose},
  };

  bool ok = true;
  for (const auto& tc : cases) {
    auto decoded = DecodedInstruction::Nullary(tc.opcode);
    SemanticInstruction sem;
    lowering.LiftFromDecoded(
        decoded, static_cast<std::uint8_t>(SourceArchitecture::kGfx1250),
        &sem);

    if (sem.lowering_kind != tc.expected_kind) {
      std::cerr << "FAIL: " << tc.opcode << " expected lowering_kind="
                << static_cast<int>(tc.expected_kind)
                << " got " << static_cast<int>(sem.lowering_kind) << "\n";
      ok = false;
    }
  }

  return ok;
}

bool TestAllVop3pBf16InstructionsLowerSuccessfully() {
  SemanticLowering lowering;

  constexpr std::string_view bf16_opcodes[] = {
      "V_PK_ADD_BF16",
      "V_PK_MUL_BF16",
      "V_PK_FMA_BF16",
      "V_PK_MAX_NUM_BF16",
      "V_PK_MIN_NUM_BF16",
  };

  bool ok = true;
  for (auto opcode : bf16_opcodes) {
    auto decoded = DecodedInstruction::Nullary(opcode);
    SemanticInstruction sem;
    lowering.LiftFromDecoded(
        decoded, static_cast<std::uint8_t>(SourceArchitecture::kGfx1250),
        &sem);

    std::vector<DecodedInstruction> output;
    bool lower_ok = lowering.LowerToTarget(
        sem, static_cast<std::uint8_t>(TargetArchitecture::kGfx950),
        &output);

    if (!lower_ok || output.size() != 1) {
      std::cerr << "FAIL: " << opcode << " lowering failed\n";
      ok = false;
    }
  }

  return ok;
}

bool TestAllFp8ScalarConversionsLowerSuccessfully() {
  SemanticLowering lowering;

  constexpr std::string_view opcodes[] = {
      "V_CVT_F32_FP8",
      "V_CVT_F32_BF8",
      "V_CVT_F16_FP8",
      "V_CVT_F16_BF8",
  };

  bool ok = true;
  for (auto opcode : opcodes) {
    auto decoded = DecodedInstruction::Nullary(opcode);
    SemanticInstruction sem;
    lowering.LiftFromDecoded(
        decoded, static_cast<std::uint8_t>(SourceArchitecture::kGfx1250),
        &sem);

    std::vector<DecodedInstruction> output;
    bool lower_ok = lowering.LowerToTarget(
        sem, static_cast<std::uint8_t>(TargetArchitecture::kGfx950),
        &output);

    if (!lower_ok || output.size() != 1) {
      std::cerr << "FAIL: " << opcode << " lowering failed\n";
      ok = false;
    } else if (output[0].opcode != opcode) {
      std::cerr << "FAIL: " << opcode << " lowered to " << output[0].opcode
                << " (expected same opcode)\n";
      ok = false;
    }
  }

  return ok;
}

bool TestCoverageStatusCountsWithPhase5() {
  // Verify that Phase 5 lowerings change the coverage distribution.
  // FP8 scalar conversions should now be requires_semantic_lowering
  // instead of coverage_only.
  SemanticLowering lowering;

  auto fp8 = DecodedInstruction::Nullary("V_CVT_F32_FP8");
  SemanticInstruction sem_fp8;
  lowering.LiftFromDecoded(
      fp8, static_cast<std::uint8_t>(SourceArchitecture::kGfx1250), &sem_fp8);

  auto transpose = DecodedInstruction::Nullary("V_TRANSPOSE_B32");
  SemanticInstruction sem_transpose;
  lowering.LiftFromDecoded(
      transpose, static_cast<std::uint8_t>(SourceArchitecture::kGfx1250),
      &sem_transpose);

  auto pk = DecodedInstruction::Nullary("V_PK_ADD_BF16");
  SemanticInstruction sem_pk;
  lowering.LiftFromDecoded(
      pk, static_cast<std::uint8_t>(SourceArchitecture::kGfx1250), &sem_pk);

  // These should all classify as requires_semantic_lowering now.
  bool ok =
      Expect(lowering.ClassifyForLowering(sem_fp8) ==
                 TranslationStatus::kRequiresSemanticLowering,
             "FP8 scalar should be requires_semantic_lowering") &&
      Expect(lowering.ClassifyForLowering(sem_transpose) ==
                 TranslationStatus::kRequiresSemanticLowering,
             "transpose should be requires_semantic_lowering") &&
      // V_PK_ADD_BF16 is vector family, goes through fast path not
      // semantic lowering classification, but has a lowering_kind set.
      Expect(sem_pk.lowering_kind == LoweringKind::kPackedBf16ToF16,
             "V_PK_ADD_BF16 should have BF16->F16 lowering kind");

  return ok;
}

}  // namespace

int main() {
  bool ok = true;

  std::cerr << "=== VOP3P packed BF16 lowering ===\n";
  ok = TestVop3pBf16LoweringKindClassification() && ok;
  ok = TestVop3pBf16AllMappings() && ok;
  ok = TestVop3pBf16LowerToTarget() && ok;
  ok = TestVop3pBf16UnmappedOpcodeHasNoLowering() && ok;

  std::cerr << "\n=== FP8/BF8 conversion lowering ===\n";
  ok = TestFp8ScalarConversionLowering() && ok;
  ok = TestFp8PackedConversionLowering() && ok;
  ok = TestBf8ConversionLowering() && ok;
  ok = TestFp8SrConversionLowering() && ok;
  ok = TestFp8WithoutKnownLoweringStaysCoverageOnly() && ok;

  std::cerr << "\n=== Scale and paired-scale lowering ===\n";
  ok = TestScaledConversionLowering() && ok;
  ok = TestScaleConversionInverse() && ok;
  ok = TestScalePairedWithoutConversionStaysCoverageOnly() && ok;

  std::cerr << "\n=== Approximate transpose lowering ===\n";
  ok = TestTransposeLoweringKind() && ok;
  ok = TestTransposeClassifiesAsSemanticLowering() && ok;
  ok = TestTransposeLowerToTargetProducesMovPlaceholder() && ok;
  ok = TestTransposeWithoutLoweringKindIsCoverageOnly() && ok;

  std::cerr << "\n=== LowerToTarget edge cases ===\n";
  ok = TestLowerToTargetFailsForNoLoweringKind() && ok;

  std::cerr << "\n=== Integration: seed catalog lowering kinds ===\n";
  ok = TestGfx1250SeedCatalogLoweringKindsPopulated() && ok;
  ok = TestAllVop3pBf16InstructionsLowerSuccessfully() && ok;
  ok = TestAllFp8ScalarConversionsLowerSuccessfully() && ok;
  ok = TestCoverageStatusCountsWithPhase5() && ok;

  if (ok) {
    std::cerr << "\nAll semantic_lowering_phase5 tests passed.\n";
  }
  return ok ? 0 : 1;
}
