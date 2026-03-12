#include "lib/sim/isa/jit/cross_arch_translator.h"

#include <algorithm>
#include <string>

namespace mirage::sim::isa::jit {

namespace {

std::uint16_t VccSgprForArch(std::uint8_t arch) {
  switch (arch) {
    case static_cast<std::uint8_t>(SourceArchitecture::kGfx950):
      return WaveAdapter::kGfx950VccSgpr;
    case static_cast<std::uint8_t>(SourceArchitecture::kGfx1201):
      return WaveAdapter::kGfx1201VccSgpr;
    case static_cast<std::uint8_t>(SourceArchitecture::kGfx1250):
      return WaveAdapter::kGfx1250VccSgpr;
    default:
      return WaveAdapter::kGfx950VccSgpr;
  }
}

}  // namespace

bool TranslationCacheKey::operator<(const TranslationCacheKey& other) const {
  if (source_code_va != other.source_code_va) {
    return source_code_va < other.source_code_va;
  }
  if (code_word_count != other.code_word_count) {
    return code_word_count < other.code_word_count;
  }
  if (source_arch != other.source_arch) {
    return source_arch < other.source_arch;
  }
  if (target_arch != other.target_arch) {
    return target_arch < other.target_arch;
  }
  if (translation_mode != other.translation_mode) {
    return translation_mode < other.translation_mode;
  }
  if (allow_approximate_lowering != other.allow_approximate_lowering) {
    return allow_approximate_lowering < other.allow_approximate_lowering;
  }
  if (wave_policy != other.wave_policy) {
    return wave_policy < other.wave_policy;
  }
  if (hazard_policy != other.hazard_policy) {
    return hazard_policy < other.hazard_policy;
  }
  if (rule_table_version != other.rule_table_version) {
    return rule_table_version < other.rule_table_version;
  }
  return source_allocation_write_version <
         other.source_allocation_write_version;
}

CrossArchTranslator::CrossArchTranslator(TranslationConfig config)
    : config_(config),
      wave_adapter_(
          VccSgprForArch(static_cast<std::uint8_t>(config.source_arch)),
          VccSgprForArch(static_cast<std::uint8_t>(config.target_arch)),
          config.wave_policy) {
  rule_table_.BuildForDirection(
      static_cast<std::uint8_t>(config_.source_arch),
      static_cast<std::uint8_t>(config_.target_arch));
}

TranslationResult CrossArchTranslator::Translate(
    std::span<const DecodedInstruction> source_program) const {
  TranslationResult result;
  result.diagnostics.reserve(source_program.size());
  result.translated_program.reserve(source_program.size());

  bool all_executable = true;
  bool has_exec_manipulating = false;
  bool has_lds = false;

  for (std::size_t i = 0; i < source_program.size(); ++i) {
    InstructionDiagnostic diagnostic;
    diagnostic.source_index = i;
    diagnostic.source_opcode = source_program[i].opcode;
    diagnostic.output_begin_index = result.translated_program.size();

    bool ok = TranslateInstruction(
        source_program[i], &result.translated_program, &diagnostic);

    diagnostic.output_instruction_count = static_cast<std::uint32_t>(
        result.translated_program.size() - diagnostic.output_begin_index);

    if (!ok || !IsExecutable(diagnostic.status)) {
      all_executable = false;
    }

    // Track whether any instruction manipulates the EXEC mask.
    if (ok) {
      const TranslationRule* rule =
          rule_table_.FindRule(source_program[i].opcode);
      if (rule != nullptr && rule->is_exec_manipulating) {
        has_exec_manipulating = true;
      }
    }

    // Track whether any instruction accesses LDS memory.
    if (IsLdsTouchingOpcode(source_program[i].opcode)) {
      has_lds = true;
    }

    switch (diagnostic.status) {
      case TranslationStatus::kIdentity:
      case TranslationStatus::kRenamed:
      case TranslationStatus::kRewrittenWithFixup:
        result.capability_summary.executable_count++;
        break;
      case TranslationStatus::kCoverageOnly:
        result.capability_summary.coverage_only_count++;
        break;
      case TranslationStatus::kBlockedOnRuntime:
        result.capability_summary.blocked_on_runtime_count++;
        break;
      case TranslationStatus::kRequiresSemanticLowering:
        result.capability_summary.coverage_only_count++;
        break;
      case TranslationStatus::kUnsupported:
        result.capability_summary.unsupported_count++;
        break;
    }

    result.diagnostics.push_back(std::move(diagnostic));
  }

  result.contains_lds_instructions = has_lds;

  if (all_executable && !source_program.empty()) {
    ApplyBranchFixup(&result.translated_program, result.diagnostics);
    result.top_level_status = TranslationStatus::kRewrittenWithFixup;
    result.is_executable = true;
    result.requires_exec_narrowing = wave_adapter_.RequiresExecNarrowing();
    result.contains_exec_manipulating_instructions = has_exec_manipulating;
  } else if (config_.translation_mode == TranslationMode::kCoverageOnly) {
    result.top_level_status = TranslationStatus::kCoverageOnly;
    result.is_executable = false;
  } else {
    result.top_level_status = TranslationStatus::kUnsupported;
    result.is_executable = false;
  }

  return result;
}

TranslationStatus CrossArchTranslator::ClassifyInstruction(
    std::string_view opcode) const {
  if (wave_adapter_.ShouldReject(opcode)) {
    return TranslationStatus::kUnsupported;
  }
  if (wave_adapter_.IsWaveSensitive(opcode) &&
      config_.translation_mode == TranslationMode::kExecutableStrict) {
    return TranslationStatus::kUnsupported;
  }

  TranslationTier tier = rule_table_.Classify(opcode);
  switch (tier) {
    case TranslationTier::kIdentity:
      return TranslationStatus::kIdentity;
    case TranslationTier::kRename:
      return TranslationStatus::kRenamed;
    case TranslationTier::kOperandFixup:
      return TranslationStatus::kRewrittenWithFixup;
    case TranslationTier::kExpansion:
      return TranslationStatus::kRequiresSemanticLowering;
    case TranslationTier::kSemanticLowering:
      return TranslationStatus::kRequiresSemanticLowering;
    case TranslationTier::kUnsupported:
      return TranslationStatus::kUnsupported;
  }
  return TranslationStatus::kUnsupported;
}

CapabilitySummary CrossArchTranslator::ComputeCoverage(
    std::span<const DecodedInstruction> source_program) const {
  CapabilitySummary summary;
  for (const auto& instr : source_program) {
    TranslationStatus status = ClassifyInstruction(instr.opcode);
    switch (status) {
      case TranslationStatus::kIdentity:
      case TranslationStatus::kRenamed:
      case TranslationStatus::kRewrittenWithFixup:
        summary.executable_count++;
        break;
      case TranslationStatus::kCoverageOnly:
        summary.coverage_only_count++;
        break;
      case TranslationStatus::kBlockedOnRuntime:
        summary.blocked_on_runtime_count++;
        break;
      case TranslationStatus::kRequiresSemanticLowering:
        summary.coverage_only_count++;
        break;
      case TranslationStatus::kUnsupported:
        summary.unsupported_count++;
        break;
    }
  }
  return summary;
}

TranslationStatus CrossArchTranslator::ClassifyInstructionWithSemantic(
    std::string_view opcode) const {
  // First try the fast path via the rule table.
  TranslationStatus rule_status = ClassifyInstruction(opcode);
  if (rule_status != TranslationStatus::kUnsupported) {
    return rule_status;
  }

  // Fall through to semantic classification for opcodes without a
  // direct rule-table entry.
  SemanticInstruction semantic;
  semantic.opcode = opcode;
  semantic.source_arch = static_cast<std::uint8_t>(config_.source_arch);
  semantic.family = ClassifyOpcodeFamily(opcode);

  // Populate prerequisites based on family.
  DecodedInstruction decoded = DecodedInstruction::Nullary(opcode);
  semantic_lowering_.LiftFromDecoded(
      decoded, static_cast<std::uint8_t>(config_.source_arch), &semantic);

  return semantic_lowering_.ClassifyForLowering(semantic);
}

CapabilitySummary CrossArchTranslator::ComputeCoverageWithSemantic(
    std::span<const DecodedInstruction> source_program) const {
  CapabilitySummary summary;
  for (const auto& instr : source_program) {
    TranslationStatus status =
        ClassifyInstructionWithSemantic(instr.opcode);
    switch (status) {
      case TranslationStatus::kIdentity:
      case TranslationStatus::kRenamed:
      case TranslationStatus::kRewrittenWithFixup:
        summary.executable_count++;
        break;
      case TranslationStatus::kCoverageOnly:
        summary.coverage_only_count++;
        break;
      case TranslationStatus::kBlockedOnRuntime:
        summary.blocked_on_runtime_count++;
        break;
      case TranslationStatus::kRequiresSemanticLowering:
        summary.coverage_only_count++;
        break;
      case TranslationStatus::kUnsupported:
        summary.unsupported_count++;
        break;
    }
  }
  return summary;
}

std::string_view CrossArchTranslator::ArchitectureName(
    SourceArchitecture arch) {
  switch (arch) {
    case SourceArchitecture::kGfx950:
      return "gfx950";
    case SourceArchitecture::kGfx1201:
      return "gfx1201";
    case SourceArchitecture::kGfx1250:
      return "gfx1250";
  }
  return "unknown";
}

std::string_view CrossArchTranslator::ArchitectureName(
    TargetArchitecture arch) {
  switch (arch) {
    case TargetArchitecture::kGfx950:
      return "gfx950";
    case TargetArchitecture::kGfx1201:
      return "gfx1201";
    case TargetArchitecture::kGfx1250:
      return "gfx1250";
  }
  return "unknown";
}

std::string_view CrossArchTranslator::TranslationModeName(
    TranslationMode mode) {
  switch (mode) {
    case TranslationMode::kExecutableStrict:
      return "executable_strict";
    case TranslationMode::kCoverageOnly:
      return "coverage_only";
    case TranslationMode::kApproximateExperimental:
      return "approximate_experimental";
  }
  return "unknown";
}

std::string_view CrossArchTranslator::TranslationStatusName(
    TranslationStatus status) {
  switch (status) {
    case TranslationStatus::kIdentity:
      return "identity";
    case TranslationStatus::kRenamed:
      return "renamed";
    case TranslationStatus::kRewrittenWithFixup:
      return "rewritten_with_fixup";
    case TranslationStatus::kRequiresSemanticLowering:
      return "requires_semantic_lowering";
    case TranslationStatus::kCoverageOnly:
      return "coverage_only";
    case TranslationStatus::kBlockedOnRuntime:
      return "blocked_on_runtime";
    case TranslationStatus::kUnsupported:
      return "unsupported";
  }
  return "unknown";
}

bool CrossArchTranslator::TranslateInstruction(
    const DecodedInstruction& source,
    std::vector<DecodedInstruction>* output,
    InstructionDiagnostic* diagnostic) const {
  if (wave_adapter_.IsWaveSensitive(source.opcode)) {
    diagnostic->status = TranslationStatus::kUnsupported;
    diagnostic->rejection_reason = RejectionReason::kWaveSensitive;
    diagnostic->message = "Wave-topology-sensitive instruction excluded from "
                          "executable translation.";
    return false;
  }

  if (IsLdsTouchingOpcode(source.opcode)) {
    diagnostic->status = TranslationStatus::kUnsupported;
    diagnostic->rejection_reason = RejectionReason::kLdsTouching;
    diagnostic->message = "LDS-touching DS opcode has no executable "
                          "translation rule.";
    return false;
  }

  const TranslationRule* rule = rule_table_.FindRule(source.opcode);
  if (rule == nullptr) {
    diagnostic->status = TranslationStatus::kUnsupported;
    diagnostic->rejection_reason = RejectionReason::kNoTranslationRule;
    diagnostic->message = "No translation rule found for opcode.";
    return false;
  }

  DecodedInstruction translated = source;
  translated.opcode = rule->target_opcode;

  if (rule->requires_vcc_remap) {
    wave_adapter_.RemapVccOperands(&translated);
  }

  switch (rule->tier) {
    case TranslationTier::kIdentity:
      diagnostic->status = TranslationStatus::kIdentity;
      break;
    case TranslationTier::kRename:
      diagnostic->status = TranslationStatus::kRenamed;
      break;
    case TranslationTier::kOperandFixup:
      diagnostic->status = TranslationStatus::kRewrittenWithFixup;
      break;
    default:
      diagnostic->status = TranslationStatus::kUnsupported;
      diagnostic->rejection_reason = RejectionReason::kRuleTierNotExecutable;
      diagnostic->message = "Rule tier not executable.";
      return false;
  }

  output->push_back(translated);
  return true;
}

namespace {

bool IsBranchOpcode(std::string_view opcode) {
  return opcode == "S_BRANCH" || opcode == "S_CBRANCH_SCC0" ||
         opcode == "S_CBRANCH_SCC1" || opcode == "S_CBRANCH_VCCZ" ||
         opcode == "S_CBRANCH_VCCNZ" || opcode == "S_CBRANCH_EXECZ" ||
         opcode == "S_CBRANCH_EXECNZ";
}

}  // namespace

bool CrossArchTranslator::FixupBranchTargets(
    std::vector<DecodedInstruction>* program,
    const std::vector<InstructionDiagnostic>& diagnostics) {
  if (program == nullptr || program->empty()) {
    return true;
  }

  // Build a mapping from source instruction index → output instruction index.
  // source_to_output[i] = output index where source instruction i begins.
  // source_to_output[source_count] = program->size() (one past last).
  const std::size_t source_count = diagnostics.size();
  std::vector<std::size_t> source_to_output(source_count + 1);
  for (std::size_t i = 0; i < source_count; ++i) {
    source_to_output[i] = diagnostics[i].output_begin_index;
  }
  source_to_output[source_count] = program->size();

  // If every source instruction produced exactly one output instruction,
  // all offsets are already correct and we can skip adjustment.
  if (program->size() == source_count) {
    return true;
  }

  // For each branch instruction, adjust its relative offset to account
  // for index shifts caused by 1:N expansions.
  for (std::size_t diag_index = 0; diag_index < source_count; ++diag_index) {
    const auto& diag = diagnostics[diag_index];

    if (!IsBranchOpcode(diag.source_opcode)) {
      continue;
    }

    const std::size_t output_branch_idx = diag.output_begin_index;
    if (output_branch_idx >= program->size()) {
      return false;
    }

    DecodedInstruction& branch_instr = (*program)[output_branch_idx];
    if (branch_instr.operand_count == 0 ||
        branch_instr.operands[0].kind != OperandKind::kImm32) {
      continue;
    }

    const auto source_delta =
        static_cast<std::int32_t>(branch_instr.operands[0].imm32);

    // Branch semantics: target_pc = current_pc + 1 + delta
    const auto source_target =
        static_cast<std::int64_t>(diag_index) + 1 +
        static_cast<std::int64_t>(source_delta);

    if (source_target < 0 ||
        static_cast<std::size_t>(source_target) > source_count) {
      return false;
    }

    const std::size_t output_target_idx =
        source_to_output[static_cast<std::size_t>(source_target)];

    // new_delta = output_target_idx - (output_branch_idx + 1)
    const auto new_delta = static_cast<std::int32_t>(
        static_cast<std::int64_t>(output_target_idx) -
        static_cast<std::int64_t>(output_branch_idx) - 1);

    branch_instr.operands[0].imm32 = static_cast<std::uint32_t>(new_delta);
  }

  return true;
}

bool CrossArchTranslator::ApplyBranchFixup(
    std::vector<DecodedInstruction>* program,
    const std::vector<InstructionDiagnostic>& diagnostics) const {
  return FixupBranchTargets(program, diagnostics);
}

}  // namespace mirage::sim::isa::jit
