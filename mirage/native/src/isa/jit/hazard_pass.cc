#include "lib/sim/isa/jit/hazard_pass.h"

#include <cstddef>

namespace mirage::sim::isa::jit {

bool HazardPass::Apply(
    HazardPolicy policy,
    const HazardModel* source_model,
    const HazardModel* target_model,
    std::vector<DecodedInstruction>* program,
    std::vector<InstructionDiagnostic>* diagnostics,
    HazardPassStats* stats) {
  if (program == nullptr || diagnostics == nullptr) {
    return false;
  }

  HazardPassStats local_stats;
  local_stats.instructions_before =
      static_cast<std::uint32_t>(program->size());

  switch (policy) {
    case HazardPolicy::kPassthrough:
      local_stats.instructions_after = local_stats.instructions_before;
      if (stats != nullptr) {
        *stats = local_stats;
      }
      return true;

    case HazardPolicy::kStripSource:
      if (source_model == nullptr) {
        return false;
      }
      if (!StripSourceHazards(source_model, program, diagnostics,
                              &local_stats)) {
        return false;
      }
      local_stats.instructions_after =
          static_cast<std::uint32_t>(program->size());
      if (stats != nullptr) {
        *stats = local_stats;
      }
      return true;

    case HazardPolicy::kRetarget:
      if (source_model == nullptr || target_model == nullptr) {
        return false;
      }
      if (!StripSourceHazards(source_model, program, diagnostics,
                              &local_stats)) {
        return false;
      }
      if (!InsertTargetHazards(target_model, program, diagnostics,
                               &local_stats)) {
        return false;
      }
      local_stats.instructions_after =
          static_cast<std::uint32_t>(program->size());
      if (stats != nullptr) {
        *stats = local_stats;
      }
      return true;
  }
  return false;
}

bool HazardPass::StripSourceHazards(
    const HazardModel* source_model,
    std::vector<DecodedInstruction>* program,
    std::vector<InstructionDiagnostic>* diagnostics,
    HazardPassStats* stats) {
  if (source_model == nullptr || program == nullptr ||
      diagnostics == nullptr) {
    return false;
  }

  // Build a removal mask.  Only strip S_NOP with nop_count > 0 (i.e.
  // operand_count > 0 with an imm32 operand), and S_WAITCNT.
  // Bare S_NOP (operand_count == 0) comes from barrier signal lowering
  // and must be preserved.
  std::vector<bool> remove_mask(program->size(), false);
  std::uint32_t nops_stripped = 0;
  std::uint32_t waitcnts_stripped = 0;

  for (std::size_t i = 0; i < program->size(); ++i) {
    const auto& instr = (*program)[i];
    InstructionHazardInfo info = source_model->Classify(instr);

    if (info.is_nop && instr.operand_count > 0) {
      remove_mask[i] = true;
      ++nops_stripped;
    } else if (info.is_waitcnt) {
      remove_mask[i] = true;
      ++waitcnts_stripped;
    }
  }

  // Compact the program by removing marked instructions.
  std::size_t write = 0;
  for (std::size_t read = 0; read < program->size(); ++read) {
    if (!remove_mask[read]) {
      if (write != read) {
        (*program)[write] = (*program)[read];
      }
      ++write;
    }
  }
  program->resize(write);

  // Update diagnostic output_instruction_count values.
  // Walk the remove_mask and decrement counts for removed instructions.
  std::size_t mask_idx = 0;
  for (auto& diag : *diagnostics) {
    std::uint32_t removed_in_diag = 0;
    for (std::uint32_t j = 0; j < diag.output_instruction_count; ++j) {
      if (mask_idx < remove_mask.size() && remove_mask[mask_idx]) {
        ++removed_in_diag;
      }
      ++mask_idx;
    }
    diag.output_instruction_count -= removed_in_diag;
  }

  RebuildDiagnosticIndices(diagnostics);

  if (stats != nullptr) {
    stats->nops_stripped += nops_stripped;
    stats->waitcnts_stripped += waitcnts_stripped;
  }

  return true;
}

bool HazardPass::InsertTargetHazards(
    const HazardModel* target_model,
    std::vector<DecodedInstruction>* program,
    std::vector<InstructionDiagnostic>* diagnostics,
    HazardPassStats* stats) {
  if (target_model == nullptr || program == nullptr ||
      diagnostics == nullptr) {
    return false;
  }

  // Conservative stub: scan for producer-consumer pairs that need NOPs
  // and insert S_NOP instructions.
  std::vector<DecodedInstruction> new_program;
  new_program.reserve(program->size());

  // Track which diagnostic each output instruction belongs to, so we
  // can update output_instruction_count after insertion.
  std::vector<std::size_t> instr_to_diag(program->size());
  {
    std::size_t idx = 0;
    for (std::size_t d = 0; d < diagnostics->size(); ++d) {
      for (std::uint32_t j = 0; j < (*diagnostics)[d].output_instruction_count;
           ++j) {
        if (idx < instr_to_diag.size()) {
          instr_to_diag[idx] = d;
        }
        ++idx;
      }
    }
  }

  std::uint32_t nops_inserted = 0;
  PipelineStage last_stage = PipelineStage::kNone;

  for (std::size_t i = 0; i < program->size(); ++i) {
    const auto& instr = (*program)[i];
    InstructionHazardInfo info = target_model->Classify(instr);

    if (last_stage != PipelineStage::kNone &&
        info.pipeline_stage != PipelineStage::kNone) {
      std::uint8_t required_nops =
          target_model->RequiredNopsBetween(last_stage, info.pipeline_stage);
      for (std::uint8_t n = 0; n < required_nops; ++n) {
        // Insert S_NOP with count.
        new_program.push_back(
            DecodedInstruction::OneOperand(
                "S_NOP", InstructionOperand::Imm32(required_nops - 1)));
        ++nops_inserted;

        // Credit this NOP to the same diagnostic as the consumer instruction.
        if (i < instr_to_diag.size()) {
          (*diagnostics)[instr_to_diag[i]].output_instruction_count++;
        }
        // Only insert one S_NOP instruction (with count encoding the wait).
        break;
      }
    }

    new_program.push_back(instr);

    if (info.pipeline_stage != PipelineStage::kNone) {
      last_stage = info.pipeline_stage;
    }
  }

  *program = std::move(new_program);
  RebuildDiagnosticIndices(diagnostics);

  if (stats != nullptr) {
    stats->nops_inserted += nops_inserted;
  }

  return true;
}

void HazardPass::RebuildDiagnosticIndices(
    std::vector<InstructionDiagnostic>* diagnostics) {
  if (diagnostics == nullptr) {
    return;
  }
  std::size_t running = 0;
  for (auto& d : *diagnostics) {
    d.output_begin_index = running;
    running += d.output_instruction_count;
  }
}

}  // namespace mirage::sim::isa::jit
