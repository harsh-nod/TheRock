#include "lib/sim/isa/jit/peephole_pass.h"

#include <cstddef>
#include <string_view>

namespace mirage::sim::isa::jit {

namespace {

bool StartsWith(std::string_view str, std::string_view prefix) {
  return str.size() >= prefix.size() && str.substr(0, prefix.size()) == prefix;
}

bool IsBareNop(const DecodedInstruction& instr) {
  return (instr.opcode == "S_NOP" || instr.opcode == "V_NOP") &&
         instr.operand_count == 0;
}

bool IsAccvgprWrite(const DecodedInstruction& instr) {
  return instr.opcode == "V_ACCVGPR_WRITE_B32";
}

bool IsAccvgprRead(const DecodedInstruction& instr) {
  return instr.opcode == "V_ACCVGPR_READ_B32";
}

bool IsMfma(const DecodedInstruction& instr) {
  return StartsWith(instr.opcode, "V_MFMA_");
}

}  // namespace

bool PeepholePass::Apply(
    std::vector<DecodedInstruction>* program,
    std::vector<InstructionDiagnostic>* diagnostics,
    PeepholeStats* stats) {
  if (program == nullptr || diagnostics == nullptr) {
    return false;
  }
  if (program->empty()) {
    return true;
  }

  std::vector<bool> remove_mask(program->size(), false);
  PeepholeStats local_stats;

  // Pattern 1: Consecutive bare S_NOP — remove second.
  for (std::size_t i = 1; i < program->size(); ++i) {
    if (IsBareNop((*program)[i]) && IsBareNop((*program)[i - 1]) &&
        !remove_mask[i - 1]) {
      remove_mask[i] = true;
      ++local_stats.redundant_nops_removed;
    }
  }

  // Pattern 2: Dead ACCVGPR transfer pair.
  // V_ACCVGPR_WRITE_B32 followed by V_ACCVGPR_READ_B32 with no MFMA
  // between them — both are dead.
  for (std::size_t i = 0; i + 1 < program->size(); ++i) {
    if (remove_mask[i]) continue;
    if (!IsAccvgprWrite((*program)[i])) continue;

    // Scan forward for matching READ, ensuring no MFMA intervenes.
    for (std::size_t j = i + 1; j < program->size(); ++j) {
      if (remove_mask[j]) continue;
      if (IsMfma((*program)[j])) break;
      if (IsAccvgprRead((*program)[j])) {
        remove_mask[i] = true;
        remove_mask[j] = true;
        local_stats.redundant_accvgpr_transfers_removed += 2;
        break;
      }
      // Other instructions are fine — keep scanning.
    }
  }

  // Pattern 3: NOP before ENDPGM — remove the NOP.
  for (std::size_t i = 0; i + 1 < program->size(); ++i) {
    if (!remove_mask[i] && IsBareNop((*program)[i]) &&
        (*program)[i + 1].opcode == "S_ENDPGM") {
      remove_mask[i] = true;
      ++local_stats.redundant_nops_removed;
    }
  }

  local_stats.total_instructions_removed =
      CompactProgram(program, diagnostics, remove_mask);

  if (stats != nullptr) {
    *stats = local_stats;
  }

  return true;
}

std::uint32_t PeepholePass::CompactProgram(
    std::vector<DecodedInstruction>* program,
    std::vector<InstructionDiagnostic>* diagnostics,
    const std::vector<bool>& remove_mask) {
  // Update diagnostic output_instruction_count values.
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

  // Compact the program.
  std::uint32_t removed = 0;
  std::size_t write = 0;
  for (std::size_t read = 0; read < program->size(); ++read) {
    if (!remove_mask[read]) {
      if (write != read) {
        (*program)[write] = (*program)[read];
      }
      ++write;
    } else {
      ++removed;
    }
  }
  program->resize(write);

  RebuildDiagnosticIndices(diagnostics);

  return removed;
}

void PeepholePass::RebuildDiagnosticIndices(
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
