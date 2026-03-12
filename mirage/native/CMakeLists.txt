cmake_minimum_required(VERSION 3.25)

project(MirageSimCore LANGUAGES CXX)

find_package(Threads REQUIRED)

enable_testing()

add_subdirectory(
  "${CMAKE_CURRENT_LIST_DIR}/../third_party/simdojo"
  "${CMAKE_CURRENT_BINARY_DIR}/third_party/simdojo"
)
target_compile_features(simdojo PUBLIC cxx_std_20)

add_library(mirage_sim_core STATIC
  generated/gfx950_instruction_catalog.cc
  src/isa/gfx950/binary_decoder.cc
  src/mirage_sim_core.cc
  src/isa/gfx950/interpreter.cc
  src/isa/jit/cross_arch_translator.cc
  src/isa/jit/translation_rules.cc
  src/isa/jit/wave_adapter.cc
  src/isa/jit/semantic_lowering.cc
  src/simdojo_dispatch_harness.cc
  src/single_gpu_simulator.cc
  src/virtual_gpu_device.cc
)

target_compile_features(mirage_sim_core PUBLIC cxx_std_20)
target_include_directories(mirage_sim_core PUBLIC
  "${CMAKE_CURRENT_LIST_DIR}/.."
)
target_link_libraries(mirage_sim_core PUBLIC simdojo Threads::Threads)

add_executable(mirage_instruction_catalog_test
  tests/instruction_catalog_test.cc
)
target_link_libraries(mirage_instruction_catalog_test PRIVATE mirage_sim_core)

add_executable(mirage_isa_common_test
  tests/isa/common/common_isa_test.cc
)
target_link_libraries(mirage_isa_common_test PRIVATE mirage_sim_core)

add_executable(mirage_gfx950_interpreter_test
  tests/isa/gfx950/interpreter_test.cc
)
target_link_libraries(mirage_gfx950_interpreter_test PRIVATE mirage_sim_core)

add_executable(mirage_gfx950_binary_decoder_test
  tests/isa/gfx950/binary_decoder_test.cc
)
target_link_libraries(mirage_gfx950_binary_decoder_test PRIVATE mirage_sim_core)

add_executable(mirage_sim_core_test
  tests/single_gpu_simulator_test.cc
)
target_link_libraries(mirage_sim_core_test PRIVATE mirage_sim_core)

add_executable(mirage_gfx950_runtime_test
  tests/isa/gfx950/runtime_test.cc
)
target_link_libraries(mirage_gfx950_runtime_test PRIVATE mirage_sim_core)

add_executable(mirage_simdojo_dispatch_harness_test
  tests/simdojo_dispatch_harness_test.cc
)
target_link_libraries(mirage_simdojo_dispatch_harness_test PRIVATE mirage_sim_core)

add_executable(mirage_simdojo_smoke_test
  tests/simdojo_smoke_test.cc
)
target_compile_features(mirage_simdojo_smoke_test PRIVATE cxx_std_20)
target_link_libraries(mirage_simdojo_smoke_test PRIVATE simdojo Threads::Threads)

add_test(NAME mirage_instruction_catalog_test COMMAND mirage_instruction_catalog_test)
add_test(NAME mirage_isa_common_test COMMAND mirage_isa_common_test)
add_test(NAME mirage_gfx950_binary_decoder_test COMMAND mirage_gfx950_binary_decoder_test)
add_test(
  NAME mirage_gfx950_interpreter_test
  COMMAND
    "${CMAKE_CURRENT_LIST_DIR}/tests/run_with_unlimited_stack.sh"
    "$<TARGET_FILE:mirage_gfx950_interpreter_test>"
)
add_test(NAME mirage_sim_core_test COMMAND mirage_sim_core_test)
add_test(NAME mirage_gfx950_runtime_test COMMAND mirage_gfx950_runtime_test)
add_test(NAME mirage_simdojo_dispatch_harness_test COMMAND mirage_simdojo_dispatch_harness_test)
add_test(NAME mirage_simdojo_smoke_test COMMAND mirage_simdojo_smoke_test)

add_executable(mirage_jit_translation_rules_test
  tests/isa/jit/translation_rules_test.cc
)
target_link_libraries(mirage_jit_translation_rules_test PRIVATE mirage_sim_core)

add_executable(mirage_jit_cross_arch_translator_test
  tests/isa/jit/cross_arch_translator_test.cc
)
target_link_libraries(mirage_jit_cross_arch_translator_test PRIVATE mirage_sim_core)

add_executable(mirage_jit_branch_fixup_test
  tests/isa/jit/branch_fixup_test.cc
)
target_link_libraries(mirage_jit_branch_fixup_test PRIVATE mirage_sim_core)

add_executable(mirage_jit_wave_adapter_test
  tests/isa/jit/wave_adapter_test.cc
)
target_link_libraries(mirage_jit_wave_adapter_test PRIVATE mirage_sim_core)

add_executable(mirage_jit_translation_coverage_test
  tests/isa/jit/translation_coverage_test.cc
)
target_link_libraries(mirage_jit_translation_coverage_test PRIVATE mirage_sim_core)

add_executable(mirage_jit_translated_execution_test
  tests/isa/jit/translated_execution_test.cc
)
target_link_libraries(mirage_jit_translated_execution_test PRIVATE mirage_sim_core)

add_test(NAME mirage_jit_translation_rules_test COMMAND mirage_jit_translation_rules_test)
add_test(NAME mirage_jit_cross_arch_translator_test COMMAND mirage_jit_cross_arch_translator_test)
add_test(NAME mirage_jit_branch_fixup_test COMMAND mirage_jit_branch_fixup_test)
add_test(NAME mirage_jit_wave_adapter_test COMMAND mirage_jit_wave_adapter_test)
add_test(NAME mirage_jit_translation_coverage_test COMMAND mirage_jit_translation_coverage_test)
add_test(NAME mirage_jit_translated_execution_test COMMAND mirage_jit_translated_execution_test)
