cmake_minimum_required(VERSION 3.11)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(beton LANGUAGES CXX)

find_package(OpenCL REQUIRED)

add_subdirectory(tests)

include_directories(
            ${CMAKE_SOURCE_DIR}/
            ${CMAKE_SOURCE_DIR}/src
)

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated")

add_compile_options(
    $<$<BOOL:${SANITIZE}>:-fsanitize=address,undefined>
)
add_link_options(
    $<$<BOOL:${SANITIZE}>:-fsanitize=address,undefined>
)

set(BITONIC_KERNEL_IN  "${CMAKE_SOURCE_DIR}/kernels/bitonic.cl")
set(BITONIC_KERNEL_OUT "${CMAKE_CURRENT_BINARY_DIR}/generated/bitonic_cl.hpp")

add_custom_command(
  OUTPUT  "${BITONIC_KERNEL_OUT}"
  COMMAND "${CMAKE_COMMAND}"
          -DINPUT=${BITONIC_KERNEL_IN}
          -DOUTPUT=${BITONIC_KERNEL_OUT}
          -DVAR_NAME=kBitonicClSrc
          -P "${CMAKE_SOURCE_DIR}/cmake/EmbedAsRawString.cmake"
  DEPENDS "${BITONIC_KERNEL_IN}" "${CMAKE_SOURCE_DIR}/cmake/EmbedAsRawString.cmake"
  VERBATIM
)

add_custom_target(bitonic_kernel DEPENDS "${BITONIC_KERNEL_OUT}")
