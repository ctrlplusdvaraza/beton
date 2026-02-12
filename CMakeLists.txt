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

function(add_cl_kernel_compilation FILENAME VAR_NAME TARGET_NAME)
    set(KERNEL_IN "${CMAKE_SOURCE_DIR}/kernels/${FILENAME}")
    
    get_filename_component(NAME_WE ${FILENAME} NAME_WE)
    set(KERNEL_OUT "${CMAKE_CURRENT_BINARY_DIR}/generated/${NAME_WE}_cl.hpp")

    add_custom_command(
        OUTPUT  "${KERNEL_OUT}"
        COMMAND "${CMAKE_COMMAND}"
                -DINPUT=${KERNEL_IN}
                -DOUTPUT=${KERNEL_OUT}
                -DVAR_NAME=${VAR_NAME}
                -P "${CMAKE_SOURCE_DIR}/cmake/EmbedAsRawString.cmake"
        DEPENDS "${KERNEL_IN}" "${CMAKE_SOURCE_DIR}/cmake/EmbedAsRawString.cmake"
        VERBATIM
    )

    add_custom_target(${TARGET_NAME} DEPENDS "${KERNEL_OUT}")
endfunction()

add_cl_kernel_compilation("naive.cl"   "kNaiveClSrc"  "naive_kernel")
add_cl_kernel_compilation("bitonic.cl" "kBitonicClSrc" "bitonic_kernel")
