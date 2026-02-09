set(SOURCE_FILES 
    main.cpp
    correctness.cpp
    performance.cpp

    ${CMAKE_SOURCE_DIR}/src/cpu_sort.cpp
    ${CMAKE_SOURCE_DIR}/src/gpu_sort.cpp
    ${CMAKE_SOURCE_DIR}/src/stupid.cpp
)

set(TARGET_NAME tests)

add_executable(${TARGET_NAME} ${SOURCE_FILES})

target_link_libraries(tests PRIVATE OpenCL::OpenCL)

target_include_directories(tests PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_BINARY_DIR}/generated
)

target_compile_definitions(${TARGET_NAME} PRIVATE
    CL_HPP_TARGET_OPENCL_VERSION=300
    CL_TARGET_OPENCL_VERSION=300
)

add_dependencies(tests bitonic_kernel)
add_dependencies(tests stupid_kernel)
