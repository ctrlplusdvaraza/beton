include_directories(
            ${CMAKE_SOURCE_DIR}/
            ${CMAKE_SOURCE_DIR}/src
            ${CMAKE_SOURCE_DIR}/tests
)

set(SOURCE_FILES 
    main.cpp
    correctness.cpp
    performance.cpp
)

set(TARGET_NAME tests)

add_executable(${TARGET_NAME} ${SOURCE_FILES})

find_package(OpenCL REQUIRED)

target_link_libraries(${TARGET_NAME} PRIVATE OpenCL::OpenCL)

target_compile_definitions(${TARGET_NAME} PRIVATE CL_TARGET_OPENCL_VERSION=300)

