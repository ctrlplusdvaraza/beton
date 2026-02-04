cmake_minimum_required(VERSION 3.11)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


project(beton LANGUAGES CXX)

include_directories(
            ${CMAKE_SOURCE_DIR}/
            ${CMAKE_SOURCE_DIR}/src
)

add_compile_options(
    $<$<BOOL:${SANITIZE}>:-fsanitize=address,undefined>
)
add_link_options(
    $<$<BOOL:${SANITIZE}>:-fsanitize=address,undefined>
)

set(SOURCE_FILES 
	src/cpu_sort.cpp
    src/main.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

find_package(OpenCL REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenCL::OpenCL)

target_compile_definitions(${PROJECT_NAME} PRIVATE CL_TARGET_OPENCL_VERSION=300)

