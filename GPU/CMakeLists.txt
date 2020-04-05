cmake_minimum_required(VERSION 3.8)

# Configure main project
project("TreeDepth" LANGUAGES CXX)
file(GLOB source_files ${CMAKE_SOURCE_DIR}/src/**)
add_executable(${PROJECT_NAME} main.cpp ${source_files})

if(WIN32)
    set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1")
    set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc.exe")
    set(CMAKE_CUDA_HOST_COMPILER cl.exe)
else()
    set(CMAKE_CUDA_HOST_COMPILER g++-8)
    find_package(CUDA REQUIRED)
endif()
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
enable_language(CUDA)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Configure test project
file(GLOB test_sources ${CMAKE_SOURCE_DIR}/tests/**)
add_executable("${PROJECT_NAME}Test" ${test_sources})

# Download and unpack googletest at configure time
configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
                 ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
                 EXCLUDE_FROM_ALL)

# The gtest/gtest_main targets carry header search path
# dependencies automatically when using CMake 2.8.11 or
# later. Otherwise we have to add them here ourselves.
if (CMAKE_VERSION VERSION_LESS 2.8.11)
  include_directories("${gtest_SOURCE_DIR}/include")
endif()

target_link_libraries("${PROJECT_NAME}Test" gtest_main)

set(unit_test "${PROJECT_NAME}Test")
add_custom_command(
     TARGET ${unit_test}
     COMMENT "Run tests"
     POST_BUILD
     COMMAND ${unit_test}
)
