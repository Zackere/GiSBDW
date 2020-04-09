cmake_minimum_required(VERSION 3.8)

project("TreeDepthCPU" LANGUAGES CXX)

file(GLOB_RECURSE PROJECT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/* ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})

find_program(CLANG_FORMAT_EXE NAMES "clang-format")
if(NOT CLANG_FORMAT_EXE)
  message(STATUS "clang-format not found. Consider isntalling it from https://llvm.org/builds/")
else()
  message(STATUS "clang-format found: ${CLANG_FORMAT_EXE}")
  add_custom_target("clang-format-cpu" COMMAND ${CLANG_FORMAT_EXE} -i -style=Google ${PROJECT_SOURCES})
  add_dependencies(${PROJECT_NAME} "clang-format-cpu")
endif()
