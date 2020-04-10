cmake_minimum_required(VERSION 3.8)

project("TreeDepthCPU" LANGUAGES CXX)
set(TEST_PROJECT_NAME "${PROJECT_NAME}Test")

file(GLOB_RECURSE PROJECT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*)
add_executable(${PROJECT_NAME} ${PROJECT_SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

file(GLOB_RECURSE TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test/*)
add_executable(${TEST_PROJECT_NAME} ${TEST_SOURCES} ${PROJECT_SOURCES})
target_link_libraries(${TEST_PROJECT_NAME} gtest_main)
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD COMMAND ${TEST_PROJECT_NAME})

find_program(CLANG_FORMAT_EXE NAMES "clang-format")
if(NOT CLANG_FORMAT_EXE)
  message(STATUS "clang-format not found. Consider isntalling it from https://llvm.org/builds/")
else()
  message(STATUS "clang-format found: ${CLANG_FORMAT_EXE}")
  add_custom_target("clang-format-cpu" COMMAND ${CLANG_FORMAT_EXE} -i -style=Chromium ${PROJECT_SOURCES} ${TEST_SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
  add_dependencies(${PROJECT_NAME} "clang-format-cpu")
endif()
