cmake_minimum_required(VERSION 3.8)

project("TreeDepthCPU" LANGUAGES CXX)
set(TEST_PROJECT_NAME "${PROJECT_NAME}Test")

file(GLOB_RECURSE PROJECT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*)
add_executable(${PROJECT_NAME} ${PROJECT_SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
target_compile_definitions(${PROJECT_NAME} PRIVATE -DTD_CHECK_ARGS)

file(GLOB_RECURSE TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test/*)
add_executable(${TEST_PROJECT_NAME} ${TEST_SOURCES} ${PROJECT_SOURCES})
set_property(TARGET ${TEST_PROJECT_NAME} PROPERTY CXX_STANDARD 17)
target_compile_definitions(${TEST_PROJECT_NAME} PRIVATE -DTD_CHECK_ARGS)
target_link_libraries(${TEST_PROJECT_NAME} gtest_main)
target_link_libraries(${TEST_PROJECT_NAME} gmock_main)
if(CMAKE_BUILD_TYPE MATCHES DEBUG)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD COMMAND ${TEST_PROJECT_NAME})
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

find_program(CLANG_FORMAT_EXE NAMES "clang-format")
if(NOT CLANG_FORMAT_EXE)
  message(STATUS "clang-format not found. Consider isntalling it from https://llvm.org/builds/")
else()
  message(STATUS "clang-format found: ${CLANG_FORMAT_EXE}")
  add_custom_command(
  TARGET ${PROJECT_NAME}
  POST_BUILD
  COMMAND ${CLANG_FORMAT_EXE} -i -style=Chromium ${PROJECT_SOURCES} ${TEST_SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
  add_custom_command(
  TARGET ${TEST_PROJECT_NAME}
  POST_BUILD
  COMMAND ${CLANG_FORMAT_EXE} -i -style=Chromium ${PROJECT_SOURCES} ${TEST_SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
endif()
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
