cmake_minimum_required(VERSION 3.8)

project("TreeDepth" LANGUAGES CXX)

file(GLOB source_files ${CMAKE_SOURCE_DIR}/src/**)
add_executable(${PROJECT_NAME} main.cpp ${source_files})

if(DEFINED ENV{"OVERRIDE_CUDA_DIR"})
    set(CUDA_TOOLKIT_ROOT_DIR $ENV{"OVERRIDE_CUDA_DIR"})
    set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc.exe")
else()
    find_package(CUDA 10.1 REQUIRED)
endif()

if(WIN32)
    set(CMAKE_CUDA_HOST_COMPILER cl.exe)
elseif(UNIX)
    set(CMAKE_CUDA_HOST_COMPILER g++-8)
endif()
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
enable_language(CUDA)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
