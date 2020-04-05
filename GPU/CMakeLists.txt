cmake_minimum_required(VERSION 3.8)

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

file(GLOB hooks ${CMAKE_SOURCE_DIR}/../hooks/*)
foreach(hook ${hooks})
   configure_file(${hook} ${CMAKE_SOURCE_DIR}/../.git/hooks COPYONLY)
endforeach()
