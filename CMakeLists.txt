cmake_minimum_required (VERSION 3.8)

project ("TreeDepth" LANGUAGES CXX)

file(GLOB source_files
    ${CMAKE_SOURCE_DIR}/*.cu
)
add_executable(${PROJECT_NAME} ${source_files})

if(WIN32)
    set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/")
    set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc.exe")
	set(CMAKE_CUDA_HOST_COMPILER cl.exe)
else()
	set(CMAKE_CUDA_HOST_COMPILER g++-8)
endif()
enable_language(CUDA)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})