cmake_minimum_required(VERSION 3.22)

project(test_eigen LANGUAGES CXX C CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

#find_package(Eigen3 REQUIRED NO_MODULE)
#find_package(spdlog REQUIRED)
#find_package(HDF5 REQUIRED COMPONENTS CXX)
find_package(OpenGL REQUIRED)
#find_package(GLEW REQUIRED)
#find_package(glfw3 REQUIRED)

#find_library(OSMESA_LIBRARY OSMesa REQUIRED)
#find_path(OSMESA_INCLUDE_DIR GL/osmesa.h REQUIRED)

add_executable(test_eigen main.cpp stb_image_write.h)
set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)


set_target_properties(test_eigen PROPERTIES CUDA_ARCHITECTURES "86")
set_target_properties(test_eigen PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)


target_link_libraries(test_eigen
    PRIVATE
    OpenGL::GL
    OpenGL::EGL
)
