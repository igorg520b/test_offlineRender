cmake_minimum_required(VERSION 3.5)

project(test_eigen LANGUAGES CXX C CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(spdlog REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS CXX)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)


add_executable(test_eigen main.cpp cuda_test.cu stb_image.h helper_gl.h)
set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)


include(GNUInstallDirs)
install(TARGETS test_eigen
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

target_link_libraries(test_eigen PRIVATE Eigen3::Eigen spdlog::spdlog)

set_target_properties(test_eigen PROPERTIES CUDA_ARCHITECTURES "89")
set_target_properties(test_eigen PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_link_libraries(test_eigen PRIVATE ${HDF5_LIBRARIES})
target_include_directories(test_eigen PRIVATE ${HDF5_INCLUDE_DIRS})

target_include_directories(test_eigen PRIVATE /home/s2/Projects-Libraries/cub/)

target_link_libraries(test_eigen
    PRIVATE
    OpenGL::GL
    GLEW::GLEW
    glfw
)
