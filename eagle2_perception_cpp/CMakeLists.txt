cmake_minimum_required(VERSION 3.0)
project(eagle2_perception)

# set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Ofast -D_GLIBCXX_USE_CXX11_ABI=0")

# tensorRTWrapper
set(TRT_WRAPPER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vendor/tensorRTWrapper/code)
add_subdirectory(${TRT_WRAPPER_DIR})
include_directories(${TRT_WRAPPER_DIR}/include)

# yaml
set(YAML_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vendor/yaml-cpp)
set(YAML_CPP_BUILD_TESTS OFF)
set(YAML_CPP_BUILD_CONTRIB OFF)
add_subdirectory(${YAML_DIR})
include_directories(${YAML_DIR}/include)

# opencv
FIND_PACKAGE(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARIES_DIRS})

# tensorflow
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(PYTHON_EXECUTABLE "python3")
find_package(TensorFlow REQUIRED)
TensorFlow_REQUIRE_C_LIBRARY()

# eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(perception perception.cpp utils.cpp)
target_link_libraries(perception TrtNet yaml-cpp Eigen3::Eigen ${OpenCV_LIBS} TensorFlow_DEP)
