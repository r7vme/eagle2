cmake_minimum_required(VERSION 2.8)
project(eagle2_perception)

# set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Ofast")

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
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
set(PYTHON_EXECUTABLE "python3")
include(FindTensorFlow)
include_directories(${TensorFlow_INCLUDE_DIR})
link_directories(${TensorFlow_LIBRARY})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(perception perception.cpp utils.cpp)
target_link_libraries(perception TrtNet yaml-cpp ${OpenCV_LIBS} ${TensorFlow_LIBRARY})
