cmake_minimum_required(VERSION 3.0)
project(eagle2_perception)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

# ROS
find_package(catkin REQUIRED COMPONENTS
  cmake_modules
  roscpp
  std_msgs
  sensor_msgs
  cv_bridge
  image_transport
  jsk_recognition_msgs
)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS
    roscpp
    std_msgs
    sensor_msgs
    cv_bridge
    image_transport
    jsk_recognition_msgs
)

# tensorRTWrapper
set(TRT_WRAPPER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vendor/tensorRTWrapper/code)
add_subdirectory(${TRT_WRAPPER_DIR})
include_directories(${TRT_WRAPPER_DIR}/include)

# yaml
set(YAML_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vendor/yaml-cpp)
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "Enable testing")
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "Enable parse tools")
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "Enable contrib stuff in library")
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

# cuda
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# tensorRT
find_library(NVINFER NAMES nvinfer)
find_library(NVPARSERS NAMES nvparsers)

# eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${catkin_INCLUDE_DIRS}
)

add_executable(eagle2_perception
  src/perception.cpp
  src/bonnet.cpp
  src/utils.cpp
)
target_link_libraries(eagle2_perception
  ${catkin_LIBRARIES} TrtNet yaml-cpp Eigen3::Eigen TensorFlow_DEP nvinfer ${OpenCV_LIBS} ${CUDA_LIBRARIES}
)

install(TARGETS eagle2_perception
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY launch/
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch)
