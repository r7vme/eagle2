cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(eagle2_perception LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# ROS
find_package(catkin REQUIRED COMPONENTS
  nodelet
  pluginlib
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
    nodelet
    pluginlib
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
# NOTE: opencv should be compiled with CUDA support,
# Default Jetpack opencv has no CUDA support.
# How to recomplie https://github.com/jetsonhacks/buildOpenCVXavier
FIND_PACKAGE(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARIES_DIRS})

# tensorRT
find_library(NVINFER NAMES nvinfer)
find_library(NVPARSERS NAMES nvparsers)

# eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# tensorrt l2norm_helper plugin
set(TRT_L2NORM_HELPER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vendor/trt_l2norm_helper)
add_subdirectory(${TRT_L2NORM_HELPER_DIR})

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${catkin_INCLUDE_DIRS}
  ${TRT_L2NORM_HELPER_DIR}
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

set(EAGLE2_PERCEPTION_TARGET_LIBS
  ${catkin_LIBRARIES}
  Eigen3::Eigen
  trt_l2norm_helper
  TrtNet
  yaml-cpp
  nvinfer
  ${OpenCV_LIBS}
  ${CUDA_LIBRARIES}
)

# perception executable
add_executable(eagle2_perception
  src/perception.cpp
  src/perception_node.cpp
  src/bonnet.cpp
  src/box3d.cpp
  src/utils.cpp
)
target_link_libraries(eagle2_perception
  ${EAGLE2_PERCEPTION_TARGET_LIBS}
)
# camera_publisher executable
add_executable(eagle2_camera_publisher
  src/camera_publisher.cpp
  src/camera_publisher_node.cpp
)
target_link_libraries(eagle2_camera_publisher
  ${catkin_LIBRARIES}
  yaml-cpp
  ${OpenCV_LIBS}
)
# perception nodelet
add_library(eagle2_perception_nodelet
  src/perception.cpp
  src/perception_nodelet.cpp
  src/camera_publisher.cpp
  src/camera_publisher_nodelet.cpp
  src/bonnet.cpp
  src/box3d.cpp
  src/utils.cpp
)
add_dependencies(eagle2_perception_nodelet ${catkin_EXPORTED_TARGETS})
target_link_libraries(eagle2_perception_nodelet
  ${EAGLE2_PERCEPTION_TARGET_LIBS}
)

install(TARGETS eagle2_perception
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(TARGETS eagle2_perception_nodelet
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)
install(DIRECTORY launch/
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch)
