cmake_minimum_required(VERSION 3.8)
message(STATUS "CMake version: ${CMAKE_VERSION}")
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

rosbuild_init()

IF(NOT ROS_BUILD_TYPE)
  SET(ROS_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${ROS_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -O3 -march=native")

# Check C++17 support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)

if(COMPILER_SUPPORTS_CXX17)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
    add_definitions(-DCOMPILEDWITHC11)
    message(STATUS "Using flag -std=c++17.")
else()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++17 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../../cmake_modules)

# for libtorch
set(CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/../../../thirdparty/pytorch/torch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
message("-- Torch Version: " ${Torch_VERSION})

find_package(OpenCV 4.2 QUIET)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Development)
include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/../../../
${PROJECT_SOURCE_DIR}/../../../include
${PROJECT_SOURCE_DIR}/../../../include/CameraModels
${Pangolin_INCLUDE_DIRS}
${Python3_INCLUDE_DIRS}
)
set(LIBS 
${OpenCV_LIBS} 
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${TORCH_LIBRARIES}
${PROJECT_SOURCE_DIR}/../../../thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/../../../thirdparty/g2o/lib/libg2o.so
${PROJECT_SOURCE_DIR}/../../../lib/libxfeatSLAM.so
-lboost_system
)

# Node for monocular camera
rosbuild_add_executable(Mono
src/ros_mono.cc
)


target_link_libraries(Mono
${LIBS}
)

# Node for monocular camera (Augmented Reality Demo)
rosbuild_add_executable(MonoAR
src/AR/ros_mono_ar.cc
src/AR/ViewerAR.h
src/AR/ViewerAR.cc
)

target_link_libraries(MonoAR
${LIBS}
)

# Node for stereo camera
rosbuild_add_executable(Stereo
src/ros_stereo.cc
)

target_link_libraries(Stereo
${LIBS}
)

# Node for RGB-D camera
rosbuild_add_executable(RGBD
src/ros_rgbd.cc
)

target_link_libraries(RGBD
${LIBS}
)

# Node for monocular-inertial camera
rosbuild_add_executable(Mono_Inertial
src/ros_mono_inertial.cc
)

target_link_libraries(Mono_Inertial
${LIBS}
)

# Node for stereo-inertial camera
rosbuild_add_executable(Stereo_Inertial
src/ros_stereo_inertial.cc
)

target_link_libraries(Stereo_Inertial
${LIBS}
)
