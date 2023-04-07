cmake_minimum_required(VERSION 3.16)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAEK_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake;${CMAKE_MODULE_PATH}")

project(2d_pose_estimation LANGUAGES CXX)

#### for TensorFlow Lite and OpenCV
include(${PROJECT_SOURCE_DIR}/cmake/thirdParty.cmake)

add_subdirectory(pose_estimation)
add_subdirectory(test)

