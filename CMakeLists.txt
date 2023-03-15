cmake_minimum_required(VERSION 3.16)
project(2d_pose_estimation)

set(CMAKE_CXX_STANDARD 17)

### for TensorFlow Lite
set(tflite ${CMAKE_CURRENT_SOURCE_DIR}/include/)
message(STATUS "tflite: ${tflite}")
include_directories(${tflite})

set(depLibs ${CMAKE_CURRENT_SOURCE_DIR}/lib/*)
file(GLOB libs ${depLibs})
message(STATUS "depLibs: ${libs}")

# pose2d
add_subdirectory(${PROJECT_SOURCE_DIR}/src)

add_executable(2d_pose_estimation main.cpp)

target_link_libraries(2d_pose_estimation pose2d ${libs} )






