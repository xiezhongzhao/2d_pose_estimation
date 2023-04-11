cmake_minimum_required(VERSION 3.16.3)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAEK_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(bind LANGUAGES CXX)

#### for TensorFlow Lite and OpenCV
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/thirdParty.cmake)

### pose estimation headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

### 测试链接库调用是否正常
#add_executable(bind src/inference.cpp)
#target_link_libraries(bind PUBLIC
#        ${CMAKE_CURRENT_SOURCE_DIR}/libs/libpose_estimation.so
#        ${thirdPartyLibs})

add_subdirectory(pybind11) ### 加载pybind11库
pybind11_add_module(inference src/inference.cpp)
target_link_libraries(inference PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/libs/libpose_estimation.so