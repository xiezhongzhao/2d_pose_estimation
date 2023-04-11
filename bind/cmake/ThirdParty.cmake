#### for TensorFlow Lite and OpenCV
set(tflite ${CMAKE_CURRENT_SOURCE_DIR}/thirdParty/include/)
message(STATUS "tflite: ${tflite}")
include_directories(${tflite})
#
set(depLibs ${CMAKE_CURRENT_SOURCE_DIR}/thirdParty/lib/*)
file(GLOB thirdPartyLibs ${depLibs})
message(STATUS "depLibs: ${thirdPartyLibs}")