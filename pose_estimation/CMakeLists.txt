file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cpp include/*.h)
add_library(pose_estimation STATIC ${srcs})
target_include_directories(pose_estimation PUBLIC include)




