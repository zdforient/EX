cmake_minimum_required(VERSION 2.8)
project(useOpenCV)

set(CMAKE_BUILD_TYPE "Release")
set(CMTKE_CXXFLAGS "-03")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
#add header files
add_executable(img img.cpp)

target_link_libraries(img ${OpenCV_LIBS})

add_executable(undistort undistort.cpp)

target_link_libraries(undistort ${OpenCV_LIBS})

