cmake_minimum_required(VERSION 2.8)
project(useEigen)

set(CMAKE_BUILD_TYPE "Release")
set(CMTKE_CXXFLAGS "-03")

#add opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#add eigen
include_directories("/usr/include/eigen3")

add_executable(gau gau.cpp)

#link opencv
target_link_libraries(gau ${OpenCV_LIBS})

