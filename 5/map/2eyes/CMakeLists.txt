cmake_minimum_required(VERSION 2.8)
project(useEigen)

set(CMAKE_BUILD_TYPE "Release")
set(CMTKE_CXXFLAGS "-03")

#add openCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#add pangolin
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

add_executable(stereo stereo.cpp)

target_link_libraries(stereo ${OpenCV_LIBS} ${Pangolin_LIBRARIES})

