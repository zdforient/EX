cmake_minimum_required(VERSION 2.8)
project(rgbd)

set(CMAKE_BUILD_TYPE "Release")
set(CMTKE_CXXFLAGS "-03")

#add opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#add Pangolin
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

#add sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

add_executable(rgbd rgbd.cpp)

#link opencv
target_link_libraries(rgbd ${OpenCV_LIBS})
#link pangolin
target_link_libraries(rgbd ${Pangolin_LIBRARIES})
#link sophus
target_link_libraries(rgbd Sophus::Sophus)
