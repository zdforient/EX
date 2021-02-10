cmake_minimum_required(VERSION 2.8)

project(ceres)

set(CMAKE_BUILD_TYPE "Release")
#set(CMTKE_CXXFLAGS "-03")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)



#add opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#add Pangolin
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

#add sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

#add ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

#add eigen
include_directories("/usr/include/eigen3")

add_executable(ceresex ceresex.cpp)

#link opencv
target_link_libraries(ceresex ${OpenCV_LIBS})
#link pangolin
target_link_libraries(ceresex ${Pangolin_LIBRARIES})
#link sophus
target_link_libraries(ceresex Sophus::Sophus)
#link Ceres
target_link_libraries(ceresex ${CERES_LIBRARIES})
