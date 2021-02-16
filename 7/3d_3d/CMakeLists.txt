cmake_minimum_required(VERSION 2.8)

project(3d-3d)

set(CMAKE_BUILD_TYPE "Release")
#set(CMTKE_CXXFLAGS "-03")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)


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
include_directories("/usr/local/include/g2o")

#add g2o
find_package(g2o REQUIRED)
include_directories("/usr/include/eigen3")

add_executable(3D_3D 3D_3D.cpp)

#link opencv
target_link_libraries(3D_3D ${OpenCV_LIBS} Sophus::Sophus g2o::core g2o::stuff)
