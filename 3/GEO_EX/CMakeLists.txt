cmake_minimum_required(VERSION 2.8)
project(useEigen)

set(CMAKE_BUILD_TYPE "Release")
set(CMTKE_CXXFLAGS "-03")

#add header files
include_directories("/usr/include/eigen3")
find_package(Pangolin)
include_directories(%{Pangolin_INCLUDE_DIRS})

add_executable(geo_EX geo_EX.cpp)
target_link_libraries(geo_EX ${Pangolin_LIBRARIES})
