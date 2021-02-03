cmake_minimum_required(VERSION 2.8)
project(useEigen)

set(CMAKE_BUILD_TYPE "Release")
set(CMTKE_CXXFLAGS "-03")

#add header files
include_directories("/usr/include/eigen3")
add_executable(eigenEX eigenEX.cpp)
