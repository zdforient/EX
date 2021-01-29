cmake_minimum_required(VERSION 2.8)

project(hello)

add_library(hello hello.cpp)

add_executable(helloslam main.cpp)

target_link_libraries(helloslam hello)

