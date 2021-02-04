find_package(Pangolin REQUIRED)

include_directories(${Pangolin_INCLUDE_DIRS})

add_executable(traError tra_error.cpp)

target_link_libraries(traError ${Pangolin_LIBRARIES})

target_link_libraries(traError Sophus::Sophus)
