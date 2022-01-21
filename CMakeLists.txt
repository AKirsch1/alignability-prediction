cmake_minimum_required(VERSION 3.0.0)
project(Masterarbeit VERSION 0.1.0)

add_subdirectory(demo)

find_package(PCL 1.8 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
