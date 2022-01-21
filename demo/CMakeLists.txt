find_package(PCL 1.8 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories ("${CMAKE_SOURCE_DIR}/alignability_prediction/include")

add_executable(demo_alignabilityPrediction demo_alignability_prediction.cpp)
target_link_libraries(demo_alignabilityPrediction boost_system ${PCL_LIBRARIES})

add_executable(demo_pointCloudVisualizer demo_point_cloud_visualizer.cpp)
target_link_libraries(demo_pointCloudVisualizer boost_system ${PCL_LIBRARIES})
