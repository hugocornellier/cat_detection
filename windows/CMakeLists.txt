cmake_minimum_required(VERSION 3.14)

set(PROJECT_NAME "cat_detection")
project(${PROJECT_NAME} LANGUAGES CXX)

set(PLUGIN_NAME "cat_detection_plugin")

add_library(${PLUGIN_NAME} SHARED
  "cat_detection_plugin.cpp"
  "cat_detection_plugin.h"
  "cat_detection_plugin_c_api.cpp"
  "include/cat_detection/cat_detection_plugin_c_api.h"
)

apply_standard_settings(${PLUGIN_NAME})

target_compile_definitions(${PLUGIN_NAME} PRIVATE FLUTTER_PLUGIN_IMPL)
target_include_directories(${PLUGIN_NAME}
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}"
  INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(${PLUGIN_NAME} PRIVATE flutter flutter_wrapper_plugin)
