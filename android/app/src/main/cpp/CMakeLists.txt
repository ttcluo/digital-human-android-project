#===============================================================================
# C++推理引擎CMakeLists.txt
#===============================================================================

cmake_minimum_required(VERSION 3.22.1)

project("digital_human_inference")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(${PROJECT_NAME} SHARED native-lib.cpp)

find_library(log-lib log)

target_link_libraries(${PROJECT_NAME}
    android
    ${log-lib}
)
