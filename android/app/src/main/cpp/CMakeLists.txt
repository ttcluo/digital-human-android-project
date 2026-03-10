#===============================================================================
# C++推理引擎CMakeLists.txt
#===============================================================================

cmake_minimum_required(VERSION 3.22.1)

project("digital_human_inference")

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加ONNX Runtime头文件
include_directories(${CMAKE_SOURCE_DIR}/include)

# 源文件
set(NATIVE_SOURCES
    src/main/cpp/digital_human_jni.cpp
    src/main/cpp/InferenceEngine.cpp
    src/main/cpp/Preprocessor.cpp
    src/main/cpp/Postprocessor.cpp
)

# 创建共享库
add_library(${PROJECT_NAME} SHARED ${NATIVE_SOURCES})

# 链接库
target_link_libraries(${PROJECT_NAME}
    android
    log
    jnigraphics
    # ONNX Runtime
    onnxruntime
)

# 编译选项
target_compile_options(${PROJECT_NAME} PRIVATE
    -Wall
    -Wextra
    -O3
    -ffast-math
    -fvisibility=hidden
)
