# 사용할 최소 CMake 버전
cmake_minimum_required(VERSION 3.20)

# 프로젝트 이름 + C++ 프로젝트임을 지정
project(GazeMonitoringCpp LANGUAGES CXX)

# C++17 사용
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV는 기존처럼 vcpkg에서 찾음
find_package(OpenCV CONFIG REQUIRED)

# 공식 ONNX Runtime을 풀어둔 위치
set(ONNXRUNTIME_DIR "C:/dev/onnxruntime")

# OpenCV 테스트 프로그램
add_executable(test_opencv
    src/test_opencv.cpp
)

target_link_libraries(test_opencv
    PRIVATE
    ${OpenCV_LIBS}
)

# ONNX Runtime 테스트 프로그램
add_executable(test_onnx
    src/test_onnx.cpp
)

# #include <onnxruntime_cxx_api.h>를 찾을 위치
target_include_directories(test_onnx
    PRIVATE
    "${ONNXRUNTIME_DIR}/include"
)

# onnxruntime.lib 연결
target_link_libraries(test_onnx
    PRIVATE
    "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib"
)

# 실행할 때 필요한 onnxruntime.dll을 exe 옆으로 자동 복사
add_custom_command(TARGET test_onnx POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
        "$<TARGET_FILE_DIR:test_onnx>"
)