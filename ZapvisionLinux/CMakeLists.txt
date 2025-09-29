cmake_minimum_required(VERSION 3.11...3.19)
project(zapvision VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Determine architecture
message(STATUS "System processor: ${CMAKE_SYSTEM_PROCESSOR}")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    set(LIB_DIR "${CMAKE_SOURCE_DIR}/lib/arm64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64)$")
    set(LIB_DIR "${CMAKE_SOURCE_DIR}/lib/x86_64")
else()
    message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

message(STATUS "Using static libraries from: ${LIB_DIR}")

#######################################################
# main zapvision c++ demo

add_executable(zapvision main.cpp)

target_link_libraries(zapvision
  PUBLIC
  ${LIB_DIR}/libZapvision.a
  ${LIB_DIR}/libopencv_video.a
  ${LIB_DIR}/libopencv_calib3d.a
  ${LIB_DIR}/libopencv_flann.a
  ${LIB_DIR}/libopencv_features2d.a
  ${LIB_DIR}/libopencv_imgproc.a
  ${LIB_DIR}/libopencv_core.a
  ${LIB_DIR}/libz.a
)

#######################################################
# pybind demo for testing

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

# Find Python interpreter, development headers, and libraries
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

pybind11_add_module(_zapvision_py zapvision_pybind.cpp)

target_link_libraries(_zapvision_py
  PRIVATE
  pybind11::module
  Python::Python

  ${LIB_DIR}/libZapvision.a
  ${LIB_DIR}/libopencv_video.a
  ${LIB_DIR}/libopencv_calib3d.a
  ${LIB_DIR}/libopencv_flann.a
  ${LIB_DIR}/libopencv_features2d.a
  ${LIB_DIR}/libopencv_imgproc.a
  ${LIB_DIR}/libopencv_core.a
  ${LIB_DIR}/libz.a
)

target_include_directories(_zapvision_py
  PRIVATE
  ${Python_INCLUDE_DIRS}
  ${pybind11_INCLUDE_DIRS}
)

install(TARGETS _zapvision_py
        LIBRARY DESTINATION zapvision_py) 
