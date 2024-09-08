cmake_minimum_required(VERSION 3.16)

set(TARGET_LIB_NAME nn)

project(${TARGET_LIB_NAME} C)

set(TARGET_LIB_DIR ${CMAKE_BINARY_DIR}/lib)

add_subdirectory(src)
add_subdirectory(sample EXCLUDE_FROM_ALL)
