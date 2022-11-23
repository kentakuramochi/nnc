#file(GLOB SOURCES ./*.c)
file(GLOB SOURCES
    ./scnn_blas.c
    ./scnn_mat.c
    ./scnn_layer.c
    ./scnn_fc.c
    ./scnn_sigmoid.c
    ./scnn_softmax.c
)

add_library(${TARGET_LIB_NAME}
    ${SOURCES}
)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED True)

target_compile_options(${TARGET_LIB_NAME}
    PUBLIC -Wall -Wextra -Wpedantic -Werror
)

target_include_directories(${TARGET_LIB_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(${TARGET_LIB_NAME}
    m
)

set_target_properties(${TARGET_LIB_NAME}
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${TARGET_LIB_DIR}
    LIBRARY_OUTPUT_DIRECTORY ${TARGET_LIB_DIR}
)
