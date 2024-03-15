set(TARGET_EXAMPLE_NAME mnist_example)

add_executable(${TARGET_EXAMPLE_NAME}
    ./mnist_example.c
)

add_custom_target(mnist
    DEPENDS ${TARGET_EXAMPLE_NAME}
)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED True)

target_compile_options(${TARGET_EXAMPLE_NAME}
    PUBLIC -Wall -Wextra -Wpedantic -Werror
)

target_link_directories(${TARGET_EXAMPLE_NAME}
    PUBLIC ${TARGET_LIB_DIR}
)

target_link_libraries(${TARGET_EXAMPLE_NAME}
    ${TARGET_LIB_NAME}
)
