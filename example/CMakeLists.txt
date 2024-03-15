add_subdirectory(xor)

add_custom_target(example
    DEPENDS xor_example
)
