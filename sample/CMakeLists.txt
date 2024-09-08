add_custom_target(sample)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED True)

file(GLOB SOURCES *.c)

# Set a build target for each sample source file
foreach(source ${SOURCES})
    get_filename_component(exec
        ${source} NAME_WE
    )

    add_executable(${exec}
        ${source}
    )

    target_compile_options(${exec}
        PRIVATE -Wall -Wextra -Wpedantic -Werror
    )

    target_link_directories(${exec}
        PRIVATE ${TARGET_LIB_DIR}
    )

    target_link_libraries(${exec}
        ${TARGET_LIB_NAME}
    )

    add_dependencies(sample
        ${exec}
    )
endforeach()
