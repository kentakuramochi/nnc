set(TEST_ROOT ${PROJECT_SOURCE_DIR}/test)
set(UNITY_ROOT ${TEST_ROOT}/unity)

include(FetchContent)

# download Unity
FetchContent_Declare(append_unity
    GIT_REPOSITORY  git@github.com:ThrowTheSwitch/Unity.git
    GIT_TAG         v2.5.2
    SOURCE_DIR      ${UNITY_ROOT}
)
FetchContent_MakeAvailable(append_unity)

set(TARGET_TEST_RUNNER_NAME test_runner)

#file(GLOB SOURCES ${TEST_ROOT}/cases/*.c ${TEST_ROOT}/runner/*.c)
file(GLOB SOURCES
    ${TEST_ROOT}/cases/test_scnn_blas.c
    ${TEST_ROOT}/cases/test_scnn_mat.c
    ${TEST_ROOT}/cases/test_scnn_layer.c
    ${TEST_ROOT}/cases/test_scnn_fc.c
    ${TEST_ROOT}/cases/test_scnn_sigmoid.c
    ${TEST_ROOT}/cases/test_scnn_softmax.c
    ${TEST_ROOT}/runner/all_tests.c
    ${TEST_ROOT}/runner/test_scnn_blas_runner.c
    ${TEST_ROOT}/runner/test_scnn_mat_runner.c
    ${TEST_ROOT}/runner/test_scnn_layer_runner.c
    ${TEST_ROOT}/runner/test_scnn_fc_runner.c
    ${TEST_ROOT}/runner/test_scnn_sigmoid_runner.c
    ${TEST_ROOT}/runner/test_scnn_softmax_runner.c
)

add_executable(${TARGET_TEST_RUNNER_NAME}
    ${SOURCES}
    ${UNITY_ROOT}/src/unity.c
    ${UNITY_ROOT}/extras/fixture/src/unity_fixture.c
)

add_custom_target(test
    COMMAND ./${TARGET_TEST_RUNNER_NAME}
    DEPENDS ${TARGET_TEST_RUNNER_NAME}
)

add_custom_target(memcheck
    COMMAND valgrind -s --tool=memcheck --leak-check=yes --show-reachable=yes ./${TARGET_TEST_RUNNER_NAME}
    DEPENDS ${TARGET_TEST_RUNNER_NAME}
)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED True)

target_compile_options(${TARGET_TEST_RUNNER_NAME}
    PUBLIC -Wall -Wextra -Wpedantic -Werror
)

target_include_directories(${TARGET_TEST_RUNNER_NAME}
    PUBLIC ${UNITY_ROOT}/src
    PUBLIC ${UNITY_ROOT}/extras/fixture/src
)

target_link_directories(${TARGET_TEST_RUNNER_NAME}
    PUBLIC ${TARGET_LIB_DIR}
)

target_link_libraries(${TARGET_TEST_RUNNER_NAME}
    ${TARGET_LIB_NAME}
)

# using Unity fixtures w/o memory handling
add_definitions(-DUNITY_FIXTURE_NO_EXTRAS)
