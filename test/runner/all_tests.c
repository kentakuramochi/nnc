/**
 * @file all_tests.c
 * @brief main module of test runner
 * 
 */
#include "unity_fixture.h"

static void RunAllTests(void)
{
    RUN_TEST_GROUP(random);

    RUN_TEST_GROUP(mat);

    RUN_TEST_GROUP(layer);

    RUN_TEST_GROUP(fc);

    RUN_TEST_GROUP(sigmoid);

    RUN_TEST_GROUP(softmax);

    RUN_TEST_GROUP(net);

    RUN_TEST_GROUP(loss);
}

int main(int argc, const char *argv[])
{
    return UnityMain(argc, argv, RunAllTests);
}
