/**
 * @file all_tests.c
 * @brief Main module of test runner
 * 
 */
#include "unity_fixture.h"

static void RunAllTests(void)
{
    RUN_TEST_GROUP(scnn_mat);

    RUN_TEST_GROUP(scnn_layer);

    //RUN_TEST_GROUP(data);

    //RUN_TEST_GROUP(util);

    //RUN_TEST_GROUP(random);

    //RUN_TEST_GROUP(mat);

    //RUN_TEST_GROUP(layer);

    //RUN_TEST_GROUP(fc);

    //RUN_TEST_GROUP(sigmoid);

    //RUN_TEST_GROUP(softmax);

    //RUN_TEST_GROUP(net);

    //RUN_TEST_GROUP(loss);

    //RUN_TEST_GROUP(trainer);
}

int main(int argc, const char *argv[])
{
    return UnityMain(argc, argv, RunAllTests);
}
