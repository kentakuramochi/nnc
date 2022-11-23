/**
 * @file all_tests.c
 * @brief Main module of test runner
 * 
 */
#include "unity_fixture.h"

static void RunAllTests(void)
{
    RUN_TEST_GROUP(scnn_blas);

    RUN_TEST_GROUP(scnn_mat);

    RUN_TEST_GROUP(scnn_layer);

    RUN_TEST_GROUP(scnn_fc);

    RUN_TEST_GROUP(scnn_sigmoid);

    RUN_TEST_GROUP(scnn_softmax);

    //RUN_TEST_GROUP(scnn_net);

    //RUN_TEST_GROUP(scnn_loss);
}

int main(int argc, const char *argv[])
{
    return UnityMain(argc, argv, RunAllTests);
}
