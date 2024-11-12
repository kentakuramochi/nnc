/**
 * @file test_net.c
 * @brief Unit tests of net.c
 */
#include "net.h"

#include "mock_layer.h"
#include "mock_random.h"
#include "unity.h"
#include "test_utils.h"

// Dummy layer type
#define LAYER_TYPE_DUMMY 1

void setUp(void) {}

void tearDown(void) {}

void test_allocate_and_free_layer(void) {
    Net net;

    // Function `layer_alloc` call isn't fully tested
    // arguments are not verified because of using malloc internally
    Layer dummy_layer;
    layer_alloc_params_ExpectAnyArgsAndReturn(&dummy_layer);
    LayerParams params = {
        .type=LAYER_TYPE_DUMMY, .batch_size=1, .in=2, .out=2
    };
    TEST_ASSERT_EQUAL_PTR(
        &net,
        net_alloc_layers(
            &net,
            (LayerParams[]){
                params,
                { LAYER_TYPE_NONE } // Empty struct is bad in C99
            }
        )
    );

    TEST_ASSERT_EQUAL_INT(1, net_size(&net));

    TEST_ASSERT_NOT_NULL(net_layers(&net));

    Layer *layer = &net_layers(&net)[0];
    TEST_ASSERT_EQUAL_PTR(layer, net_input(&net));
    TEST_ASSERT_EQUAL_PTR(layer, net_output(&net));

    TEST_ASSERT_EQUAL_MEMORY(&params, &layer->params, sizeof(LayerParams));

    // Same with `layer_alloc_params`,
    // `layer_free_params` is not fully tested
    layer_free_params_ExpectAnyArgs();
    net_free_layers(&net);
    TEST_ASSERT_NULL(net.layers);
}

void test_allocate_and_free_3layers(void) {
    Net net;

    LayerParams layer_params[] = {
        { LAYER_TYPE_DUMMY, .batch_size=1, .in=2, .out=2 },
        { LAYER_TYPE_DUMMY },
        { LAYER_TYPE_DUMMY },
        { LAYER_TYPE_NONE }
    };

    Layer dummy_layer;
    layer_alloc_params_ExpectAnyArgsAndReturn(&dummy_layer);
    layer_connect_ExpectAnyArgsAndReturn(true);
    layer_alloc_params_ExpectAnyArgsAndReturn(&dummy_layer);
    layer_connect_ExpectAnyArgsAndReturn(true);
    layer_alloc_params_ExpectAnyArgsAndReturn(&dummy_layer);
    TEST_ASSERT_EQUAL_PTR(&net, net_alloc_layers(&net, layer_params));

    TEST_ASSERT_EQUAL_INT(3, net_size(&net));

    TEST_ASSERT_EQUAL_PTR(&net_layers(&net)[0], net_input(&net));
    TEST_ASSERT_EQUAL_PTR(&net_layers(&net)[2], net_output(&net));

    for (int i = 0; i < 3; i++) {
        Layer *layer = &net_layers(&net)[i];
        TEST_ASSERT_EQUAL_MEMORY(
            &layer_params[i], &layer->params, sizeof(LayerParams)
        );
    }

    layer_free_params_ExpectAnyArgs();
    layer_free_params_ExpectAnyArgs();
    layer_free_params_ExpectAnyArgs();
    net_free_layers(&net);
    TEST_ASSERT_NULL(net.layers);
}

void test_allocation_fail_if_net_is_NULL(void) {
    TEST_ASSERT_NULL(
        net_alloc_layers(
            NULL,
            (LayerParams[]){
                { LAYER_TYPE_DUMMY, .batch_size=1, .in=2, .out=2 },
                { LAYER_TYPE_NONE }
            }
        )
    );
}

void test_allocation_fail_if_param_list_is_NULL(void) {
    Net net;
    TEST_ASSERT_NULL(net_alloc_layers(&net, NULL));
}

void test_allocation_fail_if_layer_parameter_is_empty(void) {
    Net net;
    TEST_ASSERT_NULL(net_alloc_layers(&net, (LayerParams[]){ {} }));
}

void test_free_layers_for_NULL(void) {
    Net *net = NULL;
    net_free_layers(net);
}

void test_free_layers_when_layers_are_NULL(void) {
    Net net = { .layers=NULL };
    net_free_layers(&net);
}

void test_init(void) {
    Net net = {
        .size = 3,
        .layers = (Layer[]){
            {
                .params={ .in=2, .out=3 },
                .w=TEST_UTIL_FLOAT_ZEROS(3 * 2),
                .b=TEST_UTIL_FLOAT_ZEROS(3)
            },
            { .w=NULL, .b=NULL },
            {
                .params={ .in=3, .out=1 },
                .w=TEST_UTIL_FLOAT_ZEROS(1 * 3),
                .b=TEST_UTIL_FLOAT_ZEROS(1)
            }
        }
    };

    for (int i = 0; i < (3 * 2); i++) {
        rand_norm_ExpectAndReturn(0, (1 / sqrtf(2)), 1);
    }
    for (int i = 0; i < 3; i++) {
        rand_norm_ExpectAndReturn(0, (1 / sqrtf(3)), 1);
    }

    net_init_params(&net);

    // Confirm values are set
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(
        TEST_UTIL_FLOAT_ARRAY(1, 1, 1, 1, 1, 1), net.layers[0].w, (3 * 2)
    );
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(
        TEST_UTIL_FLOAT_ARRAY(1, 1, 1), net.layers[2].w, 3
    );
    // Confirm biases are all zero
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(TEST_UTIL_FLOAT_ZEROS(3), net.layers[0].b, 3);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(TEST_UTIL_FLOAT_ZEROS(1), net.layers[2].b, 1);
}

void test_forward_layer(void) {
    Net net;

    layer_connect_IgnoreAndReturn(true);
    Layer dummy_layer;
    layer_alloc_params_IgnoreAndReturn(&dummy_layer);
    net_alloc_layers(
        &net,
        (LayerParams[]){
            { LAYER_TYPE_DUMMY, .batch_size=1, .in=2, .out=2 },
            { LAYER_TYPE_NONE }
        }
    );

    float dummy_x, dummy_y;
    layer_forward_ExpectAndReturn(
        &net_layers(&net)[0], &dummy_x, &dummy_y
    );
    TEST_ASSERT_EQUAL_PTR(&dummy_y, net_forward(&net, &dummy_x));

    layer_free_params_Ignore();
    net_free_layers(&net);
}

void test_forward_3layers(void) {
    Net net;

    layer_connect_IgnoreAndReturn(true);
    Layer dummy_layer;
    layer_alloc_params_IgnoreAndReturn(&dummy_layer);
    net_alloc_layers(
        &net,
        (LayerParams[]){
            { LAYER_TYPE_DUMMY, .batch_size=1, .in=2, .out=2 },
            { LAYER_TYPE_DUMMY },
            { LAYER_TYPE_DUMMY },
            { LAYER_TYPE_NONE }
        }
    );

    float dummy_x, dummy_y[3];
    layer_forward_ExpectAndReturn(
        &net_layers(&net)[0], &dummy_x, &dummy_y[0]
    );
    layer_forward_ExpectAndReturn(
        &net_layers(&net)[1], &dummy_y[0], &dummy_y[1]
    );
    layer_forward_ExpectAndReturn(
        &net_layers(&net)[2], &dummy_y[1], &dummy_y[2]
    );
    TEST_ASSERT_EQUAL_PTR(&dummy_y[2], net_forward(&net, &dummy_x));

    layer_free_params_Ignore();
    net_free_layers(&net);
}

void test_forward_fail_if_net_is_NULL(void) {
    float dummy_x;
    TEST_ASSERT_NULL(net_forward(NULL, &dummy_x));
}

void test_forward_fail_if_x_is_NULL(void) {
    Net net;

    layer_connect_IgnoreAndReturn(true);
    Layer dummy_layer;
    layer_alloc_params_IgnoreAndReturn(&dummy_layer);
    net_alloc_layers(
        &net,
        (LayerParams[]){
            { LAYER_TYPE_DUMMY, .batch_size=1, .in=2, .out=2 },
            { LAYER_TYPE_NONE }
        }
    );

    TEST_ASSERT_NULL(net_forward(&net, NULL));

    layer_free_params_Ignore();
    net_free_layers(&net);
}

void test_backward_layer(void) {
    Net net;

    layer_connect_IgnoreAndReturn(true);
    Layer dummy_layer;
    layer_alloc_params_IgnoreAndReturn(&dummy_layer);
    net_alloc_layers(
        &net,
        (LayerParams[]){
            { LAYER_TYPE_DUMMY, .batch_size=1, .in=2, .out=2 },
            { LAYER_TYPE_NONE }
        }
    );

    float dummy_gy, dummy_gx;
    layer_backward_ExpectAndReturn(
        &net_layers(&net)[0], &dummy_gy, &dummy_gx
    );
    TEST_ASSERT_EQUAL_PTR(&dummy_gx, net_backward(&net, &dummy_gy));

    layer_free_params_Ignore();
    net_free_layers(&net);
}

void test_backward_3layer(void) {
    Net net;

    layer_connect_IgnoreAndReturn(true);
    Layer dummy_layer;
    layer_alloc_params_IgnoreAndReturn(&dummy_layer);
    net_alloc_layers(
        &net,
        (LayerParams[]){
            { LAYER_TYPE_DUMMY, .batch_size=1, .in=2, .out=2 },
            { LAYER_TYPE_DUMMY },
            { LAYER_TYPE_DUMMY },
            { LAYER_TYPE_NONE }
        }
    );

    float dummy_gy, dummy_gx[3];
    layer_backward_ExpectAndReturn(
        &net_layers(&net)[2], &dummy_gy, &dummy_gx[2]
    );
    layer_backward_ExpectAndReturn(
        &net_layers(&net)[1], &dummy_gx[2], &dummy_gx[1]
    );
    layer_backward_ExpectAndReturn(
        &net_layers(&net)[0], &dummy_gx[1], &dummy_gx[0]
    );
    TEST_ASSERT_EQUAL_PTR(&dummy_gx[0], net_backward(&net, &dummy_gy));

    layer_free_params_Ignore();
    net_free_layers(&net);
}

void test_backward_fail_if_net_is_NULL(void) {
    float dummy_gy;
    TEST_ASSERT_NULL(net_backward(NULL, &dummy_gy));
}

void test_backward_fail_if_dy_is_NULL(void) {
    Net net;

    layer_connect_IgnoreAndReturn(true);
    Layer dummy_layer;
    layer_alloc_params_IgnoreAndReturn(&dummy_layer);
    net_alloc_layers(
        &net,
        (LayerParams[]){
            { LAYER_TYPE_DUMMY, .batch_size=1, .in=2, .out=2 },
            { LAYER_TYPE_NONE }
        }
    );

    TEST_ASSERT_NULL(net_backward(&net, NULL));

    layer_free_params_Ignore();
    net_free_layers(&net);
}

void test_clear_grad(void) {
    Net net;

    layer_connect_IgnoreAndReturn(true);
    Layer dummy_layer;
    layer_alloc_params_IgnoreAndReturn(&dummy_layer);
    net_alloc_layers(
        &net,
        (LayerParams[]){
            { LAYER_TYPE_DUMMY, .batch_size=1, .in=2, .out=2 },
            { LAYER_TYPE_DUMMY },
            { LAYER_TYPE_DUMMY },
            { LAYER_TYPE_NONE }
        }
    );

    layer_clear_grad_Expect(&net_layers(&net)[0]);
    layer_clear_grad_Expect(&net_layers(&net)[1]);
    layer_clear_grad_Expect(&net_layers(&net)[2]);
    net_clear_grad(&net);

    layer_free_params_Ignore();
    net_free_layers(&net);
}

#define TEST_JSON_FILE "/tmp/test.json"

void test_load_from_file(void) {
    test_util_create_text_file(
        TEST_JSON_FILE,
        "{\n"
        "    \"size\": 4,\n"
        "    \"layers\": [\n"
        "        {\n"
        "            \"params\": {\n"
        "                \"type\": 1,\n"
        "                \"batch_size\": 2,\n"
        "                \"in\": 3,\n"
        "                \"out\": 10\n"
        "            }\n"
        "        },\n"
        "        {\n"
        "            \"params\": {\n"
        "                \"type\": 2,\n"
        "                \"batch_size\": 2,\n"
        "                \"in\": 10,\n"
        "                \"out\": 10\n"
        "            }\n"
        "        },\n"
        "        {\n"
        "            \"params\": {\n"
        "                \"type\": 1,\n"
        "                \"batch_size\": 2,\n"
        "                \"in\": 10,\n"
        "                \"out\": 5\n"
        "            }\n"
        "        },\n"
        "        {\n"
        "            \"params\": {\n"
        "                \"type\": 3,\n"
        "                \"batch_size\": 2,\n"
        "                \"in\": 5,\n"
        "                \"out\": 5\n"
        "            }\n"
        "        }\n"
        "    ]\n"
        "}"
    );

    Net net;

    Layer dummy;
    layer_connect_IgnoreAndReturn(true);
    layer_alloc_params_IgnoreAndReturn(&dummy);
    net_load_from_file(&net, TEST_JSON_FILE);

    TEST_ASSERT_EQUAL(4, net.size);

    TEST_ASSERT_EQUAL(LAYER_TYPE_FC, net.layers[0].params.type);
    TEST_ASSERT_EQUAL(2, net.layers[0].params.batch_size);
    TEST_ASSERT_EQUAL(3, net.layers[0].params.in);
    TEST_ASSERT_EQUAL(10, net.layers[0].params.out);

    TEST_ASSERT_EQUAL(LAYER_TYPE_SIGMOID, net.layers[1].params.type);
    TEST_ASSERT_EQUAL(2, net.layers[1].params.batch_size);
    TEST_ASSERT_EQUAL(10, net.layers[1].params.in);
    TEST_ASSERT_EQUAL(10, net.layers[1].params.out);

    TEST_ASSERT_EQUAL(LAYER_TYPE_FC, net.layers[2].params.type);
    TEST_ASSERT_EQUAL(2, net.layers[2].params.batch_size);
    TEST_ASSERT_EQUAL(10, net.layers[2].params.in);
    TEST_ASSERT_EQUAL(5, net.layers[2].params.out);

    TEST_ASSERT_EQUAL(LAYER_TYPE_SOFTMAX, net.layers[3].params.type);
    TEST_ASSERT_EQUAL(2, net.layers[3].params.batch_size);
    TEST_ASSERT_EQUAL(5, net.layers[3].params.in);
    TEST_ASSERT_EQUAL(5, net.layers[3].params.out);

    layer_free_params_Ignore();
    net_free_layers(&net);
}

void test_load_from_file_fail_when_NULL_input(void) {
    test_util_create_text_file(
        TEST_JSON_FILE,
        "{\n"
        "    \"size\": 4,\n"
        "    \"layers\": [\n"
        "        {\n"
        "            \"params\": {\n"
        "                \"type\": 1,\n"
        "                \"batch_size\": 2,\n"
        "                \"in\": 3,\n"
        "                \"out\": 10\n"
        "            }\n"
        "        },\n"
        "        {\n"
        "            \"params\": {\n"
        "                \"type\": 2,\n"
        "                \"batch_size\": 2,\n"
        "                \"in\": 10,\n"
        "                \"out\": 10\n"
        "            }\n"
        "        },\n"
        "        {\n"
        "            \"params\": {\n"
        "                \"type\": 1,\n"
        "                \"batch_size\": 2,\n"
        "                \"in\": 10,\n"
        "                \"out\": 5\n"
        "            }\n"
        "        },\n"
        "        {\n"
        "            \"params\": {\n"
        "                \"type\": 3,\n"
        "                \"batch_size\": 2,\n"
        "                \"in\": 5,\n"
        "                \"out\": 5\n"
        "            }\n"
        "        }\n"
        "    ]\n"
        "}"
    );

    Net *net = NULL;
    net_load_from_file(net, NULL);
    TEST_ASSERT_NULL(net);

    // Just return, nothing occured
    net_load_from_file(NULL, TEST_JSON_FILE);
}

void test_load_from_file_fail_when_file_is_missing(void) {
    // Just return, nothing occured
    Net *net = NULL;
    net_load_from_file(net, "/non/exsiting/file.json");
}
