/**
 * @file net.c
 * @brief Network structure
 */
#include "net.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

#include "random.h"

int net_size(const Net *net) {
    return net->size;
}

Layer *net_layers(Net *net) {
    return net->layers;
}

Layer *net_input(const Net *net) {
    return &net->layers[0];
}
 
Layer *net_output(const Net *net) {
    if (net->size == 0) {
        return NULL;
    }
    return &net->layers[net->size - 1];
}

Net *net_alloc_layers(
    Net *net, LayerParams *param_list
) {
    if ((net == NULL) || (param_list == NULL)) {
        return NULL;
    }

    int num_layers = 0;
    while (param_list[num_layers].type != LAYER_TYPE_NONE) {
        num_layers++;
    }

    if (num_layers == 0) {
        return NULL;
    }

    Layer *layers = malloc(sizeof(Layer) * num_layers);
    if (layers == NULL) {
        return NULL;
    }

    net->layers = layers;

    // Initialize new layers
    net->size = 0;
    for (int i = 0; i < num_layers; i++) {
        Layer *layer = &layers[i];

        layer->params = param_list[i];

        // Connect layers
        if (i > 0) {
            layer_connect(&layers[i - 1], &layers[i]);
        }

        // Initialize pointers
        layer->x = NULL;
        layer->y = NULL;
        layer->w = NULL;
        layer->b = NULL;

        layer->gx = NULL;
        layer->gw = NULL;
        layer->gb = NULL;

        if (layer_alloc_params(layer) == NULL) {
            goto FREE_LAYERS;
        }

        net->size++;
    }

    return net;

FREE_LAYERS:
    net_free_layers(net);

    return NULL;
}

void net_free_layers(Net *net) {
    if ((net == NULL) || (net->layers == NULL)) {
        return;
    }

    for (int i = 0; i < net->size; i++) {
        layer_free_params(&net->layers[i]);
    }

    free(net->layers);
    net->layers = NULL;
}

void net_init_params(Net *net) {
    srand(time(NULL));

    for (int i = 0; i < net->size; i++) {
        Layer *layer = &net_layers(net)[i];
        LayerParams *params = &layer->params;

        // Initialize weights by Xavier initialization
        const float n = (float)params->in;
        if (layer->w != NULL) {
            for (int j = 0; j < (params->in * params->out); j++) {
                layer->w[j] = rand_norm(0, (1 / sqrtf(n)));
            }
        }

        // Initialize biases by 0
        if (layer->b != NULL) {
            for (int j = 0; j < params->out; j++) {
                layer->b[j] = 0;
            }
        }
    }
}

float *net_forward(Net *net, const float *x) {
    if ((net == NULL) || (x == NULL)) {
        return NULL;
    }

    float *in = (float*)x;
    float *out = NULL;
    for (int i = 0; i < net->size; i++) {
        out = layer_forward(&net->layers[i], in);
        in = out;
    }

    return out;
}

float *net_backward(Net *net, const float *dy) {
    if ((net == NULL) || (dy == NULL)) {
        return NULL;
    }

    float *din = (float*)dy;
    float *dout = NULL;
    for (int i = (net->size - 1); i >= 0; i--) {
        dout = layer_backward(&net->layers[i], din);
        din = dout;
    }

    return dout;
}

void net_clear_grad(Net *net) {
    for (int i = 0; i < net->size; i++) {
        layer_clear_grad(&net->layers[i]);
    }
}

void net_load_from_file(Net *net, const char *config_file) {
    FILE *fp = fopen(config_file, "r");
    if (fp == NULL) {
        return;
    }

    char c;
    char buf[256];
    size_t size = 0;
    char *p_buf = buf;
    while ((c = fgetc(fp)) != EOF) {
        if ((c == '{') || (c == '}') || (c == '[') || (c == ']') ||
            (c == ' ')) {
            // Skip
        }
        if ((c == ':') || (c == ',') || (c == '\n')) {
            *p_buf = '\0';
            if (size > 0) {
                // printf("%s\n", buf);
            }
            p_buf = buf;
            size = 0;
        } else {
            *p_buf = c;
            p_buf++;
            size++;
        }
    }

    // Dummy
    net_alloc_layers(
        net,
        LAYER_PARAMS_LIST(
            { .type=LAYER_TYPE_FC, .batch_size=2, .in=3, .out=10 },
            { .type=LAYER_TYPE_SIGMOID, .batch_size=2, .in=10, .out=10 },
            { .type=LAYER_TYPE_FC, .batch_size=2, .in=10, .out=5 },
            { .type=LAYER_TYPE_SOFTMAX, .batch_size=2, .in=5, .out=5 }
        )
    );

    fclose(fp);
}
