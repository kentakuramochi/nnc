/**
 * @file net.c
 * @brief Network structure
 */
#include "net.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
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

/**
 * @brief Check equality of 2 strings
 *
 * @param[in] s1 NULL-terminated string #1
 * @param[in] s2 NULL-terminated string #2
 * @return true if s1 == s2, otherwise false
 */
static inline bool str_equal(const char *s1, const char *s2) {
    return !strcmp(s1, s2) ? true : false;
}

/**
 * @brief Get a JSON token from a stream
 *
 * @param[out] buffer Buffer to store a NULL-terminated token
 * @param[in] fp Input stream
 * @param[in] buf_size Size of the buffer
 * @return Length of the token string
 * @note Implementation is incomplete, DOES NOT conform to the JSON specification
 */
static size_t get_json_token(char *buffer, FILE *fp, const size_t buf_size) {
    size_t token_length = 0;

    bool in_string = false;
    char c;
    while ((c = fgetc(fp)) != EOF) {
        if (in_string) {
            // In string
            if (c == '"') {
                in_string = false;
                buffer[token_length] = '\0';
                break;
            }
        } else {
            // Whitespace
            if ((c == ' ') || (c == '\n') || (c == '\r') || (c == '\t')) {
                if (token_length > 0) {
                    buffer[token_length] = '\0';
                    break;
                }
                continue;
            }

            // Start string
            if (c == '"') {
                in_string = true;
                continue;
            }

            // Separate token
            if ((c == ':') || (c == ',')) {
                if (token_length > 0) {
                    buffer[token_length] = '\0';
                    break;
                }
                continue;
            }
        }

        buffer[token_length] = c;
        token_length++;

        // Reach to a max length
        if (token_length == (buf_size - 1)) {
            buffer[token_length] = '\0';
            break;
        }
    }

    return token_length;
}

void net_load_from_file(Net *net, const char *config_file) {
    if ((net == NULL) || (config_file == NULL)) {
        return;
    }

    FILE *fp = fopen(config_file, "r");
    if (fp == NULL) {
        return;
    }

    int params_index = 0;
    bool in_params = false;
    LayerParams *layer_params_list = NULL;

    char buffer[256]; // Temporal buffer for a token
    size_t size;
    while ((size = get_json_token(buffer, fp, sizeof(buffer))) > 0) {
        if (str_equal(buffer, "size")) {
            get_json_token(buffer, fp, sizeof(buffer));
            size_t size = strtoul(buffer, NULL, 10);
            layer_params_list = malloc(sizeof(LayerParams) * (size + 1));
        } else if (str_equal(buffer, "params")) {
            in_params = true;
        } else if (str_equal(buffer, "type")) {
            get_json_token(buffer, fp, sizeof(buffer));
            layer_params_list[params_index].type = strtoul(buffer, NULL, 10);
        } else if (str_equal(buffer, "batch_size")) {
            get_json_token(buffer, fp, sizeof(buffer));
            layer_params_list[params_index].batch_size =
                strtoul(buffer, NULL, 10);
        } else if (str_equal(buffer, "in")) {
            get_json_token(buffer, fp, sizeof(buffer));
            layer_params_list[params_index].in =
                strtoul(buffer, NULL, 10);
        } else if (str_equal(buffer, "out")) {
            get_json_token(buffer, fp, sizeof(buffer));
            layer_params_list[params_index].out =
                strtoul(buffer, NULL, 10);
        } else if (str_equal(buffer, "}")) {
            if (in_params) {
                in_params = false;
                params_index++;
            }
        }
    }

    // Terminate the parameter list
    layer_params_list[params_index].type = LAYER_TYPE_NONE;

    net_alloc_layers(net, layer_params_list);

    free(layer_params_list);

    fclose(fp);
}
