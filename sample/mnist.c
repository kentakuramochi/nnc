#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "layers.h"
#include "losses.h"
#include "trainer.h"
#include "util.h"

// The number of data
#define TRAIN_DATA_NUM 60000
#define TEST_DATA_NUM 10000

// Size (width/height) of a image
#define IMAGE_SIZE 28

// The number of classes
#define CLASS_NUM 10

// Load MNIST label data: "***-labels-idx1-ubyte"
static bool load_mnist_labels(
    const char *filename, float **labels, const int num
) {
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL) {
        fprintf(stderr, "Error: failed to open file: %s\n", filename);
        return false;
    }

    size_t n_read;
    (void)n_read;

    // Check a magic number
    int32_t magic_number = 0;
    for (int i = 0; i < 4; i++) {
        uint8_t byte;
        n_read = fread(&byte, sizeof(uint8_t), 1, fp);
        magic_number = (magic_number << 8) | byte;
    }

    if (magic_number != 2049) {
        fprintf(
            stderr, "Error: magic number %d mismatch to 2049d\n", magic_number
        );
        fclose(fp);
        return false;
    }

    // Check the number of data
    int32_t num_data = 0;
    for (int i = 0; i < 4; i++) {
        uint8_t byte;
        n_read = fread(&byte, sizeof(uint8_t), 1, fp);
        num_data = (num_data << 8) | byte;
    }

    if (num_data != num) {
        fprintf(
            stderr,
            "Error: the number of items %d mismatch to %d\n", num_data, num
        );
        fclose(fp);
        return false;
    }

    // Get labels as one-hot vectors
    for (int i = 0; i < num_data; i++) {
        uint8_t label;
        n_read = fread(&label, sizeof(uint8_t), 1, fp);
        // Convert to a one-hot vector
        for (int j = 0; j < CLASS_NUM; j++) {
            labels[i][j] = (j == label) ? 1.0f : 0.0f;
        }
    }

    return true;
}

// Load MNIST image data: "***-images-idx3-ubyte"
static bool load_mnist_images(
    const char *filename, float **images, const int num
) {
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL) {
        fprintf(stderr, "Error: failed to open file: %s\n", filename);
        return false;
    }

    size_t n_read;
    (void)n_read;

    // Check a magic number
    int32_t magic_number = 0;
    for (int i = 0; i < 4; i++) {
        uint8_t byte;
        n_read = fread(&byte, sizeof(uint8_t), 1, fp);
        magic_number = (magic_number << 8) | byte;
    }

    if (magic_number != 2051) {
        fprintf(
            stderr, "Error: magic number %d mismatch to 2051\n", magic_number
        );
        fclose(fp);
        return false;
    }

    // Check the number of data
    int32_t num_data = 0;
    for (int i = 0; i < 4; i++) {
        uint8_t byte;
        n_read = fread(&byte, sizeof(uint8_t), 1, fp);
        num_data = (num_data << 8) | byte;
    }

    if (num_data != num) {
        fprintf(
            stderr,
            "Error: the number of items %d mismatch to %d\n", num_data, num
        );
        fclose(fp);
        return false;
    }

    // Get the number of rows/cols
    int32_t num_rows = 0;
    for (int i = 0; i < 4; i++) {
        uint8_t byte;
        n_read = fread(&byte, sizeof(uint8_t), 1, fp);
        num_rows = (num_rows << 8) | byte;
    }

    if (num_rows != IMAGE_SIZE) {
        fprintf(
            stderr,
            "Error: number of rows %d mismatch to %d\n", num_rows, IMAGE_SIZE
        );
        fclose(fp);
        return false;
    }

    int32_t num_cols = 0;
    for (int i = 0; i < 4; i++) {
        uint8_t byte;
        n_read = fread(&byte, sizeof(uint8_t), 1, fp);
        num_cols = (num_cols << 8) | byte;
    }

    if (num_cols != IMAGE_SIZE) {
        fprintf(
            stderr,
            "Error: number of columns %d mismatch to %d\n", num_cols, IMAGE_SIZE
        );
        fclose(fp);
        return false;
    }

    // Get pixels as vectors
    for (int i = 0; i < num_data; i++) {
        for (int j = 0; j < (IMAGE_SIZE * IMAGE_SIZE); j++) {
            uint8_t pixel;
            n_read = fread(&pixel, sizeof(uint8_t), 1, fp);
            // Normalize values to [0,1]
            images[i][j] = (float)pixel / 255;
        }
    }

    return true;
}

int main(int argc, char *argv[]) {
    // Check arguments: path to the MNIST data files
    if (argc < 5) {
        fprintf(stderr, "Error: arguments are required\n");
        fprintf(stderr, "    ./mnist PATH_TO_TRAIN-LABELS PATH_TO_TRAIN-IMAGES ");
        fprintf(stderr, "PATH_TO_T10K-LABELS PATH_TO_T10K-IMAGES\n");
        return EXIT_FAILURE;
    }

    // Allocate memories for the dataset
    float **train_labels = NULL;
    float **train_images = NULL;
    float **test_labels = NULL;
    float **test_images = NULL;

    train_labels = alloc_dataset(TRAIN_DATA_NUM, CLASS_NUM);
    if (train_labels == NULL) {
        fprintf(
            stderr, "Error: failed to allocate memory for a training dataset\n"
        );
        return EXIT_FAILURE;
    }

    train_images = alloc_dataset(TRAIN_DATA_NUM, (IMAGE_SIZE * IMAGE_SIZE));
    if (train_images == NULL) {
        fprintf(
            stderr, "Error: failed to allocate memory for a training dataset\n"
        );
        free_dataset(&train_labels, TRAIN_DATA_NUM);
        return EXIT_FAILURE;
    }

    test_labels = alloc_dataset(TEST_DATA_NUM, CLASS_NUM);
    if (test_labels == NULL) {
        fprintf(
            stderr, "Error: failed to allocate memory for a test dataset\n"
        );
        free_dataset(&train_labels, TRAIN_DATA_NUM);
        free_dataset(&train_images, TRAIN_DATA_NUM);
        return EXIT_FAILURE;
    }

    test_images = alloc_dataset(TEST_DATA_NUM, (IMAGE_SIZE * IMAGE_SIZE));
    if (test_images == NULL) {
        fprintf(
            stderr, "Error: failed to allocate memory for a test dataset\n"
        );
        free_dataset(&train_labels, TRAIN_DATA_NUM);
        free_dataset(&train_images, TRAIN_DATA_NUM);
        free_dataset(&test_labels, TEST_DATA_NUM);
        return EXIT_FAILURE;
    }

    // Load the dataset
    if (!load_mnist_labels(argv[1], train_labels, TRAIN_DATA_NUM)) {
        fprintf(stderr, "Error: failed to load a training dataset\n");
        free_dataset(&train_labels, TRAIN_DATA_NUM);
        free_dataset(&train_images, TRAIN_DATA_NUM);
        free_dataset(&test_labels, TEST_DATA_NUM);
        free_dataset(&test_images, TEST_DATA_NUM);
        return EXIT_FAILURE;
    }
    if (!load_mnist_images(argv[2], train_images, TRAIN_DATA_NUM)) {
        fprintf(stderr, "Error: failed to load a training dataset\n");
        free_dataset(&train_labels, TRAIN_DATA_NUM);
        free_dataset(&train_images, TRAIN_DATA_NUM);
        free_dataset(&test_labels, TEST_DATA_NUM);
        free_dataset(&test_images, TEST_DATA_NUM);
        return EXIT_FAILURE;
    }
    if (!load_mnist_labels(argv[3], test_labels, TEST_DATA_NUM)) {
        fprintf(stderr, "Error: failed to load a test dataset\n");
        free_dataset(&train_labels, TRAIN_DATA_NUM);
        free_dataset(&train_images, TRAIN_DATA_NUM);
        free_dataset(&test_labels, TEST_DATA_NUM);
        free_dataset(&test_images, TEST_DATA_NUM);
        return EXIT_FAILURE;
    }
    if (!load_mnist_images(argv[4], test_images, TEST_DATA_NUM)) {
        fprintf(stderr, "Error: failed to load a test dataset\n");
        free_dataset(&train_labels, TRAIN_DATA_NUM);
        free_dataset(&train_images, TRAIN_DATA_NUM);
        free_dataset(&test_labels, TEST_DATA_NUM);
        free_dataset(&test_images, TEST_DATA_NUM);
        return EXIT_FAILURE;
    }

    // Create a network
    Net net;
    net_alloc_layers(
        &net,
        LAYER_PARAMS_LIST(
            {
                .type=LAYER_TYPE_FC,
                .batch_size=1, .in=IMAGE_SIZE*IMAGE_SIZE, .out=100
            },
            { .type=LAYER_TYPE_SIGMOID },
            { .type=LAYER_TYPE_FC, .out=10 },
            { .type=LAYER_TYPE_SOFTMAX }
        )
    );

    net_init_params(&net);

    printf("====================\n");
    printf("Start training\n");
    printf("====================\n");
    const float lr = 0.01;
    const int epochs = 5;
    LossFunc loss_func = ce_loss();

    // Train the network with simple SGD
    for (int i = 0; i < epochs; i++) {
        float total_loss = 0;
        int corrects = 0;
        float grad[CLASS_NUM];

        printf("Epoch %d\n", (i + 1));
        for (int j = 0; j < TRAIN_DATA_NUM; j++) {
            net_clear_grad(&net);

            float *y = net_forward(&net, train_images[j]);
            float loss = loss_func.forward(y, train_labels[j], 1, CLASS_NUM);

            loss_func.backward(grad, y, train_labels[j], 1, CLASS_NUM);
            net_backward(&net, grad);

            train_step(&net, lr);

            if (argmax(y, CLASS_NUM) == argmax(train_labels[j], CLASS_NUM)) {
                corrects++;
            }

            total_loss += loss;

            int data_num = j + 1;
            // Display metrics for the training data
            if (data_num % 6000 == 0) {
                printf(
                    "(%d/%d) train loss=%f, train accuracy=%f\n",
                    data_num, TRAIN_DATA_NUM,
                    (total_loss / data_num), ((float)corrects / data_num)
                );
            }
        }

        // Display metrics for the test data
        total_loss = 0;
        corrects = 0;
        for (int j = 0; j < TEST_DATA_NUM; j++) {
            float *y = net_forward(&net, test_images[j]);
            float loss = loss_func.forward(y, test_labels[j], 1, CLASS_NUM);

            if (argmax(y, CLASS_NUM) == argmax(test_labels[j], CLASS_NUM)) {
                corrects++;
            }

            total_loss += loss;
        }

        printf(
            "test loss=%f, test accuracy=%f\n",
            (total_loss / TEST_DATA_NUM), ((float)corrects / TEST_DATA_NUM)
        );
        printf("====================\n");
    }

    printf("Finished\n");

    net_free_layers(&net);

    free_dataset(&train_labels, TRAIN_DATA_NUM);
    free_dataset(&train_images, TRAIN_DATA_NUM);
    free_dataset(&test_labels, TEST_DATA_NUM);
    free_dataset(&test_images, TEST_DATA_NUM);

    return EXIT_SUCCESS;
}
