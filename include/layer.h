/**
 * @file layer.h
 * @brief basic layer structure and operations
 * 
 */
#ifndef LAYER_H
#define LAYER_H

#define LAYER_NAME_MAX_LENGTH 64    //!< max length of layer name

/**
 * @struct 
 * @brief basic layer structure
 * 
 */
typedef struct Layer {
    char name[LAYER_NAME_MAX_LENGTH + 1];   //!< layer name

    const float *x;   //!< layer input matrix
    int in;     //!< num of layer input
    int in_h;   //!< input height
    int in_w;   //!< input width

    float *y;   //!< layer output matrix
    int out;    //!< num of layer output
    int out_h;  //!< output height
    int out_w;  //!< output width

    float *w;   //!< layer weight
    float *b;   //!< layer bias

    float *dx;  //!< diff of layer input
    float *dw;  //!< diff of layer weight
    float *db;  //!< diff of layer bias

    struct Layer *prev; //!< pointer to previous layer
    struct Layer *next; //!< pointer to next layer

    void (*forward)(struct Layer *self, const float *x);     //!< forward propagation
    void (*backward)(struct Layer *self, const float *dy);   //!< backward propagation
} Layer;

/**
 * @brief layer parameter structure
 * 
 */
typedef struct LayerParameter {
    char name[LAYER_NAME_MAX_LENGTH];   //!< layer name

    int in;     //!< num of layer input
    int in_h;   //!< height of layer output
    int in_w;   //!< width of layer input

    int out;    //!< num of layer output
} LayerParameter;

/**
 * @brief allocate basic layer structure
 * 
 * @param[in] layer_param layer parameter
 * @return Layer* poiner to layer structure
 */
Layer *layer_alloc(const LayerParameter layer_param);

/**
 * @brief deallocate layer structure
 * 
 * @param[in,out] layer layer structure to be deallocated
 */
void layer_free(Layer **layer);

#endif // LAYER_H
