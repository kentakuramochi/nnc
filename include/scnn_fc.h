/**
 * @file scnn_fc.h
 * @brief Fully connected layer
 * 
 */
#ifndef SCNN_FC_H
#define SCNN_FC_H

#include "scnn_layer.h"

/**
 * @brief Allocate fully connected layer
 * 
 * @param[in] params    Layer parameters
 * @return              Pointer to layer, NULL if failed
 */
scnn_layer *scnn_fc_layer(const scnn_layer_params params);

scnn_dtype *scnn_fc(const scnn_dtype* x, const scnn_dtype *w, const scnn_dtype *b, scnn_dtype *y);

scnn_dtype *scnn_fc_diff(const scnn_dtype* dy, const scnn_dtype *w, const scnn_dtype *b, scnn_dtype *dx, scnn_dtype *dw, scnn_dtype *db);

#endif // SCNN_FC_H
