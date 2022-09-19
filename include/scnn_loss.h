/**
 * @file scnn_loss.h
 * @brief Loss function
 * 
 */
#ifndef SCNN_LOSS_H
#define SCNN_LOSS_H

/**
 * @brief Mean squared error (MSE)
 * 
 * @param[in]   y       Pointer to predicted values 
 * @param[in]   t       Pointer to correct labels 
 * @param[in]   size    Size of vector
 * @return              Mean squared error
 */
float scnn_mean_squared_error(const float *y, const float *t, const int size);

/**
 * @brief Cross entropy loss
 * 
 * @param[in]   y       Pointer to predicted values 
 * @param[in]   t       Pointer to correct labels 
 * @param[in]   size    Size of vector
 * @return              Cross entropy loss
 */
float scnn_cross_entropy_loss(const float *y, const float *t, const int size);

#endif // SCNN_LOSS_H
