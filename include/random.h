/**
 * @file random.h
 * @brief Simple PRNG
 */
#ifndef RANDOM_H
#define RANDOM_H

/**
 * @brief Get a random value from a uniform distribution
 * 
 * @return float Random value within [0, 1)
 * @note Need to initialize by srand in stdlib.h beforehand
 */
float rand_uniform(void);

/**
 * @brief Get a random value from a normal distribution
 * 
 * @param[in] mean Mean of the distribution
 * @param[in] stddev Std. dev. for the distribution
 * @return float Random value
 * @note Need to initialize by srand in stdlib.h beforehand
 */
float rand_norm(const float mean, const float stddev);

#endif // RANDOM_H
