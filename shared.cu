/*
 * shared.cu
 *
 *      Author: David Lawrie
 *      for cuda and rand functions used by both go_fish and by sfs
 */

#include "shared.cuh"

__device__ int RNG::ApproxRandBinomHelper(unsigned int i, float mean, float var, float N){
	if(mean <= MEAN_BOUNDARY){ return poiscdfinv(uint_float_01(i), mean); }
	else if(mean >= N-MEAN_BOUNDARY){ return N - poiscdfinv(uint_float_01(i), N-mean); } //flip side of binomial, when 1-p is small
	return round(normcdfinv(uint_float_01(i))*sqrtf(var)+mean);
}
