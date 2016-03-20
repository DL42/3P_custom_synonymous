/*
 * sample.cuh
 *
 *  Created on: Mar 20, 2016
 *      Author: dlawrie
 */

#ifndef SAMPLE_CUH_
#define SAMPLE_CUH_

struct no_sample
{
	__host__ __forceinline__ int operator()(const int generation) const{ return 0; }
};


#endif /* SAMPLE_CUH_ */
