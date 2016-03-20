/*
 * demography.h
 *
 *  Created on: Mar 20, 2016
 *      Author: dlawrie
 */

#ifndef DEMOGRAPHY_H_
#define DEMOGRAPHY_H_

struct const_demography
{
	int N;
	const_demography();
	const_demography(int N);
	__host__ __device__ __forceinline__ int operator()(const int population, const int generation) const;
};

#include "demography.cuh"

#endif /* DEMOGRAPHY_CUH_ */
