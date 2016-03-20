/*
 * inbreeding.h
 *
 *  Created on: Mar 20, 2016
 *      Author: dlawrie
 */

#ifndef INBREEDING_H_
#define INBREEDING_H_


struct const_inbreeding
{
	float F;
	const_inbreeding();
	const_inbreeding(float F);
	__host__ __forceinline__ float operator()(const int population, const int generation) const;
};

#include "inbreeding.cuh"

#endif /* INBREEDING_H_ */
