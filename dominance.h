/*
 * dominance.h
 *
 *  Created on: Mar 20, 2016
 *      Author: dlawrie
 */

#ifndef DOMINANCE_H_
#define DOMINANCE_H_

struct const_dominance
{
	float h;
	const_dominance();
	const_dominance(float h);
	__host__ __forceinline__ float operator()(const int population, const int generation) const;
};

#include "dominance.cuh"

#endif /* DOMINANCE_H_ */
