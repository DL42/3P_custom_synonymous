/*
 * mutation.h
 *
 *  Created on: Mar 20, 2016
 *      Author: dlawrie
 */

#ifndef MUTATION_H_
#define MUTATION_H_


struct const_mutation
{
	float mu;
	const_mutation();
	const_mutation(float mu);
	__host__ __forceinline__ float operator()(const int population, const int generation) const;
};

#include "mutation.cuh"

#endif /* MUTATION_H_ */
