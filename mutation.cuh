/*
 * mutation.cuh
 *
 *  Created on: Mar 20, 2016
 *      Author: dlawrie
 */

#ifndef MUTATION_CUH_
#define MUTATION_CUH_


struct const_mutation
{
	float mu;
	const_mutation() : mu(0){ }
	const_mutation(float mu) : mu(mu){ }
	__host__ __forceinline__ float operator()(const int population, const int generation) const{
		return mu;
	}
};

#endif /* MUTATION_CUH_ */
