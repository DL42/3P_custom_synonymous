/*
 * dominance.cuh
 *
 *  Created on: Mar 20, 2016
 *      Author: dlawrie
 */

#ifndef DOMINANCE_CUH_
#define DOMINANCE_CUH_

struct const_dominance
{
	float h;
	const_dominance() : h(0) {}
	const_dominance(float h) : h(h){ }
	__host__ __forceinline__ float operator()(const int population, const int generation) const{
		return h;
	}
};

#endif /* DOMINANCE_CUH_ */
