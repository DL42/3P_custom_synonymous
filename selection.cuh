/*
 * selection.cuh
 *
 *  Created on: Mar 20, 2016
 *      Author: dlawrie
 */

#ifndef SELECTION_CUH_
#define SELECTION_CUH_

struct const_selection
{
	float s;
	const_selection() : s(0) {}
	const_selection(float s) : s(s){ }
	__device__ __forceinline__ float operator()(const int population, const int generation, const float freq) const{
		return s;
	}
};

#endif /* SELECTION_CUH_ */
