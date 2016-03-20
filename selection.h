/*
 * selection.h
 *
 *  Created on: Mar 20, 2016
 *      Author: dlawrie
 */

#ifndef SELECTION_H_
#define SELECTION_H_

struct const_selection
{
	float s;
	const_selection();
	const_selection(float s);
	__device__ __forceinline__ float operator()(const int population, const int generation, const float freq) const;
};

#include "selection.cuh"

#endif /* SELECTION_H_ */
