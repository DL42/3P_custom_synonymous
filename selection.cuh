/*
 * selection.cuh
 *
 *      Author: David Lawrie
 */

#ifndef SELECTION_CUH_
#define SELECTION_CUH_

const_selection::const_selection() : s(0) {}
const_selection::const_selection(float s) : s(s){ }
__device__ __forceinline__ float const_selection::operator()(const int population, const int generation, const float freq) const{ return s; }

#endif /* SELECTION_CUH_ */
