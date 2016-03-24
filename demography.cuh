/*
 * demography.cuh
 *
 *      Author: David Lawrie
 */

#ifndef DEMOGRAPHY_CUH_
#define DEMOGRAPHY_CUH_

GO_Fish::const_demography::const_demography() : N(0){ }
GO_Fish::const_demography::const_demography(int N) : N(N){ }
__host__ __device__ __forceinline__ int GO_Fish::const_demography::operator()(const int population, const int generation) const{ return N; }

#endif /* DEMOGRAPHY_CUH_ */
