/*
 * dominance.cuh
 *
 *      Author: David Lawrie
 */

#ifndef DOMINANCE_CUH_
#define DOMINANCE_CUH_

GO_Fish::const_dominance::const_dominance() : h(0) {}
GO_Fish::const_dominance::const_dominance(float h) : h(h){ }
__host__ __forceinline__ float GO_Fish::const_dominance::operator()(const int population, const int generation) const{ return h; }

#endif /* DOMINANCE_CUH_ */
