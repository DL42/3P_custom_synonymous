/*
 * mutation.cuh
 *
 *      Author: David Lawrie
 */

#ifndef MUTATION_CUH_
#define MUTATION_CUH_

const_mutation::const_mutation() : mu(0){ }
const_mutation::const_mutation(float mu) : mu(mu){ }
__host__ __forceinline__ float const_mutation::operator()(const int population, const int generation) const{ return mu; }

#endif /* MUTATION_CUH_ */
