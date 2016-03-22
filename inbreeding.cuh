/*
 * inbreeding.cuh
 *
 *      Author: David Lawrie
 */

#ifndef INBREEDING_CUH_
#define INBREEDING_CUH_

const_inbreeding::const_inbreeding() : F(0){ }
const_inbreeding::const_inbreeding(float F) : F(F){ }
__host__ __forceinline__ float const_inbreeding::operator()(const int population, const int generation) const{ return F; }

#endif /* INBREEDING_CUH_ */
