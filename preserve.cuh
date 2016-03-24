/*
 * preserve.cuh
 *
 *      Author: dlawrie
 *
 *      preserve_function(generation) = 0 => don't preserve current mutations; preserve_function(generation) = 1 => compact + preserve
 */

#ifndef PRESERVE_CUH_
#define PRESERVE_CUH_

__host__ __forceinline__ bool GO_Fish::no_preserve::operator()(const int generation) const{ return 0; }

#endif /* PRESERVE_CUH_ */
