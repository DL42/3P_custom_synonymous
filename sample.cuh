/*
 * sample.cuh
 *
 *      Author: David Lawrie
 *
 *      sample_function(generation) = 0 => no sample; sample_function(generation) = 1 => sample;
 */

#ifndef SAMPLE_CUH_
#define SAMPLE_CUH_

__host__ __forceinline__ bool no_sample::operator()(const int generation) const{ return 0; }

#endif /* SAMPLE_CUH_ */
