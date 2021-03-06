/*
 * rng.cuh
 *
 *      Author: David Lawrie
 *      for RNG functions
 */

#ifndef RNG_CUH_
#define RNG_CUH_

#include <cuda_runtime.h>
#include "../_outside_libraries/helper_math.h"
#include <limits.h>
#include <math.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "../_outside_libraries/Random123/philox.h"
#include "../_outside_libraries/Random123/features/compilerfeatures.h"

/* ----- random number generation ----- */

namespace RNG{
constexpr int RNG_MEAN_BOUNDARY_NORM = 6;
constexpr int RNG_N_BOUNDARY_POIS_BINOM = 100;  //binomial calculation starts to become numerically unstable for large values of N, not sure where that starts but is between 200 and 200,000

// uint_float_01: Input is a W-bit integer (unsigned).  It is multiplied
// by Float(2^-W) and added to Float(2^(-W-1)).  A good compiler should
// optimize it down to an int-to-float conversion followed by a multiply
// and an add, which might be fused, depending on the architecture.
//
// If the input is a uniformly distributed integer, then the
// result is a uniformly distributed floating point number in (0, 1].
// The result is never exactly 0.0.
// The smallest value returned is 2^-W.
// Let M be the number of mantissa bits in Float.
// If W>M  then the largest value retured is 1.0.
// If W<=M then the largest value returned is the largest Float less than 1.0.
// __host__ __device__ __forceinline__ float uint_float_01(unsigned int in){
// 	//(mostly) stolen from Philox code "uniform.hpp"
// 	R123_CONSTEXPR float factor = 1.f/(UINT_MAX + 1.f);
// 	R123_CONSTEXPR float halffactor = 0.5f*factor;
//     return in*factor + halffactor;
// }

// uint_float_01:  Return a "fixed point" number in (0,1).  Let:
//   W = width of Itype, e.g., 32 or 64, regardless of signedness.
//   M = mantissa bits of Ftype, e.g., 24, 53 or 64
//   B = min(M, W)
// Then the 2^(B-1) possible output values are:
//    2^-B*{1, 3, 5, ..., 2^B - 1}
// The smallest output is: 2^-B
// The largest output is:  1 - 2^-B
// The output is never exactly 0.0, nor 0.5, nor 1.0.
// The 2^(B-1) possible outputs:
//   - are equally likely,
//   - are uniformly spaced by 2^-(B-1),
//   - are balanced around 0.5

__host__ __device__ __forceinline__ float uint_float_01(unsigned int in){
	//(mostly) stolen from Philox code "uniform.hpp" bit-shifts 'in' and UINT_MAX by the difference W-M (std::numeric_limits<T>::digits)
	constexpr float factor = 1.f/(1.f + ((UINT_MAX>>8)));
    return (1 | (in>>8)) * factor;
}

__host__ __device__ __forceinline__  uint4 Philox(int2 seed, unsigned int k, unsigned int step, unsigned int population, unsigned int round){
	typedef r123::Philox4x32_R<10> P; //can change the 10 rounds of bijection down to 7 (lowest safe limit) to get possible extra speed!
	P rng;

	unsigned int seedx = seed.x;
	unsigned int seedy = seed.y;
	P::key_type key = {{seedx, seedy}}; //random int to set key space + seed
	P::ctr_type count = {{k, step, population, round}};

	union {
		P::ctr_type c;
		uint4 i;
	}u;

	u.c = rng(count, key);

	return u.i;
}

__host__ __device__ __forceinline__ void binom_iter(float j, float x, float n, float & emu, float & cdf){
	emu *= ((n+1.f-j)*x)/(j*(1-x));
	cdf += emu;
}

__host__ __device__ __forceinline__ int binomcdfinv(float r, float mean, float x, float n){
	float emu = powf(1-x,n);
	if(emu == 1) { emu = expf(-1 * mean);  }
	float cdf = emu;
	if(cdf >= r){ return 0; }

	binom_iter(1.f, x, n, emu, cdf); if(cdf >= r){ return 1; }
	binom_iter(2.f, x, n, emu, cdf); if(cdf >= r){ return 2; }
	binom_iter(3.f, x, n, emu, cdf); if(cdf >= r){ return 3; }
	binom_iter(4.f, x, n, emu, cdf); if(cdf >= r){ return 4; }
	binom_iter(5.f, x, n, emu, cdf); if(cdf >= r){ return 5; }
	binom_iter(6.f, x, n, emu, cdf); if(cdf >= r){ return 6; }
	binom_iter(7.f, x, n, emu, cdf); if(cdf >= r){ return 7; }
	binom_iter(8.f, x, n, emu, cdf); if(cdf >= r){ return 8; }
	binom_iter(9.f, x, n, emu, cdf); if(cdf >= r){ return 9; }
	binom_iter(10.f, x, n, emu, cdf); if(cdf >= r){ return 10; }
	binom_iter(11.f, x, n, emu, cdf); if(cdf >= r || mean <= 1){ return 11; }
	binom_iter(12.f, x, n, emu, cdf); if(cdf >= r){ return 12; }
	binom_iter(13.f, x, n, emu, cdf); if(cdf >= r){ return 13; }
	binom_iter(14.f, x, n, emu, cdf); if(cdf >= r || mean <= 2){ return 14; }
	binom_iter(15.f, x, n, emu, cdf); if(cdf >= r){ return 15; }
	binom_iter(16.f, x, n, emu, cdf); if(cdf >= r){ return 16; }
	binom_iter(17.f, x, n, emu, cdf); if(cdf >= r || mean <= 3){ return 17; }
	binom_iter(18.f, x, n, emu, cdf); if(cdf >= r){ return 18; }
	binom_iter(19.f, x, n, emu, cdf); if(cdf >= r){ return 19; }
	binom_iter(20.f, x, n, emu, cdf); if(cdf >= r || mean <= 4){ return 20; }
	binom_iter(21.f, x, n, emu, cdf); if(cdf >= r){ return 21; }
	binom_iter(22.f, x, n, emu, cdf); if(cdf >= r || mean <= 5){ return 22; }
	binom_iter(23.f, x, n, emu, cdf); if(cdf >= r){ return 23; }
	binom_iter(24.f, x, n, emu, cdf); if(cdf >= r || mean <= 6){ return 24; }
	binom_iter(25.f, x, n, emu, cdf); if(cdf >= r){ return 25; }
	binom_iter(26.f, x, n, emu, cdf); if(cdf >= r || mean <= 7){ return 26; }
	binom_iter(27.f, x, n, emu, cdf); if(cdf >= r){ return 27; }
	binom_iter(28.f, x, n, emu, cdf); if(cdf >= r || mean <= 8){ return 28; }
	binom_iter(29.f, x, n, emu, cdf); if(cdf >= r){ return 29; }
	binom_iter(30.f, x, n, emu, cdf); if(cdf >= r || mean <= 9){ return 30; }
	binom_iter(31.f, x, n, emu, cdf); if(cdf >= r){ return 31; }
	binom_iter(32.f, x, n, emu, cdf); if(cdf >= r || mean <= 10){ return 32; }
	binom_iter(33.f, x, n, emu, cdf); if(cdf >= r){ return 33; }
	binom_iter(34.f, x, n, emu, cdf); if(cdf >= r || mean <= 11){ return 34; }
	binom_iter(35.f, x, n, emu, cdf); if(cdf >= r){ return 35; }
	binom_iter(36.f, x, n, emu, cdf); if(cdf >= r || mean <= 12){ return 36; }
	binom_iter(37.f, x, n, emu, cdf); if(cdf >= r){ return 37; }
	binom_iter(38.f, x, n, emu, cdf); if(cdf >= r || mean <= 13){ return 38; }
	binom_iter(39.f, x, n, emu, cdf); if(cdf >= r){ return 39; }
	binom_iter(40.f, x, n, emu, cdf); if(cdf >= r || mean <= 14){ return 40; }
	binom_iter(41.f, x, n, emu, cdf); if(cdf >= r || mean <= 15){ return 41; }
	binom_iter(42.f, x, n, emu, cdf); if(cdf >= r){ return 42; }
	binom_iter(43.f, x, n, emu, cdf); if(cdf >= r || mean <= 16){ return 43; }
	binom_iter(44.f, x, n, emu, cdf); if(cdf >= r){ return 44; }
	binom_iter(45.f, x, n, emu, cdf); if(cdf >= r || mean <= 17){ return 45; }
	binom_iter(46.f, x, n, emu, cdf); if(cdf >= r || mean <= 18){ return 46; }
	binom_iter(47.f, x, n, emu, cdf); if(cdf >= r){ return 47; }
	binom_iter(48.f, x, n, emu, cdf); if(cdf >= r || mean <= 19){ return 48; }
	binom_iter(49.f, x, n, emu, cdf); if(cdf >= r){ return 49; }
	binom_iter(50.f, x, n, emu, cdf); if(cdf >= r || mean <= 20){ return 50; }
	binom_iter(51.f, x, n, emu, cdf); if(cdf >= r || mean <= 21){ return 51; }
	binom_iter(52.f, x, n, emu, cdf); if(cdf >= r){ return 52; }
	binom_iter(53.f, x, n, emu, cdf); if(cdf >= r || mean <= 22){ return 53; }
	binom_iter(54.f, x, n, emu, cdf); if(cdf >= r){ return 54; }
	binom_iter(55.f, x, n, emu, cdf); if(cdf >= r || mean <= 23){ return 55; }
	binom_iter(56.f, x, n, emu, cdf); if(cdf >= r || mean <= 24){ return 56; }
	binom_iter(57.f, x, n, emu, cdf); if(cdf >= r){ return 57; }
	binom_iter(58.f, x, n, emu, cdf); if(cdf >= r || mean <= 25){ return 58; }
	binom_iter(59.f, x, n, emu, cdf); if(cdf >= r || mean <= 26){ return 59; }
	binom_iter(60.f, x, n, emu, cdf); if(cdf >= r){ return 60; }
	binom_iter(61.f, x, n, emu, cdf); if(cdf >= r || mean <= 27){ return 61; }
	binom_iter(62.f, x, n, emu, cdf); if(cdf >= r || mean <= 28){ return 62; }
	binom_iter(63.f, x, n, emu, cdf); if(cdf >= r){ return 63; }
	binom_iter(64.f, x, n, emu, cdf); if(cdf >= r || mean <= 29){ return 64; }
	binom_iter(65.f, x, n, emu, cdf); if(cdf >= r || mean <= 30){ return 65; }
	binom_iter(66.f, x, n, emu, cdf); if(cdf >= r){ return 66; }
	binom_iter(67.f, x, n, emu, cdf); if(cdf >= r || mean <= 31){ return 67; }
	binom_iter(68.f, x, n, emu, cdf); if(cdf >= r || mean <= 32){ return 68; }
	binom_iter(69.f, x, n, emu, cdf); if(cdf >= r){ return 69; }

	return 70; //17 for mean <= 3, 24 limit for mean <= 6, 32 limit for mean <= 10, 36 limit for mean <= 12, 41 limit for mean <= 15, 58 limit for mean <= 25, 70 limit for mean <= 33; max float between 0 and 1 is 0.99999999
}

__host__ __device__ __forceinline__ void pois_iter(float j, float mean, float & emu, float & cdf){
	emu *= mean*j;
	cdf += emu;
}

__host__ __device__ __forceinline__ int poiscdfinv(float r, float mean){
	float emu = expf(-1 * mean);
	float cdf = emu;
	if(cdf >= r){ return 0; }

	pois_iter(1.f, mean, emu, cdf); if(cdf >= r){ return 1; }
	pois_iter(1.f/2.f, mean, emu, cdf); if(cdf >= r){ return 2; }
	pois_iter(1.f/3.f, mean, emu, cdf); if(cdf >= r){ return 3; }
	pois_iter(1.f/4.f, mean, emu, cdf); if(cdf >= r){ return 4; }
	pois_iter(1.f/5.f, mean, emu, cdf); if(cdf >= r){ return 5; }
	pois_iter(1.f/6.f, mean, emu, cdf); if(cdf >= r){ return 6; }
	pois_iter(1.f/7.f, mean, emu, cdf); if(cdf >= r){ return 7; }
	pois_iter(1.f/8.f, mean, emu, cdf); if(cdf >= r){ return 8; }
	pois_iter(1.f/9.f, mean, emu, cdf); if(cdf >= r){ return 9; }
	pois_iter(1.f/10.f, mean, emu, cdf); if(cdf >= r){ return 10; }
	pois_iter(1.f/11.f, mean, emu, cdf); if(cdf >= r || mean <= 1){ return 11; }
	pois_iter(1.f/12.f, mean, emu, cdf); if(cdf >= r){ return 12; }
	pois_iter(1.f/13.f, mean, emu, cdf); if(cdf >= r){ return 13; }
	pois_iter(1.f/14.f, mean, emu, cdf); if(cdf >= r || mean <= 2){ return 14; }
	pois_iter(1.f/15.f, mean, emu, cdf); if(cdf >= r){ return 15; }
	pois_iter(1.f/16.f, mean, emu, cdf); if(cdf >= r){ return 16; }
	pois_iter(1.f/17.f, mean, emu, cdf); if(cdf >= r || mean <= 3){ return 17; }
	pois_iter(1.f/18.f, mean, emu, cdf); if(cdf >= r){ return 18; }
	pois_iter(1.f/19.f, mean, emu, cdf); if(cdf >= r){ return 19; }
	pois_iter(1.f/20.f, mean, emu, cdf); if(cdf >= r || mean <= 4){ return 20; }
	pois_iter(1.f/21.f, mean, emu, cdf); if(cdf >= r){ return 21; }
	pois_iter(1.f/22.f, mean, emu, cdf); if(cdf >= r || mean <= 5){ return 22; }
	pois_iter(1.f/23.f, mean, emu, cdf); if(cdf >= r){ return 23; }
	pois_iter(1.f/24.f, mean, emu, cdf); if(cdf >= r || mean <= 6){ return 24; }
	pois_iter(1.f/25.f, mean, emu, cdf); if(cdf >= r){ return 25; }
	pois_iter(1.f/26.f, mean, emu, cdf); if(cdf >= r || mean <= 7){ return 26; }
	pois_iter(1.f/27.f, mean, emu, cdf); if(cdf >= r){ return 27; }
	pois_iter(1.f/28.f, mean, emu, cdf); if(cdf >= r || mean <= 8){ return 28; }
	pois_iter(1.f/29.f, mean, emu, cdf); if(cdf >= r){ return 29; }
	pois_iter(1.f/30.f, mean, emu, cdf); if(cdf >= r || mean <= 9){ return 30; }
	pois_iter(1.f/31.f, mean, emu, cdf); if(cdf >= r){ return 31; }
	pois_iter(1.f/32.f, mean, emu, cdf); if(cdf >= r || mean <= 10){ return 32; }
	pois_iter(1.f/33.f, mean, emu, cdf); if(cdf >= r){ return 33; }
	pois_iter(1.f/34.f, mean, emu, cdf); if(cdf >= r || mean <= 11){ return 34; }
	pois_iter(1.f/35.f, mean, emu, cdf); if(cdf >= r){ return 35; }
	pois_iter(1.f/36.f, mean, emu, cdf); if(cdf >= r || mean <= 12){ return 36; }
	pois_iter(1.f/37.f, mean, emu, cdf); if(cdf >= r){ return 37; }
	pois_iter(1.f/38.f, mean, emu, cdf); if(cdf >= r || mean <= 13){ return 38; }
	pois_iter(1.f/39.f, mean, emu, cdf); if(cdf >= r){ return 39; }
	pois_iter(1.f/40.f, mean, emu, cdf); if(cdf >= r || mean <= 14){ return 40; }
	pois_iter(1.f/41.f, mean, emu, cdf); if(cdf >= r || mean <= 15){ return 41; }
	pois_iter(1.f/42.f, mean, emu, cdf); if(cdf >= r){ return 42; }
	pois_iter(1.f/43.f, mean, emu, cdf); if(cdf >= r || mean <= 16){ return 43; }
	pois_iter(1.f/44.f, mean, emu, cdf); if(cdf >= r){ return 44; }
	pois_iter(1.f/45.f, mean, emu, cdf); if(cdf >= r || mean <= 17){ return 45; }
	pois_iter(1.f/46.f, mean, emu, cdf); if(cdf >= r || mean <= 18){ return 46; }
	pois_iter(1.f/47.f, mean, emu, cdf); if(cdf >= r){ return 47; }
	pois_iter(1.f/48.f, mean, emu, cdf); if(cdf >= r || mean <= 19){ return 48; }
	pois_iter(1.f/49.f, mean, emu, cdf); if(cdf >= r){ return 49; }
	pois_iter(1.f/50.f, mean, emu, cdf); if(cdf >= r || mean <= 20){ return 50; }
	pois_iter(1.f/51.f, mean, emu, cdf); if(cdf >= r || mean <= 21){ return 51; }
	pois_iter(1.f/52.f, mean, emu, cdf); if(cdf >= r){ return 52; }
	pois_iter(1.f/53.f, mean, emu, cdf); if(cdf >= r || mean <= 22){ return 53; }
	pois_iter(1.f/54.f, mean, emu, cdf); if(cdf >= r){ return 54; }
	pois_iter(1.f/55.f, mean, emu, cdf); if(cdf >= r || mean <= 23){ return 55; }
	pois_iter(1.f/56.f, mean, emu, cdf); if(cdf >= r || mean <= 24){ return 56; }
	pois_iter(1.f/57.f, mean, emu, cdf); if(cdf >= r){ return 57; }
	pois_iter(1.f/58.f, mean, emu, cdf); if(cdf >= r || mean <= 25){ return 58; }
	pois_iter(1.f/59.f, mean, emu, cdf); if(cdf >= r || mean <= 26){ return 59; }
	pois_iter(1.f/60.f, mean, emu, cdf); if(cdf >= r){ return 60; }
	pois_iter(1.f/61.f, mean, emu, cdf); if(cdf >= r || mean <= 27){ return 61; }
	pois_iter(1.f/62.f, mean, emu, cdf); if(cdf >= r || mean <= 28){ return 62; }
	pois_iter(1.f/63.f, mean, emu, cdf); if(cdf >= r){ return 63; }
	pois_iter(1.f/64.f, mean, emu, cdf); if(cdf >= r || mean <= 29){ return 64; }
	pois_iter(1.f/65.f, mean, emu, cdf); if(cdf >= r || mean <= 30){ return 65; }
	pois_iter(1.f/66.f, mean, emu, cdf); if(cdf >= r){ return 66; }
	pois_iter(1.f/67.f, mean, emu, cdf); if(cdf >= r || mean <= 31){ return 67; }
	pois_iter(1.f/68.f, mean, emu, cdf); if(cdf >= r || mean <= 32){ return 68; }
	pois_iter(1.f/69.f, mean, emu, cdf); if(cdf >= r){ return 69; }

	return 70; //17 for mean <= 3, 24 limit for mean <= 6, 32 limit for mean <= 10, 36 limit for mean <= 12, 41 limit for mean <= 15, 58 limit for mean <= 25, 70 limit for mean <= 33; max float between 0 and 1 is 0.99999999
}

__host__ __device__ __forceinline__ int ApproxRandPois1(float mean, float var, float p, float N, int2 seed, int id, int generation, int population){
	uint4 i = Philox(seed, id, generation, population, 0);
	if(mean <= RNG_MEAN_BOUNDARY_NORM){ return poiscdfinv(uint_float_01(i.x), mean); }
	else if(mean >= N-RNG_MEAN_BOUNDARY_NORM){ return N - poiscdfinv(uint_float_01(i.x), N-mean); } //flip side of poisson, when 1-p is small
	return round(normcdfinv(uint_float_01(i.x))*sqrtf(var)+mean);
}

__host__ __device__ __forceinline__ int ApproxRandBinom1(float mean, float var, float p, float N, int2 seed, int id, int generation, int population){
	uint4 i = Philox(seed, id, generation, population, 0);
	if(mean <= RNG_MEAN_BOUNDARY_NORM){
		if(N < RNG_N_BOUNDARY_POIS_BINOM){ return binomcdfinv(uint_float_01(i.x), mean, mean/N, N); } else{ return poiscdfinv(uint_float_01(i.x), mean); }
	}
	else if(mean >= N-RNG_MEAN_BOUNDARY_NORM){ //flip side of binomial, when 1-p is small
		if(N < RNG_N_BOUNDARY_POIS_BINOM){ return N - binomcdfinv(uint_float_01(i.x), N-mean, (N-mean)/N, N); } else{ return N - poiscdfinv(uint_float_01(i.x), N-mean); }
	}
	return round(normcdfinv(uint_float_01(i.x))*sqrtf(var)+mean);
}

//faster if don't keep as separate compilation on both 780 and 980 GPUs! using static keyword to allow for fast compilation, works on 980, with whole program compilation works on 780 and even better on 980
 __device__ __noinline__ static int ApproxRandBinomHelper(unsigned int i, float mean, float var, float N){
	if(mean <= RNG_MEAN_BOUNDARY_NORM){
		if(N < RNG_N_BOUNDARY_POIS_BINOM){ return binomcdfinv(uint_float_01(i), mean, mean/N, N); } else{ return poiscdfinv(uint_float_01(i), mean); }
	}
	else if(mean >= N-RNG_MEAN_BOUNDARY_NORM){ //flip side of binomial, when 1-p is small
		if(N < RNG_N_BOUNDARY_POIS_BINOM){ return N - binomcdfinv(uint_float_01(i), N-mean, (N-mean)/N, N); } else{ return N - poiscdfinv(uint_float_01(i), N-mean); }
	}
	return round(normcdfinv(uint_float_01(i))*sqrtf(var)+mean);
}

__device__ __forceinline__ int4 ApproxRandBinom4(float4 mean, float4 var, float4 p, float N, int2 seed, int id, int generation, int population){
	uint4 i = Philox(seed, id, generation, population, 0);
	return make_int4(ApproxRandBinomHelper(i.x, mean.x, var.x, N), ApproxRandBinomHelper(i.y, mean.y, var.y, N), ApproxRandBinomHelper(i.z, mean.z, var.z, N), ApproxRandBinomHelper(i.w, mean.w, var.w, N));
}
/* ----- end random number generation ----- */

} /* ----- end namespace RNG ----- */

#endif /* SHARED_CUH_ */
