/*
 * fw_sim_api.h
 *
 *  Created on: Mar 20, 2016
 *      Author: dlawrie
 */

#ifndef FW_SIM_API_H_
#define FW_SIM_API_H_
#include <cuda_runtime.h>

/* ----- mutation models ----- */
struct const_mutation
{
	float mu;
	const_mutation();
	const_mutation(float mu);
	__host__ __forceinline__ float operator()(const int population, const int generation) const;
};
/* ----- end mutation models ----- */

/* ----- selection models ----- */
struct const_selection
{
	float s;
	const_selection();
	const_selection(float s);
	__device__ __forceinline__ float operator()(const int population, const int generation, const float freq) const;
};
/* ----- end selection models ----- */

/* ----- dominance models ----- */
struct const_dominance
{
	float h;
	const_dominance();
	const_dominance(float h);
	__host__ __forceinline__ float operator()(const int population, const int generation) const;
};
/* ----- end of dominance models ----- */

/* ----- demography models ----- */
struct const_demography
{
	int N;
	const_demography();
	const_demography(int N);
	__host__ __device__ __forceinline__ int operator()(const int population, const int generation) const;
};
/* ----- end of demography models ----- */

/* ----- migration models ----- */
struct const_migration
{
	float m;
	int num_pop;
	const_migration();
	const_migration(int n);
	const_migration(float m, int n);
	__device__ __forceinline__ float operator()(const int pop_FROM, const int pop_TO, const int generation) const;
};
/* ----- end of migration models ----- */

/* ----- inbreeding models ----- */
struct const_inbreeding
{
	float F;
	const_inbreeding();
	const_inbreeding(float F);
	__host__ __forceinline__ float operator()(const int population, const int generation) const;
};
/* ----- end of inbreeding models ----- */

/* ----- sampling functions ----- */
struct no_sample
{
	__host__ __forceinline__ int operator()(const int generation) const;
};
/* ----- end of sampling functions ----- */

#include "mutation.cuh"
#include "selection.cuh"
#include "dominance.cuh"
#include "demography.cuh"
#include "migration.cuh"
#include "inbreeding.cuh"
#include "sample.cuh"

#endif /* FW_SIM_API_H_ */
