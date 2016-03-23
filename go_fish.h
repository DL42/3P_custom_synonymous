/*
 * go_fish.h
 *
 *      Author: David Lawrie
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

/* ----- preserving functions ----- */
struct no_preserve
{
	__host__ __forceinline__ bool operator()(const int generation) const;
};
/* ----- end of preserving functions ----- */

/* ----- sampling functions ----- */
struct no_sample
{
	__host__ __forceinline__ bool operator()(const int generation) const;
};
/* ----- end of sampling functions ----- */

/* ----- importing functor implementations ----- */
#include "mutation.cuh"
#include "selection.cuh"
#include "dominance.cuh"
#include "demography.cuh"
#include "migration.cuh"
#include "inbreeding.cuh"
#include "preserve.cuh"
#include "sample.cuh"
/* ----- end importing functor implementations ----- */

/* ----- importing run_sim  ----- */
#include "shared.cuh"

template <typename Functor_mutation, typename Functor_demography, typename Functor_migration, typename Functor_selection, typename Functor_inbreeding, typename Functor_dominance, typename Functor_preserve, typename Functor_timesample>
__host__ sim_result * run_sim(const Functor_mutation mu_rate, const Functor_demography demography, const Functor_migration mig_prop, const Functor_selection sel_coeff, const Functor_inbreeding FI, const Functor_dominance dominance, const int num_generations, const float num_sites, const int num_populations, const int seed1, const int seed2, Functor_preserve preserve_mutations, Functor_timesample take_sample, int max_samples = 0, const bool init_mse = true, const sim_result & prev_sim = sim_result(), const int compact_rate = 35, const int cuda_device = -1);

#include "go_fish_impl.cuh"
/* ----- end importing run_sim ----- */

#endif /* FW_SIM_API_H_ */
