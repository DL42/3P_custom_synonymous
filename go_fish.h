/*
 * go_fish.h
 *
 *      Author: David Lawrie
 */

#ifndef FW_SIM_API_H_
#define FW_SIM_API_H_
#include <cuda_runtime.h>
#include "shared.cuh"

namespace GO_Fish{

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

struct linear_frequency_dependent_selection
{
	float slope;
	float intercept;
	linear_frequency_dependent_selection();
	linear_frequency_dependent_selection(float slope, float intercept);
	__device__ __forceinline__ float operator()(const int population, const int generation, const float freq) const;
};

//models selection as a sine wave through time
struct seasonal_selection
{
	float amplitude;
	float frequency;
	float phase;
	float offset;
	int generation_shift;

	seasonal_selection();
	seasonal_selection(float frequency, float amplitude, float offset, float phase = 0, int generation_shift = 0);
	__device__ __forceinline__ float operator()(const int population, const int generation, const float freq) const;
};

//one population, pop, has a different, selection functor, s_pop
template <typename Functor_sel, typename Functor_sel_pop>
struct population_specific_selection
{
	int pop, generation_shift;
	Functor_sel s;
	Functor_sel_pop s_pop;
	population_specific_selection();
	population_specific_selection(Functor_sel s_in, Functor_sel_pop s_pop_in, int pop, int generation_shift = 0);
	__device__ __forceinline__ float operator()(const int population, const int generation, const float freq) const;
};

//selection function changes at inflection_point
template <typename Functor_sel1, typename Functor_sel2>
struct piecewise_selection
{
	int inflection_point, generation_shift;
	Functor_sel1 s1;
	Functor_sel2 s2;
	piecewise_selection();
	piecewise_selection(Functor_sel1 s1_in, Functor_sel2 s2_in, int inflection_point, int generation_shift = 0);
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

//one population, pop, has a different, dominance functor, h_pop
template <typename Functor_h, typename Functor_h_pop>
struct population_specific_dominance
{
	int pop, generation_shift;
	Functor_h h;
	Functor_h_pop h_pop;
	population_specific_dominance();
	population_specific_dominance(Functor_h h_in, Functor_h_pop h_pop_in, int pop, int generation_shift = 0);
	__device__ __forceinline__ float operator()(const int population, const int generation, const float freq) const;
};

//dominance function changes at inflection_point
template <typename Functor_h1, typename Functor_h2>
struct piecewise_dominance
{
	int inflection_point, generation_shift;
	Functor_h1 h1;
	Functor_h2 h2;
	piecewise_dominance();
	piecewise_dominance(Functor_h1 h1, Functor_h2 h2, int inflection_point, int generation_shift = 0);
	__device__ __forceinline__ float operator()(const int population, const int generation, const float freq) const;
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
	__host__ __device__ __forceinline__ float operator()(const int pop_FROM, const int pop_TO, const int generation) const;
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
struct no_preserve{ __host__ __forceinline__ bool operator()(const int generation) const; };
/* ----- end of preserving functions ----- */

/* ----- sampling functions ----- */
struct no_sample{ __host__ __forceinline__ bool operator()(const int generation) const; };
/* ----- end of sampling functions ----- */

/* ----- go_fish_impl  ----- */
template <typename Functor_mutation, typename Functor_demography, typename Functor_migration, typename Functor_selection, typename Functor_inbreeding, typename Functor_dominance, typename Functor_preserve, typename Functor_timesample>
__host__ sim_result * run_sim(const Functor_mutation mu_rate, const Functor_demography demography, const Functor_migration mig_prop, const Functor_selection sel_coeff, const Functor_inbreeding FI, const Functor_dominance dominance, const int num_generations, const float num_sites, const int num_populations, const int seed1, const int seed2, Functor_preserve preserve_mutations, Functor_timesample take_sample, int max_samples = 0, const bool init_mse = true, const sim_result & prev_sim = sim_result(), const int compact_rate = 35, int cuda_device = -1);
/* ----- end go_fish_impl ----- */

} /* ----- end namespace GO_Fish ----- */

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

/* ----- importing go_fish_impl  ----- */
#include "go_fish_impl.cuh"
/* ----- end importing go_fish_impl ----- */


#endif /* GO_FISH_H_ */
