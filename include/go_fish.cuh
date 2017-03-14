/*!\file
\brief GO Fish Simulation API
*/
/*
 * 		go_fish.cuh
 *
 *      Author: David Lawrie
 *      GO Fish Simulation API
 */

#ifndef GO_FISH_API_H_
#define GO_FISH_API_H_
#include <cuda_runtime.h>
#include "../outside_libraries/helper_math.h"
#include "../include/go_fish_data_struct.h"

///Functions for controlling GO_Fish simulations
namespace Sim_Model{

/** \defgroup selection Simulation Models: Selection Group*//**@{*/

/* ----- mutation, dominance, & inbreeding models ----- */
///functor: models selection coefficient \p s as a constant across populations and over time
struct selection_constant
{
	float s; /**<\brief selection coefficient */ /**<\t*/
	inline selection_constant(); /**<\brief default constructor */ /**<`s = 0`*/
	inline selection_constant(float s); /**<\brief constructor */ /**<\t*/
	template <typename Functor_demography, typename Functor_inbreeding>
	inline selection_constant(float gamma, Functor_demography demography, Functor_inbreeding F, int forward_generation_shift = 0); /**<\brief constructor: effective selection */
	__device__ __forceinline__ float operator()(const int population, const int generation, const float freq) const; /**<\brief selection operator, returns selection coefficient for a given `population, generation, freq` */ /**<\t*/
};

/**\brief functor: models selection coefficient as linearly dependent on frequency */
struct selection_linear_frequency_dependent
{
	float slope; /**<\brief slope of selection coefficient's linear dependence on frequency */ /**<\t*/
	float intercept; /**<\brief selection coefficient's intercept with frequency 0 */ /**<\t*/
	inline selection_linear_frequency_dependent(); /**<\brief default constructor */ /**<`slope = 0, intercept = 0`*/
	inline selection_linear_frequency_dependent(float slope, float intercept); /**<\brief constructor */ /**<\t*/
	template <typename Functor_demography, typename Functor_inbreeding>
	inline selection_linear_frequency_dependent(float gamma_slope, float gamma_intercept, Functor_demography demography, Functor_inbreeding F, int forward_generation_shift = 0); /**<\brief constructor: effective selection */
	__device__ __forceinline__ float operator()(const int population, const int generation, const float freq) const; //!<\copybrief Sim_Model::selection_constant::operator()(const int population, const int generation, const float freq) const
};

/**\brief functor: models selection as a sine wave through time */ /**useful for modeling cyclical/seasonal behavior over time*/
struct selection_sine_wave
{
	float A; /**<\brief Amplitude of sine wave */ /**<\t*/
	float pi; /**<\brief Frequency of sine wave */ /**<\t*/
	float rho; /**<\brief Phase of sine wave */ /**<\t*/
	float D; /**<\brief Offset of sine wave */ /**<\t*/
	int generation_shift; /**<\brief number of generations to shift function backwards */ /**< useful if you are starting the simulation from a previous simulation state and this function is expecting to start at 0 or any scenario where you want to shift the generation of the function relative to the simulation generation */

	inline selection_sine_wave(); /**<\brief default constructor */ /**<all parameters set to `0`*/
	inline selection_sine_wave(float A, float pi, float D, float rho = 0, int generation_shift = 0); /**<\brief constructor */
	template <typename Functor_demography, typename Functor_inbreeding>
	inline selection_sine_wave(float gamma_A, float pi, float gamma_D, Functor_demography demography, Functor_inbreeding F, float rho = 0, int generation_shift = 0, int forward_generation_shift = 0); /**<\brief constructor: effective selection `gamma_A, gamma_D` */
	__device__ __forceinline__ float operator()(const int population, const int generation, const float freq) const; //!<\copybrief Sim_Model::selection_constant::operator()(const int population, const int generation, const float freq) const
};

///functor: one population, \p pop, has a different, selection function, \p s_pop, all other have function \p s
template <typename Functor_sel, typename Functor_sel_pop>
struct selection_population_specific
{
	int pop; /**<\brief population with specific selection function */ /**<\t*/
	int generation_shift; //!<\copydoc Sim_Model::selection_sine_wave::generation_shift
	Functor_sel s; /**<\brief selection function applied to all other populations */ /**<\t*/
	Functor_sel_pop s_pop; /**<\brief population specific selection function for \p pop */ /**<\t*/
	inline selection_population_specific(); /**<\brief default constructor */
	inline selection_population_specific(Functor_sel s_in, Functor_sel_pop s_pop_in, int pop, int generation_shift = 0); /**<\brief constructor */ /**<\t*/
	__device__ __forceinline__ float operator()(const int population, const int generation, const float freq) const; //!<\copybrief Sim_Model::selection_constant::operator()(const int population, const int generation, const float freq) const
};

///functor: selection function changes from \p s1 to \p s2 at generation \p inflection_point
template <typename Functor_sel1, typename Functor_sel2>
struct selection_piecewise
{
	int inflection_point; /**<\brief generation in which the selection function switches from `s1` to `s2` */ /**<\t*/
	int generation_shift; //!<\copydoc Sim_Model::selection_sine_wave::generation_shift
	Functor_sel1 s1; /**<\brief first selection function */ /**<\t*/
	Functor_sel2 s2; /**<\brief second selection function */ /**<\t*/
	inline selection_piecewise(); /**<\brief default constructor */
	inline selection_piecewise(Functor_sel1 s1_in, Functor_sel2 s2_in, int inflection_point, int generation_shift = 0); /**<\brief constructor */ /**<\t*/
	__device__ __forceinline__ float operator()(const int population, const int generation, const float freq) const; //!<\copybrief Sim_Model::selection_constant::operator()(const int population, const int generation, const float freq) const
};
/* ----- end selection models ----- *//** @} */

/** \defgroup in_mut_dom Simulation Models: Inbreeding, Mutation, and Dominance Group *//**@{*/

/* ----- inbreeding, mutation, & dominance models ----- */
///functor: models parameter \p p as a constant across populations and over time
struct F_mu_h_constant
{
	float p; /**<\brief parameter constant */ /**<\t*/
	inline F_mu_h_constant(); /**<\brief default constructor */ /**<`p = 0`*/
	inline F_mu_h_constant(float p); /**<\brief constructor */ /**<\t*/
	__host__ __forceinline__ float operator()(const int population, const int generation) const; /**<\brief Inbreeding/Mutation/Dominance operator, returns parameter \p p for a given `population, generation` */ /**<\t*/
};

///functor: models parameter as a sine wave through time
struct F_mu_h_sine_wave
{
	float A; //!<\copydoc Sim_Model::selection_sine_wave::A
	float pi; //!<\copydoc Sim_Model::selection_sine_wave::pi
	float rho; //!<\copydoc Sim_Model::selection_sine_wave::rho
	float D; //!<\copydoc Sim_Model::selection_sine_wave::D
	int generation_shift; //!<\copydoc Sim_Model::selection_sine_wave::generation_shift

	inline F_mu_h_sine_wave(); //!<\copydoc Sim_Model::selection_sine_wave::selection_sine_wave()
	inline F_mu_h_sine_wave(float A, float pi, float D, float rho = 0, int generation_shift = 0); //!<\copydoc Sim_Model::selection_sine_wave::selection_sine_wave(float A, float pi, float D, float rho = 0, int generation_shift = 0)
	__host__ __forceinline__ float operator()(const int population, const int generation) const; //!<\copybrief Sim_Model::F_mu_h_constant::operator()(const int population, const int generation) const
};

///functor: one population, \p pop, has a different, parameter function, \p p_pop, all others have function \p p
template <typename Functor_p, typename Functor_p_pop>
struct F_mu_h_population_specific
{
	int pop; /**<\brief population with specific parameter function */ /**<\t*/
	int generation_shift; //!<\copydoc Sim_Model::selection_sine_wave::generation_shift
	Functor_p p; /**<\brief parameter function applied to all other populations */ /**<\t*/
	Functor_p_pop p_pop; /**<\brief population specific parameter function for \p pop */ /**<\t*/
	inline F_mu_h_population_specific(); /**<\brief default constructor */
	inline F_mu_h_population_specific(Functor_p p_in, Functor_p_pop p_pop_in, int pop, int generation_shift = 0); /**<\brief constructor */  /**<\t*/
	__host__ __forceinline__ float operator()(const int population, const int generation) const; //!<\copybrief Sim_Model::F_mu_h_constant::operator()(const int population, const int generation) const
};

///functor: parameter function changes from \p p1 to \p p2 at generation \p inflection_point
template <typename Functor_p1, typename Functor_p2>
struct F_mu_h_piecewise
{
	int inflection_point; /**<\brief generation in which the Inbreeding/Mutation/Dominance function switches from `p1` to `p2` */ /**<\t*/
	int generation_shift; //!<\copydoc Sim_Model::selection_sine_wave::generation_shift
	Functor_p1 p1; /**<\brief first parameter function */ /**<\t*/
	Functor_p2 p2; /**<\brief second parameter function */ /**<\t*/
	inline F_mu_h_piecewise(); /**<\brief default constructor */
	inline F_mu_h_piecewise(Functor_p1 p1_in, Functor_p2 p2_in, int inflection_point, int generation_shift = 0); /**<\brief constructor */
	__host__ __forceinline__ float operator()(const int population, const int generation) const; //!<\copybrief Sim_Model::F_mu_h_constant::operator()(const int population, const int generation) const
};
/* ----- end of inbreeding, mutation, & dominance models ----- */ /** @} */

/** \defgroup demography Simulation Models: Demography Group *//**@{*/

/* ----- demography models ----- */
///functor: single, constant population size (\p d individuals) across populations and over time
struct demography_constant
{
	int d;
	inline demography_constant();
	inline demography_constant(int p);
	__host__ __device__  __forceinline__ int operator()(const int population, const int generation) const;
};

///functor: models population size (individuals) as a sine wave through time
struct demography_sine_wave
{
	float A; //Amplitude
	float pi; //Frequency
	float rho; //Phase
	int D; //Offset
	int generation_shift; //!<\copydoc Sim_Model::selection_sine_wave::generation_shift

	inline demography_sine_wave();
	inline demography_sine_wave(float A, float pi, int D, float rho = 0, int generation_shift = 0);
	__host__ __device__  __forceinline__ int operator()(const int population, const int generation) const;
};

///functor: models exponential growth of population size (individuals) over time
struct demography_exponential_growth
{
	float rate;
	int initial_population_size;
	int generation_shift; //!<\copydoc Sim_Model::selection_sine_wave::generation_shift

	inline demography_exponential_growth();
	inline demography_exponential_growth(float rate, int initial_population_size, int generation_shift = 0);
	__host__ __device__ __forceinline__ int operator()(const int population, const int generation) const;
};

///functor: models logistic growth of population size (individuals) over time
struct demography_logistic_growth
{
	float rate;
	int initial_population_size;
	int carrying_capacity;
	int generation_shift; //!<\copydoc Sim_Model::selection_sine_wave::generation_shift

	inline demography_logistic_growth();
	inline demography_logistic_growth(float rate, int initial_population_size, int carrying_capacity, int generation_shift = 0);
	__host__ __device__ __forceinline__ int operator()(const int population, const int generation) const;
};

///functor: one population, \p pop, has a different, demography function, \p d_pop, all others have function, \p d
template <typename Functor_d, typename Functor_d_pop>
struct demography_population_specific
{
	int pop;
	int generation_shift; //!<\copydoc Sim_Model::selection_sine_wave::generation_shift
	Functor_d d;
	Functor_d_pop d_pop;
	inline demography_population_specific();
	inline demography_population_specific(Functor_d d_in, Functor_d_pop d_pop_in, int pop, int generation_shift = 0);
	__host__ __device__ __forceinline__ int operator()(const int population, const int generation) const;
};

///functor: demography function changes from \p d1 to \p d2 at generation \p inflection_point
template <typename Functor_d1, typename Functor_d2>
struct demography_piecewise
{
	int inflection_point;
	int generation_shift; //!<\copydoc Sim_Model::selection_sine_wave::generation_shift
	Functor_d1 d1;
	Functor_d2 d2;
	inline demography_piecewise();
	inline demography_piecewise(Functor_d1 d1_in, Functor_d2 d2_in, int inflection_point, int generation_shift = 0);
	__host__ __device__ __forceinline__ int operator()(const int population, const int generation) const;
};
/* ----- end of demography models ----- *//** @} */

/** \defgroup migration Simulation Models: Migration Group *//**@{*/

/* ----- migration models ----- */
///functor: migration flows at rate \p m from pop i to pop j =/= i and 1-num_pop*m for i == j
struct migration_constant_equal
{
	float m;
	int num_pop;
	inline migration_constant_equal();
	inline migration_constant_equal(int n);
	inline migration_constant_equal(float m, int n);
	__host__ __device__ __forceinline__ float operator()(const int pop_FROM, const int pop_TO, const int generation) const;
};

///functor: migration flows at rate \p m from \p pop1 to \p pop2 and function \p rest for all other migration rates
template <typename Functor_m1>
struct migration_constant_directional
{
	float m;
	int pop1, pop2;
	Functor_m1 rest;
	inline migration_constant_directional();
	inline migration_constant_directional(float m, int pop1, int pop2, Functor_m1 rest_in);
	__host__ __device__ __forceinline__ float operator()(const int pop_FROM, const int pop_TO, const int generation) const;
};

///functor: migration function changes from \p m1 to \p m2 at generation \p inflection_point
template <typename Functor_m1, typename Functor_m2>
struct migration_piecewise
{
	int inflection_point;
	int generation_shift; //!<\copydoc Sim_Model::selection_sine_wave::generation_shift
	Functor_m1 m1;
	Functor_m2 m2;
	inline migration_piecewise();
	inline migration_piecewise(Functor_m1 m1_in, Functor_m2 m2_in, int inflection_point, int generation_shift = 0);
	__host__ __device__ __forceinline__ int operator()(const int pop_FROM, const int pop_TO, const int generation) const;
};
/* ----- end of migration models ----- *//** @} */

/** \defgroup pres_samp Simulation Models: Preserve and Sampling Group *//**@{*/

/* ----- preserving & sampling functions ----- */
/**\brief functor: turns sampling and preserving off (for every generation except the final one which is always sampled) */ /**returns false*/
struct bool_off{ __host__ __forceinline__ bool operator()(const int generation) const; };

/**\brief functor: turns sampling and preserving on (for every generation except the final one which is always sampled) */ /**returns true*/
struct bool_on{__host__ __forceinline__ bool operator()(const int generation) const; };

//fix - use vectors
/*struct bool_array{
	const bool * array;
	int length;
	int generation_shift;
	inline bool_array();
	inline bool_array(const bool * const in_array, int length, int generation_shift = 0);
	__host__ __forceinline__ bool operator()(const int generation) const;
	~do_array();
}; */

///functor: returns the result of function \p f1 except at generation \p pulse returns the result of function \p f2
template <typename Functor_stable, typename Functor_action>
struct bool_pulse{
	int pulse;
	int generation_shift; //!<\copydoc Sim_Model::selection_sine_wave::generation_shift
	Functor_stable f1;
	Functor_action f2;
	bool_pulse();
	bool_pulse(Functor_stable f1_in, Functor_action f2_in, int pulse, int generation_shift = 0);
	__host__ __forceinline__ bool operator()(const int generation) const;
};

///functor: returns the result of function \p f1 until generation \p inflection_point, then returns the result of function \p f2
template <typename Functor_first, typename Functor_second>
struct bool_piecewise{
	int inflection_point;
	int generation_shift; //!<\copydoc Sim_Model::selection_sine_wave::generation_shift
	Functor_first f1;
	Functor_second f2;
	inline bool_piecewise();
	inline bool_piecewise(Functor_first f1_in, Functor_second f2_in, int inflection_point, int generation_shift = 0);
	__host__ __forceinline__ bool operator()(const int generation) const;
};
/* ----- end of preserving & sampling functions ----- *//** @} */

} /* ----- end namespace Sim_Model ----- */

//!Namespace for single locus, forward Wright-Fisher simulation and output data structures
namespace GO_Fish{

/* ----- go_fish_impl  ----- */
///runs a single-locus Wright-Fisher simulation specified by the given simulation functions and sim_constants, storing the results into \p all_results
template <typename Functor_mutation, typename Functor_demography, typename Functor_migration, typename Functor_selection, typename Functor_inbreeding, typename Functor_dominance, typename Functor_preserve, typename Functor_timesample>
__host__ void run_sim(allele_trajectories & all_results, const Functor_mutation mu_rate, const Functor_demography demography, const Functor_migration mig_prop, const Functor_selection sel_coeff, const Functor_inbreeding FI, const Functor_dominance dominance, const Functor_preserve preserve_mutations, const Functor_timesample take_sample);
///runs a single-locus Wright-Fisher simulation specified by the given simulation functions and sim_constants, storing the results into \p all_results
template <typename Functor_mutation, typename Functor_demography, typename Functor_migration, typename Functor_selection, typename Functor_inbreeding, typename Functor_dominance, typename Functor_preserve, typename Functor_timesample>
__host__ void run_sim(allele_trajectories & all_results, const Functor_mutation mu_rate, const Functor_demography demography, const Functor_migration mig_prop, const Functor_selection sel_coeff, const Functor_inbreeding FI, const Functor_dominance dominance, const Functor_preserve preserve_mutations, const Functor_timesample take_sample, const allele_trajectories & prev_sim);
/* ----- end go_fish_impl ----- */

} /* ----- end namespace GO_Fish ----- */

/* ----- importing functor implementations ----- */
#include "../source/template_inline_simulation_functors.cuh"
/* ----- end importing functor implementations ----- */

/* ----- importing go_fish_impl  ----- */
#include "../source/go_fish_impl.cuh"
/* ----- end importing go_fish_impl ----- */


#endif /* GO_FISH_API_H_ */
