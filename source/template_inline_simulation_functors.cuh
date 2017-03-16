/*
 * template_simulation_functors.cuh
 *
 *      Author: David Lawrie
 *      implementation of template and inline functions for GO Fish evolutionary models and sampling schemes
 */

#ifndef TEMPLATE_INLINE_SIMULATION_FUNCTORS_CUH_
#define TEMPLATE_INLINE_SIMULATION_FUNCTORS_CUH_

/**Functions for modeling selection, inbreeding, mutation, dominance, demography, migration across populations and over time as well as functions to preserve mutations in a generation and to sample time points in the simulation. \n
* \n Use of these functions is optional as users can supply their own with the same given format, for details on how to write your own simulation functions, go to the <a href="modules.html">Modules</a> page, click on the simulation function group which describes the function you wish to write, and read its detailed description. */
namespace Sim_Model{

/** \addtogroup selection
*  \brief Functions that model selection coefficients (\p s) across populations and over time
*
*  Selection coefficients are defined by:
*
*  AA | Aa | aa
*  -- | -- | --
*  `1` | `1+hs` | `1+s`
*
*  where \p h is the dominance coefficient and AA, Aa, aa represent the various alleles.
*  Thus Effective Selection, \p gamma, is defined by `N_chromosome_e*s` which for outbreeding diploids is `2*N*s` and haploid is `N*s`
*  where N is the number of individuals (as returned by demography functions).
*  Diploids with inbreeding will have an effective strength of selection of `2*N*s/(1+F)`.
*
*  Minimum selection is `s >= -1` (lethal). Currently the program will not throw an error if the selection is less than -1, but will simply take the `max(s,-1)`.
*
*  ###Writing your own Selection functions###
*  These can be functions or functors (or soon, with C++11 support, lambdas). However, the selection function must be of the form:\n
*  \code __device__ float your_function(int population, int generation, float freq){ ... return selection_coeff; } \endcode
*  This returns the selection coefficient in population \p population at generation \p generation for a mutation at frequency \p freq.
*  The `__device__` flag is to ensure the nvcc compiler knows the function must be compiled for the device (GPU).
*  Because of this flag, the function must be defined in CUDA source file (*.cu) or declared/defined header file (*.h, *.hpp, *.cuh, etc...) which is included in a CUDA source file.
*  Since this code will be compiled on the GPU, do not use dynamically allocated arrays in your function (e.g. `float * f = new float[5]`) unless you know CUDA.
*  And even then avoid them as they will slow the code down (parameters have to be pulled from the GPU's global memory (vRAM), which is slow).
*  Statically allocated arrays (e.g. `float f[5]`) are fine.
*/

/* ----- constant selection model ----- */
inline selection_constant::selection_constant() : s(0) {}
inline selection_constant::selection_constant(float s) : s(s){ }
/**`s = gamma/(2*demography(0,forward_generation_shift)/(1+F(0,forward_generation_shift)))`\n
 * \param forward_generation_shift (optional input) default `0` \n allows you to push the population size and inbreeding coefficient value to the state forward in time - useful
 * if you are starting the simulation from a previous simulation state and are using the same functions as the previous simulation or any time you want to shift the generation of the demography and inbreeding functions from 0 \n */
template <typename Functor_demography, typename Functor_inbreeding>
inline selection_constant::selection_constant(float gamma, Functor_demography demography, Functor_inbreeding F, int forward_generation_shift /*= 0*/){ s = gamma/(2*demography(0,forward_generation_shift)/(1+F(0,forward_generation_shift))); }
__device__ __forceinline__ float selection_constant::operator()(const int population, const int generation, const float freq) const{ return s; }
/* ----- end constant selection model ----- */

/* ----- linear frequency dependent selection model ----- */
/**\struct selection_linear_frequency_dependent
 * `(slope < 0)` := balancing selection model (negative frequency-dependent selection) \n
 * `(slope = 0)` := constant selection \n
 * `(slope > 0)` := reinforcement selection model (positive frequency-dependent selection) \n
 * */
inline selection_linear_frequency_dependent::selection_linear_frequency_dependent() : slope(0), intercept(0) {}
inline selection_linear_frequency_dependent::selection_linear_frequency_dependent(float slope, float intercept) : slope(slope), intercept(intercept) { }
/**`slope = gamma_slope/(2*demography(0,forward_generation_shift)/(1+F(0,forward_generation_shift)))`\n
 * `intercept = gamma_intercept/(2*demography(0,forward_generation_shift)/(1+F(0,forward_generation_shift)))`\n
 * \param forward_generation_shift (optional input) default `0` \n allows you to push the population size and inbreeding coefficient value to the state forward in time - useful
 * if you are starting the simulation from a previous simulation state and are using the same functions as the previous simulation or any time you want to shift the generation of the demography and inbreeding functions from 0 \n */
template <typename Functor_demography, typename Functor_inbreeding>
inline selection_linear_frequency_dependent::selection_linear_frequency_dependent(float gamma_slope, float gamma_intercept, Functor_demography demography, Functor_inbreeding F, int forward_generation_shift /*= 0*/){
	slope = gamma_slope/(2*demography(0,forward_generation_shift)/(1+F(0,forward_generation_shift)));
	intercept = gamma_intercept/(2*demography(0,forward_generation_shift)/(1+F(0,forward_generation_shift)));
}
/** `s = slope*freq + intercept` */
__device__ __forceinline__ float selection_linear_frequency_dependent::operator()(const int population, const int generation, const float freq) const{ return slope*freq+intercept; }
/* ----- end linear frequency dependent selection model ----- */

/* ----- seasonal selection model ----- */
inline selection_sine_wave::selection_sine_wave() : A(0), pi(0), rho(0), D(0), generation_shift(0) {}
/**\param rho (optional input) default `0` \param generation_shift (optional input) default `0` */
inline selection_sine_wave::selection_sine_wave(float A, float pi, float D, float rho /*= 0*/, int generation_shift /*= 0*/) : A(A), pi(pi), rho(rho), D(D), generation_shift(generation_shift) {}
/**`A = gamma_A/(2*demography(0,forward_generation_shift)/(1+F(0,forward_generation_shift)))`\n
 * `D = gamma_D/(2*demography(0,forward_generation_shift)/(1+F(0,forward_generation_shift)))`\n
 * \param rho (optional input) default `0` \param generation_shift (optional input) default `0`
 * \param forward_generation_shift (optional input) default `0` \n allows you to push the population size and inbreeding coefficient value to the state forward in time - useful
 * if you are starting the simulation from a previous simulation state and are using the same functions as the previous simulation or any time you want to shift the generation of the demography and inbreeding functions from 0 \n */
template <typename Functor_demography, typename Functor_inbreeding>
inline selection_sine_wave::selection_sine_wave(float gamma_A, float pi, float gamma_D, Functor_demography demography, Functor_inbreeding F, float rho /*= 0*/, int generation_shift /*= 0*/, int forward_generation_shift /*= 0*/) : pi(pi), rho(rho), generation_shift(generation_shift) {
	A = gamma_A/(2*demography(0,forward_generation_shift)/(1+F(0,forward_generation_shift)));
	D = gamma_D/(2*demography(0,forward_generation_shift)/(1+F(0,forward_generation_shift)));
}
/** `s = A*sin(pi*(generation-generation_shift) + rho) + D` */
__device__ __forceinline__ float selection_sine_wave::operator()(const int population, const int generation, const float freq) const{ return A*sin(pi*(generation-generation_shift) + rho) + D;}
/* ----- end seasonal selection model ----- */

/* ----- population specific selection model ----- */
/** \struct selection_population_specific
 * Takes in two template types: the function to be returned for the rest of the populations and the function for the specific population, `pop` \n
 * Population specific selection functors can be nested within each other and with piecewise selection functors for multiple populations and multiple time functions, e.g.:\n\n
 * 3 populations with different selection coefficients where mutations in the 1st are neutral, in the 2nd are deleterious, and in the 3rd are beneficial up to generation 300 when selection relaxes and they become neutral
 * \code typedef Sim_Model::selection_constant constant;
	typedef Sim_Model::selection_piecewise<constant,constant> piecewise_constant;
	typedef Sim_Model::selection_population_specific<constant,constant> population_specific_constant;
	constant neutral;
	constant purifying(-0.01);
	constant positive(0.01);
	population_specific_constant first_second(neutral,purifying,1);
	piecewise_constant third(positive,neutral,300);
	Sim_Model::selection_population_specific<population_specific_constant,piecewise_constant> selection_model(first_second,third,2); \endcode
	The modularity of these functor templates allow selection models to be extended to any number of populations and piecewise selection functions (including user defined functions).
 * */
/**`pop = 0` \n `generation_shift = 0` \n
Function `s` assigned default constructor of `Functor_sel`\n
Function `s_pop` assigned default constructor of `Functor_sel_pop`*/
template <typename Functor_sel, typename Functor_sel_pop>
inline selection_population_specific<Functor_sel,Functor_sel_pop>::selection_population_specific() : pop(0), generation_shift(0) { s = Functor_sel(); s_pop = Functor_sel_pop(); }
/** \param generation_shift (optional input) default `0` */
template <typename Functor_sel, typename Functor_sel_pop>
inline selection_population_specific<Functor_sel,Functor_sel_pop>::selection_population_specific(Functor_sel s_in, Functor_sel_pop s_pop_in, int pop, int generation_shift /*= 0*/) : pop(pop), generation_shift(generation_shift){  s = s_in; s_pop = s_pop_in; }
/** `if(pop == population) s = s_pop(population, generation-generation_shift, freq)`\n
	`else s = s(population, generation-generation_shift, freq)` */
template <typename Functor_sel, typename Functor_sel_pop>
__device__ __forceinline__ float selection_population_specific<Functor_sel,Functor_sel_pop>::operator()(const int population, const int generation, const float freq) const{
	if(pop == population) return s_pop(population, generation-generation_shift, freq);
	return s(population, generation-generation_shift, freq);
}
/* ----- end population specific selection model ----- */

/* ----- piecewise selection model ----- */
/** \struct selection_piecewise
 *  Takes in two template types: the function to be returned before the `inflection_point` and the function for after the `inflection_point` \n
 * Piecewise selection functors can be nested within each other and with population specific selection functors for multiple populations and multiple time functions, e.g.:\n\n
 * 3 populations with different selection coefficients where mutations in the 1st are neutral, in the 2nd are deleterious, and in the 3rd are beneficial up to generation 300 when selection relaxes and they become neutral
 * \code typedef Sim_Model::selection_constant constant;
	typedef Sim_Model::selection_piecewise<constant,constant> piecewise_constant;
	typedef Sim_Model::selection_population_specific<constant,constant> population_specific_constant;
	constant neutral;
	constant purifying(-0.01);
	constant positive(0.01);
	population_specific_constant first_second(neutral,purifying,1);
	piecewise_constant third(positive,neutral,300);
	Sim_Model::selection_population_specific<population_specific_constant,piecewise_constant> selection_model(first_second,third,2);\endcode
	The modularity of these functor templates allow selection models to be extended to any number of populations and piecewise selection functions (including user defined functions).
 */
/**`inflection_point = 0` \n `generation_shift = 0` \n
Function `s1` assigned default constructor of `Functor_sel1`\n
Function `s2` assigned default constructor of `Functor_sel2`*/
template <typename Functor_sel1, typename Functor_sel2>
inline selection_piecewise<Functor_sel1, Functor_sel2>::selection_piecewise() : inflection_point(0), generation_shift(0) { s1 = Functor_sel1(); s2 = Functor_sel2(); }
/** \param generation_shift (optional input) default `0` */
template <typename Functor_sel1, typename Functor_sel2>
inline selection_piecewise<Functor_sel1, Functor_sel2>::selection_piecewise(Functor_sel1 s1_in, Functor_sel2 s2_in, int inflection_point, int generation_shift /* = 0*/) : inflection_point(inflection_point), generation_shift(generation_shift) { s1 = s1_in; s2 = s2_in; }
/** `if(generation >= inflection_point+generation_shift) s = s2(population, generation-generation_shift, freq)`\n
	`else s = s1(population, generation-generation_shift, freq)` */
template <typename Functor_sel1, typename Functor_sel2>
__device__ __forceinline__ float selection_piecewise<Functor_sel1, Functor_sel2>::operator()(const int population, const int generation, const float freq) const{
	if(generation >= inflection_point+generation_shift){ return s2(population, generation-generation_shift, freq) ; }
	return s1(population, generation-generation_shift, freq);
};
/* ----- end piecewise selection model ----- */
/* ----- end selection models ----- */

/** \addtogroup in_mut_dom
*  \brief Functions that model inbreeding coefficients (\p F), per-site mutation rates (\p mu), and dominance coefficients (\p h) across populations and over time
*
*  Inbreeding coefficients, \p F, must be between [0,1]:
*  > `(F = 0)` := diploid, outbred; `(F = 1)` := haploid, inbred
*  Per-site mutation rate, \p mu, must be `> 0`. \n\n Dominance coefficients, \p h, can be any real number:
*  > `(h < 0)` := underdominance; `(h > 1)` := overdominance; `(h = 0)` := recessive; `(h = 1)` := dominant; `(0 < h < 1)` := co-dominant
*  \n
*  ###Writing your own Inbreeding, Mutation, and Dominance functions###
*  These can be functions or functors (or soon, with C++11 support, lambdas). However, the parameter function must be of the form:\n
*  \code float your_function(int population, int generation){ ... return parameter; } \endcode
*  This returns the parameter in population \p population at generation \p generation.
*  Adding a `__host__` flag is optional, but if done, the function must be defined in CUDA source file (*.cu) or declared/defined header file (*.h, *.hpp, *.cuh, etc...) which is included in a CUDA source file.
*  If no flag added, function can be defined in a regular C/C++ source file (e.g. *.c, *.cpp).
*  Note: run_sim is required to be in a CUDA source file to be compiled.
*/

/* ----- inbreeding, mutation, & dominance models ----- */
/* ----- constant parameter model ----- */
inline F_mu_h_constant::F_mu_h_constant() : p(0) {}
inline F_mu_h_constant::F_mu_h_constant(float p) : p(p){ }
__host__ __forceinline__ float F_mu_h_constant::operator()(const int population, const int generation) const{ return p; }
/* ----- end constant parameter model ----- */

/* ----- seasonal parameter model ----- */
inline F_mu_h_sine_wave::F_mu_h_sine_wave() : A(0), pi(0), rho(0), D(0), generation_shift(0) {}
inline F_mu_h_sine_wave::F_mu_h_sine_wave(float A, float pi, float D, float rho /*= 0*/, int generation_shift /*= 0*/) : A(A), pi(pi), rho(rho), D(D), generation_shift(generation_shift) {}
/** `p = A*sin(pi*(generation-generation_shift) + rho) + D` */
__host__ __forceinline__ float F_mu_h_sine_wave::operator()(const int population, const int generation) const{ return A*sin(pi*(generation-generation_shift) + rho) + D;}
/* ----- end seasonal parameter model ----- */

/* ----- population specific parameter model ----- */
/** \struct F_mu_h_population_specific
 * Takes in two template types: the function to be returned for the rest of the populations and the function for the specific population, `pop` \n
 * Population specific parameter functors can be nested within each other and with piecewise parameter functors for multiple populations and multiple time functions, e.g.:\n\n
 * 3 populations with different Inbreeding coefficients where populations in the 1st are outbred, in the 2nd are partially inbred, and in the 3rd are completely inbred until population becomes completely outcrossing at generation 300
 * \code typedef Sim_Model::F_mu_h_constant F_constant;
	typedef Sim_Model::F_mu_h_piecewise<F_constant,F_constant> F_piecewise_constant;
	typedef Sim_Model::F_mu_h_population_specific<F_constant,F_constant> F_population_specific_constant;
	F_constant outbred;
	F_constant inbred(1);
	F_constant mixed(0.5);
	F_population_specific_constant first_second(outbred,mixed,1);
	F_piecewise_constant third(inbred,outbred,300);
	Sim_Model::F_mu_h_population_specific<F_population_specific_constant,F_piecewise_constant> inbreeding_model(first_second,third,2); \endcode
	The modularity of these functor templates allow parameter models to be extended to any number of populations and piecewise parameter functions (including user defined functions).
 * */
/**`pop = 0` \n `generation_shift = 0` \n
Function `p` assigned default constructor of `Functor_p`\n
Function `p_pop` assigned default constructor of `Functor_p_pop`*/
template <typename Functor_p, typename Functor_p_pop>
inline F_mu_h_population_specific<Functor_p,Functor_p_pop>::F_mu_h_population_specific() : pop(0), generation_shift(0) { p = Functor_p(); p_pop = Functor_p_pop(); }
/** \param generation_shift (optional input) default `0` */
template <typename Functor_p, typename Functor_p_pop>
inline F_mu_h_population_specific<Functor_p,Functor_p_pop>::F_mu_h_population_specific(Functor_p p_in, Functor_p_pop p_pop_in, int pop, int generation_shift /*= 0*/) : pop(pop), generation_shift(generation_shift){  p = p_in; p_pop = p_pop_in; }
/** `if(pop == population) p = p_pop(population, generation-generation_shift)`\n
	`else p = p(population, generation-generation_shift)` */
template <typename Functor_p, typename Functor_p_pop>
__host__ __forceinline__ float F_mu_h_population_specific<Functor_p,Functor_p_pop>::operator()(const int population, const int generation) const{
	if(pop == population) return p_pop(population, generation-generation_shift);
	return p(population, generation-generation_shift);
}
/* ----- end population specific parameter model ----- */

/* ----- piecewise parameter model ----- */
/** \struct F_mu_h_piecewise
 * Takes in two template types: the function to be returned before the `inflection_point` and the function for after the `inflection_point` \n
 * Piecewise parameter functors can be nested within each other and with population specific parameter functors for multiple populations and multiple time functions, e.g.:\n\n
 * 3 populations with different Inbreeding coefficients where populations in the 1st are outbred, in the 2nd are partially inbred, and in the 3rd are completely inbred until population becomes completely outcrossing at generation 300
 * \code typedef Sim_Model::F_mu_h_constant F_constant;
	typedef Sim_Model::F_mu_h_piecewise<F_constant,F_constant> F_piecewise_constant;
	typedef Sim_Model::F_mu_h_population_specific<F_constant,F_constant> F_population_specific_constant;
	F_constant outbred;
	F_constant inbred(1);
	F_constant mixed(0.5);
	F_population_specific_constant first_second(outbred,mixed,1);
	F_piecewise_constant third(inbred,outbred,300);
	Sim_Model::F_mu_h_population_specific<F_population_specific_constant,F_piecewise_constant> inbreeding_model(first_second,third,2); \endcode
	The modularity of these functor templates allow parameter models to be extended to any number of populations and piecewise parameter functions (including user defined functions).
 * */
/**`inflection_point = 0` \n `generation_shift = 0` \n
Function `p1` assigned default constructor of `Functor_p1`\n
Function `p2` assigned default constructor of `Functor_p2`*/
template <typename Functor_p1, typename Functor_p2>
inline F_mu_h_piecewise<Functor_p1, Functor_p2>::F_mu_h_piecewise() : inflection_point(0), generation_shift(0) { p1 = Functor_p1(); p2 = Functor_p2(); }
/** \param generation_shift (optional input) default `0` */
template <typename Functor_p1, typename Functor_p2>
inline F_mu_h_piecewise<Functor_p1, Functor_p2>::F_mu_h_piecewise(Functor_p1 p1_in, Functor_p2 p2_in, int inflection_point, int generation_shift /* = 0*/) : inflection_point(inflection_point), generation_shift(generation_shift) { p1 = p1_in; p2 = p2_in; }
/** `if(generation >= inflection_point+generation_shift) p = p2(population, generation-generation_shift)`\n
	`else p = p1(population, generation-generation_shift)` */
template <typename Functor_p1, typename Functor_p2>
__host__ __forceinline__ float F_mu_h_piecewise<Functor_p1, Functor_p2>::operator()(const int population, const int generation) const{
	if(generation >= inflection_point+generation_shift){ return p2(population, generation-generation_shift) ; }
	return p1(population, generation-generation_shift);
};
/* ----- end piecewise parameter model ----- */
/* ----- end of inbreeding, mutation, & dominance models models ----- */

/** \addtogroup demography
*  \brief Functions that model population size in number of individuals across populations and over time
*
*  If a population with a previously positive number of individuals hits 0 individuals over the course of a simulation, it will be declared extinct.
*  If the population size of an extinct population becomes non-zero after the extinction, an error will be thrown.
*  Populations that start the simulation at 0 individuals are not considered extinct.
*
*  ###Writing your own Demography functions###
*  These can be functions or functors (or soon, with C++11 support, lambdas). However, the demographic function must be of the form:\n
*  \code __host__ __device__ int your_function(int population, int generation){ ... return number_of_individuals; } \endcode
*  This returns the number of individuals in population \p population at generation \p generation.
*  The `__host__` and `__device__` flags are to ensure the nvcc compiler knows the function must be compiled for both the host (CPU) and device (GPU).
*  Because of these flags, the function must be defined in CUDA source file (*.cu) or declared/defined header file (*.h, *.hpp, *.cuh, etc...) which is included in a CUDA source file.
*  Since this code will be compiled on the GPU, do not use dynamically allocated arrays in your function (e.g. `int * f = new int[5]`) unless you know CUDA.
*  And even then avoid them as they will slow the code down (parameters have to be pulled from the GPU's global memory (vRAM), which is slow).
*  Statically allocated arrays (e.g. `int f[5]`) are fine.
*/

/* ----- demography models ----- */
/* ----- constant demography model ----- */
inline demography_constant::demography_constant() : N(0) {}
inline demography_constant::demography_constant(int N) : N(N){ }
__host__ __device__  __forceinline__ int demography_constant::operator()(const int population, const int generation) const{ return N; }
/* ----- end constant demography model ----- */

/* ----- seasonal demography model ----- */
inline demography_sine_wave::demography_sine_wave() : A(0), pi(0), rho(0), D(0), generation_shift(0) {}
inline demography_sine_wave::demography_sine_wave(float A, float pi, int D, float rho /*= 0*/, int generation_shift /*= 0*/) : A(A), pi(pi), rho(rho), D(D), generation_shift(generation_shift) {}
/** `N = A*sin(pi*(generation-generation_shift) + rho) + D` */
__host__ __device__  __forceinline__ int demography_sine_wave::operator()(const int population, const int generation) const{ return (int)A*sin(pi*(generation-generation_shift) + rho) + D;}
/* ----- end seasonal parameter model ----- */

/* ----- exponential growth model ----- */
inline demography_exponential_growth::demography_exponential_growth() : rate(0), initial_population_size(0), generation_shift(0) {}
/** \param generation_shift (optional input) default `0` */
inline demography_exponential_growth::demography_exponential_growth(float rate, int initial_population_size, int generation_shift /*= 0*/) : rate(rate), initial_population_size(initial_population_size), generation_shift(generation_shift) {}
/** `N = round(initial_population_size*`\f$e^{\textrm{rate*(generation-generation_shift)}} \f$`)` */
__host__ __device__  __forceinline__ int demography_exponential_growth::operator()(const int population, const int generation) const{ return (int)round(initial_population_size*exp(rate*(generation-generation_shift))); }
/* ----- end exponential growth model ----- */

/* ----- logistic growth model ----- */
inline demography_logistic_growth::demography_logistic_growth() : rate(0), initial_population_size(0), carrying_capacity(0), generation_shift(0) {}
inline demography_logistic_growth::demography_logistic_growth(float rate, int initial_population_size, int carrying_capacity, int generation_shift /*= 0*/) : rate(rate), initial_population_size(initial_population_size), carrying_capacity(carrying_capacity), generation_shift(generation_shift) {}
/** `exp_term = `\f$e^{\textrm{rate*(generation-generation_shift)}} \f$ \n
 * `N = round((carrying_capacity*initial_population_size*exp_term)`\f$\div\f$`(carrying_capacity + initial_population_size*(exp_term-1)))` */
__host__ __device__  __forceinline__ int demography_logistic_growth::operator()(const int population, const int generation) const{
	float term = exp(rate*(generation-generation_shift));
	return (int)round(carrying_capacity*initial_population_size*term/(carrying_capacity + initial_population_size*(term-1)));
}
/* ----- end logistic growth model ----- */

/* ----- population specific demography model ----- */
/**`pop = 0` \n `generation_shift = 0` \n
Function `d` assigned default constructor of `Functor_d`\n
Function `d_pop` assigned default constructor of `Functor_d_pop`*/
template <typename Functor_d, typename Functor_d_pop>
inline demography_population_specific<Functor_d,Functor_d_pop>::demography_population_specific() : pop(0), generation_shift(0) { d = Functor_d(); d_pop = Functor_d_pop(); }
/** \param generation_shift (optional input) default `0` */
template <typename Functor_d, typename Functor_d_pop>
inline demography_population_specific<Functor_d,Functor_d_pop>::demography_population_specific(Functor_d d_in, Functor_d_pop d_pop_in, int pop, int generation_shift /*= 0*/) : pop(pop), generation_shift(generation_shift){  d = d_in; d_pop = d_pop_in; }
/** `if(pop == population) N = d_pop(population, generation-generation_shift)`\n
	`else N = d(population, generation-generation_shift)` */
template <typename Functor_d, typename Functor_d_pop>
__host__ __device__  __forceinline__ int demography_population_specific<Functor_d,Functor_d_pop>::operator()(const int population, const int generation) const{
	if(pop == population) return d_pop(population, generation-generation_shift);
	return d(population, generation-generation_shift);
}
/* ----- end population specific demography model ----- */

/* ----- piecewise demography model ----- */
/**`inflection_point = 0` \n `generation_shift = 0` \n
Function `d1` assigned default constructor of `Functor_d1`\n
Function `d2` assigned default constructor of `Functor_d2`*/
template <typename Functor_d1, typename Functor_d2>
inline demography_piecewise<Functor_d1, Functor_d2>::demography_piecewise() : inflection_point(0), generation_shift(0) { d1 = Functor_d1(); d2 = Functor_d2(); }
/** \param generation_shift (optional input) default `0` */
template <typename Functor_d1, typename Functor_d2>
inline demography_piecewise<Functor_d1, Functor_d2>::demography_piecewise(Functor_d1 d1_in, Functor_d2 d2_in, int inflection_point, int generation_shift /* = 0*/) : inflection_point(inflection_point), generation_shift(generation_shift) { d1 = d1_in; d2 = d2_in; }
/** `if(generation >= inflection_point+generation_shift) N = d2(population, generation-generation_shift)`\n
	`else N = d1(population, generation-generation_shift)` */
template <typename Functor_d1, typename Functor_d2>
__host__ __device__  __forceinline__ int demography_piecewise<Functor_d1, Functor_d2>::operator()(const int population, const int generation) const{
	if(generation >= inflection_point+generation_shift){ return d2(population, generation-generation_shift) ; }
	return d1(population, generation-generation_shift);
};
/* ----- end piecewise demography model ----- */
/* ----- end of demography models ----- */

/** \addtogroup migration
*  \brief Functions that model migration rates over time (conservative model of migration)
*
*  In the conservative model of migration, migration rate from population i to population j is expressed as the fraction of population j originally from i:\n
*  > e.g. in a 2 population model, a migration rate of \f$ m_{ij} = 0.1 \f$ ==> 10% of population j is originally from population i \n
*  > and the frequency, \f$x_{mig,j}\f$, in the next generation of an allele is \f$x_{mig,j} = 0.1*x_i + 0.9*x_j\f$
*  Thus, in general, the following must be true: \f$\sum_{i=1}^n m_{ij} = 1\f$ (program will throw error elsewise).
*  However the sum of migration rates FROM a population need not sum to anything in particular.
*  This is also the set of functions that are used to specify a single population splitting into (two or more) populations. \n \n
*
*  ###Writing your own Migration functions###
*  These can be functions or functors (or soon, with C++11 support, lambdas). However, the migration function must be of the form:\n
*  \code __host__ __device__ float your_function(int population_FROM, int population_TO, int generation){ ... return migration_rate; } \endcode
*  This returns the rate of migration from population \p population_FROM to population \p population_TO at generation \p generation.
*  The `__host__` and `__device__` flags are to ensure the nvcc compiler knows the function must be compiled for both the host (CPU) and device (GPU).
*  Because of these flags, the function must be defined in CUDA source file (*.cu) or declared/defined header file (*.h, *.hpp, *.cuh, etc...) which is included in a CUDA source file.
*  Since this code will be compiled on the GPU, do not use dynamically allocated arrays in your function (e.g. `float * f = new float[5]`) unless you know CUDA.
*  And even then avoid them as they will slow the code down (parameters have to be pulled from the GPU's global memory (vRAM), which is slow).
*  Statically allocated arrays (e.g. `float f[5]`) are fine.
*/

/* ----- migration models ----- */
/* ----- constant equal migration model ----- */
inline migration_constant_equal::migration_constant_equal() : m(0), num_pop(1){ }
inline migration_constant_equal::migration_constant_equal(float m, int num_pop) : m(m), num_pop(min(num_pop,1)){ }
/**`if(pop_FROM == pop_TO) mig_rate = 1-(num_pop-1)*m` \n
 * `else mig_rate = m`
 *  */
__host__ __device__ __forceinline__ float migration_constant_equal::operator()(const int pop_FROM, const int pop_TO, const int generation) const{
		if(pop_FROM == pop_TO){ return 1-(num_pop-1)*m; }
		return (num_pop > 1) * m;
}
/* ----- end constant equal migration model ----- */

/* ----- constant directional migration model ----- */
/**`m = 0` \n `pop1 = 0` \n `pop2 = 0` \n
Function `rest` assigned default constructor of `Functor_m1`\n*/
template <typename Functor_m1>
inline migration_constant_directional<Functor_m1>::migration_constant_directional() : m(0), pop1(0), pop2(0) { rest = Functor_m1(); }
template <typename Functor_m1>
inline migration_constant_directional<Functor_m1>::migration_constant_directional(float m, int pop1, int pop2, Functor_m1 rest_in) : m(m), pop1(pop1), pop2(pop2) { rest = rest_in; }
/**`if(pop_FROM == pop1 && pop_TO == pop2) mig_rate = m` \n
 * `else mig_rate = rest(pop_FROM,pop_TO,generation)`
 *  */
template <typename Functor_m1>
__host__ __device__ __forceinline__ float migration_constant_directional<Functor_m1>::operator()(const int pop_FROM, const int pop_TO, const int generation) const{
	if(pop_FROM == pop1 && pop_TO == pop2) return m;
	return rest(pop_FROM, pop_TO, generation);
}
/* ----- end constant directional migration model ----- */

/* ----- piecewise migration model ----- */
/**`inflection_point = 0` \n `generation_shift = 0` \n
Function `m1` assigned default constructor of `Functor_m1`\n
Function `m2` assigned default constructor of `Functor_m2`*/
template <typename Functor_m1, typename Functor_m2>
inline migration_piecewise<Functor_m1,Functor_m2>::migration_piecewise() : inflection_point(0), generation_shift(0) { m1 = Functor_m1(); m2 = Functor_m2(); }
/** \param generation_shift (optional input) default `0` */
template <typename Functor_m1, typename Functor_m2>
inline migration_piecewise<Functor_m1,Functor_m2>::migration_piecewise(Functor_m1 m1_in, Functor_m2 m2_in, int inflection_point, int generation_shift /*= 0*/) : inflection_point(inflection_point), generation_shift(generation_shift) { m1 = m1_in; m2 = m2_in; }
/** `if(generation >= inflection_point+generation_shift) mig_rate = m2(pop_FROM, pop_TO, generation-generation_shift)`\n
	`else mig_rate = m1(pop_FROM, pop_TO, generation-generation_shift)` */
template <typename Functor_m1, typename Functor_m2>
__host__ __device__ __forceinline__ int migration_piecewise<Functor_m1,Functor_m2>::operator()(const int pop_FROM, const int pop_TO, const int generation) const{
	if(generation >= inflection_point+generation_shift){ return m2(pop_FROM,pop_TO,generation-generation_shift); }
	return m1(pop_FROM,pop_TO,generation-generation_shift);
}
/* ----- end piecewise migration model ----- */
/* ----- end of migration models ----- */

/** \addtogroup pres_samp
*  \brief Functions that control at when to flag currently segregating mutations as preserved and when to sample the mutation frequencies
*
*  When the preserve function returns true for a given generation, the simulation will compact out any lost or fixed mutations that are not already flagged preserved - provided that `GO_Fish::allele_trajectories::sim_constants::compact_interval > 0`.
*  (Though if `compact_interval == 0`, then there is no reason to employ the preserve function.)
*  The remaining mutations will then be flagged as preserved so that compact will not remove them from the simulation.\n\n
*
*  Sampling a simulation writes out the mutation frequencies from the GPU's memory vRAM to the CPU's RAM for storage and later analysis. Sampling a simulation at a time point other than the final generation (the final generation is always sampled) will also trigger the preserve flag (and thus compact if `compact_interval > 0`).
*  Thus, those mutations sampled at generation \p t will be preserved in all samples after generation \p t in order to ease frequency comparisons and track allele frequencies.
*
*  \n
*  ###Writing your own Preserving and Sampling Functions###
*  These can be functions or functors (or soon, with C++11 support, lambdas). However, the preserving/sampling function must be of the form:
*
*  \code bool your_function(int generation){ ... return true/false; } \endcode
*  This returns true or false (preserve or not preserve, sample or do not sample) at generation \p generation.
*  Adding a `__host__` flag is optional, but if done, the function must be defined in CUDA source file (*.cu) or declared/defined header file (*.h, *.hpp, *.cuh, etc...) which is included in a CUDA source file.
*  If no flag added, function can be defined in a regular C/C++ source file (e.g. *.c, *.cpp).
*  Note: run_sim is required to be in a CUDA source file to be compiled.
*/

/* ----- preserving & sampling functions ----- */
/* ----- off ----- */
__host__ __forceinline__ bool bool_off::operator()(const int generation) const{ return false; }
/* ----- end off ----- */

/* ----- on ----- */
__host__ __forceinline__ bool bool_on::operator()(const int generation) const{ return true; }
/* ----- end on ----- */

/* ----- bool_pulse_array ----- */
/* will switch to variadic templates/initializer lists when switching to C++11
bool_pulse_array::bool_pulse_array(): num_generations(0), generation_start(0) { array = NULL; }
bool_pulse_array::bool_pulse_array(const bool default_return, const int generation_start, const int num_generations, int generation_pulse...): num_generations(num_generations), generation_start(generation_start) {
	array = new bool[num_generations];
	memset(&array, default_return, num_generations*sizeof(bool));
	array[generation_pulse-generation_start] = !default_return;
}
__host__ __forceinline__ bool bool_pulse_array::operator()(const int generation) const {
	int gen = generation - generation_start;
	if((gen < 0) | (gen > num_generations)){ fprintf(stderr,"do_array functor generation error,\t generation %d\t shifted generation %d\t array length %d\n",generation,gen,num_generations); exit(1); }
	return array[gen];
}
bool_pulse_array::~bool_pulse_array(){ delete [] array; array = NULL; }*/
/* ----- end on_off_array ----- */

/* ----- pulse ----- */
/**`pulse = 0` \n `generation_shift = 0` \n
Function `f_default` assigned default constructor of `Functor_default`\n
Function `f_action` assigned default constructor of `Functor_action`*/
template <typename Functor_default, typename Functor_action>
inline bool_pulse<Functor_default,Functor_action>::bool_pulse() : pulse(0), generation_shift(0) { f_default = Functor_default(); f_action = Functor_action(); }
/**Function `f_default` assigned default constructor of `Functor_default`\n
Function `f_action` assigned default constructor of `Functor_action`\n
\param generation_shift (optional input) default `0` */
template <typename Functor_default, typename Functor_action>
inline bool_pulse<Functor_default,Functor_action>::bool_pulse(int pulse, int generation_shift/*= 0*/) : pulse(pulse), generation_shift(generation_shift) { f_default = Functor_default(); f_action = Functor_action(); }
/**\param generation_shift (optional input) default `0` */
template <typename Functor_default, typename Functor_action>
inline bool_pulse<Functor_default,Functor_action>::bool_pulse(Functor_default f_default_in, Functor_action f_action_in, int pulse, int generation_shift/*= 0*/) : pulse(pulse), generation_shift(generation_shift) { f_default = f_default_in; f_action = f_action_in; }
/**`if(generation == pulse + generation_shift) b = f_action(generation)` \n
 * `else b = f_default(generation)`
 *  */
template <typename Functor_default, typename Functor_action>
__host__ __forceinline__ bool bool_pulse<Functor_default,Functor_action>::operator()(const int generation) const{ if(generation == pulse + generation_shift){ return f_action(generation); } return f_default(generation); }
/* ----- end pulse ----- */

/* ----- switch_function ----- */
/**`inflection_point = 0` \n `generation_shift = 0` \n
Function `f1` assigned default constructor of `Functor_first`\n
Function `f2` assigned default constructor of `Functor_second`*/
template <typename Functor_first, typename Functor_second>
inline bool_piecewise<Functor_first,Functor_second>::bool_piecewise() : inflection_point(0), generation_shift(0) { f1 = Functor_first(); f2 = Functor_second(); }
/**Function `f1` assigned default constructor of `Functor_first`\n
Function `f2` assigned default constructor of `Functor_second`
\param generation_shift (optional input) default `0` */
template <typename Functor_first, typename Functor_second>
inline bool_piecewise<Functor_first,Functor_second>::bool_piecewise(int inflection_point, int generation_shift /*= 0*/) : inflection_point(inflection_point), generation_shift(generation_shift) { f1 = Functor_first(); f2 = Functor_second(); }
/**\param generation_shift (optional input) default `0` */
template <typename Functor_first, typename Functor_second>
inline bool_piecewise<Functor_first,Functor_second>::bool_piecewise(Functor_first f1_in, Functor_second f2_in, int inflection_point, int generation_shift /*= 0*/) : inflection_point(inflection_point), generation_shift(generation_shift) { f1 = f1_in; f2 = f2_in; }
/**`if(generation >= inflection_point+generation_shift)  b = f2(generation)` \n
 * `else b = f1(generation)`
 *  */
template <typename Functor_first, typename Functor_second>
__host__ __forceinline__ bool bool_piecewise<Functor_first,Functor_second>::operator()(const int generation) const{ if(generation >= inflection_point+generation_shift){ return f2(generation); } return f1(generation); }
/* ----- end switch_function ----- */
/* ----- end of preserving & sampling functions ----- */

}/* ----- end namespace Sim_Model ----- */

#endif /* TEMPLATE_INLINE_SIMULATION_FUNCTORS_CUH_ */
