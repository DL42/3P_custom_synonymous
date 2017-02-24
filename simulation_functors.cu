/*
 * simulation_functors.cu
 *
 *      Author: David Lawrie
 *      implementation of non-template and non-inline functions for GO Fish evolutionary scenarios
 */

#include "go_fish.h"

namespace GO_Fish{

/* ----- selection models ----- */
/* ----- constant selection model ----- */
const_selection::const_selection() : s(0) {}
const_selection::const_selection(float s) : s(s){ }
/* ----- end constant selection model ----- */

/* ----- linear frequency dependent selection model ----- */
linear_frequency_dependent_selection::linear_frequency_dependent_selection() : slope(0), intercept(0) {}
linear_frequency_dependent_selection::linear_frequency_dependent_selection(float slope, float intercept) : slope(slope), intercept(intercept) { }
/* ----- end linear frequency dependent selection model ----- */

/* ----- seasonal selection model ----- */
seasonal_selection::seasonal_selection() : A(0), pi(0), rho(0), D(0), generation_shift(0) {}
seasonal_selection::seasonal_selection(float A, float pi, float D, float rho /*= 0*/, int generation_shift /*= 0*/) : A(A), pi(pi), rho(rho), D(D), generation_shift(generation_shift) {}
/* ----- end seasonal selection model ----- */
/* ----- end selection models ----- */

/* ----- mutation, dominance, & inbreeding models ----- */
/* ----- constant parameter model ----- */
const_parameter::const_parameter() : p(0) {}
const_parameter::const_parameter(float p) : p(p){ }
/* ----- end constant parameter model ----- */

/* ----- seasonal parameter model ----- */
seasonal_parameter::seasonal_parameter() : A(0), pi(0), rho(0), D(0), generation_shift(0) {}
seasonal_parameter::seasonal_parameter(float A, float pi, float D, float rho /*= 0*/, int generation_shift /*= 0*/) : A(A), pi(pi), rho(rho), D(D), generation_shift(generation_shift) {}
/* ----- end seasonal parameter model ----- */
/* ----- end of mutation, dominance, & inbreeding models ----- */

/* ----- demography models ----- */
/* ----- constant demography model ----- */
const_demography::const_demography() : p(0) {}
const_demography::const_demography(int p) : p(p){ }
/* ----- end constant demography model ----- */

/* ----- seasonal demography model ----- */
seasonal_demography::seasonal_demography() : A(0), pi(0), rho(0), D(0), generation_shift(0) {}
seasonal_demography::seasonal_demography(float A, float pi, int D, float rho /*= 0*/, int generation_shift /*= 0*/) : A(A), pi(pi), rho(rho), D(D), generation_shift(generation_shift) {}
/* ----- end seasonal parameter model ----- */

/* ----- exponential growth model ----- */
exponential_growth::exponential_growth() : rate(0), initial_population_size(0), generation_shift(0) {}
exponential_growth::exponential_growth(float rate, int initial_population_size, int generation_shift /*= 0*/) : rate(rate), initial_population_size(initial_population_size), generation_shift(generation_shift) {}
/* ----- end exponential growth model ----- */

/* ----- logistic growth model ----- */
logistic_growth::logistic_growth() : rate(0), initial_population_size(0), carrying_capacity(0), generation_shift(0) {}
logistic_growth::logistic_growth(float rate, int initial_population_size, int carrying_capacity, int generation_shift /*= 0*/) : rate(rate), initial_population_size(initial_population_size), carrying_capacity(carrying_capacity), generation_shift(generation_shift) {}
/* ----- end logistic growth model ----- */
/* ----- end of demography models ----- */

/* ----- migration models ----- */
/* ----- constant equal migration model ----- */
const_equal_migration::const_equal_migration() : m(0), num_pop(0){ }
const_equal_migration::const_equal_migration(int n) : m(0), num_pop(n){ }
const_equal_migration::const_equal_migration(float m, int n) : m(m), num_pop(n){ }
/* ----- end constant equal migration model ----- */
/* ----- end of migration models ----- */

/* ----- preserving & sampling functions ----- */
do_array::do_array(): length(0), generation_shift(0) { array = NULL; }
do_array::do_array(const bool * const in_array, int length, int generation_shift/* = 0*/): length(length), generation_shift(generation_shift) { array = in_array; }
/* ----- end of preserving & sampling functions ----- */

}
