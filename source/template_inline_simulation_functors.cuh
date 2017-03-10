/*
 * template_simulation_functors.cuh
 *
 *      Author: David Lawrie
 *      implementation of template and inline functions for GO Fish evolutionary models and sampling schemes
 */

#ifndef TEMPLATE_INLINE_SIMULATION_FUNCTORS_CUH_
#define TEMPLATE_INLINE_SIMULATION_FUNCTORS_CUH_

namespace GO_Fish{

/* ----- selection models ----- */
/* ----- constant selection model ----- */
inline const_selection::const_selection() : s(0) {}
inline const_selection::const_selection(float s) : s(s){ }
template <typename Functor_demography, typename Functor_inbreeding>
inline const_selection::const_selection(float gamma, Functor_demography demography, Functor_inbreeding F, int forward_generation_shift /*= 0*/){ s = gamma/(2*demography(0,forward_generation_shift)/(1+F(0,forward_generation_shift))); }
__host__ __device__ __forceinline__ float const_selection::operator()(const int population, const int generation, const float freq) const{ return s; }
/* ----- end constant selection model ----- */

/* ----- linear frequency dependent selection model ----- */
inline linear_frequency_dependent_selection::linear_frequency_dependent_selection() : slope(0), intercept(0) {}
inline linear_frequency_dependent_selection::linear_frequency_dependent_selection(float slope, float intercept) : slope(slope), intercept(intercept) { }
template <typename Functor_demography, typename Functor_inbreeding>
inline linear_frequency_dependent_selection::linear_frequency_dependent_selection(float gamma_slope, float gamma_intercept, Functor_demography demography, Functor_inbreeding F, int forward_generation_shift /*= 0*/){
	slope = gamma_slope/(2*demography(0,forward_generation_shift)/(1+F(0,forward_generation_shift)));
	intercept = gamma_intercept/(2*demography(0,forward_generation_shift)/(1+F(0,forward_generation_shift)));
}
__host__ __device__ __forceinline__ float linear_frequency_dependent_selection::operator()(const int population, const int generation, const float freq) const{ return slope*freq+intercept; }
/* ----- end linear frequency dependent selection model ----- */

/* ----- seasonal selection model ----- */
inline seasonal_selection::seasonal_selection() : A(0), pi(0), rho(0), D(0), generation_shift(0) {}
inline seasonal_selection::seasonal_selection(float A, float pi, float D, float rho /*= 0*/, int generation_shift /*= 0*/) : A(A), pi(pi), rho(rho), D(D), generation_shift(generation_shift) {}
template <typename Functor_demography, typename Functor_inbreeding>
inline seasonal_selection::seasonal_selection(float gamma_A, float pi, float gamma_D, Functor_demography demography, Functor_inbreeding F, float rho /*= 0*/, int generation_shift /*= 0*/, int forward_generation_shift /*= 0*/) : pi(pi), rho(rho), generation_shift(generation_shift) {
	A = gamma_A/(2*demography(0,forward_generation_shift)/(1+F(0,forward_generation_shift)));
	D = gamma_D/(2*demography(0,forward_generation_shift)/(1+F(0,forward_generation_shift)));
}
__host__ __device__ __forceinline__ float seasonal_selection::operator()(const int population, const int generation, const float freq) const{ return A*sin(pi*(generation-generation_shift) + rho) + D;}
/* ----- end seasonal selection model ----- */

/* ----- population specific selection model ----- */
template <typename Functor_sel, typename Functor_sel_pop>
inline population_specific_selection<Functor_sel,Functor_sel_pop>::population_specific_selection() : pop(0), generation_shift(0) { s = Functor_sel(); s_pop = Functor_sel_pop(); }
template <typename Functor_sel, typename Functor_sel_pop>
inline population_specific_selection<Functor_sel,Functor_sel_pop>::population_specific_selection(Functor_sel s_in, Functor_sel_pop s_pop_in, int pop, int generation_shift /*= 0*/) : pop(pop), generation_shift(generation_shift){  s = s_in; s_pop = s_pop_in; }
template <typename Functor_sel, typename Functor_sel_pop>
__host__ __device__ __forceinline__ float population_specific_selection<Functor_sel,Functor_sel_pop>::operator()(const int population, const int generation, const float freq) const{
	if(pop == population) return s_pop(population, generation-generation_shift, freq);
	return s(population, generation-generation_shift, freq);
}
/* ----- end population specific selection model ----- */

/* ----- piecewise selection model ----- */
template <typename Functor_sel1, typename Functor_sel2>
inline piecewise_selection<Functor_sel1, Functor_sel2>::piecewise_selection() : inflection_point(0), generation_shift(0) { s1 = Functor_sel1(); s2 = Functor_sel2(); }
template <typename Functor_sel1, typename Functor_sel2>
inline piecewise_selection<Functor_sel1, Functor_sel2>::piecewise_selection(Functor_sel1 s1_in, Functor_sel2 s2_in, int inflection_point, int generation_shift /* = 0*/) : inflection_point(inflection_point), generation_shift(generation_shift) { s1 = s1_in; s2 = s2_in; }
template <typename Functor_sel1, typename Functor_sel2>
__host__ __device__ __forceinline__ float piecewise_selection<Functor_sel1, Functor_sel2>::operator()(const int population, const int generation, const float freq) const{
	if(generation >= inflection_point+generation_shift){ return s2(population, generation-generation_shift, freq) ; }
	return s1(population, generation-generation_shift, freq);
};
/* ----- end piecewise selection model ----- */
/* ----- end selection models ----- */

/* ----- mutation, dominance, & inbreeding models ----- */
/* ----- constant parameter model ----- */
inline const_parameter::const_parameter() : p(0) {}
inline const_parameter::const_parameter(float p) : p(p){ }
__host__ __forceinline__ float const_parameter::operator()(const int population, const int generation) const{ return p; }
/* ----- end constant parameter model ----- */

/* ----- seasonal parameter model ----- */
inline seasonal_parameter::seasonal_parameter() : A(0), pi(0), rho(0), D(0), generation_shift(0) {}
inline seasonal_parameter::seasonal_parameter(float A, float pi, float D, float rho /*= 0*/, int generation_shift /*= 0*/) : A(A), pi(pi), rho(rho), D(D), generation_shift(generation_shift) {}
__host__ __forceinline__ float seasonal_parameter::operator()(const int population, const int generation) const{ return A*sin(pi*(generation-generation_shift) + rho) + D;}
/* ----- end seasonal parameter model ----- */

/* ----- population specific parameter model ----- */
template <typename Functor_p, typename Functor_p_pop>
inline population_specific_parameter<Functor_p,Functor_p_pop>::population_specific_parameter() : pop(0), generation_shift(0) { p = Functor_p(); p_pop = Functor_p_pop(); }
template <typename Functor_p, typename Functor_p_pop>
inline population_specific_parameter<Functor_p,Functor_p_pop>::population_specific_parameter(Functor_p p_in, Functor_p_pop p_pop_in, int pop, int generation_shift /*= 0*/) : pop(pop), generation_shift(generation_shift){  p = p_in; p_pop = p_pop_in; }
template <typename Functor_p, typename Functor_p_pop>
__host__ __forceinline__ float population_specific_parameter<Functor_p,Functor_p_pop>::operator()(const int population, const int generation) const{
	if(pop == population) return p_pop(population, generation-generation_shift);
	return p(population, generation-generation_shift);
}
/* ----- end population specific parameter model ----- */

/* ----- piecewise parameter model ----- */
template <typename Functor_p1, typename Functor_p2>
inline piecewise_parameter<Functor_p1, Functor_p2>::piecewise_parameter() : inflection_point(0), generation_shift(0) { p1 = Functor_p1(); p2 = Functor_p2(); }
template <typename Functor_p1, typename Functor_p2>
inline piecewise_parameter<Functor_p1, Functor_p2>::piecewise_parameter(Functor_p1 p1_in, Functor_p2 p2_in, int inflection_point, int generation_shift /* = 0*/) : inflection_point(inflection_point), generation_shift(generation_shift) { p1 = p1_in; p2 = p2_in; }
template <typename Functor_p1, typename Functor_p2>
__host__ __forceinline__ float piecewise_parameter<Functor_p1, Functor_p2>::operator()(const int population, const int generation) const{
	if(generation >= inflection_point+generation_shift){ return p2(population, generation-generation_shift) ; }
	return p1(population, generation-generation_shift);
};
/* ----- end piecewise parameter model ----- */
/* ----- end of mutation, dominance, & inbreeding models ----- */

/* ----- demography models ----- */
/* ----- constant demography model ----- */
inline const_demography::const_demography() : p(0) {}
inline const_demography::const_demography(int p) : p(p){ }
__host__ __device__  __forceinline__ int const_demography::operator()(const int population, const int generation) const{ return p; }
/* ----- end constant demography model ----- */

/* ----- seasonal demography model ----- */
inline seasonal_demography::seasonal_demography() : A(0), pi(0), rho(0), D(0), generation_shift(0) {}
inline seasonal_demography::seasonal_demography(float A, float pi, int D, float rho /*= 0*/, int generation_shift /*= 0*/) : A(A), pi(pi), rho(rho), D(D), generation_shift(generation_shift) {}
__host__ __device__  __forceinline__ int seasonal_demography::operator()(const int population, const int generation) const{ return (int)A*sin(pi*(generation-generation_shift) + rho) + D;}
/* ----- end seasonal parameter model ----- */

/* ----- exponential growth model ----- */
inline exponential_growth::exponential_growth() : rate(0), initial_population_size(0), generation_shift(0) {}
inline exponential_growth::exponential_growth(float rate, int initial_population_size, int generation_shift /*= 0*/) : rate(rate), initial_population_size(initial_population_size), generation_shift(generation_shift) {}
__host__ __device__  __forceinline__ int exponential_growth::operator()(const int population, const int generation) const{ return (int)round(initial_population_size*exp(rate*(generation-generation_shift))); }
/* ----- end exponential growth model ----- */

/* ----- logistic growth model ----- */
inline logistic_growth::logistic_growth() : rate(0), initial_population_size(0), carrying_capacity(0), generation_shift(0) {}
inline logistic_growth::logistic_growth(float rate, int initial_population_size, int carrying_capacity, int generation_shift /*= 0*/) : rate(rate), initial_population_size(initial_population_size), carrying_capacity(carrying_capacity), generation_shift(generation_shift) {}
__host__ __device__  __forceinline__ int logistic_growth::operator()(const int population, const int generation) const{
	float term = exp(rate*(generation-generation_shift));
	return (int)round(carrying_capacity*initial_population_size*term/(carrying_capacity + initial_population_size*(term-1)));
}
/* ----- end logistic growth model ----- */

/* ----- population specific demography model ----- */
template <typename Functor_p, typename Functor_p_pop>
inline population_specific_demography<Functor_p,Functor_p_pop>::population_specific_demography() : pop(0), generation_shift(0) { p = Functor_p(); p_pop = Functor_p_pop(); }
template <typename Functor_p, typename Functor_p_pop>
inline population_specific_demography<Functor_p,Functor_p_pop>::population_specific_demography(Functor_p p_in, Functor_p_pop p_pop_in, int pop, int generation_shift /*= 0*/) : pop(pop), generation_shift(generation_shift){  p = p_in; p_pop = p_pop_in; }
template <typename Functor_p, typename Functor_p_pop>
__host__ __device__  __forceinline__ int population_specific_demography<Functor_p,Functor_p_pop>::operator()(const int population, const int generation) const{
	if(pop == population) return p_pop(population, generation-generation_shift);
	return p(population, generation-generation_shift);
}
/* ----- end population specific demography model ----- */

/* ----- piecewise demography model ----- */
template <typename Functor_p1, typename Functor_p2>
inline piecewise_demography<Functor_p1, Functor_p2>::piecewise_demography() : inflection_point(0), generation_shift(0) { p1 = Functor_p1(); p2 = Functor_p2(); }
template <typename Functor_p1, typename Functor_p2>
inline piecewise_demography<Functor_p1, Functor_p2>::piecewise_demography(Functor_p1 p1_in, Functor_p2 p2_in, int inflection_point, int generation_shift /* = 0*/) : inflection_point(inflection_point), generation_shift(generation_shift) { p1 = p1_in; p2 = p2_in; }
template <typename Functor_p1, typename Functor_p2>
__host__ __device__  __forceinline__ int piecewise_demography<Functor_p1, Functor_p2>::operator()(const int population, const int generation) const{
	if(generation >= inflection_point+generation_shift){ return p2(population, generation-generation_shift) ; }
	return p1(population, generation-generation_shift);
};
/* ----- end piecewise demography model ----- */

/* ----- end of demography models ----- */

/* ----- migration models ----- */
/* ----- constant equal migration model ----- */
inline const_equal_migration::const_equal_migration() : m(0), num_pop(0){ }
inline const_equal_migration::const_equal_migration(int n) : m(0), num_pop(n){ }
inline const_equal_migration::const_equal_migration(float m, int n) : m(m), num_pop(n){ }
__host__ __device__ __forceinline__ float const_equal_migration::operator()(const int pop_FROM, const int pop_TO, const int generation) const{
		if(pop_FROM == pop_TO){ return 1-(num_pop-1)*m; }
		return (num_pop > 1) * m;
}
/* ----- end constant equal migration model ----- */

/* ----- constant directional migration model ----- */
template <typename Functor_m1>
inline const_directional_migration<Functor_m1>::const_directional_migration() : m(0), pop1(0), pop2(0) { rest = Functor_m1(); }
template <typename Functor_m1>
inline const_directional_migration<Functor_m1>::const_directional_migration(float m, int pop1, int pop2, Functor_m1 rest_in) : m(m), pop1(pop1), pop2(pop2) { rest = rest_in; }
template <typename Functor_m1>
__host__ __device__ __forceinline__ float const_directional_migration<Functor_m1>::operator()(const int pop_FROM, const int pop_TO, const int generation) const{
	if(pop_FROM == pop1 && pop_TO == pop2) return m;
	return rest(pop_FROM, pop_TO, generation);
}
/* ----- end constant directional migration model ----- */

/* ----- piecewise migration model ----- */
template <typename Functor_m1, typename Functor_m2>
inline piecewise_migration<Functor_m1,Functor_m2>::piecewise_migration() : inflection_point(0), generation_shift(0) { m1 = Functor_m1(); m2 = Functor_m2(); }
template <typename Functor_m1, typename Functor_m2>
inline piecewise_migration<Functor_m1,Functor_m2>::piecewise_migration(Functor_m1 m1_in, Functor_m2 m2_in, int inflection_point, int generation_shift /*= 0*/) : inflection_point(inflection_point), generation_shift(generation_shift) { m1 = m1_in; m2 = m2_in; }
template <typename Functor_m1, typename Functor_m2>
__host__ __device__ __forceinline__ int piecewise_migration<Functor_m1,Functor_m2>::operator()(const int pop_FROM, const int pop_TO, const int generation) const{
	if(generation >= inflection_point+generation_shift){ return m2(pop_FROM,pop_TO,generation); }
	return m1(pop_FROM,pop_TO,generation);
}
/* ----- end piecewise migration model ----- */
/* ----- end of migration models ----- */

/* ----- preserving & sampling functions ----- */
/* ----- off ----- */
__host__ __forceinline__ bool off::operator()(const int generation) const{ return false; }
/* ----- end off ----- */

/* ----- on ----- */
__host__ __forceinline__ bool on::operator()(const int generation) const{ return true; }
/* ----- end on ----- */

/* ----- on_off_array ----- */
//on_off_array::on_off_array(): length(0), generation_shift(0) { array = NULL; }
//on_off_array::on_off_array(const bool * const in_array, int length, int generation_shift/* = 0*/): length(length), generation_shift(generation_shift) { array = in_array; }
/*__host__ __forceinline__ bool on_off_array::operator()(const int generation) const {
	int gen = generation - generation_shift;
	if((gen < 0) | (gen > length)){ fprintf(stderr,"do_array functor generation error,\t generation %d\t shifted generation %d\t array length %d\n",generation,gen,length); exit(1); }
	return array[gen];
}*/
/* ----- end on_off_array ----- */

/* ----- pulse ----- */
template <typename Functor_stable, typename Functor_action>
inline pulse<Functor_stable,Functor_action>::pulse() : Fgen(0), generation_shift(0) { f1 = Functor_stable(); f2 = Functor_action(); }
template <typename Functor_stable, typename Functor_action>
inline pulse<Functor_stable,Functor_action>::pulse(Functor_stable f1_in, Functor_action f2_in, int Fgen, int generation_shift/*= 0*/) : Fgen(Fgen), generation_shift(generation_shift) { f1 = f1_in; f2 = f2_in; }
template <typename Functor_stable, typename Functor_action>
__host__ __forceinline__ bool pulse<Functor_stable,Functor_action>::operator()(const int generation) const{ if(generation-generation_shift == Fgen){ return f2(generation); } return f1(generation); }
/* ----- end pulse ----- */

/* ----- switch_function ----- */
template <typename Functor_first, typename Functor_second>
inline switch_function<Functor_first,Functor_second>::switch_function() : Fgen(0), generation_shift(0) { f1 = Functor_first(); f2 = Functor_second(); }
template <typename Functor_first, typename Functor_second>
inline switch_function<Functor_first,Functor_second>::switch_function(Functor_first f1_in, Functor_second f2_in, int Fgen, int generation_shift /*= 0*/) : Fgen(Fgen), generation_shift(generation_shift) { f1 = f1_in; f2 = f2_in; }
template <typename Functor_first, typename Functor_second>
__host__ __forceinline__ bool switch_function<Functor_first,Functor_second>::operator()(const int generation) const{ if(generation-generation_shift < Fgen){ return f1(generation); } return f2(generation); }
/* ----- end switch_function ----- */
/* ----- end of preserving & sampling functions ----- */

}/* ----- end namespace GO_Fish ----- */

#endif /* TEMPLATE_INLINE_SIMULATION_FUNCTORS_CUH_ */
