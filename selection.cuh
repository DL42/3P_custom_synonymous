/*
 * selection.cuh
 *
 *      Author: David Lawrie
 */

#ifndef SELECTION_CUH_
#define SELECTION_CUH_

namespace GO_Fish{

/* ----- constant selection model ----- */
const_selection::const_selection() : s(0) {}
const_selection::const_selection(float s) : s(s){ }
__device__ __forceinline__ float const_selection::operator()(const int population, const int generation, const float freq) const{ return s; }
/* ----- end constant selection model ----- */

/* ----- linear frequency dependent selection model ----- */
linear_frequency_dependent_selection::linear_frequency_dependent_selection() : slope(0), intercept(0) {}
linear_frequency_dependent_selection::linear_frequency_dependent_selection(float slope, float intercept) : slope(slope), intercept(intercept) { }
__device__ __forceinline__ float linear_frequency_dependent_selection::operator()(const int population, const int generation, const float freq) const{ return slope*freq+intercept; }
/* ----- end linear frequency dependent selection model ----- */

/* ----- seasonal selection model ----- */
seasonal_selection::seasonal_selection() : amplitude(0), frequency(0), phase(0), offset(0), generation_shift(0) {}
seasonal_selection::seasonal_selection(float frequency, float amplitude, float offset, float phase /*= 0*/, int generation_shift /*= 0*/) : amplitude(amplitude), frequency(frequency), phase(phase), offset(offset), generation_shift(generation_shift) {}
__device__ __forceinline__ float seasonal_selection::operator()(const int population, const int generation, const float freq) const{ return amplitude*sin(frequency*(generation-generation_shift) + phase) + offset;}
/* ----- end seasonal selection model ----- */

/* ----- population specific selection model ----- */
template <typename Functor_sel, typename Functor_sel_pop>
population_specific_selection<Functor_sel,Functor_sel_pop>::population_specific_selection() : pop(0), generation_shift(0) { s = Functor_sel(); s_pop = Functor_sel_pop(); }
template <typename Functor_sel, typename Functor_sel_pop>
population_specific_selection<Functor_sel,Functor_sel_pop>::population_specific_selection(Functor_sel s_in, Functor_sel_pop s_pop_in, int pop, int generation_shift /*= 0*/) : pop(pop), generation_shift(generation_shift){  s = s_in; s_pop = s_pop_in; }
template <typename Functor_sel, typename Functor_sel_pop>
__device__ __forceinline__ float population_specific_selection<Functor_sel,Functor_sel_pop>::operator()(const int population, const int generation, const float freq) const{
	if(pop == population) return s_pop(population, generation-generation_shift, freq);
	return s(population, generation-generation_shift, freq);
}
/* ----- end population specific selection model ----- */

/* ----- piecewise selection model ----- */
template <typename Functor_sel1, typename Functor_sel2>
piecewise_selection<Functor_sel1, Functor_sel2>::piecewise_selection() : inflection_point(0), generation_shift(0) { s1 = Functor_sel1(); s2 = Functor_sel2(); }
template <typename Functor_sel1, typename Functor_sel2>
piecewise_selection<Functor_sel1, Functor_sel2>::piecewise_selection(Functor_sel1 s1_in, Functor_sel2 s2_in, int inflection_point, int generation_shift /* = 0*/) : inflection_point(inflection_point), generation_shift(generation_shift) { s1 = s1_in; s2 = s2_in(); }
template <typename Functor_sel1, typename Functor_sel2>
__device__ __forceinline__ float piecewise_selection<Functor_sel1, Functor_sel2>::operator()(const int population, const int generation, const float freq) const{
	if(generation >= inflection_point+generation_shift){ return s2(population, generation-generation_shift, freq) ; }
	return s1(population, generation-generation_shift, freq);
};
/* ----- end piecewise selection model ----- */

}/* ----- end namespace GO_Fish ----- */

#endif /* SELECTION_CUH_ */
