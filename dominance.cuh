/*
 * dominance.cuh
 *
 *      Author: David Lawrie
 */

#ifndef DOMINANCE_CUH_
#define DOMINANCE_CUH_

namespace GO_Fish{

const_dominance::const_dominance() : h(0) {}
const_dominance::const_dominance(float h) : h(h){ }
__host__ __forceinline__ float const_dominance::operator()(const int population, const int generation) const{ return h; }

/* ----- population specific dominance model ----- */
template <typename Functor_h, typename Functor_h_pop>
population_specific_dominance<Functor_h,Functor_h_pop>::population_specific_dominance() : pop(0), generation_shift(0) { h = Functor_h(); h_pop = Functor_h_pop(); }
template <typename Functor_h, typename Functor_h_pop>
population_specific_dominance<Functor_h,Functor_h_pop>::population_specific_dominance(Functor_h h_in, Functor_h_pop h_pop_in, int pop, int generation_shift /*= 0*/) : pop(pop), generation_shift(generation_shift){  h = h_in; h_pop = h_pop_in; }
template <typename Functor_h, typename Functor_h_pop>
__device__ __forceinline__ float population_specific_dominance<Functor_h,Functor_h_pop>::operator()(const int population, const int generation, const float freq) const{
	if(pop == population) return h_pop(population, generation-generation_shift, freq);
	return h(population, generation-generation_shift, freq);
}
/* ----- end population specific dominance model ----- */

/* ----- piecewise dominance model ----- */
template <typename Functor_h1, typename Functor_h2>
piecewise_dominance<Functor_h1, Functor_h2>::piecewise_dominance() : inflection_point(0), generation_shift(0) { h1 = Functor_h1(); h2 = Functor_h2(); }
template <typename Functor_h1, typename Functor_h2>
piecewise_dominance<Functor_h1, Functor_h2>::piecewise_dominance(Functor_h1 h1_in, Functor_h2 h2_in, int inflection_point, int generation_shift /* = 0*/) : inflection_point(inflection_point), generation_shift(generation_shift) { h1 = h1_in; h2 = h2_in(); }
template <typename Functor_h1, typename Functor_h2>
__device__ __forceinline__ float piecewise_dominance<Functor_h1, Functor_h2>::operator()(const int population, const int generation, const float freq) const{
	if(generation >= inflection_point+generation_shift){ return h2(population, generation-generation_shift, freq) ; }
	return h1(population, generation-generation_shift, freq);
};
/* ----- end piecewise dominance model ----- */

}/* ----- end namespace GO_Fish ----- */

#endif /* DOMINANCE_CUH_ */
