/*
 * migration.cuh
 *
 *  Created on: Mar 20, 2016
 *      Author: dlawrie
 */

#ifndef MIGRATION_CUH_
#define MIGRATION_CUH_

struct const_migration
{
	float m;
	int num_pop;
	const_migration() : m(0), num_pop(0){ }
	const_migration(int n) : m(0), num_pop(n){ }
	const_migration(float m, int n) : m(m), num_pop(n){ }
	__device__ __forceinline__ float operator()(const int pop_FROM, const int pop_TO, const int generation) const{
		if(pop_FROM == pop_TO){ return 1-(num_pop-1)*m; }
		return (num_pop > 1) * m;
	}
};


#endif /* MIGRATION_CUH_ */
