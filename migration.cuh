/*
 * migration.cuh
 *
 *      Author: David Lawrie
 */

#ifndef MIGRATION_CUH_
#define MIGRATION_CUH_

GO_Fish::const_migration::const_migration() : m(0), num_pop(0){ }
GO_Fish::const_migration::const_migration(int n) : m(0), num_pop(n){ }
GO_Fish::const_migration::const_migration(float m, int n) : m(m), num_pop(n){ }
__host__ __device__ __forceinline__ float GO_Fish::const_migration::operator()(const int pop_FROM, const int pop_TO, const int generation) const{
		if(pop_FROM == pop_TO){ return 1-(num_pop-1)*m; }
		return (num_pop > 1) * m;
	}

#endif /* MIGRATION_CUH_ */
