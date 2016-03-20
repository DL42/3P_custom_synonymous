/*
 * migration.h
 *
 *  Created on: Mar 20, 2016
 *      Author: dlawrie
 */

#ifndef MIGRATION_H_
#define MIGRATION_H_

struct const_migration
{
	float m;
	int num_pop;
	const_migration();
	const_migration(int n);
	const_migration(float m, int n);
	__device__ __forceinline__ float operator()(const int pop_FROM, const int pop_TO, const int generation) const;
};

#include "migration.cuh"

#endif /* MIGRATION_H_ */
