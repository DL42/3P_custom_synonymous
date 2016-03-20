/*
 * sample.h
 *
 *  Created on: Mar 20, 2016
 *      Author: dlawrie
 */

#ifndef SAMPLE_H_
#define SAMPLE_H_

struct no_sample
{
	__host__ __forceinline__ int operator()(const int generation) const;
};

#include "sample.cuh"

#endif /* SAMPLE_H_ */
