#pragma once

#include "cuda_runtime.h"
#include <thrust/device_vector.h>
#include <thrust/functional.h>

#include <stdio.h>
#include <iostream>
#include <cmath>

#define BLOCK_SIZE 16

void print2DArray(const double* array, int n, int m);
void print2DArray(const double* array, int n);

struct abs_difference
{
	__host__ __device__
		double operator()(const thrust::tuple<double, double>& t) const {
		return fabs(thrust::get<0>(t) - thrust::get<1>(t));
	}
};

struct AbsoluteValue {
	__host__ __device__
		double operator()(const double& x) const {
		return fabs(x);
	}
};
