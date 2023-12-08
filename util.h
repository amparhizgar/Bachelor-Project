#pragma once

#include "cuda_runtime.h"
#include <thrust/functional.h>

#include <stdio.h>
#include <iostream>
#include <cmath>


void print2DArray(const double* array, int n, int m);
void print2DArray(const double* array, int n);

struct abs_difference
{
	__host__ __device__
		double operator()(const thrust::tuple<double, double>& t) const {
		return fabs(thrust::get<0>(t) - thrust::get<1>(t));
	}
};
