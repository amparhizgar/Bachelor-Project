#pragma once

#include "cuda_runtime.h"
#include <thrust/device_vector.h>
#include <thrust/functional.h>

#include <stdio.h>
#include <iostream>
#include <cmath>

#define BLOCK_SIZE 32

const double pi = 3.141592653589793;

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

struct ConvergenceCriteria {
	const double error;
	const long iterations;

	ConvergenceCriteria(double _error, long _iterations) : error(_error), iterations(_iterations) {}

	bool hasConverged(double currentError, long currentIteration) {
		if (error > 0 && iterations > 0)
			return currentError < error || currentIteration >= iterations;
		else if (error > 0)
			return currentError < error;
		else
			return  currentIteration >= iterations;
	}
};

