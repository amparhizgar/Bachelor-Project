#ifndef UTILsdfsdfhyfa_H
#define UTILsdfsdfhyfa_H

#include "cuda_runtime.h"
#include <thrust/device_vector.h>
#include <thrust/functional.h>

#include <stdio.h>
#include <iostream>
#include <cmath>

#define BLOCK_SIZE 4

const double pi = 3.141592653589793;

void print2DArray(const double* array, int n, int m);
void print2DArray(const double* array, int n);
void writeToFile(const double* array, int n, int m, int p, std::string algname, int time, int iterations);

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

struct Result {
	thrust::device_vector<double>* soloution;
	const double error;
	const long iterations;

	Result(thrust::device_vector<double>* _soloution,double _error, long _iterations) : soloution(_soloution), error(_error), iterations(_iterations) {}
};

void checkForError();



#endif // UTILsdfsdfhyfa_H
