#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/fill.h>
#include "util.h"

extern void jacubi();
extern void jacubi_redblack();
extern void sor();
extern void sor_separated();
extern thrust::device_vector<double>* conjugate_gradient(thrust::device_vector<double>& u, int n, int m);


int main() {
	int n = 5;
	int m = 5;
	int size = n * m;

	thrust::device_vector<double> u(size, 0);
	thrust::fill(u.begin(), u.begin() + n, 2.0);
	thrust::fill(u.begin() + n * (n - 1), u.end(), 1.0);
	thrust::device_vector<double>* (*algorithm)(thrust::device_vector<double>&, int, int);

	switch (4)
	{
	case 0:
		jacubi();
		break;
	case 1:
		jacubi_redblack();
		break;
	case 2:
		sor();
		break;
	case 3:
		sor_separated();
		break;
	case 4:
		algorithm = &conjugate_gradient;
		break;
	default:
		break;
	}

	thrust::device_vector<double> result = *algorithm(u, n, m);

	return 0;
}
