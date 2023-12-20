#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/fill.h>
#include "util.h"
#include <chrono>
#include <iomanip>

extern void jacubi();
extern void jacubi_redblack();
extern void sor();
extern void sor_separated();
extern thrust::device_vector<double>* conjugate_gradient(thrust::device_vector<double>& u, int n, int m, ConvergenceCriteria cc);


int main() {
	int n = 1000;
	int m = 1000;
	int size = n * m;

	thrust::device_vector<double> u(size, 0);
	thrust::fill(u.begin(), u.begin() + n, 2.0);
	thrust::fill(u.begin() + n * (n - 1), u.end(), 1.0);
	thrust::device_vector<double>* (*algorithm)(thrust::device_vector<double>&, int, int, ConvergenceCriteria);

	ConvergenceCriteria cc(0.0, 1000);

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

	auto start = std::chrono::high_resolution_clock::now();

	thrust::device_vector<double> result = *algorithm(u, n, m, cc);

	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	
	std::cout.imbue(std::locale(""));
	// Print the duration with commas as thousands separators
	std::cout << "Time taken by the algorithm: "
		<< std::fixed << std::setprecision(3) << duration.count() / 1000.0 << " milliseconds"
		<< std::endl;


	return 0;
}
