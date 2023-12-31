#include <stdio.h>
#include <iostream>

#include "util.h"


void print2DArray(const double* array, int n, int m) {
	for (int j = 0; j < m; ++j) {
		for (int i = 0; i < n; ++i) {
			std::cout << std::fixed << array[j * n + i] << "   ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}


void print2DArray(const double* array, int n) {
	print2DArray(array, n, n);
}

void checkForError()
{
	cudaError_t	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "\033[31mCuda error: %s\033[0m\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
}