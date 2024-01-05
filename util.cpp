#include <stdio.h>
#include <iostream>

#include "util.h"
#include <fstream>
#include <sstream>

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

void writeToFile(const double* array, int n, int m, int p) {
	std::ofstream dimFile(".\\plot3d\\dim.csv");
	std::ofstream outputFile(".\\plot3d\\vector_data.csv");

	dimFile << n << "," << m << "," << p << std::endl;
	dimFile.close();

	std::stringstream rowString;

	for (int k = 0; k < p; ++k) {
		for (int j = 0; j < m; ++j) {
			for (int i = 0; i < n; ++i) {
				int index = k * n * m + j * n + i;
					/*if (index % 1000 == 0) {
						
					}*/
				outputFile << array[index] << ",";
			}
		}
	}

	outputFile.seekp(-1, std::ios_base::end);
	outputFile << std::endl;
	std::cout << std::endl;
	outputFile.close();
}



void checkForError()
{
	cudaError_t	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "\033[31mCuda error: %s\033[0m\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
}