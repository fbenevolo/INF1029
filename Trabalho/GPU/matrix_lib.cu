#include <stdio.h>
#include <stdlib.h>
#include "matrix_lib.h"


int threads_per_block = THREADS_PER_BLOCK_DEFAULT;
int max_blocks_per_grid = MAX_BLOCKS_PER_GRID_DEFAULT;
int deviceSize;

cudaError_t cudaError;

void set_device_size(int device_size) {
    // divide by sizeof(float) to get number of elements from the number of bytes
    deviceSize = device_size / sizeof(float);
}


int set_grid_size(int threads_per_block, int max_blocks_per_grid) {
    if (threads_per_block > THREADS_PER_BLOCK_LIMIT ||
        max_blocks_per_grid > MAX_BLOCKS_PER_GRID_LIMIT) {
            return 0;
        }

    threads_per_block = threads_per_block;
    max_blocks_per_grid = max_blocks_per_grid;

    return 1;
}


int is_matrix_valid(struct matrix* matrix) {
    if (matrix == NULL || matrix->height <= 0 || matrix->width <= 0) {
        return 0;
    }
    return 1;
}


__global__
void multiply_rows(int n, float scalar_value, float* d_rows){

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = index; i < n; i += stride) {
        d_rows[index] = d_rows[index] * scalar_value;
    }
    
    return;
}


int scalar_matrix_mult_gpu(float scalar_value, struct matrix *matrix) {
    if (!is_matrix_valid(matrix)) return 0;

    int matrix_size = matrix->height * matrix->width;
    int num_loops = (matrix_size + deviceSize - 1) / deviceSize;
    int chunk_size = deviceSize;
    int block_size;
    int num_blocks;

    for (int i = 0; i < num_loops; i++) {
        if (i == num_loops - 1 && matrix_size % deviceSize) {
            chunk_size = matrix_size % deviceSize;
        }

        cudaError = cudaMemcpy(matrix->d_rows, matrix->h_rows+(i*chunk_size), chunk_size*sizeof(float), cudaMemcpyHostToDevice);
        if (cudaError != cudaSuccess) {
            printf("cudaMemcpy returned error %s (code %d)\n",
            cudaGetErrorString(cudaError), cudaError);
            return 0;
        }

        block_size = threads_per_block;
        num_blocks = (matrix_size + block_size - 1) / block_size;
        if (num_blocks > max_blocks_per_grid){
            num_blocks = max_blocks_per_grid;
        }

        multiply_rows<<<MAX_BLOCKS_PER_GRID_DEFAULT, THREADS_PER_BLOCK_DEFAULT>>>(chunk_size, scalar_value, matrix->d_rows);

        cudaDeviceSynchronize();

        cudaError = cudaMemcpy(matrix->h_rows+(i*chunk_size), matrix->d_rows, chunk_size*sizeof(float), cudaMemcpyDeviceToHost);
        if (cudaError != cudaSuccess) {
            printf("cudaMemcpy returned error %s (code %d)\n",
            cudaGetErrorString(cudaError), cudaError);
            return 0;
        }
    }
    
    return 1;
}


int matrix_matrix_mult_gpu(struct matrix *matrixA, struct matrix * matrixB, struct matrix * matrixC) {


    return 1;
}