#include <stdio.h>
#include <stdlib.h>
#include "matrix_lib.h"


int threads_per_block = THREADS_PER_BLOCK_DEFAULT;
int max_blocks_per_grid = MAX_BLOCKS_PER_GRID_DEFAULT;


int set_grid_size(int threads_per_block, int max_blocks_per_grid) {
    if (threads_per_block > THREADS_PER_BLOCK_LIMIT ||
        max_blocks_per_grid > MAX_BLOCKS_PER_GRID_LIMIT) {
            return 0;
        }

    threads_per_block = threads_per_block;
    max_blocks_per_grid = max_blocks_per_grid;

    return 1;
}


void initialize_matrix(struct matrix* matrix){
    for (unsigned long int i = 0; i < matrix->width*matrix->height; i++) {
        matrix->h_rows[i] = 0;
    }
    return;
}


int is_matrix_valid(struct matrix* matrix) {
    if (matrix == NULL || matrix->height <= 0 || matrix->width <= 0) {
        return 0;
    }

    return 1;
}


int is_matrix_mult_valid(struct matrix* matrixA, struct matrix* matrixB) {
    if (is_matrix_valid(matrixA) && is_matrix_valid(matrixB)) {
        if (matrixA->width == matrixB->height) return 1;
        
    }
    return 0;
}


__global__
void multiply_rows_by_scalar(int n, float scalar_value, float* d_rows){

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = index; i < n; i += stride) {
        d_rows[i] = d_rows[i] * scalar_value;
    }
    
    return;
}


int scalar_matrix_mult_gpu(float scalar_value, struct matrix *matrix) {
    if (!is_matrix_valid(matrix)) return 0;

    int chunk_size;
    int matrix_size = matrix->height * matrix->width;

    // all matrix
    if (matrix->alloc_mode == FULL_ALLOC) chunk_size = matrix_size;
    // one line from matrix
    else chunk_size = matrix->width;
     
    int num_loops = matrix_size / chunk_size;

    for (int i = 0; i < num_loops; i++) {

        cudaError = cudaMemcpy(matrix->d_rows, matrix->h_rows+(i*chunk_size), chunk_size*sizeof(float), cudaMemcpyHostToDevice);
        if (cudaError != cudaSuccess) {
            printf("cudaMemcpy returned error %s (code %d)\n",
            cudaGetErrorString(cudaError), cudaError);
            return 0;
        }

        multiply_rows_by_scalar<<<max_blocks_per_grid, threads_per_block>>>(chunk_size, scalar_value, matrix->d_rows);

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


__global__
void multiply_rows_by_rows(int n, float* d_rows_a, float* d_rows_b, float* d_rows_c, unsigned long int hA, unsigned long int wA,
                           unsigned long int hB, unsigned long int wB) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride) {
        for (int j = 0; j < wA; j++) {
            d_rows_c[i] += d_rows_a[wA*(i/hA) + j] * d_rows_b[(i%hA) + j*wB];
        }
        
    }
    
    return;
}


int matrix_matrix_mult_gpu(struct matrix *matrixA, struct matrix * matrixB, struct matrix * matrixC) {
    if (!is_matrix_mult_valid(matrixA, matrixB)) return 0;

    int matrix_size_A = matrixA->height * matrixA->width;
    int matrix_size_B = matrixB->height * matrixB->width;
    int matrix_size_C = matrixA->height * matrixB->width;
    int chunk_size_A;
    int chunk_size_C;
    int num_loops;

    if (matrixA->alloc_mode == FULL_ALLOC) {
        chunk_size_A = matrix_size_A;
        chunk_size_C = matrix_size_C;
    }
    else {
        // Only one line from matrixes A and C
        chunk_size_A = matrixA->width;
        chunk_size_C = matrixC->width;
    }

    num_loops = matrix_size_A / chunk_size_A;

    // Copying full matrix B to GPU
    cudaError = cudaMemcpy(matrixB->d_rows, matrixB->h_rows, matrix_size_B*sizeof(float), cudaMemcpyHostToDevice);
    if (cudaError != cudaSuccess) {
        printf("cudaMemcpy returned error %s (code %d)\n",
        cudaGetErrorString(cudaError), cudaError);
        return 0;
    }
    
    // Filling matrix C with zeros
    initialize_matrix(matrixC);
    for (int i = 0; i < num_loops; i++) {
        
       // Copying matrix A to GPU (full or one line)
        cudaError = cudaMemcpy(matrixA->d_rows, matrixA->h_rows+(i*chunk_size_A), chunk_size_A*sizeof(float), cudaMemcpyHostToDevice);
        if (cudaError != cudaSuccess) {
            printf("cudaMemcpy returned error %s (code %d)\n",
            cudaGetErrorString(cudaError), cudaError);
            return 0;
        }

        // Copying matrix C to GPU (full or one line)
        cudaError = cudaMemcpy(matrixC->d_rows, matrixC->h_rows+(i*chunk_size_C), chunk_size_C*sizeof(float), cudaMemcpyHostToDevice);
        if (cudaError != cudaSuccess) {
            printf("cudaMemcpy returned error %s (code %d)\n",
            cudaGetErrorString(cudaError), cudaError);
            return 0;
        }

        multiply_rows_by_rows<<<max_blocks_per_grid, threads_per_block>>>(chunk_size_A, matrixA->d_rows, matrixB->d_rows, matrixC->d_rows, matrixA->height, matrixA->width,
                                                                        matrixB->height, matrixB->width);

        cudaDeviceSynchronize();

        // Copiando matrix C from GPU to CPU (full or one line)
        cudaError = cudaMemcpy(matrixC->h_rows+(i*chunk_size_C), matrixC->d_rows, chunk_size_C*sizeof(float), cudaMemcpyDeviceToHost);
        if (cudaError != cudaSuccess) {
            printf("cudaMemcpy returned error %s (code %d)\n",
            cudaGetErrorString(cudaError), cudaError);
            return 0;
        }
    }

    return 1;
}