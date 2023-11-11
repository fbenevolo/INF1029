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

    printf("%d %d\n", threads_per_block, max_blocks_per_grid);
    return 1;
}


int scalar_matrix_mult_gpu(float scalar_value, struct matrix *matrix) {


    return 1;
}


int matrix_matrix_mult_gpu(struct matrix *matrixA, struct matrix * matrixB, struct matrix * matrixC) {


    return 1;
}