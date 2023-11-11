#define FULL_ALLOC 1
#define PARTIAL_ALLOC 0
#define THREADS_PER_BLOCK_DEFAULT 256
#define MAX_BLOCKS_PER_GRID_DEFAULT 4096
#define THREADS_PER_BLOCK_LIMIT 1024
#define MAX_BLOCKS_PER_GRID_LIMIT 65535

struct matrix {
    unsigned long int height;
    unsigned long int width;
    float* h_rows;
    float* d_rows;
    int alloc_mode;
};

int set_grid_size(int threads_per_block, int max_blocks_per_grid);

int scalar_matrix_mult_gpu(float scalar_value, struct matrix *matrix);

int matrix_matrix_mult_gpu(struct matrix *matrixA, struct matrix * matrixB, struct matrix * matrixC);