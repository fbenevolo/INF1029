#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "matrix_lib.h"
extern "C" {
    #include "timer.h"
}

#define NUM_ARGS 13
#define MAX_DISPLAY 256
#define MiB 1048576

/* Headers of auxiliary functions */
struct matrix* inicialize_matrix(unsigned long int width, unsigned long int height,
                                 int alloc_mode);
struct matrix* inicialize_empty_matrix(unsigned long int width, unsigned long int height,
                                       int alloc_modes);
int load_matrix(struct matrix* matrix, FILE* file);
int display_matrix(struct matrix* matrix);


int main(int argc, char* argv[]) {

    float scalar;
    int device_max_size,
        wA, hA, wB, hB, wC, hC,
        min_alloc, max_alloc, bool_alloc;
    struct matrix* matrixA, * matrixB, * matrixC;
    struct timeval start, stop, start_program, stop_program;  
    FILE* file_matrix_A, * file_matrix_B, * file_result1, * file_result2; 

    // Getting time of program execution start
    gettimeofday(&start_program, NULL);

    // Setting scalar value
    scalar = atoi(argv[1]);

    // Setting width-height to each matrix 
    wA = atoi(argv[2]);
    hA = wC = atoi(argv[3]);
    wB = hC = atoi(argv[4]);
    hB = atoi(argv[5]);

    // Setting minimum and maximum allocations sizes
    min_alloc = ((wA + (wB * hB) + wC) * sizeof(float));
    max_alloc = (((wA * hA) + (wB * hB) + (wC * hC)) * sizeof(float));

    // Checking if threads per block and max blocks per grid args were received
    int received_gpu_info = 0;
    if (argc == NUM_ARGS) {
        received_gpu_info = 2;
        // threads_per_block > 1024 or max_blocs_per_grid > 65535 
        if ((set_grid_size(atoi(argv[6]), atoi(argv[7])) == 0)) {
            printf("Threads per block or maximum blocks per grid values were bigger than what is supported."
                    "The values will be set to default\n");
         }
    }
    else {
         printf("Threads per block and maximum blocks per grid values were not received."
                    "The values will be set to default\n");
    }

    // Defining device maximum size
    device_max_size  = atoi(argv[6+received_gpu_info]) * MiB;

    // Check if devide allows allocation of matrixes
    // Full allocation
    if (device_max_size >= max_alloc) {
        printf("Could do full alocation - device memory admits full"
        " matrixes A, B and C\n\n");
        bool_alloc = FULL_ALLOC;
    }
    // Partial allocation
    else if (device_max_size >= min_alloc) {
        printf("Could only do partial alocation - device memory admits only"
        " one line of A, one line of C and full B\n\n");
        bool_alloc = PARTIAL_ALLOC;
    }
    // Can't do partial allocation
    else  {
        perror("Can't do minimum allocation on GPU device\n\n");
        exit(3);
    }

    // Opening files
    file_matrix_A = fopen(argv[7+received_gpu_info], "rb");
    file_matrix_B = fopen(argv[8+received_gpu_info], "rb");
    file_result1  = fopen(argv[9+received_gpu_info], "wb");
    file_result2  = fopen(argv[10+received_gpu_info], "wb");

    // Initializing matrixes
    matrixA = inicialize_matrix(wA, hA, bool_alloc);
    matrixB = inicialize_matrix(wB, hB, FULL_ALLOC);
    matrixC = inicialize_empty_matrix(wC, hC, bool_alloc); 

    // Loading matrixes with values
    load_matrix(matrixA, file_matrix_A);
    load_matrix(matrixB, file_matrix_B);

    // starting scalar matrix mult
    gettimeofday(&start, NULL);
    scalar_matrix_mult_gpu(scalar, matrixA);
    gettimeofday(&stop, NULL);

    display_matrix(matrixA);

    printf("Scalar matrix mult: %f msec\n\n", timedifference_msec(start, stop));

    // starting matrix matrix mult
    gettimeofday(&start, NULL);
    matrix_matrix_mult_gpu(matrixA, matrixB, matrixC);
    gettimeofday(&stop, NULL);

    display_matrix(matrixC);

    printf("Matrix matrix mult: %f msec\n\n", timedifference_msec(start, stop));
    
    // Closing files
    fclose(file_matrix_A);
    fclose(file_matrix_B);
    fclose(file_result1);
    fclose(file_result2);

    // Free memory
    free(matrixA->h_rows);
    free(matrixB->h_rows);
    free(matrixC->h_rows);
    cudaFree(matrixA->d_rows);
    cudaFree(matrixB->d_rows);
    cudaFree(matrixC->d_rows);
    free(matrixA);
    free(matrixB);
    free(matrixC);
    
    // Getting time of program execution stop
    gettimeofday(&stop_program, NULL); 

    printf("Overall time of program: %.f msec\n\n", timedifference_msec(start_program, stop_program));

    return 0;
}


/*
    Inicializing matrix 
*/
struct matrix* inicialize_matrix(unsigned long int width, unsigned long int height, int alloc_mode) {

    struct matrix* matrix = (struct matrix*)malloc(sizeof(struct matrix));
    if (!matrix) {
        perror("Could not allocate matrix struct on CPU\n");
        exit(-10);
    }

    matrix->h_rows = (float*)malloc(width * height * sizeof(float));
    if (!matrix->h_rows) {
        perror("Could not allocate rows of matrix A or B on CPU\n");
        exit(1);
    }

    if (alloc_mode == PARTIAL_ALLOC) {
        cudaMalloc(&(matrix->d_rows), width * sizeof(float));
    }
    else {
        cudaMalloc(&(matrix->d_rows), width * height * sizeof(float));
    }


    if (cudaError != cudaSuccess) {
        perror("Could not allocate matrix A or B on GPU device\n");
        exit(4);
    }


    matrix->width = width;
    matrix->height = height;
    matrix->alloc_mode = alloc_mode;

    return matrix;
}


/*
    Load the matrix with values in the file
*/
int load_matrix(struct matrix* matrix, FILE* file) {

    for (int i = 0; i < matrix->width * matrix->height; i++) {
        fread(&matrix->h_rows[i], sizeof(float), 1, file);
    }

    return 1;
}


/*
    Used only to matrix C
*/
struct matrix* inicialize_empty_matrix(unsigned long int width, unsigned long int height, int alloc_mode) {

    struct matrix* matrix = (struct matrix*)malloc(sizeof(struct matrix));
     if (!matrix) {
        perror("Could not allocate matrix C struct on CPU\n");
        exit(-12);
    }

    matrix->h_rows = (float*)malloc(width * height * sizeof(float));
    if (!matrix->h_rows) {
        perror("Could not allocate matrix C on CPU\n");
        exit(2);
    }

    if (alloc_mode == PARTIAL_ALLOC) {
        cudaMalloc(&(matrix->d_rows), width * sizeof(float));
    }
    else {
        cudaMalloc(&(matrix->d_rows), width * height * sizeof(float));
    }

    if (cudaError != cudaSuccess) {
        perror("Could not allocate matrix C on GPU device\n");
        exit(4);
    }

    matrix->width = width;
    matrix->height = height;
    matrix->alloc_mode = alloc_mode;

    
    for (int i = 0; i < width * height; i++) { 
        matrix->h_rows[i] = 0;
    }
    

    return matrix;
} 


int display_matrix(struct matrix* matrix) {

    for (int i = 0; i < MAX_DISPLAY; i++) printf("%.f ", matrix->h_rows[i]);
    printf("\n\n");
    return 1;
}