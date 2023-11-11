#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "matrix_lib.h"
extern "C" {
    #include "timer.h"
}

#define NUM_ARGS 13
#define MAX_DISPLAY 256 

/* Headers of auxiliary functions */
struct matrix* inicialize_matrix(unsigned long int width, unsigned long int height,
                                 int alloc_mode);
struct matrix* inicialize_empty_matrix(unsigned long int width, unsigned long int height, 
                                       int alloc_mode);
int load_matrix(struct matrix* matrix, FILE* file);
int display_matrix(struct matrix* matrix);


int main(int argc, char* argv[]) {

    int scalar, gpu_max_size;
    struct matrix* matrixA, * matrixB, * matrixC;
    struct timeval start, stop, start_program, stop_program;  
    FILE* file_matrix_A, * file_matrix_B, * file_result1, * file_result2; 

    // Getting time of program execution start
    gettimeofday(&start_program, NULL);

    // setting scalar value
    scalar = atoi(argv[1]);

    // checking if threads per block and max blocks per grid args were received
    int received_gpu_info = 0;
    if (argc == NUM_ARGS) {
        received_gpu_info = 2;
        // threads_per_block > 1024 or max_blocs_per_grid > 65535 
        if (set_grid_size(atoi(argv[6]), atoi(argv[7])) == 0) {
            fprintf(stderr, "Threads per block or maximum blocks per grid values
                             were bigger than what is supported.
                             The values will be set to default\n");
         }
    }

    // defining gpu maximum size
    gpu_max_size  = atoi(argv[6+received_gpu_info]);

    // opening files
    file_matrix_A = fopen(argv[7+received_gpu_info], "rb");
    file_matrix_B = fopen(argv[8+received_gpu_info], "rb");
    file_result1  = fopen(argv[9+received_gpu_info], "wb");
    file_result2  = fopen(argv[10+received_gpu_info], "wb");

    // initializing matrixes TODO: verificar condição de alocação
    matrixA = inicialize_matrix(atoi(argv[2]), atoi(argv[3]), FULL_ALLOC);
    matrixB = inicialize_matrix(atoi(argv[4]), atoi(argv[5]), FULL_ALLOC);
    matrixC = inicialize_empty_matrix(atoi(argv[3]), atoi(argv[5]), FULL_ALLOC); 

    // loading matrixes with values
    load_matrix(matrixA, file_matrix_A);
    load_matrix(matrixB, file_matrix_B);

    display_matrix(matrixA);
    display_matrix(matrixB);
    display_matrix(matrixC);


    // Closing files
    fclose(file_matrix_A);
    fclose(file_matrix_B);
    fclose(file_result1);
    fclose(file_result2);

    // Getting time of program execution stop
    gettimeofday(&stop_program, NULL); 

    //printf("Overall time of program: %.f msec", timedifference_msec(start_program, stop_program));

    return 0;
}


/*
    Inicializing matrix 
*/
struct matrix* inicialize_matrix(unsigned long int width, unsigned long int height, int alloc_mode) {

    struct matrix* matrix = (struct matrix*)malloc(sizeof(struct matrix));
    matrix->h_rows = (float*)malloc(width * height * sizeof(float));
    if (!matrix->h_rows) {
        fprintf(stderr, "Nao foi possivel alocar a matriz");
        exit(1);
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
    matrix->h_rows = (float*)malloc(width * height * sizeof(float));
    if (!matrix->h_rows) {
        fprintf(stderr, "Could not allocate matrix C");
        exit(2);
    }

    matrix->width = width;
    matrix->height = height;
    for (int i = 0; i < width * height; i++) { 
        matrix->h_rows[i] = 0 
    }

    matrix->alloc_mode = alloc_mode;

    return matrix;
} 


int display_matrix(struct matrix* matrix) {

    for (int i = 0; i < MAX_DISPLAY; i++) printf("%.f ", matrix->h_rows[i]);
    printf("\n\n\n");
    return 1;
}