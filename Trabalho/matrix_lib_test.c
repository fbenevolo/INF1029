#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "matrix_lib.h"
#include "timer.h"
#define MAX_DISPLAY 256
#define NUM_ARGS 11

// Cabeçalhos
int load_matrix(struct matrix* matrix, FILE* file);
int store_matrix(struct matrix* matrix, FILE* file);
int display_matrix(struct matrix* matrix);
struct matrix* inicialize_matrix(unsigned long int width, unsigned long int height);
struct matrix* inicialize_empty_matrix(unsigned long int width, unsigned long int height);


int main(int argc, char* argv[]) {

    // Declaração de variáveis
    float scalar = atof(argv[1]);
    struct matrix * matrixA, * matrixB, * matrixC;
    struct timeval start, stop, start_program, stop_program;
    FILE* file_matrix_A, * file_matrix_B, * file_result1, * file_result2; 

    // Iniciando o tempo do programa
    gettimeofday(&start_program, NULL); 

     // definindo número de threads
    if(argc == NUM_ARGS - 1){
        set_number_threads(1);
    }
    set_number_threads(atoi(argv[NUM_ARGS-1]));

    // Abrindo arquivos
    file_matrix_A = fopen(argv[6], "rb");
    file_matrix_B = fopen(argv[7], "rb");
    file_result1  = fopen(argv[8], "wb");
    file_result2  = fopen(argv[9], "wb");

    // Inicializando matrizes A e B 
    matrixA = inicialize_matrix(atoi(argv[2]), atoi(argv[3]));
    matrixB = inicialize_matrix(atoi(argv[4]), atoi(argv[5]));
    // Inicializando matriz C com colunas de A e linhas de B e preenchendo zeros
    matrixC = inicialize_empty_matrix(atoi(argv[3]), atoi(argv[4]));

    // Preenchendo as matrizes A e B
    load_matrix(matrixA, file_matrix_A);
    load_matrix(matrixB, file_matrix_B);
    
    // Start da multiplicação por um escalar 
    gettimeofday(&start, NULL);

    // Multiplicação por escalar
    scalar_matrix_mult(scalar, matrixA);
    gettimeofday(&stop, NULL); // Stop da multiplicação por escalar
    store_matrix(matrixA, file_result1);
    display_matrix(matrixA);
    
    printf("Overall time scalar matrix mult: %f ms\n\n\n", timedifference_msec(start, stop));

    // Multiplicação de matrizes NÃO OTIMIZADO
    gettimeofday(&start, NULL);
    matrix_matrix_mult(matrixA, matrixB, matrixC);
    gettimeofday(&stop, NULL); // Stop da multiplicação
    store_matrix(matrixC, file_result2);
    display_matrix(matrixC);

    printf("Overall time matrix matrix mult non-optimized: %f ms\n\n\n", timedifference_msec(start, stop));

    // Multiplicação de matrizes OTIMIZADO
    gettimeofday(&start, NULL);
    matrix_matrix_mult_opt(matrixA, matrixB, matrixC);
    gettimeofday(&stop, NULL);
    display_matrix(matrixC);

    printf("Overall time matrix matrix mult optimized: %f ms\n", timedifference_msec(start, stop));


    // Fechando arquivos
    fclose(file_matrix_A);
    fclose(file_matrix_B);
    fclose(file_result1);
    fclose(file_result2);
    

    gettimeofday(&stop_program, NULL);

    printf("\nOverall time of program: %f ms\n", timedifference_msec(start_program, stop_program));

    return 0;
}


/*
    Inicializa as matrizes, utilizando malloc para o campo 'rows' 
    e atribuição para os campos 'width' e 'height' 
*/
struct matrix* inicialize_matrix(unsigned long int width, unsigned long int height) {

    struct matrix* matrix = (struct matrix*)aligned_alloc(32, sizeof(struct matrix));
    matrix->rows = (float*)aligned_alloc(32, width * height * sizeof(float));
    if (!matrix->rows) {
        fprintf(stderr, "Nao foi possivel alocar a matriz");
        exit(1);
    }

    matrix->width = width;
    matrix->height = height;

    return matrix;
}


/* 
    Carrega na matriz os valores contidos no arquivo binário
*/
int load_matrix(struct matrix* matrix, FILE* file) {

    for (int i = 0; i < matrix->width * matrix->height; i++) {
        fread(&matrix->rows[i], sizeof(float), 1, file);
    }

    return 1;
}


/*
    Guarda os valores da matriz em um arquivo binário
*/
int store_matrix(struct matrix* matrix, FILE* file) {

    for (int i = 0; i < matrix->height * matrix->width; i++) {
        fwrite(&matrix->rows[i], sizeof(float), 1, file);
    }
    
    return 1;
}


/*
    Exibe a matriz    
*/
int display_matrix(struct matrix* matrix) {

    for (int i = 0; i < MAX_DISPLAY; i++) printf("%.f ", matrix->rows[i]);
    printf("\n\n\n");
    return 1;
}


/*
    Função utilizada apenas para a matriz C
*/
struct matrix* inicialize_empty_matrix(unsigned long int width, unsigned long int height) {

    struct matrix* matrix = (struct matrix*)aligned_alloc(32, sizeof(struct matrix));
    matrix->rows = (float*)aligned_alloc(32, width * height * sizeof(float));
    if (!matrix->rows) {
        fprintf(stderr, "Nao foi possivel alocar a matriz C");
        exit(2);
    }

    matrix->width = width;
    matrix->height = height;
    for (int i = 0; i < width * height; i++) matrix->rows[i] = 0;

    return matrix;
} 
