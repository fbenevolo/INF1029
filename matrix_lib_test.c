#include <stdio.h>
#include <stdlib.h>
#include "matrix_lib.h"
#include "timer.h"

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
    FILE* file_matrix_A, * file_matrix_B, * file_result1, * file_result2; 
    struct timeval start, stop; 

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
    
    // Multiplicação por escalar
    scalar_matrix_mult(scalar, matrixA);
    store_matrix(matrixA, file_result1);
    //display_matrix(matrixA);

    // Start da multiplicação
    gettimeofday(&start, NULL);

    // Multiplicação de matrizes
    matrix_matrix_mult(matrixA, matrixB, matrixC);
    gettimeofday(&stop, NULL); // Stop da multiplicação
    store_matrix(matrixC, file_result2);
    display_matrix(matrixC);

    printf("Overall time: %f ms\n", timedifference_msec(start, stop));

    // Fechando arquivos
    fclose(file_matrix_A);
    fclose(file_matrix_B);
    fclose(file_result1);
    fclose(file_result2);
    
    return 0;
}


/*
    Inicializa as matrizes, utilizando malloc para o campo 'rows' 
    e atribuição para os campos 'width' e 'height' 
*/
struct matrix* inicialize_matrix(unsigned long int width, unsigned long int height) {

    struct matrix* matrix = (struct matrix*)malloc(sizeof(struct matrix));
    matrix->rows = (float*)malloc(width * height * sizeof(float));
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

    for (int i = 0; i < matrix->height * matrix->width; i++) printf("%f  ", matrix->rows[i]);
    printf("\n\n\n");
    return 1;
}


/*
    Função utilizada apenas para a matriz C
*/
struct matrix* inicialize_empty_matrix(unsigned long int width, unsigned long int height) {

    struct matrix* matrix = (struct matrix*)malloc(sizeof(struct matrix));
    matrix->rows = (float*)malloc(width * height * sizeof(float));
    if (!matrix->rows) {
        fprintf(stderr, "Nao foi possivel alocar a matriz C");
        exit(2);
    }


    matrix->width = width;
    matrix->height = height;
    for (int i = 0; i < width * height; i++) matrix->rows[i] = 0;

    return matrix;
} 
