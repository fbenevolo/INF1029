#include <stdio.h>
#include <stdlib.h>
#include "time.h"
#include "matrix_lib.h"

int load_matrix(struct matrix* matrix, FILE* file);
int store_matrix(struct matrix* matrix, FILE* file);
int inicialize_matrix(struct matrix* matrix, unsigned long int width, unsigned long int height);
int inicialize_empty_matrix(struct matrix* matrix, int width, int height);
int display_matrix(struct matrix* matrix);

int main(int argc, char* argv[]) {

    // Declaração de variáveis
    float scalar = atof(argv[1]);
    struct matrix matrixA, matrixB, matrixC;
    FILE* file_matrix_A, file_matrix_B, file_result1, file_result2; 

    // Abrindo arquivos
    file_matrix_A = fopen(&argv[6], "rb");
    file_matrix_B = fopen(&argv[7], "rb");
    file_result1  = fopen(&argv[8], "wb");
    file_result2  = fopen(&argv[9], "wb");

    // Inicializando matrizes A e B 
    inicialize_matrix(matrixA, atoi(argv[2]), atoi(argv[3]));
    inicialize_matrix(matrixB, atoi(argv[4]), atoi(argv[5]));

    // Inicializando matriz C com colunas de A e linhas de B e preenchendo zeros
    inicialize_empty_matrix(matrixC, atoi(argv[3]), atoi(argv[4]);)

    // Preenchendo as matrizes A e B
    load_matrix(matrixA, file_matrix_A);
    load_matrix(matrixB, file_matrix_B);
    
    // Multiplicação por escalar
    scalar_matrix_mult(scalar, matrixA);
    load_matrix(matrixC, file_result2);

    // Multiplicação de matrizes
    matrix_matrix_mult(matrixA, matrixB, matrixC);
    load_matrix(matrixC, file_result2);
    
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
int inicialize_matrix(struct matrix* matrix, unsigned long int width, unsigned long int height) {

    matrix->rows = (float*)malloc*(width * height * sizeof(float));
    if (!matrix->rows) {
        fprintf(stderr, "Nao foi possivel alocar a matriz");
        exit(1);
    }

    matrix->width = width;
    matrix->height = height;

    return 1;
}


/* 
    Carrega na matriz os valores contidos no arquivo binário
*/
int load_matrix(struct matrix* matrix, FILE* file) {

    fread(matrix->rows, sizeof(float), matrix->height * matrix->width, file);
    return 1;
}


/*
    Guarda os valores da matriz em um arquivo binário
*/
int store_matrix(struct matrix* matrix, FILE* file) {

    fwrite(matrix->rows, sizeof(float), matrix->height * matrix->width, file)
    return 1;
}


/*
    Exibe a matriz    
*/
int display_matrix(struct matrix* matrix) {

    for (int i = 0; i < matrix->height * matrix->rows; i++) printf("%f", matrix->rows[i];
    return 1;
}


/*
    Função utilizada apenas para a matriz C
*/
int inicialize_empty_matrix(struct matrix* matrix, int width, int height) {
    matrixC.rows = (float*)malloc(matrixC.width * matrixC.height * sizeof(float))
    if (!matrixC.rows) {
        fprintf(stderr, "Nao foi possivel alocar a matriz C");
        exit(2);
    }

    for (int i = 0; i < matrixC.width * matrixC.height; i++) matrixC.rows[i] = 0;

    return 1;
} 