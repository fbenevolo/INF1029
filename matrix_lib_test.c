#include <stdio.h>
#include <stdlib.h>
#include "matrix_lib.h"

int main(int argc, char* argv[]) {

    // Declaração de variáveis
    float scalar = atof(argv[1]);
    struct matrix matrixA, matrixB, matrixC;
    FILE* file_matrix_A, file_matrix_B, file_result1, file_result2; 

    // Abrindo arquivos
    file_matrix_A = fopen(&argv[6], "rb");
    file_matrix_B = fopen(&argv[7], "rb");
    file_result1  = fwrite(&argv[8], "wb");
    file_result2  = fwrite(&argv[9], "wb");

    // Preenchendo campos 'linha' e 'coluna' da struct
    matrixA.height = atoi(argv[2]);
    matrixA.width  = atoi(argv[3]);
    matrixB.height = atoi(argv[4]);
    matrixB.width  = atoi(argv[5]);
    
    // Inicializando as matrizes
    matrixA.rows = (float*)malloc(matrixA.height * matrixA.width * sizeof(float));
    matrixB.rows = (float*)malloc(matrixB.height * matrixB.width * sizeof(float));
    if (!matrixA.rows || !matrixB.rows) {
        fprintf(stderr, "Naoi foi possivel alocar a memoria de uma das matrizes\n");
        exit(1);
    }

    // Preenchendo a matriz em si a partir dos valores lidos do arquivo
    fread(matrixA.rows, sizeof(float), matrixA.height * matrixA.width, file_matrix_A);
    fread(matrixB.rows, sizeof(float), matrixB.height * matrixB.width, file_matrix_B);

    fclose(file_matrix_A);
    fclose(file_matrix_B);
    fclose(file_result1);
    fclose(file_result2);

   return 0;
}