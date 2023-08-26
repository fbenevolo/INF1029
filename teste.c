#include <stdio.h>
#include <stdlib.h>

struct matrix {
    unsigned long int height;
    unsigned long int width;
    float *rows;
};


// Qual seria um caso de erro?
int scalar_matrix_mult(float scalar_value, struct matrix *matrix){
    for(unsigned long int i=0; i<(matrix->height*matrix->width);i++){
        matrix->rows[i]*=scalar_value;
    }
    return 1;
}

int matrix_matrix_mult(struct matrix *matrixA, struct matrix * matrixB, struct matrix * matrixC){
    if(matrixA->width!=matrixB->height){
        return 0;
    }
    //multiplicar elementos da linha de A pelos elementos da coluna de B
    int n=matrixA->width;
    int k;
    for(unsigned long int i=0;i<matrixA->height;i++){
        for(unsigned long int j=0;j<matrixB->width;j++){
            k=0;
            while(k<n){
                matrixC->rows[i*matrixC->width + j]+= matrixA->rows[i*matrixA->width + k]*matrixB->rows[k*matrixB->width + j];
                k++;
            }
        }

    }
    return 1;
}

void print_matrix(struct matrix* matrix){
    printf("[");
     for(unsigned long int i=0; i<(matrix->height*matrix->width);i++){
        printf(" %.1f", matrix->rows[i]);
    }
    printf(" ]\n");
}

int main(void){
    int saida;

    struct matrix *matrixA = (struct matrix*)malloc(sizeof(struct matrix));
    matrixA->height = 3;
    matrixA->width = 2;
    matrixA->rows = (float*)malloc(sizeof(float)*matrixA->height*matrixA->width);
    matrixA->rows[0] = 2;
    matrixA->rows[1] = 4;
    matrixA->rows[2] = 7;
    matrixA->rows[3] = 6;
    matrixA->rows[4] = 8;
    matrixA->rows[5] = 9;


    struct matrix *matrixB = (struct matrix*)malloc(sizeof(struct matrix));
    matrixB->height = 2;
    matrixB->width = 2;
    matrixB->rows = (float*)malloc(sizeof(float)*matrixB->height*matrixB->width);
    matrixB->rows[0] = 1;
    matrixB->rows[1] = 2;
    matrixB->rows[2] = 2;
    matrixB->rows[3] = 1;

    print_matrix(matrixA);
    saida = scalar_matrix_mult(2, matrixA);
    print_matrix(matrixA);

    struct matrix *matrixC = (struct matrix*)malloc(sizeof(struct matrix));
    matrixC->height = 3;
    matrixC->width = 2;
    matrixC->rows = (float*)malloc(sizeof(float)*matrixC->height*matrixC->width);
    matrixC->rows[0] = 0;
    matrixC->rows[1] = 0;
    matrixC->rows[2] = 0;
    matrixC->rows[3] = 0;
    matrixC->rows[4] = 0;
    matrixC->rows[5] = 0;
    saida = matrix_matrix_mult(matrixA, matrixB, matrixC);
    print_matrix(matrixC);
    return 0;

}