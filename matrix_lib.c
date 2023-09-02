# include "matrix_lib.h"

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

int matrix_matrix_mult_opt(struct matrix *matrixA, struct matrix * matrixB, struct matrix * matrixC){
    if(matrixA->width!=matrixB->height){
        return 0;
    }

    for(unsigned long int i=0;i<matrixA->height;i++){
        inicializa_linha(matrixC, i);
        for(unsigned long int j=0;j<matrixA->width;j++){
            for(unsigned long int k=0;k<matrixB->width;k++){
                matrixC->rows[i*matrixC->width + k] += matrixA->rows[i*matrixA->width + j] * matrixB->rows[j*matrixB->width + k];
            }
        }
    }

    return 1; 
            
}

void inicializa_linha(struct matrix *matrix, int height){
    for(int i=0;i<matrix->width;i++){
        matrix->rows[height*matrix->width + i] = 0;
    }
    return;
}
