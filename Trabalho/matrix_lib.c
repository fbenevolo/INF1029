#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include "matrix_lib.h"

/*
void initialize_row(struct matrix* matrix, unsigned long int height) {
    for (unsigned long int i = 0;i < matrix->width;i++) {
        matrix->rows[height * matrix->width + i] = 0;
    }
    return;
}
*/


void initialize_row(struct matrix* matrix, unsigned long int height) {
    __m256 vec_zeros = _mm256_setzero_ps();
    float* aux;
    
    for (unsigned long int i = 0; i < matrix->width; i+=8, aux+=8) {
        aux = matrix->rows + (height * matrix->width + i);
        _mm256_store_ps(aux, vec_zeros);
    } 
   
   return;
}

int is_matrix_valid(struct matrix* matrix) {
    if (matrix == NULL || matrix->height <= 0 || matrix->width <= 0) {
        return 0;
    }
    return 1;
}


int is_matrix_mult_valid(struct matrix* matrixA, struct matrix* matrixB) {
    if (is_matrix_valid(matrixA) && is_matrix_valid(matrixB)) {
        if (matrixA->width == matrixB->height) {
            return 1;
        }
    }
    return 0;
}


int scalar_matrix_mult(float scalar_value, struct matrix* matrix) {
    if (!is_matrix_valid(matrix)) {
        return 0;
    }

    __m256 vec_scalar = _mm256_set1_ps(scalar_value);
    float* aux = matrix->rows;

    for (unsigned long int i = 0; i < (matrix->height * matrix->width); i+=8, aux+=8) {
        __m256 vec_twos = _mm256_load_ps(aux);
        __m256 vec_mult = _mm256_mul_ps(vec_twos, vec_scalar);
        
        _mm256_store_ps(aux, vec_mult);
    } 
    
    return 1;
}


int matrix_matrix_mult(struct matrix* matrixA, struct matrix* matrixB, struct matrix* matrixC) {
    if (!is_matrix_mult_valid(matrixA, matrixB)) {
        return 0;
    }

    //multiplicar elementos da linha de A pelos elementos da coluna de B
    int n = matrixA->width;
    int k;
    for (unsigned long int i = 0;i < matrixA->height;i++) {
        for (unsigned long int j = 0;j < matrixB->width;j++) {
            k = 0;
            while (k < n) {
                matrixC->rows[i * matrixC->width + j] += matrixA->rows[i * matrixA->width + k] * matrixB->rows[k * matrixB->width + j];
                k++;
            }
        }

    }
    return 1;
}

int matrix_matrix_mult_opt(struct matrix* matrixA, struct matrix* matrixB, struct matrix* matrixC) {
    if (!is_matrix_mult_valid(matrixA, matrixB)) {
        return 0;
    }

    unsigned long int num_elements_A = (matrixA->height) * (matrixA->width);
    unsigned long int current_element_B;
    unsigned long int current_element_C;

    float* vecANext = matrixA->rows;
    float* vecBNext;
    float* vecCNext;
    
    for (unsigned long int i = 0;i < num_elements_A; i++, vecANext++) {
        __m256 vecA = _mm256_set1_ps(*vecANext);
        current_element_B = (i % matrixA->width) * (matrixB->width);
        current_element_C = (i / matrixA->width) * (matrixC->width);

        
        if (i % matrixA->width == 0) {
            initialize_row(matrixC, (current_element_C / matrixC->width));
        } 

        vecBNext = matrixB->rows+current_element_B;
        vecCNext = matrixC->rows+current_element_C;
        for (unsigned long int j = 0;j < matrixB->width;j+=8, vecBNext+=8, vecCNext+=8) {
            __m256 vecB = _mm256_load_ps(vecBNext);
            __m256 vecC = _mm256_load_ps(vecCNext);
            __m256 vecRes = _mm256_fmadd_ps(vecA, vecB, vecC);
            _mm256_store_ps(vecCNext, vecRes);
        }
    }

    return 1;

}