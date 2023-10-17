#include <immintrin.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include "matrix_lib.h"


static int NUM_THREADS;


void set_number_threads(int num_threads) { 
    NUM_THREADS = num_threads;
}


void initialize_row(struct matrix* matrix, unsigned long int height) {
    __m256 vec_zeros = _mm256_setzero_ps();
    float* aux;
    
    for (unsigned long int i = 0; i < matrix->width; i+=8, aux+=8) {
        aux = matrix->rows + (height * matrix->width + i);
        _mm256_store_ps(aux, vec_zeros);
    } 
   
   return;
}


void* scalarMatrixMultThread(void* threadarg) {
    scalar_data* data = (scalar_data*) threadarg;

    __m256 vec_scalar = _mm256_set1_ps(data->scalar);
    float* aux = &data->matrixA->rows[data->begin];

    for (unsigned long int i = data->begin; i < data->end; i+=8, aux+=8) {
        __m256 vec_twos = _mm256_load_ps(aux);
        __m256 vec_mult = _mm256_mul_ps(vec_twos, vec_scalar);
        
        _mm256_store_ps(aux, vec_mult);
    } 

    pthread_exit(NULL);
}


void* matrixMatrixMultThread(void* threadarg) {

    matrix_data* data = (matrix_data*)threadarg;

    unsigned long int current_element_B;
    unsigned long int current_element_C;

    struct matrix* matrixA = data->matrixA;
    struct matrix* matrixB = data->matrixB;
    struct matrix* matrixC = data->matrixC;

    float* vecANext = matrixA->rows;
    float* vecBNext;
    float* vecCNext;

    for (int i = data->begin; i < data->end; i++, vecANext++) {
        __m256 vecA = _mm256_set1_ps(*vecANext);
        current_element_B = (i % matrixA->width) * (matrixB->width);
        current_element_C = (i / matrixA->width) * (matrixC->width);

        if (i % matrixA->width == 0) {
            initialize_row(matrixC, (current_element_C / matrixC->width));
        } 

        vecBNext = matrixB->rows+current_element_B;
        vecCNext = matrixC->rows+current_element_C;

        for (unsigned long int j = 0; j < matrixB->width; j+=8, vecBNext+=8, vecCNext+=8) {
            __m256 vecB = _mm256_load_ps(vecBNext);
            __m256 vecC = _mm256_load_ps(vecCNext);
            __m256 vecRes = _mm256_fmadd_ps(vecA, vecB, vecC);
            _mm256_store_ps(vecCNext, vecRes);
        }
    }

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
    
    if (!is_matrix_valid(matrix)) return 0;

    pthread_t threads[NUM_THREADS];
    scalar_data data[NUM_THREADS];
    pthread_attr_t attr;
    struct matrix* auxMatrix = matrix;
    int lines = matrix->width * matrix->height;
    void* status;

    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    for (int i = 0; i < NUM_THREADS; i++) {
        data[i].begin = i * (lines / NUM_THREADS);
        data[i].end = (i+1) * (lines / NUM_THREADS);
        data[i].scalar = scalar_value;
        data[i].matrixA = auxMatrix;

        pthread_create(&threads[i], &attr, scalarMatrixMultThread, (void*)&data[i]);
    }

    pthread_attr_destroy(&attr);

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], &status);
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
    
    if (!is_matrix_mult_valid(matrixA, matrixB)) return 0;
    

    unsigned long int num_elements_A = (matrixA->height) * (matrixA->width);
    unsigned long int current_element_B;
    unsigned long int current_element_C;

    float* vecANext = matrixA->rows;
    float* vecBNext;
    float* vecCNext;

    pthread_t threads[NUM_THREADS];
    matrix_data data[NUM_THREADS];
    pthread_attr_t attr;
    void* status;
    
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    for (int i = 0; i < NUM_THREADS; i++) {
        data[i].begin = i * (num_elements_A / NUM_THREADS);
        data[i].end = (i+1) * (num_elements_A / NUM_THREADS);
        data[i].matrixA = matrixA;
        data[i].matrixB = matrixB;
        data[i].matrixC = matrixC;

        pthread_create(&threads[i], &attr, matrixMatrixMultThread, (void*)&data[i]);
    }

    pthread_attr_destroy(&attr);

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], &status);
    }

    return 1;

}