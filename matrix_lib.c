#include "matrix_lib.h"
#include <stdlib.h>

void initialize_row(struct matrix* matrix, unsigned long int height) {
    for (unsigned long int i = 0;i < matrix->width;i++) {
        matrix->rows[height * matrix->width + i] = 0;
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
    for (unsigned long int i = 0; i < (matrix->height * matrix->width);i++) {
        matrix->rows[i] *= scalar_value;
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

    for (unsigned long int i = 0;i < num_elements_A;i++) {
        current_element_B = (i % matrixA->width) * (matrixB->width);
        current_element_C = (i / matrixA->width) * (matrixC->width);
        if (i % matrixA->width == 0) {
            initialize_row(matrixC, (current_element_C / matrixC->width));
        }
        for (unsigned long int j = 0;j < matrixB->width;j++) {
            matrixC->rows[current_element_C + j] += matrixA->rows[i] * matrixB->rows[current_element_B + j];
        }
    }

    return 1;

}


