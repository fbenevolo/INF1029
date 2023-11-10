struct matrix {
    unsigned long int height;
    unsigned long int width;
    float *rows;
};


struct scalar_thread_data {
    float scalar;
    long unsigned int begin;
    long unsigned int end;
    struct matrix* matrixA;
} typedef scalar_data;



struct matrix_thread_data {
    long unsigned int begin;
    long unsigned int end;
    struct matrix* matrixA;
    struct matrix* matrixB;
    struct matrix* matrixC;
} typedef matrix_data;

void set_number_threads(int num_threads);

int scalar_matrix_mult(float scalar_value, struct matrix *matrix);

int matrix_matrix_mult(struct matrix *matrixA, struct matrix * matrixB, struct matrix * matrixC);

int matrix_matrix_mult_opt(struct matrix *matrixA, struct matrix * matrixB, struct matrix * matrixC);