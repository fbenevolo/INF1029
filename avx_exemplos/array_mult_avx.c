#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>

#define SIZE 2056

int main(void) {

    float* fives, * twos, * threes, * result;
    float* aux_five, * aux_two, * aux_three, * aux_result;
 
    fives  = (float*)aligned_alloc(32, SIZE * sizeof(float));
    twos   = (float*)aligned_alloc(32, SIZE * sizeof(float));
    threes = (float*)aligned_alloc(32, SIZE * sizeof(float));
    result = (float*)aligned_alloc(32, SIZE * sizeof(float));

    __m256 vec_fives  = _mm256_set1_ps(5.0f);
    __m256 vec_twos   = _mm256_set1_ps(2.0f);
    __m256 vec_threes = _mm256_set1_ps(3.0f);
    __m256 vec_result = _mm256_set1_ps(0.0f);

    aux_five = fives; 
    aux_three = threes; 
    aux_two = twos;
    for (int i = 0; i < SIZE; i+=8, aux_five+=8, aux_two+=8, aux_three+=8) {
        _mm256_store_ps(aux_five, vec_fives);
        _mm256_store_ps(aux_two, vec_twos);
        _mm256_store_ps(aux_three, vec_threes);
    }


    aux_five = fives; aux_three = threes; aux_two = twos;
    aux_result = result; 
    for (int i = 0; i < SIZE; i+=8, aux_five+=8, aux_two+=8, aux_three+=8, aux_result+=8) {
        __m256 vec_fives  = _mm256_load_ps(aux_five);
        __m256 vec_twos   = _mm256_load_ps(aux_two);
        __m256 vec_threes = _mm256_load_ps(aux_three);

        __m256 vec_result = _mm256_fmadd_ps(vec_fives, vec_twos, vec_threes);
        _mm256_store_ps(aux_result, vec_result);
    }

    for (int i = 0; i < SIZE; i++) printf("%.f\n", result[i]);
    
    return 0;
}