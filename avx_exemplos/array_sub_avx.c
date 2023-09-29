#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>

#define SIZE 64

int main(void) {

    float* evens, * odds, * result;
    float* aux_pointer_evens, * aux_pointer_odds, * aux_pointer_result;

    /* alocação dos três arrays em memória */
    evens  = (float*)aligned_alloc(32, SIZE * sizeof(float));
    odds   = (float*)aligned_alloc(32, SIZE * sizeof(float));
    result = (float*)aligned_alloc(32, SIZE * sizeof(float));
    printf("%p\n", result);

    /* inicialização dos vetores */
    __m256 vec_evens = _mm256_set1_ps(8.0f);
    __m256 vec_odds = _mm256_set1_ps(5.0f);
    
    aux_pointer_evens = evens;
    aux_pointer_odds  = odds;
    /* carregando os valores no array através dos vetores */
    for (int i = 0; i < SIZE; i+=8, aux_pointer_evens+=8, aux_pointer_odds+=8) {
        _mm256_store_ps(aux_pointer_evens, vec_evens);
        _mm256_store_ps(aux_pointer_odds, vec_odds);
    }

    aux_pointer_evens = evens;
    aux_pointer_odds = odds;
    aux_pointer_result = result;
    
    for (int i = 0; i < SIZE; i+=8, aux_pointer_evens+=8, aux_pointer_odds+=8, aux_pointer_result+=8) {
        __m256 vec_evens = _mm256_load_ps(aux_pointer_evens);
        __m256 vec_odds = _mm256_load_ps(aux_pointer_odds);

        __m256 vec_result = _mm256_sub_ps(vec_evens, vec_odds);
        _mm256_store_ps(aux_pointer_result, vec_result);
    }


    for (int i = 0; i < SIZE; i++) printf("%.f\n", result[i]); 
    

    return 0;
}