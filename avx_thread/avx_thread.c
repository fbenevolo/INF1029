#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <pthread.h>

struct thread_data {
    int id;
    float* twos; 
    float* fives;
    float* tens;
    int begin;
    int end;
} typedef threadData;


void* mult_arrays(void* threadarg);
void* init_arrays(void* threadarg);


int main(void) {

    float* twos = aligned_alloc(32, 64 * sizeof(float));
    float* fives = aligned_alloc(32, 64 * sizeof(float));
    float * tens = aligned_alloc(32, 64 * sizeof(float));
    float* aux_twos, * aux_fives, * aux_tens;

    pthread_t threads[4];
    threadData dataArray[4]; 

    aux_twos = twos; aux_fives = fives; aux_tens = tens;

    for (int i = 0; i < 4; i++) {
        dataArray[i].id = i+1;
        dataArray[i].begin = i * 16;
        dataArray[i].end = (i+1) * 16;
        dataArray[i].fives = aux_fives + (i*16);
        dataArray[i].twos = aux_twos + (i*16);

        pthread_create(&threads[i], NULL, init_arrays, (void*)&dataArray[i]);
    }

    //for (int i = 0; i < 4; i++) pthread_join(threads[i], NULL);

    for (int i = 0; i < 4; i++) {
        dataArray[i].id = i+1;
        dataArray[i].begin = i * 16;
        dataArray[i].end = (i+1) * 16;
        dataArray[i].fives = aux_fives + (i*16);
        dataArray[i].twos = aux_twos + (i*16);
        dataArray[i].tens = aux_tens + (i*16);

        pthread_create(&threads[i], NULL, mult_arrays, (void*)&dataArray[i]);
    }

    //for (int i = 0; i < 4; i++) pthread_join(threads[i], NULL);
    
    for (int i = 0; i < 64; i++) printf("%.f\n", *(tens+i));
    
    return 0;
}


void* init_arrays(void* threadarg) {
    threadData* data = (threadData*)threadarg;

    for (int i = data->begin; i < data->end; i += 8, data->twos+=8, data->fives+=8) {
        __m256 vec_twos = _mm256_set1_ps(2.0f);
        __m256 vec_fives = _mm256_set1_ps(5.0f);

        _mm256_store_ps(data->twos, vec_twos);
        _mm256_store_ps(data->fives, vec_fives);
    }
}


void* mult_arrays(void* threadarg) {
    threadData* data = (threadData*)threadarg;

    for (int i = data->begin; i < data->end; i+= 8, 
    data->twos +=8, data->fives += 8, data->tens +=8) {
        __m256 vec_twos = _mm256_load_ps(data->twos);
        __m256 vec_fives = _mm256_load_ps(data->fives);

        __m256 vec_tens = _mm256_mul_ps(vec_twos, vec_fives);
        _mm256_store_ps(data->tens, vec_tens); 
    }
}