//------------------------------------------------------------------------------------
// Matrix multiplication code adapted from an SME programming example provided
// by Arm Ltd.
//
// Assumptions :
//  - Input and Output matricies are in row-major
//  - nbr in matLeft (M): any
//  - nbc in matLeft, nbr in matRight (K): any K > 2
//  - nbc in matRight (N): any
//------------------------------------------------------------------------------------

#include <iostream>
#include <random>
#include <chrono>
#include <cstdint>
#include <stdlib.h>
#include <sys/time.h>
#include <pthread.h>
#include <Accelerate/Accelerate.h>
#include "megemm.h"

#define ERROR_TOLERANCE 0.0005f

void matmul_ref( uint64_t rows_l,  uint64_t cols_l, uint64_t cols_r,  float * input_left,
     float * input_right, float * output) {
    for (uint64_t x = 0; x < rows_l; ++x) {
      for (uint64_t y = 0; y < cols_r; ++y) {
        float acc = 0.0f;
        for (uint64_t z = 0; z < cols_l; ++z) {
          acc += input_left[(x * cols_l) + z] * input_right[(z * cols_r) + y];
        }
        output[(x * cols_r) + y] = acc;
      }
    }
}

void matmul_ref_col(int m, int n, int k, int lda, int ldb, int ldc,   float * input_left,
   float * input_right, float * output){
    for (int j = 0; j < n; j ++ ) {
      for (int i = 0; i < m; i ++ ) {
          float acc = 0.0f;
          for (int p = 0; p < k; p ++ ) {
             acc += input_left[ p * lda + i ] * input_right[ j * ldb + p ];
          }
          output[i + j * ldc] = acc;
      }
  }
}




void random_matrix( int m, int n, float *a)
{
  //#pragma omp parallel for num_threads(num)
  for(int i =0 ; i < m * n; i ++)
  {
      //a[i] = 1;
      //a[i] = i+1;
      a[i]= 2.0 * (float)drand48() - 1.0 + 0.0000001 * (i/m);
  }
}


// extern "C"
// {
//   void sgemm_sme2_preprocess_col( uint64_t rows,  uint64_t cols,
//      float * a, float * a_mod);

//   void sgemm_sme2_4VLx4VL_col( uint64_t rows_l,  uint64_t cols_l,
//     uint64_t cols_r,  float * a,
//     float * b, float * c, float * packa ,uint64_t lda);
// }

int main(int argc, char **argv) {
  int M, N, K, N_mod, iterations,num_threads;

  /* Size parameters */
  if (argc == 5) {
    iterations = strtoul(argv[1], NULL, 0);
    M = strtoul(argv[2], NULL, 0);
    N = strtoul(argv[3], NULL, 0);
    K = strtoul(argv[4], NULL, 0);
    num_threads = 1;
    // num_threads = strtoul(argv[5], NULL, 0);
  } else {
    /* Default: 128x128x128 */
    M = 32;
    N = 32;
    K = 32;
    iterations = 1;
    num_threads = 1;
  }

  // M = 64;
  // N = 24576;
  // K = 1536;
  // iterations = 500;

  if (!(K > 2)) {
    printf("Invalid Matrix Dimensions. Please ensure that :\n\t- K dimension "
           "is Greater-Than 2 \n\nRuntime args ==> ./BINARY_NAME iterations M "
           "K N\n");
    exit(1);
  }

  printf("Left Matrix Dimensions = %d x %d\nRight Matrix Dimensions = %d x "
         "%d\nSME Loop iterations = %d\n\n",
         M, K, K, N, iterations);

  /* Calculate M of transformed matLeft.  */
  uint64_t vl_elms = 16;
  printf("%llu\n", vl_elms);

  N_mod = ceil((double)N / (double)vl_elms) * vl_elms; // ceil(N/cntw)*cntw
  printf("N_mod = %d\n", N_mod);

  BLASSetThreading(BLAS_THREADING_SINGLE_THREADED);
  //BLASSetThreading(BLAS_THREADING_MULTI_THREADED);
  pthread_set_qos_class_self_np( QOS_CLASS_USER_INTERACTIVE, 0 );
  
  float *matLeft = (float*)malloc(M * K * sizeof(float));
  float *matRight = (float*)malloc(K * N * sizeof(float));
  float *matResult_ref = (float*)malloc(M * N * sizeof(float));
  float *matResult_opt = (float*)malloc(M * N * sizeof(float));

  float *matLeft_mod = (float*)malloc(M * K * sizeof(float));

  //srand((unsigned int)time(0));

  random_matrix(M, K, matLeft);
  random_matrix(K, N, matRight);
  memset(matResult_opt, 0, M * N * sizeof(float));
  

  double ops = (double )M *  N  * K * 1.0e-09 * 2 * 1;

  //Calculate matrix multiply naively
  
  printf("Calculating Reference Matrix Multiply...\n");
  double best = 1e9;
  std::chrono::steady_clock::time_point  l_start,l_end;
  for (int i = 0; i < iterations; i++) {
    //matmul_ref_col(M, N, K, M, K, M, matLeft, matRight, matResult_ref);
    l_start = std::chrono::steady_clock::now();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, matLeft, K, matRight, N, 0.0, matResult_ref, N);
    l_end = std::chrono::steady_clock::now();
    best = min(best, std::chrono::duration_cast< std::chrono::duration<double> >( l_end - l_start ).count());
  }
  //matmul_ref_col(M, K, N, M, K, M, matLeft, matRight, matResult_ref);
  printf("SGEMM:  M= %d N=%d K=%d cost=%lf Gflops = %lf \n", M, N, K, best ,  ops/best);
  printf("Done\n");

  // for(int i=0;i<K*N; i++){
  //   printf(" %.2f ",matRight[i]);
  // }

  printf("\n");
  printf("\n");
  
  // for(int z=0;z< M;z++){
  //   for(int j=0;j<K;j++){
  //         printf(" %.2f ",matLeft[ z * K + j]);
  //   }
  //   printf("\n");
  // }
      

  printf("\n");
  printf("\n");

  // std::chrono::steady_clock::time_point  l_start_pack,l_end_pack;
  // l_start_pack = std::chrono::steady_clock::now();

  //row_packa_opt(M, K, matLeft, K, matLeft_mod);  //32,16384 / 1024,16384 / 512, 512/ 512, 1024
  
  // for(int i=0;i<100;i++){
  //   row_packa1(M, K, matLeft, K, matLeft_mod); 
  // }
  //row_packa(M, K, matLeft, K, matLeft_mod); 
  //32,1024:0.000069 
  //512,512:0.000104
  //512,1024:0.000153
  //1024,1024:0.000227
  //1024,16384:0.003145
  //32,16384:0.000369
  // l_end_pack = std::chrono::steady_clock::now();
  // double pack_time = std::chrono::duration_cast< std::chrono::duration<double> >( l_end - l_start ).count();
  // printf("pack_time: %lf\n", pack_time);


  // for(int i=0;i< M * K; i++){
  //   if( i % (K*32) ==0)
  //   {
  //     printf("\n");
  //     printf("\n");
  //   }
  //   printf(" %.2f ",matLeft_mod[i]);
  // }
  // printf("\n");
  // printf("\n");

  


  // for (int i = 0; i < 3; i++) {
  //   sgemm_direct_sme1_preprocess_col(K, N, matRight, matRight_mod);
  //   sgemm_direct_sme1_2VLx2VL_col(M, K, N, matLeft, matRight_mod, matResult_opt);
  // }
  
  // Calculate matrix multiply using SME
  // Looping multiple times to isolate performance to this function call
  printf("Calculating %d iterations of SME Matrix Multiply...\n", iterations);
  double best1 = 1e9;
  
  for (int i = 0; i < iterations; i++) {
       l_start = std::chrono::steady_clock::now();
       row_sgemm(M, N, K, matLeft, matRight, matResult_opt);
       l_end = std::chrono::steady_clock::now();
       best1 = min(best1 , std::chrono::duration_cast< std::chrono::duration<double> >( l_end - l_start ).count());
  }
  printf("SGEMMxsme:  M=%d N=%d K=%d cost=%lf ops=%f Gflops = %lf \n", M, N, K, best1, ops, ops/best1);
  printf("Done\n\n");

  //clang -march=armv9-a+sme+sme2 -O0 -c matmul_opt.s preprocess.s

  //clang -O3 -c main.c -o main.o
  //clang -march=armv9-a+sme+sme2 -O0 -c matmul_opt.s preprocess.s
  //clang main.o matmul_opt.o preprocess.o -o matmul
  

  printf("ERROR TOLERANCE = %.4f%%\n", (float)ERROR_TOLERANCE);
  int error = 0;

  // for(int z=0;z< M;z++){
  //   for(int j=0;j< K;j++){
  //         printf(" %.2f ",matLeft[ z + j * M]);
  //   }
  // }

  // printf("\n"); 
  // printf("\n");

  // for(int i=0;i < 32 * K; i++){
  //   if(i % 32 == 0)
  //     printf("\n");
  //   printf(" %.2f ",packa[i]);
  // }

  // printf("\n"); 
  // printf("\n");

  // for(int z=0;z< M;z++){
  //   for(int j=0;j<N;j++){
  //         printf(" %.2f ",matResult_opt[ z + j * M]);
  //   }
  //  printf("\n");
  //  printf("\n");
  // }

  // printf("\n");
  // printf("\n");
  // printf("\n");
  // printf("\n");
  
  // for(int z=0;z< M;z++){
  //   for(int j=0;j<N;j++){
  //         printf(" %.2f ",matResult_ref[ z + j * M]);
  //   }
  //   printf("\n");
  //   printf("\n");
  // }
  

//   printf("\n");
//   printf("\n");

//   for(int z=0;z< M*N;z++){
//     printf(" %.2f ",matResult_opt[ z]);
//   }
 

  // for (uint64_t y = 0; y < M; y++) {
  //   for (uint64_t x = 0; x < N; x++) {
  //     if (fabsf(matResult_ref[x * N + y] - matResult_opt[x * N + y]) >
  //         fabsf((float)ERROR_TOLERANCE * matResult_ref[x * N + y])) {
  //       error = 1;
  //       printf("%lu, %f, %f\n", y * N + x, matResult_ref[x * N + y],matResult_opt[x * N + y]);
  //     }
  //   }
  // }


  for (uint64_t i = 0; i < M * N; i++) {
    if (fabsf(matResult_ref[i] - matResult_opt[i]) > fabsf((float)ERROR_TOLERANCE * matResult_ref[i])) {
        error = 1;
        printf("%llu, %f, %f\n", i, matResult_ref[i],matResult_opt[i]);
      }
  }

  printf("SGEMMxsme:  M=%d N=%d K=%d cost=%lf ops=%f Gflops = %lf \n", M, N, K, best1, ops, ops/best1);
  printf("Done\n\n");
  
  if (error)
  {
     printf("FAILED\n");
     printf("100 , %f, %f\n", matResult_ref[100],matResult_opt[100]);
  }
  else
  {
     printf("PASS\n");
     printf("100 , %f, %f\n", matResult_ref[100],matResult_opt[100]);
  }

  free(matLeft_mod);
  free(matLeft);
  free(matRight);
  free(matResult_ref);
  free(matResult_opt);

  return 0;
}