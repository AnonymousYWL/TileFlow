#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include "mkl.h"
#include <omp.h>
// #include <cblas.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <cstdint>
#include <numeric> // for std::accumulate
#include <random>
#include <algorithm>

#define NUM 64
static double ref_time_sec = 0.0;

double dclock()
{
    struct timespec ts;
    double now;

    clock_gettime(CLOCK_MONOTONIC, &ts);

    now = (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;

    if (ref_time_sec == 0.0)
        ref_time_sec = now;

    return now - ref_time_sec;
}

// 将 FP32 转换为 BF16
uint16_t FP32_toBF16(float fp32_value) 
{
    // 将浮点数的二进制表示强制转换为无符号 32 位整数
    uint32_t fp32_bits;
    memcpy(&fp32_bits, &fp32_value, sizeof(fp32_bits));
    
    // 截取高 16 位作为 BF16 格式，低 16 位丢弃（精度截断）
    uint16_t bf16_bits = (uint16_t)(fp32_bits >> 16);
    
    return bf16_bits;
}

// 将 BF16 转换回 FP32
float BF16_toFP32(uint16_t bf16_value) {
    // 创建一个 32 位的无符号整数，将 BF16 的高 16 位放入 FP32 的高位
    uint32_t fp32_bits = (uint32_t)bf16_value << 16;
    
    // 将二进制表示强制转换回浮点数
    float fp32_value;
    memcpy(&fp32_value, &fp32_bits, sizeof(fp32_value));
    
    return fp32_value;
}


void random_matrix(int M, int N, uint16_t *src1) 
{
    #pragma omp parallel for num_threads(NUM)
    for (int i = 0; i < M*N; ++i) {
    // 每个线程有自己的生成器（通过 thread_local 或局部变量）
    thread_local std::mt19937 gen(std::random_device{}());
    thread_local std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    src1[i] = FP32_toBF16(dis(gen));
}
}


int main()
{
    mkl_set_num_threads(NUM);
    int i, j, k, jj, pc, m;
    int loop = 100;
    long M, N, K;
    double start, cost;
    double gflops;
    int flag = 0;
    float alpha = 1.0, beta = 0.0;
    
    void *ptr, *ptr1;

    FILE *fp;
    if ((fp = fopen("MKL_NT_FP32_6430_exp2.txt", "w")) == NULL)
    {
        puts("Fail to open file!");
        exit(0);
    }

    long Ns[] = {
    28672, 36864, 25600, 16384,
    13824, 14336, 11008, 7168, 
    8192, 4608, 5120, 6144,
    5120, 4096, 4096,
    10240, 16545, 11243
    };

    long Ks[] = {
    8192, 4608, 5120, 6144,
    5120, 4096, 4096, 4096,
    28672, 36864, 25600, 16384,
    13824, 14336, 7168,
    192, 128, 96
    };

    for (j = 0; j < 18; j++)
    {

        N = Ns[j];
        K = Ks[j];

        int TEST_REPEAT=500;
        if(j<15)
        {
            TEST_REPEAT = 10;
        }
        for(m = 1; m < 129; m++)
        {
            M = m;
            double ops = (double)M * N * K * 1.0e-12 * 2;
            fprintf(fp, "%d %d %d", M, N, K);

        	for(pc =0; pc < 5; pc++)
        	{
         
                double start=0, cost = 0;
                posix_memalign(&ptr, 64, ((M*K)*sizeof( uint16_t )));
                uint16_t *A = ( uint16_t * ) ptr;
                posix_memalign(&ptr, 8192, (N*K*TEST_REPEAT*sizeof( uint16_t )));
                uint16_t *B = ( uint16_t * ) ptr;
                posix_memalign(&ptr, 64, (M*N*TEST_REPEAT*sizeof( float )));
                float *C = ( float *) ptr;
                
                // init matrix
                random_matrix(M, K, A);
                for(int ii =0;ii < TEST_REPEAT; ii++)
                    random_matrix(K, N, B+ii*N*K);



                // warm up
               // for (i = 0; i < 5; i++)
               for(int ii = 0; ii <TEST_REPEAT; ii++)
               cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasTrans,
                                       M, N, K, alpha, A, K, B+ii *N*K, K, beta, C +ii *N*M , N);

                start = dclock();
                for (i = 0; i < loop/TEST_REPEAT; i++)
                {
                    for(int ii = 0; ii <TEST_REPEAT; ii++)
                    cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasTrans,
                                       M, N, K, alpha, A, K, B+ii *N*K, K, beta, C +ii *N*M , N);

                }
    	        cost = (dclock() - start) / loop;

                printf("bf16_mkl:  M= %-6d N=%-6d K=%-6d Tflops = %-10.3lf\n",
                         M, N, K, ops / cost);
                fprintf(fp, " %.3f", ops / cost);


                free(A);
                free(B);
                free(C);
            }
            fprintf(fp, "\n");
    	}
	   
    }
    fclose(fp);
    return 0;
}
