#include <immintrin.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "ukernel.h"
#include <sys/syscall.h>
#include <unistd.h>
#include <stdbool.h>
#include <sys/time.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <mkl.h>
#include <numa.h>
#include <numeric> // for std::accumulate
#include <random>
#include <algorithm>

#define MAX_ROWS 16
#define MAX_COLS 64
#define STRIDE 64
#define ARCH_GET_XCOMP_PERM     0x1022
#define ARCH_REQ_XCOMP_PERM     0x1023
#define XFEATURE_XTILECFG       17
#define XFEATURE_XTILEDATA      18


#define NUM 64

#define GEMM_K 2048
#define GEMM_M 256
#define GEMM_N 10240

//Define tile config data structure 
typedef struct __tile_config
{
  uint8_t palette_id;
  uint8_t start_row;
  uint8_t reserved_0[14];
  uint16_t colsb[16]; 
  uint8_t rows[16]; 
} __tilecfg;

/* Initialize tile config */
static void init_tile_config (__tilecfg *tileinfo)
{
    int i;
    tileinfo->palette_id = 1;
    tileinfo->start_row = 0;
  
    for (i = 0; i < 14; i++)
    {
        tileinfo->reserved_0[i] = 0;
    }

    for (i = 0; i < 8; i++)
    {
        tileinfo->colsb[i] = MAX_COLS;
        tileinfo->rows[i] =  MAX_ROWS;
    }

    _tile_loadconfig(tileinfo);
}

/* Set_tiledata_use() - Invoke syscall to set ARCH_SET_STATE_USE */
static bool set_tiledata_use()
{
   if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) 
   {
      printf("\n Fail to do XFEATURE_XTILEDATA \n\n");
      return false;
   }
   else
   {
      printf("\n TILE DATA USE SET - OK \n\n");
      return true;
   }

   return true;
}

static double gtod_ref_time_sec = 0.0;
double dclock()
{
        double the_time, norm_sec;
        struct timeval tv;

        gettimeofday( &tv, NULL );

        if ( gtod_ref_time_sec == 0.0 )
                gtod_ref_time_sec = ( double ) tv.tv_sec;

        norm_sec = ( double ) tv.tv_sec - gtod_ref_time_sec;
        the_time = norm_sec + tv.tv_usec * 1.0e-6;
        return the_time;
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

void random_matrix1(int M, int N, uint16_t *src1) 
{
    #pragma omp parallel for num_threads(NUM)
    for (int i = 0; i < M*N; ++i) {
    // 每个线程有自己的生成器（通过 thread_local 或局部变量）
    src1[i] = FP32_toBF16(1.0);
}
}


void transform_layout(uint16_t* Ac, uint16_t* A, int M, int K)
{
    int kk, k, ii, i, kc, mr;
    int index = 0;

    for (kk = 0; kk < K; kk += kc)
    {
        kc = GEMM_K;
        if (K - kk < GEMM_K)
            kc = K - kk;

        for (ii = 0; ii < M; ii += mr)
        {
            mr = 16;
            if (M - ii < 16)
                mr = M - ii;

            for (k = kk; k < kk + kc; k += 2)
            {
                for (i = ii; i < ii + mr; i++)
                {
                    Ac[index++] = A[i*K + k];
                    Ac[index++] = A[i*K + k + 1];
                }
            }
        }
    }
}
