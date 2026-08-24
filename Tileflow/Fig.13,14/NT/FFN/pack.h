#include <immintrin.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "ukernel.h"
#include "../utils.h"
#include <sys/syscall.h>
#include <unistd.h>
#include <stdbool.h>
#include <sys/time.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <mkl.h>
#include <vector>
#include <cstdint>
#include <random>

using namespace std;

#define MAX_ROWS 16
#define MAX_COLS 64
#define STRIDE 64
#define ARCH_GET_XCOMP_PERM     0x1022
#define ARCH_REQ_XCOMP_PERM     0x1023
#define XFEATURE_XTILECFG       17
#define XFEATURE_XTILEDATA      18


#define GEMM_K 6144
#define GEMM_M 512
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
static void init_tile_config (__tilecfg *tileinfo, uint8_t M)
{
    int i;
    tileinfo->palette_id = 1;
    tileinfo->start_row = 0;
  
    for (i = 0; i < 14; i++)
    {
        tileinfo->reserved_0[i] = 0;
    }


    tileinfo->colsb[0] = M<<2;
    tileinfo->rows[0] =  16;

    tileinfo->colsb[2] = 64;
    tileinfo->rows[2] =  16;

    tileinfo->colsb[3] = 64;
    tileinfo->rows[3] =  16;

    tileinfo->colsb[4] = M<<2;
    tileinfo->rows[4] =  16;

    tileinfo->colsb[5] = M<<2;
    tileinfo->rows[5] =  16;

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
    #pragma omp parallel for num_threads(64)
    for (int i = 0; i < M*N; ++i) {
    // 每个线程有自己的生成器（通过 thread_local 或局部变量）
    thread_local std::mt19937 gen(std::random_device{}());
    thread_local std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    src1[i] = FP32_toBF16(dis(gen));
}
}

void random_matrix1(int M, int N, uint16_t *src1) 
{
    #pragma omp parallel for num_threads(64)
    for (int i = 0; i < M*N; ++i) {
    // 每个线程有自己的生成器（通过 thread_local 或局部变量）
    src1[i] = FP32_toBF16(1.0);
}
}

void transform_filter(uint16_t* trans_F, uint16_t* F, int K, int C, int R, int S)
{

    int kk,k,c,r,s, cc, kr, cr, k1,k2;

    int index=0;

    for(kk = 0; kk < K; kk = kk + kr)
    {
        kr = 32;
        if(K - kk < 32)
        {
            kr = K - kk;
        }
        for(r= 0;r < R; r++)
        {
            for(s = 0;s < S; s++)
            {
                for(cc = 0; cc < C; cc = cc + cr)
                {
                    cr = 32;
                    if(C - cc < 32)
                    {
                        cr = C - cc;
                    }
                    for(k = kk; k < kk+kr; k = k + k1)
                    {

                        k1 = 16;
                        if(kk+kr-k < 16)
                        {
                            k1= kk+kr - k;
                        }

                        for(c = cc; c < cc+cr; c = c + 2)
                        {

                            for(k2= k; k2 < k + k1; k2++)
                            {
                                trans_F[index++] = F[k2*C*R*S+c*R*S+r*S+s];
                                trans_F[index++] = F[k2*C*R*S+(c+1)*R*S+r*S+s];
                            }
                        }
                    }
                }

            }
        }
    }

}

namespace mydata {
    vector<uint32_t>index = {0,    2048, 1, 0, 
                             2048, 2048, 1, 0,
                             0,    2048, 4, 2048, 16384, 2048, 3, 2048,
                             2048, 2048, 3, 2048, 14336, 2048, 4, 2048};
    uint8_t vpermt2d_control[] = {0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23,
                            8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31};//交叉数据

    float Va[]={0x1.715476p+0f, -0x1.62E430p-1f, 
              0x1.0F9F9Cp-7f, 0x1.573A1Ap-5f, 0x1.555A80p-3f, 0x1.FFFDC6p-2f, 
              0x1.FFFFF6p-1f, 1.0f};   // exp计算

}