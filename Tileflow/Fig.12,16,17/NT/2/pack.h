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


#define NUM 32

#define GEMM_K 2048
#define GEMM_M 320
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
static void init_tile_config (__tilecfg *tileinfo, uint8_t M, uint8_t N)
{
    int i;
    tileinfo->palette_id = 1;
    tileinfo->start_row = 0;
  
    for (i = 0; i < 14; i++)
    {
        tileinfo->reserved_0[i] = 0;
    }

    tileinfo->colsb[0] = MAX_COLS;
    tileinfo->rows[0] =  MAX_ROWS;

    tileinfo->colsb[1] = M*4;
    tileinfo->rows[1] =  MAX_ROWS;

    tileinfo->colsb[2] = MAX_COLS;
    tileinfo->rows[2] =  MAX_ROWS;

    tileinfo->colsb[3] = MAX_COLS;
    tileinfo->rows[3] =  N;

    tileinfo->colsb[4] = MAX_COLS;
    tileinfo->rows[4] =  MAX_ROWS;

    tileinfo->colsb[5] = MAX_COLS;
    tileinfo->rows[5] =  N;

    tileinfo->colsb[6] = M*4;
    tileinfo->rows[6] =  MAX_ROWS;

    tileinfo->colsb[7] = M*4;
    tileinfo->rows[7] =  N;

    //_tile_loadconfig(tileinfo);
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


void transpose(uint16_t* Ac, uint16_t* A, int mr, int K, long stride_K)
{

    asm volatile(

    "   mov       %[A], %%rax                       \n"
    "   mov       %[Ac], %%rbx                      \n"
    "   movl      %[mr], %%r8d                      \n"
    "   movl      %[K], %%r12d                      \n"
    "   mov       %[stride_K], %%r11                \n"

    "   mov       %%r8, %%rsi                       \n"
    "   shl       $2, %%rsi                         \n"

    "   mov      $1, %%r13                          \n"
    "   shlx     %%r8, %%r13, %%r13                 \n"
    "   subq     $1, %%r13                          \n"
    "   kmovq    %%r13, %%k2                        \n"

    "   mov       $4, %%rdx                         \n"
    "   shlx      %%rdx, %%rsi, %%rdx               \n"

    "0:                                             \n"

    "   vmovdqu16      (%%rax), %%zmm0              \n"
    "   cmp          $2, %%r8                       \n"
    "   jl           1f                             \n"
    "   vmovdqu16      (%%rax, %%r11), %%zmm1       \n"
    "   cmp          $3, %%r8                       \n"
    "   jl           1f                             \n"
    "   vmovdqu16      (%%rax, %%r11, 2), %%zmm2    \n"

    "   cmp          $4, %%r8                       \n"
    "   jl           1f                             \n"
    "   leaq         (%%rax, %%r11, 2), %%rcx       \n"
    "   vmovdqu16      (%%rcx, %%r11), %%zmm3       \n"
    "   cmp          $5, %%r8                       \n"
    "   jl           1f                             \n"
    "   vmovdqu16      (%%rcx, %%r11, 2), %%zmm4    \n"
    "   cmp          $6, %%r8                       \n"
    "   jl           1f                             \n"
    "   leaq         (%%rcx, %%r11, 2), %%rcx       \n"
    "   vmovdqu16      (%%rcx, %%r11), %%zmm5       \n"
    "   cmp          $7, %%r8                       \n"
    "   jl           1f                             \n"
    "   vmovdqu16      (%%rcx, %%r11, 2), %%zmm6    \n"
    "   cmp          $8, %%r8                       \n"
    "   jl           1f                             \n"
    "   leaq         (%%rcx, %%r11, 2), %%rcx       \n"
    "   vmovdqu16      (%%rcx, %%r11), %%zmm7       \n"
    "   cmp          $9, %%r8                       \n"
    "   jl           1f                             \n"
    "   vmovdqu16      (%%rcx, %%r11, 2), %%zmm8    \n"
    "   cmp          $10, %%r8                      \n"
    "   jl           1f                             \n"
    "   leaq         (%%rcx, %%r11, 2), %%rcx       \n"
    "   vmovdqu16      (%%rcx, %%r11), %%zmm9       \n"

    "   cmp          $11, %%r8                      \n"
    "   jl           1f                             \n"
    "   vmovdqu16      (%%rcx, %%r11, 2), %%zmm10   \n"

    "   cmp          $12, %%r8                      \n"
    "   jl           1f                             \n"
    "   leaq         (%%rcx, %%r11, 2), %%rcx       \n"
    "   vmovdqu16      (%%rcx, %%r11), %%zmm11      \n"
    "   cmp          $13, %%r8                      \n"
    "   jl           1f                             \n"
    "   vmovdqu16      (%%rcx, %%r11, 2), %%zmm12   \n"
    "   cmp          $14, %%r8                      \n"
    "   jl           1f                             \n"
    "   leaq         (%%rcx, %%r11, 2), %%rcx       \n"
    "   vmovdqu16      (%%rcx, %%r11), %%zmm13      \n"
    "   cmp          $15, %%r8                      \n"
    "   jl           1f                             \n"
    "   vmovdqu16      (%%rcx, %%r11, 2), %%zmm14   \n"
    "   cmp          $16, %%r8                      \n"
    "   jl           1f                             \n"
    "   leaq         (%%rcx, %%r11, 2), %%rcx       \n"
    "   vmovdqu16      (%%rcx, %%r11), %%zmm15      \n"

    "1:                                             \n"

    "   vunpcklps %%zmm1, %%zmm0, %%zmm16           \n"
    "   vunpcklps %%zmm3, %%zmm2, %%zmm17           \n"
    "   vunpcklpd %%zmm17, %%zmm16, %%zmm18         \n"
    "   vunpckhpd %%zmm17, %%zmm16, %%zmm28         \n"

    "   vunpcklps %%zmm5, %%zmm4, %%zmm16           \n"
    "   vunpcklps %%zmm7, %%zmm6, %%zmm17           \n"
    "   vunpcklpd %%zmm17, %%zmm16, %%zmm19         \n"
    "   vunpckhpd %%zmm17, %%zmm16, %%zmm29         \n"

    "   vunpcklps %%zmm9, %%zmm8, %%zmm16           \n"
    "   vunpcklps %%zmm11, %%zmm10, %%zmm17         \n"
    "   vunpcklpd %%zmm17, %%zmm16, %%zmm20         \n"
    "   vunpckhpd %%zmm17, %%zmm16, %%zmm30         \n"

    "   vunpcklps %%zmm13, %%zmm12, %%zmm16         \n"
    "   vunpcklps %%zmm15, %%zmm14, %%zmm17         \n"
    "   vunpcklpd %%zmm17, %%zmm16, %%zmm21         \n"
    "   vunpckhpd %%zmm17, %%zmm16, %%zmm31         \n"


    "   vinsertf32x4    $1, %%xmm19, %%zmm18, %%zmm22\n"
    "   vinsertf32x4    $1, %%xmm21, %%zmm20, %%zmm23\n"
    "   vinsertf32x8    $1, %%ymm23, %%zmm22, %%zmm24\n"

    "   vextractf32x4   $1, %%zmm18, %%xmm22         \n"
    "   vinsertf32x4    $0, %%xmm22, %%zmm19, %%zmm22\n"
    "   vextractf32x4   $1, %%zmm20, %%xmm23         \n"
    "   vinsertf32x4    $0, %%xmm23, %%zmm21, %%zmm23\n"
    "   vinsertf32x8    $1, %%ymm23, %%zmm22, %%zmm25\n"

    "   vextractf32x4   $2, %%zmm18, %%xmm22         \n"
    "   vextractf32x4   $2, %%zmm19, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm26\n"
    "   vextractf32x4   $2, %%zmm20, %%xmm22         \n"
    "   vextractf32x4   $2, %%zmm21, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm22\n"
    "   vinsertf32x8    $1, %%ymm22, %%zmm26, %%zmm26\n"

    "   vextractf32x4   $3, %%zmm18, %%xmm22         \n"
    "   vextractf32x4   $3, %%zmm19, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm27\n"
    "   vextractf32x4   $3, %%zmm20, %%xmm22         \n"
    "   vextractf32x4   $3, %%zmm21, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm22\n"
    "   vinsertf32x8    $1, %%ymm22, %%zmm27, %%zmm27\n"


    "   vmovups         %%zmm24, (%%rbx)%{%%k2}      \n"
    "   vmovups         %%zmm25, (%%rbx, %%rsi, 4)%{%%k2}   \n"
    "   vmovups         %%zmm26, (%%rbx, %%rsi, 8)%{%%k2}   \n"
    "   leaq            (%%rbx, %%rsi, 8), %%r14     \n"
    "   vmovups         %%zmm27, (%%r14, %%rsi, 4)%{%%k2}   \n"

//----------------------------------------------------

    "2:                                              \n"

    "   vinsertf32x4    $1, %%xmm29, %%zmm28, %%zmm22\n"
    "   vinsertf32x4    $1, %%xmm31, %%zmm30, %%zmm23\n"
    "   vinsertf32x8    $1, %%ymm23, %%zmm22, %%zmm24\n"

    "   vextractf32x4   $1, %%zmm28, %%xmm22         \n"
    "   vinsertf32x4    $0, %%xmm22, %%zmm29, %%zmm22\n"
    "   vextractf32x4   $1, %%zmm30, %%xmm23         \n"
    "   vinsertf32x4    $0, %%xmm23, %%zmm31, %%zmm23\n"
    "   vinsertf32x8    $1, %%ymm23, %%zmm22, %%zmm25\n"

    "   vextractf32x4   $2, %%zmm28, %%xmm22         \n"
    "   vextractf32x4   $2, %%zmm29, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm26\n"
    "   vextractf32x4   $2, %%zmm30, %%xmm22         \n"
    "   vextractf32x4   $2, %%zmm31, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm22\n"
    "   vinsertf32x8    $1, %%ymm22, %%zmm26, %%zmm26\n"

    "   vextractf32x4   $3, %%zmm28, %%xmm22         \n"
    "   vextractf32x4   $3, %%zmm29, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm27\n"
    "   vextractf32x4   $3, %%zmm30, %%xmm22         \n"
    "   vextractf32x4   $3, %%zmm31, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm22\n"
    "   vinsertf32x8    $1, %%ymm22, %%zmm27, %%zmm27\n"

    "   vmovups         %%zmm24, (%%rbx, %%rsi)%{%%k2}      \n"
    "   leaq            (%%rbx, %%rsi), %%r14        \n"
    "   vmovups         %%zmm25, (%%r14, %%rsi, 4)%{%%k2}   \n"
    "   vmovups         %%zmm26, (%%r14, %%rsi, 8)%{%%k2}   \n"
    "   leaq            (%%r14, %%rsi, 8), %%r14     \n"
    "   vmovups         %%zmm27, (%%r14, %%rsi, 4)%{%%k2}   \n"


    "3:                                             \n"
    //----------------------------------------------------
    
    "   vunpckhps %%zmm1, %%zmm0, %%zmm16           \n"
    "   vunpckhps %%zmm3, %%zmm2, %%zmm17           \n"
    "   vunpcklpd %%zmm17, %%zmm16, %%zmm18         \n"
    "   vunpckhpd %%zmm17, %%zmm16, %%zmm28         \n"

    "   vunpckhps %%zmm5, %%zmm4, %%zmm16           \n"
    "   vunpckhps %%zmm7, %%zmm6, %%zmm17           \n"
    "   vunpcklpd %%zmm17, %%zmm16, %%zmm19         \n"
    "   vunpckhpd %%zmm17, %%zmm16, %%zmm29         \n"

    "   vunpckhps %%zmm9, %%zmm8, %%zmm16           \n"
    "   vunpckhps %%zmm11, %%zmm10, %%zmm17         \n"
    "   vunpcklpd %%zmm17, %%zmm16, %%zmm20         \n"
    "   vunpckhpd %%zmm17, %%zmm16, %%zmm30         \n"

    "   vunpckhps %%zmm13, %%zmm12, %%zmm16         \n"
    "   vunpckhps %%zmm15, %%zmm14, %%zmm17         \n"
    "   vunpcklpd %%zmm17, %%zmm16, %%zmm21         \n"
    "   vunpckhpd %%zmm17, %%zmm16, %%zmm31         \n"


    "   vinsertf32x4    $1, %%xmm19, %%zmm18, %%zmm22\n"
    "   vinsertf32x4    $1, %%xmm21, %%zmm20, %%zmm23\n"
    "   vinsertf32x8    $1, %%ymm23, %%zmm22, %%zmm24\n"

    "   vextractf32x4   $1, %%zmm18, %%xmm22         \n"
    "   vinsertf32x4    $0, %%xmm22, %%zmm19, %%zmm22\n"
    "   vextractf32x4   $1, %%zmm20, %%xmm23         \n"
    "   vinsertf32x4    $0, %%xmm23, %%zmm21, %%zmm23\n"
    "   vinsertf32x8    $1, %%ymm23, %%zmm22, %%zmm25\n"

    "   vextractf32x4   $2, %%zmm18, %%xmm22         \n"
    "   vextractf32x4   $2, %%zmm19, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm26\n"
    "   vextractf32x4   $2, %%zmm20, %%xmm22         \n"
    "   vextractf32x4   $2, %%zmm21, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm22\n"
    "   vinsertf32x8    $1, %%ymm22, %%zmm26, %%zmm26\n"

    "   vextractf32x4   $3, %%zmm18, %%xmm22         \n"
    "   vextractf32x4   $3, %%zmm19, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm27\n"
    "   vextractf32x4   $3, %%zmm20, %%xmm22         \n"
    "   vextractf32x4   $3, %%zmm21, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm22\n"
    "   vinsertf32x8    $1, %%ymm22, %%zmm27, %%zmm27\n"


    "   vmovups         %%zmm24, (%%rbx, %%rsi, 2)%{%%k2}   \n"
    "   leaq            (%%rbx, %%rsi, 2), %%r14     \n"
    "   vmovups         %%zmm25, (%%r14, %%rsi, 4)%{%%k2}   \n"
    "   vmovups         %%zmm26, (%%r14, %%rsi, 8)%{%%k2}   \n"
    "   leaq            (%%r14, %%rsi, 8), %%r14     \n"
    "   vmovups         %%zmm27, (%%r14, %%rsi, 4)%{%%k2}   \n"


    "4:                                             \n"

    "   vinsertf32x4    $1, %%xmm29, %%zmm28, %%zmm22\n"
    "   vinsertf32x4    $1, %%xmm31, %%zmm30, %%zmm23\n"
    "   vinsertf32x8    $1, %%ymm23, %%zmm22, %%zmm24\n"

    "   vextractf32x4   $1, %%zmm28, %%xmm22         \n"
    "   vinsertf32x4    $0, %%xmm22, %%zmm29, %%zmm22\n"
    "   vextractf32x4   $1, %%zmm30, %%xmm23         \n"
    "   vinsertf32x4    $0, %%xmm23, %%zmm31, %%zmm23\n"
    "   vinsertf32x8    $1, %%ymm23, %%zmm22, %%zmm25\n"

    "   vextractf32x4   $2, %%zmm28, %%xmm22         \n"
    "   vextractf32x4   $2, %%zmm29, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm26\n"
    "   vextractf32x4   $2, %%zmm30, %%xmm22         \n"
    "   vextractf32x4   $2, %%zmm31, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm22\n"
    "   vinsertf32x8    $1, %%ymm22, %%zmm26, %%zmm26\n"

    "   vextractf32x4   $3, %%zmm28, %%xmm22         \n"
    "   vextractf32x4   $3, %%zmm29, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm27\n"
    "   vextractf32x4   $3, %%zmm30, %%xmm22         \n"
    "   vextractf32x4   $3, %%zmm31, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm22\n"
    "   vinsertf32x8    $1, %%ymm22, %%zmm27, %%zmm27\n"


    "   leaq            (%%rbx, %%rsi, 2), %%r14     \n"
    "   leaq            (%%r14, %%rsi), %%r14        \n"
    "   vmovups         %%zmm24, (%%r14)%{%%k2}             \n"
    "   vmovups         %%zmm25, (%%r14, %%rsi, 4)%{%%k2}   \n"
    "   vmovups         %%zmm26, (%%r14, %%rsi, 8)%{%%k2}   \n"
    "   leaq            (%%r14, %%rsi, 8), %%r14     \n"
    "   vmovups         %%zmm27, (%%r14, %%rsi, 4)%{%%k2}   \n"


    "5:                                             \n"

    "   addq        $64, %%rax                      \n"
    "   addq        %%rdx, %%rbx                    \n"
    "   subq        $32, %%r12                      \n"
    "   cmp         $0, %%r12                       \n"
    "   jg          0b                              \n"


    :
    :
      [A] "m" (A),
      [Ac] "m" (Ac),
      [mr] "m" (mr),
      [K] "m" (K),
      [stride_K] "m" (stride_K)
    :"rax", "rbx", "rcx", "rdx", "rdi", "rsi","rbp","r8", "r9", "r10", "r11", "r12",
     "r13", "r14", "r15","k1","k2","k3","zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
     "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
     "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
     "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
     "zmm30", "zmm31", "memory"
    );

}