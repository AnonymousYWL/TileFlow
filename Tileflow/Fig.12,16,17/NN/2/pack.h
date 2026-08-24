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

#define GEMM_K 2560
#define GEMM_M 256
#define GEMM_N 10240

#define PAGE_ELEM 2048
#define PAIR_ELEM 4096
#define GRAIN 16
#define MAX_SEG 3

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


    tileinfo->colsb[0] = MAX_COLS;
    tileinfo->rows[0] =  MAX_ROWS;
    tileinfo->colsb[1] = MAX_COLS;
    tileinfo->rows[1] =  M;

    tileinfo->colsb[2] = MAX_COLS;
    tileinfo->rows[2] =  MAX_ROWS;
    tileinfo->colsb[3] = MAX_COLS;
    tileinfo->rows[3] =  MAX_ROWS;

    tileinfo->colsb[4] = MAX_COLS;
    tileinfo->rows[4] =  MAX_ROWS;
    tileinfo->colsb[5] = MAX_COLS;
    tileinfo->rows[5] =  MAX_ROWS;

    tileinfo->colsb[6] = MAX_COLS;
    tileinfo->rows[6] =  M;
    tileinfo->colsb[7] = MAX_COLS;
    tileinfo->rows[7] =  M;
    

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
    double drand48();
    for (int i = 0; i < M*N; ++i) 
    {
        src1[i] = FP32_toBF16( 1.0 );
    }
}
/*
static inline long numa_offset_to_global(long off, int node)
{
    long page = off / PAGE_ELEM;
    long in_page = off % PAGE_ELEM;
    long real_page = page * 2 + node;
    return real_page * PAGE_ELEM + in_page;
}



void schedule_thread(long N, int Tn, int tid, thread_task *t)
{
    int T = Tn / 2;
    int node = tid / T;
    int local_tid = tid % T;

    long Np = (N + PAGE_ELEM - 1) / PAGE_ELEM;

    long Ne = 0;
    for(long p=node; p<Np; p+=2)
    {
        long s = p * PAGE_ELEM;
        long e = s + PAGE_ELEM;
        if(e > N) e = N;
        Ne += (e - s);
    }

    long chunk = (Ne / T) / GRAIN * GRAIN;

    long start = local_tid * chunk;
    long end   = start + chunk;

    long page0 = start / PAGE_ELEM;
    long page1 = (end-1) / PAGE_ELEM;

    if(page0 == page1)
    {
        t->nseg = 1;
        t->js[0] = numa_offset_to_global(start,node);
        t->nb[0] = chunk;
    }
    else
    {
        long cut = (page0+1) * PAGE_ELEM;

        t->nseg = 2;

        t->js[0] = numa_offset_to_global(start,node);
        t->nb[0] = cut - start;

        t->js[1] = numa_offset_to_global(cut,node);
        t->nb[1] = end - cut;
    }
}
*/

typedef struct {
    int nseg;
    long js[MAX_SEG];
    long nb[MAX_SEG];
} thread_task;


static inline long numa_to_global(long off, int node)
{
    long page = off / PAGE_ELEM;
    long in_page = off % PAGE_ELEM;

    long real_page = page*2 + node;

    return real_page * PAGE_ELEM + in_page;
}

void schedule_thread(long N, int Tn, int tid, thread_task *t)
{
    int T = Tn/2;

    long bulk = (N / PAIR_ELEM) * PAIR_ELEM;
    long tail = N - bulk;

    int node = tid / T;
    int local = tid % T;

    t->nseg = 0;

    /* ---------- bulk ---------- */

    long pages = bulk / PAGE_ELEM;

    long pages_node = (pages + (node ? 0 : 1)) / 2;

    long size = pages_node * PAGE_ELEM;

    long chunk = (size / T) / GRAIN * GRAIN;

    long start = local * chunk;
    long end   = start + chunk;

    if(chunk > 0)
    {
        long p0 = start / PAGE_ELEM;
        long p1 = (end-1) / PAGE_ELEM;

        if(p0 == p1)
        {
            t->js[t->nseg] = numa_to_global(start,node);
            t->nb[t->nseg] = chunk;
            t->nseg++;
        }
        else
        {
            long cut = (p0+1)*PAGE_ELEM;

            t->js[t->nseg] = numa_to_global(start,node);
            t->nb[t->nseg] = cut-start;
            t->nseg++;

            t->js[t->nseg] = numa_to_global(cut,node);
            t->nb[t->nseg] = end-cut;
            t->nseg++;
        }
    }

    /* ---------- tail ---------- */

    if(tail)
    {
        long base = (tail/Tn)/GRAIN*GRAIN;

        long rem = (tail - base*Tn)/GRAIN;

        long my = base;
        if(tid < rem)
            my += GRAIN;

        if(my)
        {
            long offset;

            if(tid < rem)
                offset = tid*(base+GRAIN);
            else
                offset = rem*(base+GRAIN) + (tid-rem)*base;

            t->js[t->nseg] = bulk + offset;
            t->nb[t->nseg] = my;
            t->nseg++;
        }
    }
}
void classify_range(uint32_t start, uint32_t end, uint32_t N,
                    uint8_t *flag0, uint8_t *flag1,
                    uint32_t *count0, uint32_t *count1)
{
    uint32_t c0 = 0, c1 = 0;

    for(uint32_t A=start; A<end; A++)
    {
        if((A*N+1024)/2048%2==0)
        {
            flag0[A] = 1;
            c0++;
        }
        else
        {
            flag1[A] = 1;
            c1++;
        }
    }

    *count0 = c0;
    *count1 = c1;
}