//Fp32 output

#include "pack.h"

void Check_result(uint16_t *C, float *C1, long M, long N)
{

    int i, j, flag = 0;
    for( i= 0; i< M; i++)
    {
        for( j= 0 ;j < N; j++)
        {
            if( fabs( BF16_toFP32(C[i*N + j])- C1[i*N + j]) > 5)
            {
                printf("i = %d, j= %d\n", i,j);
                printf("C= %lf , C1= %lf\n", BF16_toFP32(C[i*N + j]), C1[i*N + j]);
                flag =1;
                printf("error\n");
                break;
            }           
        }
        if(flag == 1)
            break;
    }

    if(flag == 0)
        printf("结果正确\n");

}


void bf16gemm_amx(uint16_t *C, uint16_t *A, uint16_t *B, long M, long N, long K)
{


    int Tm = 1;
    int Tk = 2;
    int Tn = NUM/Tm/Tk;

    uint8_t vpermt2d_control[] = {0, 32, 1, 33, 2, 34, 3, 35, 4, 36, 5, 37, 6, 38, 7, 39,
                                8, 40, 9, 41, 10, 42, 11, 43, 12, 44, 13, 45, 14, 46, 15, 47};
    void *ptr;
    posix_memalign(&ptr, 64, NUM*GEMM_K*64*sizeof( uint16_t ));
    uint16_t *Bc = ( uint16_t * ) ptr;

    uint16_t *Cc[NUM/2];

    #pragma omp parallel num_threads(NUM)
    {
        // Load tile configuration 
        __tilecfg tile_data={0};
        init_tile_config(&tile_data);
        int j,k,i, ii,jj,kk,js, is, ks;
        int mc, nc, kc, nr, kr;
        int id = omp_get_thread_num();
        int nb = (N>>4)/Tn<<4;
        int mb = M / Tm;
        int kb= K/Tk;
        long size = M * N;

        int numa_id = id / 32;  //每个NUMA节点32个线程
        int core_id = id % 32;

        js = core_id * nb;
        is = mb * (id/Tn/Tk);

        if(core_id < (N>>4)%Tn)
        {
            nb += 16;
            js += (core_id<<4);
        }
        else
        {
            js += ((N>>4)%Tn<<4);
        }

        ks = numa_id * kb;

        if(numa_id)
        {
            posix_memalign(
            (void**)&Cc[core_id],
            64,
            M*nb*sizeof( uint16_t )
            );

        }

        for(jj = js; jj < js + nb; jj = jj + nc)
        {
            nc = GEMM_N;
            if(js + nb - jj < GEMM_N)
            {
                nc = js + nb - jj;
            }

            for(kk = ks; kk < ks+kb; kk = kk + kc)
            {
                kc = GEMM_K;
                if(ks + kb - kk < GEMM_K)
                    kc = ks + kb - kk;


                for(ii = is; ii < is + mb; ii = ii + mc)
                {
                    mc = GEMM_M;
                    if(is + mb - ii < GEMM_M)
                        mc = is + mb - ii;


                    uint16_t *A0 = A + ii*K + kk;
                    for(j = jj; j < jj + nc; j = j + nr)
                    {
                        nr = 32;
                        if(jj+nc-j<32)
                        {
                            nr = jj+nc-j;
                        }

                        //pack B
                        uint16_t *B0 = B + kk*N + j;
                        uint16_t *C0 = C + ii * N + j;
                        long stride_C = N<<1;
                        if(numa_id)
                        {
                            C0 = Cc[core_id] + ii*nb + j-jj;
                            stride_C = nb<<1;
                        }


                        if(nr==32)
                        {
                            mKernel_32x32x32(C0, A0, B0, Bc + id*GEMM_K*64 + kc*(j%64),
                                            mc, kc, stride_C, N<<1, K<<1, kk-ks, 
                                            vpermt2d_control, j %64);
                        }
                    }
                }
            }
        }

        _tile_release ();


        #pragma omp barrier
        /* ---------- 双 NUMA 并行 reduce：从 mb（M 行）维度等分 ---------- */
        
        uint16_t *src = Cc[core_id];   // numa_id==0 读远程 Cc；numa_id==1 读本地 Cc
        if (src != NULL)               // 仅多 NUMA 场景下执行（单 NUMA 时 Cc 全为 NULL）
        {
            long m_start = (mb * numa_id) / 2;        // 当前 NUMA 节点负责的起始行
            long m_end   = (mb * (numa_id + 1)) / 2;  // 负责结束行（开区间）

            uint16_t *dst = C + js;                   // 全局 C 中该 core_id 对应的列块起始

            for (long m = m_start; m < m_end; m++)
            {
                int n = 0;
                // AVX-512 向量化：Cc 行跨度 nb，C 行跨度 N
                for (; n + 16 <= nb; n += 16)
                {
                    __m256bh vc16 = (__m256bh)_mm256_loadu_si256((__m256i *)(dst + m * N + n));
                    __m512   vc   = _mm512_cvtpbh_ps(vc16);

                    __m256bh vs16 = (__m256bh)_mm256_loadu_si256((__m256i *)(src + m * nb + n));
                    __m512   vs   = _mm512_cvtpbh_ps(vs16);

                    vc = _mm512_add_ps(vc, vs);

                    __m256bh out = _mm512_cvtneps_pbh(vc);
                    _mm256_storeu_si256((__m256i *)(dst + m * N + n), (__m256i)out);
                }
                // 尾部兜底（nb 通常已是 16 倍数，这里保险）
                for (; n < nb; n++)
                {
                    uint32_t c_u = ((uint32_t)dst[m * N + n]) << 16;
                    uint32_t s_u = ((uint32_t)src[m * nb + n]) << 16;
                    float c_f = *(float *)&c_u;
                    float s_f = *(float *)&s_u;
                    float sum = c_f + s_f;
                    dst[m * N + n] = (uint16_t)((*(uint32_t *)&sum) >> 16);
                }
            }
        }
        
    }

    free(Bc);
    for (int i = 0; i < NUM/2; i++)
    {
        free(Cc[i]);
    }
    
}

int main()
{

    long loop = 100;
    long M, N, K;
    int i,j;
    void *ptr, *ptr1;

    // Request permission to linux kernel to run AMX 
    if (!set_tiledata_use())
    exit(-1);

    long Ms[] = {1024};
    long Ks[] = {8192*2};
    long Ns[] = {256*NUM};

    FILE *fp;
    if( (fp=fopen("TileFlow_NN_BF16_6430.txt","w+")) == NULL )
    {
        puts("Fail to open file!");
        exit(0);
    }
       
    for(j = 0 ;j < 32; j++)
    {

        N = 16384;
        K = 16384;
        M = 32 + 32 * j;
        fprintf(fp, "%d %d %d", M,N,K);
        for(int pc =0; pc < 5; pc++)
        {
            double ops = 2.0*(double)M*N*K*1.0e-12; 

            posix_memalign(&ptr, 64, ((M*K)*sizeof( uint16_t )));
            uint16_t *A = ( uint16_t * ) ptr;
            posix_memalign(&ptr, 8192, (N*K*sizeof( uint16_t )));
            uint16_t *B = ( uint16_t * ) ptr;
            posix_memalign(&ptr, 64, (M*N*sizeof( uint16_t )));
            uint16_t *C = ( uint16_t *) ptr;
            //posix_memalign(&ptr, 64, (M*N*sizeof( float )));
            //float *C1 = ( float *) ptr;
            
            random_matrix(M, K, A);
            random_matrix(K, N, B);

            bf16gemm_amx(C, A, B, M, N, K);

            double start = dclock();
            for(i = 0;i < loop; i++)
            {
                bf16gemm_amx(C, A, B, M, N, K);
     
            } 
       
            double cost = (dclock() -start)/loop;
          
            printf("M = %d, N = %d, K = %d, ms = %.3f, Tflops = %.3f\n", M, N, K, cost*1000, ops/cost);
            fprintf(fp, " %.3f", ops/cost);

/*
            cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        M, N, K, 1, A, K, B, N, 0, C1, N);

            Check_result(C, C1, M, N);

            printf("%f %f\n", BF16_toFP32(C[11]), C1[11]);

  */   

            free(A);
            free(B);
            free(C);
    //        free(C1);      
        }
        fprintf(fp, "\n"); 
    }

    fclose(fp);
    return 0;
}
