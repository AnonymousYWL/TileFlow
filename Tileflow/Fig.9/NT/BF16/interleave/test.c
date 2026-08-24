//Fp32 output
// page_interleaving

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
                flag = 1;
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
    int Tn = NUM / Tk;

    void *ptr;

    posix_memalign(&ptr,64,(M*K)*sizeof(uint16_t));
    uint16_t *Ac = (uint16_t *)ptr;

    uint16_t *Cc[NUM/2];

    long size = M * N;
    int M_blocks = (M + 15) / 16;
    int K_blocks = (K + GEMM_K - 1) / GEMM_K;

    #pragma omp parallel num_threads(NUM)
    {

        __tilecfg tile_data={0};
        init_tile_config(&tile_data);

        int i,j,k,kk,ii,jj,js,is,ks,ke;
        int mc,nc,mr,nr,kc,nb;
        int id = omp_get_thread_num();
        int n_tid = id / Tk;
        int k_tid = id % Tk;

        int numa_id = id / 32;
        int core_id = id % 32;

        int mapped_id = core_id * 2 + numa_id;
        int total_tasks = K_blocks * M_blocks;

        is = 0;

        nb = (N >> 4) / 32 << 4;
        js = nb * core_id;

        if(core_id < (N >> 4) % 32)
        {
            nb += 16;
            js += (core_id << 4);
        }
        else
        {
            js += ((N >> 4) % 32 << 4);
        }

        int mb = M;

        if(numa_id)
        {
            posix_memalign(
            (void**)&Cc[core_id],
            64,
            M*nb*sizeof( uint16_t )
            );

        }

        for(int task = mapped_id; task < total_tasks; task += NUM)
        {

            int ii_block = task / K_blocks;
            int k_slice  = task % K_blocks;

            ii = ii_block * 16;

            mr = 16;
            if(M - ii < 16)
                mr = M - ii;

            ks = k_slice * GEMM_K;
            ke = ks + GEMM_K;

            if(ke > K)
                ke = K;

            for(kk = ks; kk < ke; kk += GEMM_K)
            {

                kc = GEMM_K;

                if(ke - kk < GEMM_K)
                    kc = ke - kk;


                int index = kk * M + ii * kc;

                transpose(
                    Ac + index,
                    A + ii * K + kk,
                    mr,
                    kc,
                    K << 1
                );
            }
        }

        #pragma omp barrier

        for(jj = js; jj < js + nb; jj = jj + nc)
        {

            nc = GEMM_N;

            if(js + nb - jj < GEMM_N)
                nc = js + nb - jj;


            ks = numa_id * GEMM_K;


            for(kk = ks; kk < K; kk = kk + GEMM_K * 2)
            {

                kc = GEMM_K;


                for(ii = is; ii < is + mb; ii = ii + mc)
                {

                    mc = GEMM_M;

                    if(is + mb - ii < GEMM_M)
                        mc = is + mb - ii;

                    uint16_t *A0 = Ac + ii*kc + kk*M;

                    for(j = jj; j < jj + nc; j = j + nr)
                    {

                        nr = 32;

                        if(jj + nc - j < 32)
                            nr = jj + nc - j;


                        uint16_t *B0 = B + j*K + kk;
                        uint16_t *C0 = C + ii*N + j;
                        long stride_C = N<<1;
                        if(numa_id)
                        {
                            C0 = Cc[core_id] + ii*nb + j-jj;
                            stride_C = nb<<1;
                        }
                        if(nr == 32)
                        {
                            mKernel_32x32x32(C0, A0, B0, mc, kc, stride_C, K<<1, kk-ks);
                        }

                    }

                }

            }

        }

        _tile_release();

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


    for (int i = 0; i < NUM/2; i++)
    {
        free(Cc[i]);
    }
    free(Ac);

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


    FILE *fp;
    if( (fp=fopen("MeGEMM_NT_BF16_6430_interleave.txt","w+")) == NULL )
    {
        puts("Fail to open file!");
        exit(0);
    }
       
    for(j = 0; j < 32; j++)
    {

        N = 16384;
        K = 16384;
        M = 32 + j * 32;
        fprintf(fp, "%d %d %d", M,N,K);
        for(int pc =0; pc < 5; pc++)
        {
            double ops = 2.0*(double)M*N*K*1.0e-12; 
            double DV = (double)M*K*2*2/1024/1024/1024;
            double start=0, cost = 0;

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

            // warm up
            bf16gemm_amx(C, A, B, M, N, K);

            start = dclock();
            for(i = 0; i < loop; i++)
            {   
                bf16gemm_amx(C, A, B, M, N, K);
            } 
            cost = (dclock() -start)/loop;
          
            printf("M = %d, N = %d, K = %d, ms = %.3f, GB/s = %.3f, Tflops = %.3f\n", 
                M, N, K, cost*1000,DV/cost,ops/cost);
            fprintf(fp, " %.3f", ops/cost);

            /*
            cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasTrans,
                        M, N, K, 1, A, K, B, K, 0, C1, N);

            Check_result(C, C1, M, N);
            */
     

            free(A);
            free(B);
            free(C);
            //free(C1); 
        }
        fprintf(fp, "\n"); 
    }

    fclose(fp);
    return 0;
}
