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
            if( fabs( BF16_toFP32( C[i*N + j])- C1[i*N + j]) > 3.0e-1)
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

void QKV_project(uint16_t *C, uint16_t *A, uint16_t *B, long M, long N, long K)
{


    int Tm = 1;
    int Tk = 2;
    int Tn = NUM /Tk;
    void *ptr;
    posix_memalign(&ptr, 64, (M*K)*sizeof( uint16_t ));
    uint16_t *Inter_data = ( uint16_t * ) ptr;
    float *Cc[NUM];

    long size = M * N;
    int M_blocks = (M + 15) / 16;
    int K_blocks = (K + 255) / 256;

    #pragma omp parallel num_threads(NUM)
    {
        // Load tile configuration 
        __tilecfg tile_data={0};
        init_tile_config(&tile_data,M);
        int i,j,k,kk,ii,jj,js, is,ks,ke;
        int mc, nc, mr, nr, kr, kc;
        int nb = (N>>4)/Tn<<4;
        int mb = M / Tm;
        int kb = GEMM_K;

        int id = omp_get_thread_num();
        int n_tid = id / Tk;
        int k_tid = id % Tk;
        
        int numa_id = id / 32;  //每个NUMA节点32个线程
        int core_id = id % 32;
        int mapped_id = core_id * 2 + numa_id;  // 2个NUMA节点

        posix_memalign((void**)&Cc[id],64,M*nb*sizeof( float ));

        uint16_t *Ac = A;
        if(M>1)
        {
            Ac = Inter_data;
            int total_tasks = K_blocks * M_blocks;
            for(int task = mapped_id; task < total_tasks; task += NUM)
            {
                int ii_block = task / K_blocks;
                int k_slice  = task % K_blocks;

                ii = ii_block * 16;

                mr = 16;
                if (M - ii < 16)
                    mr = M - ii;

                ks = k_slice * 256;
                ke = ks + 256;
                
                for (kk = ks; kk < ke; kk += GEMM_K)
                {
                    kc = GEMM_K;
                    if (ke - kk < GEMM_K)
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
        }

        is = 0;
        n_tid = core_id;
        nb = (N>>4)/32<<4;  //每个NUMA有32个线程
        js = nb*n_tid;

        if(n_tid < (N>>4)%32)
        {
            nb += 16;
            js += (n_tid<<4);
        }
        else
        {
            js += ((N>>4)%32<<4);
        }


        for(jj = js; jj < js + nb; jj = jj + nc)
        {
            nc = GEMM_N;
            if(js + nb - jj < GEMM_N)
            {
                nc = js + nb - jj;
            }

            ks = numa_id * kb;

            for(kk = ks; kk < K; kk = kk + kb*2)
            {
                kc = kb;

                for(ii = is; ii < is + mb; ii = ii + mc)
                {
                    mc = GEMM_M;
                    if(is + mb - ii < GEMM_M)
                        mc = is + mb - ii;

                    uint16_t *A0 = Ac + ii*kc + kk*M;
                    for(j = jj; j < jj + nc; j = j + nr)
                    {
                        nr = 32;
                        if(jj+nc-j<32)
                        {
                            nr = jj+nc-j;
                        }

                        uint16_t *B0 = B  + j*K + kk;
                        float *C0 = Cc[id] + ii*nb + j-jj;
                        long stride_C = nb<<2;
                        mKernel_16x32x32(C0, A0, B0, mc, kc,
                                stride_C, K<<1, kk-ks);

                    }
                
                }
            }
        }
        _tile_release ();


        #pragma omp barrier

        /* ---------- 双 NUMA 并行 reduce：从 mb（M 行）维度等分 ---------- */
        long m_start = (mb * numa_id) / 2;      // 当前 NUMA 节点负责的起始行
        long m_end   = (mb * (numa_id + 1)) / 2;// 负责结束行（开区间）

        // 所有线程（numa_id == 0 或 1）都参与
        // numa_id==1 读本地 Cc[core_id]；numa_id==0 读远程 Cc[core_id]
        float *src1 = Cc[core_id];
        float *src2 = Cc[32+core_id];
        uint16_t *dst = C + js;                    // 全局 C 中该 core_id 对应的列块起始
    
        for(i = m_start; i < m_end; i++)
        {
            // 主循环：一次处理 32 个 bf16（加载 2×16 float，转成 512bit bf16）
            for (j=0; j + 32 <= nb; j += 32)
            {
                __m512 vc_lo = _mm512_loadu_ps(src1 + i * nb + j);
                __m512 vc_hi = _mm512_loadu_ps(src1 + i * nb + j + 16);
                __m512 vs_lo = _mm512_loadu_ps(src2 + i * nb + j);
                __m512 vs_hi = _mm512_loadu_ps(src2 + i * nb + j + 16);

                vc_lo = _mm512_add_ps(vc_lo, vs_lo);
                vc_hi = _mm512_add_ps(vc_hi, vs_hi);

                // vc_hi 存结果高 16 个 bf16，vc_lo 存低 16 个
                __m512bh vbf16 = _mm512_cvtne2ps_pbh(vc_hi, vc_lo);
                _mm512_storeu_si512(dst + i * N + j, (__m512i)vbf16);
            }

        }
    }

    free(Inter_data);
    for (int i = 0; i < NUM; i++)
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


    FILE *fp;
    if( (fp=fopen("TileFlow_QKV_6430_interleave.txt","w+")) == NULL )
    {
        puts("Fail to open file!");
        exit(0);
    }
       
    for(j = 1; j < 17; j++)
    {

        N = 6144;
        K = 4096;
        M = j;
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
            posix_memalign(&ptr, 64, (M*N*sizeof( float )));
            float *C1 = ( float *) ptr;
            
            random_matrix(M, K, A);
            random_matrix(K, N, B);

            // warm up
            QKV_project(C, A, B, M, N, K);

            start = dclock();
            for(i = 0; i < loop; i++)
            {   
                QKV_project(C, A, B, M, N, K);
            } 
            cost = (dclock() -start)/loop;
          
            printf("M = %d, N = %d, K = %d, ms = %.3f, GB/s = %.3f, Tflops = %.3f\n", 
                M, N, K, cost*1000,DV/cost,ops/cost);
            fprintf(fp, " %.3f", ops/cost);

            
            cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasTrans,
                        M, N, K, 1, A, K, B, K, 0, C1, N);

            Check_result(C, C1, M, N);

            printf("%f %f\n", BF16_toFP32( C[10]),C1[10]);
            
     

            free(A);
            free(B);
            free(C);
            free(C1); 
        }
        fprintf(fp, "\n"); 
    }

    fclose(fp);
    return 0;
}
