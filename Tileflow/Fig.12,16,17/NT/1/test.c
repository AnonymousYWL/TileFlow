//Fp32 output

#include "pack.h"
#define TEST_REPEAT 1

void Check_result(float *C, float *C1, long M, long N)
{

    int i, j, flag = 0;
    for( i= 0; i< M; i++)
    {
        for( j= 0 ;j < N; j++)
        {
            if( fabs( C[i*N + j]- C1[i*N + j]) > 1.0e-3)
            {
                printf("i = %d, j= %d\n", i,j);
                printf("C= %lf , C1= %lf\n", C[i*N + j], C1[i*N + j]);
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

void bf16gemm_amx(float *C, uint16_t *A, uint16_t *B, long M, long N, long K)
{


    int Tm = 1;
    int Tk=1;
    if(K>N && M>16)
    {
        Tk = 4;
    }
    int Tn = NUM /Tk;
    void *ptr;
    posix_memalign(&ptr, 64, (M*K)*sizeof( uint16_t ));
    uint16_t *Ac = ( uint16_t * ) ptr;

    //posix_memalign(&ptr, 64, ((Tk-1)*M*N)*sizeof( float ));
    //float *Cc = ( float * ) ptr;
    float **Cc = (float **)malloc((Tk-1) * sizeof(float*));

    int Int_M = M - M % 32;
    uint32_t tail_M = M - (M-1)/16*16;
    long size = M * N;
    int M_blocks = (M + 15) / 16;
    int K_blocks = (K + GEMM_K - 1) / GEMM_K;
    if(Tk > 1)
        K_blocks = Tk;

    // Load tile configuration 
    __tilecfg tile_data1={0}, tile_data2={0};
    init_tile_config(&tile_data1, 16);
    init_tile_config(&tile_data2, tail_M);


    #pragma omp parallel num_threads(NUM)
    {

        int i,j,k,kk,ii,jj,js, is,ks,ke;
        int mc, nc, mr, nr, kr, kc;
        int nb = (N>>4)/Tn<<4;
        int mb = M / Tm;
        int kb = K/ Tk;

        int id = omp_get_thread_num();
        int n_tid = id / Tk;
        int k_tid = id % Tk;
        int total_tasks = K_blocks * M_blocks;

        int numa_id = id / 32;  //每个NUMA节点32个线程
        int core_id = id % 32;
        int mapped_id = core_id * 2 + numa_id;  // 2个NUMA节点

        for(int task = mapped_id; task < total_tasks; task += NUM)
        //for (int task = id; task < total_tasks; task += NUM)
        {
            int ii_block = task / K_blocks;
            int k_slice  = task % K_blocks;

            ii = ii_block * 16;

            mr = 16;
            if (M - ii < 16)
                mr = M - ii;

            if (Tk > 1)
            {
                ks = k_slice * kb;
                ke = ks + kb;
            }
            else
            {
                ks = k_slice * GEMM_K;
                ke = ks + GEMM_K;
                if (ke > K)
                    ke = K;
            }

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


        if (id && id < Tk) 
        {
            posix_memalign((void**)&Cc[id - 1], 64, M * N * sizeof(float));
        }
        

        #pragma omp barrier
        is = 0;
        js = nb*n_tid;
        ks = id%Tk * kb;

        if( n_tid%Tn < (N>>4)%Tn) 
        {
            nb = nb + 16;
            js = js + (n_tid%Tn<<4);
        }
        else
        {
            js = js + ((N>>4)%Tn<<4);
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
                if(ks+kb - kk < GEMM_K)
                {
                    kc = ks+kb - kk;
                }

                for(ii = is; ii < is + mb; ii = ii + mc)
                {
                    mc = GEMM_M;
                    if(is + mb - ii < GEMM_M)
                        mc = is + mb - ii;


                    uint16_t *A0 =  Ac + ii*kc + kk*M;
                    for(j = jj; j < jj + nc; j = j + nr)
                    {
                        nr = 32;
                        if(jj+nc-j<32)
                        {
                            nr = jj+nc-j;
                        }

                        uint16_t *B0 = B  + j*K + kk;
                        float *C0 = C + ii * N + j;
                        if(id%Tk)
                        {
                            //C0 = Cc + (id%Tk-1)*M*N + ii * N + j;
                            C0 = Cc[id%Tk-1] + ii * N + j;
                        }

                        //init_tile_config(&tile_data, 16);
                        _tile_loadconfig(&tile_data1);
                        if(Int_M >= 32)
                        {
                            if(nr==32)
                            {
                                mKernel_32x32x32(C0, A0, B0, Int_M, kc,
                                        N<<2, K<<1, kk-ks);
                            }
                            else
                            {
                                mKernel_32x16x32(C0, A0, B0, Int_M, kc,
                                        N<<2, K<<1, kk-ks);
                            }
                        }
                        // EGDE_M
                        if(Int_M < M)
                        {
                            uint32_t edge_M;
                            for(i = Int_M; i < M; i = i + edge_M)
                            {
                                edge_M = 16;
                                if(M - i < 16)
                                {
                                    edge_M= M - i;
                                    //init_tile_config(&tile_data, tail_M);
                                    _tile_loadconfig(&tile_data2);
                                }

                                if(nr==32)
                                {
                                    mKernel_16x32x32(C0 + i*N, A0 + i*kc, B0, edge_M, kc,
                                            N<<2, K<<1, kk-ks);
                                }
                                else
                                {
                                    mKernel_16x16x32(C0 + i*N, A0 + i*kc, B0, edge_M, kc,
                                            N<<2, K<<1, kk-ks);
                                }
                            }

                        }
                    }
                }
            }
        }
        _tile_release ();

        // reduce
        if(Tk>1)
        {
            #pragma omp barrier
            long chunk = (size + NUM - 1) / NUM;
            long start = id * chunk;
            long end   = start + chunk;

            if(end > size)
                end = size;
            i = start;
            // vector part: 16 floats
            for(; i + 16 <= end; i += 16)
            {
                __m512 vc = _mm512_loadu_ps(C + i);

                for(int k = 0; k < Tk-1; k++)
                {
                    __m512 v = _mm512_loadu_ps(Cc[k] + i);
                    vc = _mm512_add_ps(vc, v);
                }

                _mm512_storeu_ps(C + i, vc);
            }


            // boundary: 8 floats
            if(i + 8 <= end)
            {
                __m256 vc = _mm256_loadu_ps(C + i);

                for(int k = 0; k < Tk-1; k++)
                {
                    __m256 v = _mm256_loadu_ps(Cc[k] + i);
                    vc = _mm256_add_ps(vc, v);
                }

                _mm256_storeu_ps(C + i, vc);

                i += 8;
            }
        }
    }
    free(Ac);
    for (int i = 0; i < Tk-1; i++) 
    {
        free(Cc[i]);
    }
    free(Cc);
    
}


int main()
{

    long loop = 100;
    long M, N, K;
    int i,j,m;
    void *ptr, *ptr1;

    // Request permission to linux kernel to run AMX 
    if (!set_tiledata_use())
    exit(-1);

    long Ms[] = {256};
    long Ns[] = {
    28672, 36864, 25600, 16384,
    13824, 14336, 11008, 7168, 
    8192, 4608, 5120, 6144,
    5120, 4096, 4096
    };

    long Ks[] = {
    8192, 4608, 5120, 6144,
    5120, 4096, 4096, 4096, 
    28672, 36864, 25600, 16384, 
    13824, 14336, 7168
    };

    FILE *fp;
    if( (fp=fopen("TileFlow_NT_FP32_6430_exp2_2.txt","w+")) == NULL )
    {
        puts("Fail to open file!");
        exit(0);
    }
       
    for(j = 0; j < 15; j++)
    {

        N = Ns[j];
        K = Ks[j];
        for(m = 1; m < 129; m = m + 1)
        {
            M = m;
            fprintf(fp, "%d %d %d", M,N,K);
            for(int pc =0; pc < 5; pc++)
            {
                double ops = 2.0*(double)M*N*K*1.0e-12; 
                double DV = (double)M*K*2*2/1024/1024/1024;

                posix_memalign(&ptr, 64, ((M*K)*sizeof( uint16_t )));
                uint16_t *A = ( uint16_t * ) ptr;
                posix_memalign(&ptr, 8192, (N*K*TEST_REPEAT*sizeof( uint16_t )));
                uint16_t *B = ( uint16_t * ) ptr;
                posix_memalign(&ptr, 64, (M*N*TEST_REPEAT*sizeof( float )));
                float *C = ( float *) ptr;
                posix_memalign(&ptr, 64, (M*N*sizeof( float )));
                float *C1 = ( float *) ptr;
                
                random_matrix(M, K, A);
                for(int ii =0;ii < TEST_REPEAT; ii++)
                    random_matrix(K, N, B+ii*N*K);

                // warm up
                for(int ii =0;ii < TEST_REPEAT; ii++)
                {
                    bf16gemm_amx(C+ii*M*N, A, B+ii*N*K, M, N, K);
                }
                
                double start = dclock();

                for(i = 0;i < loop/TEST_REPEAT; i++)
                {
                    for(int ii =0;ii < TEST_REPEAT; ii++)
                    {
                        bf16gemm_amx(C+ii*M*N, A, B+ii*N*K, M, N, K);
                    }
                } 
           
                double cost = (dclock() -start)/loop;
              
                printf("M = %d, N = %d, K = %d, ms = %.3f, GB/s = %.3f, Tflops = %.3f\n", 
                    M, N, K, cost*1000,DV/cost,ops/cost);
                fprintf(fp, " %.3f", ops/cost);



                cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasTrans,
                            M, N, K, 1, A, K, B, K, 0, C1, N);

                Check_result(C, C1, M, N);
      

                free(A);
                free(B);
                free(C);
                free(C1); 
            }
            fprintf(fp, "\n"); 
        }
        
    }

    fclose(fp);
    return 0;
}
