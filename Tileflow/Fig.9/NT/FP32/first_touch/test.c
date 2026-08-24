//Fp32 output

#include "pack.h"

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
        printf("result correct\n");

}

void bf16gemm_amx(float *C, uint16_t *A, uint16_t *B, long M, long N, long K)
{


    int Tm = 1;
    int Tk = 1;
    int Tn = NUM /Tk;
    void *ptr;
    posix_memalign(&ptr, 64, (M*K)*sizeof( uint16_t ));
    uint16_t *Ac = ( uint16_t * ) ptr;

    posix_memalign(&ptr, 64, ((Tk-1)*M*N)*sizeof( float ));
    float *Cc = ( float * ) ptr;

    long size = M * N;
    int M_blocks = (M + 15) / 16;
    int K_blocks = (K + GEMM_K - 1) / GEMM_K;
    if(Tk > 1)
        K_blocks = Tk;

    #pragma omp parallel num_threads(NUM)
    {

        // Load tile configuration 
        __tilecfg tile_data={0};
        init_tile_config(&tile_data);
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


                    uint16_t *A0 = Ac + ii*kc + kk*M;
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
                            C0 = Cc + (id%Tk-1)*M*N + ii * N + j;
                        }

                        if(nr==32)
                        {
                            mKernel_32x32x32(C0, A0, B0, mc, kc,
                                    N<<2, K<<1, kk-ks);
                        }
                        else
                        {
                            mKernel_32x16x32(C0, A0, B0, mc, kc,
                                    N<<2, K<<1, kk-ks);
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
            // vector part 
            for(; i + 16 <= end; i += 16)
            {
                __m512 vc = _mm512_loadu_ps(C + i);

                for(int k = 0; k < Tk-1; k++)
                {
                    __m512 v = _mm512_loadu_ps(Cc + k*size + i);
                    vc = _mm512_add_ps(vc, v);
                }

                _mm512_storeu_ps(C + i, vc);
            }
        }
    }
    free(Ac);
    free(Cc);
    
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
    long Ns[] = {
    28672, 8192, 36864, 4608, 25600,
    5120, 16384, 6144, 13824, 5120,
    14336, 4096, 11008, 7168, 4096};
    long Ks[] = {
    8192, 28672, 4608, 36864, 5120,
    25600, 6144, 16384, 5120, 13824,
    4096, 14336, 4096, 4096, 7168};

    FILE *fp;
    if( (fp=fopen("MeGEMM_NT_FP32_6430.txt","w+")) == NULL )
    {
        puts("Fail to open file!");
        exit(0);
    }
       
    for(j = 0 ;j < 32; j++)
    {

        N = 16384;
        K = 16384;
        M = 32 + j*32;
        fprintf(fp, "%d %d %d", M,N,K);
        for(int pc =0; pc < 1; pc++)
        {
            double ops = 2.0*(double)M*N*K*1.0e-12; 
            double DV = (double)M*K*2*2/1024/1024/1024;
            double start=0, cost = 0;

            posix_memalign(&ptr, 64, ((M*K)*sizeof( uint16_t )));
            uint16_t *A = ( uint16_t * ) ptr;
            posix_memalign(&ptr, 8192, (N*K*sizeof( uint16_t )));
            uint16_t *B = ( uint16_t * ) ptr;
            posix_memalign(&ptr, 64, (M*N*sizeof( float )));
            float *C = ( float *) ptr;
            posix_memalign(&ptr, 64, (M*N*sizeof( float )));
            float *C1 = ( float *) ptr;
            
            random_matrix(M, K, A);
            random_matrix(K, N, B);

            // warm up
            for(i = 0; i < 3; i++)
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

    fclose(fp);
    return 0;
}
