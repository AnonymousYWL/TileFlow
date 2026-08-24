//Fp32 output

#include "pack.h"

void Check_result(uint16_t *C, float *C1, long M, long N)
{

    int i, j, flag = 0;
    for( i= 0; i< M; i++)
    {
        for( j= 0 ;j < N; j++)
        {
            if( fabs( BF16_toFP32 (C[i*N + j])- C1[i*N + j]) > 4)
            {
                printf("i = %d, j= %d\n", i,j);
                printf("C= %lf , C1= %lf\n", BF16_toFP32(C[i*N + j]), 
                    C1[i*N + j]);
                flag =1;
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


void bf16gemm_amx(uint16_t *C, uint16_t *A, uint16_t *B, long M, long N, long K)
{


    int Tm = 1;
    int Tn = NUM/Tm;
    int nb = N / Tn;
    int mb = M / Tm;
    void *ptr;
    posix_memalign(&ptr, 64, M*K*sizeof( uint16_t ));
    uint16_t *Ac = ( uint16_t * ) ptr;

    int K_blocks = (K + GEMM_K - 1) / GEMM_K;
    int M_blocks = (M + 15) / 16;

    #pragma omp parallel num_threads(NUM)
    {
        // Load tile configuration 
        __tilecfg tile_data={0};
        init_tile_config(&tile_data);
        int i,j,k,ii,jj,kk,js, is;
        int mc, nc, kc, nr, kr;
        int nb = (N>>4)/Tn<<4;
        int mb = M / Tm;

        int id = omp_get_thread_num();
        
        int total_tasks = K_blocks * M_blocks;
        for (int task = id; task < total_tasks; task += NUM)
        {
            int ii_block = task / K_blocks;
            int kk_block = task % K_blocks;

            kk = kk_block * GEMM_K;
            ii = ii_block * 16;

            kc = GEMM_K;
            if (K - kk < GEMM_K)
                kc = K - kk;

            int mr = 16;
            if (M - ii < 16)
                mr = M - ii;

            int index = kk * M + ii * kc;
            transpose(Ac + index, A + ii*K + kk, mr, kc, K<<1);
        }

        #pragma omp barrier

        js = nb*(id%Tn);
        is = 0;//mb * (id/Tn);

        if( id%Tn < (N>>4)%Tn) 
        {
            nb = nb + 16;
            js = js + (id%Tn<<4);
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

            for(kk = 0; kk < K; kk = kk + kc)
            {
                kc = GEMM_K;
                if(K - kk < GEMM_K)
                    kc = K - kk;


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
                        uint16_t *C0 = C + ii * N + j;
                        if(nr==32)
                        {
                            mKernel_32x32x32(C0, A0, B0, mc, kc,
                                    N<<1, K<<1, kk);
                        }

                    }
                }
            }
            //#pragma omp barrier
        }

        _tile_release ();
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

    long Ms[] = {1024};
    long Ks[] = {8192*2/*1024*/};
    long Ns[] = {256*NUM};

    FILE *fp;
    if( (fp=fopen("MeGEMM_NT_BF16_6430.txt","w+")) == NULL )
    {
        puts("Fail to open file!");
        exit(0);
    }
       
    for(j = 0 ;j < 32; j++)
    {

        N = Ns[0];
        K = Ks[0];
        M = 32 + 32 * j;
        fprintf(fp, "%d %d %d", M,N,K);
        for(int pc =0; pc < 5; pc++)
        {
            double ops = 2.0*(double)M*N*K*1.0e-12; 

            posix_memalign(&ptr, 64, ((M*K)*sizeof( uint16_t )));
            uint16_t *A = ( uint16_t * ) ptr;
            posix_memalign(&ptr, 64, ((M*K)*sizeof( uint16_t )));
            uint16_t *Ac = ( uint16_t * ) ptr;
            posix_memalign(&ptr, 8192, (N*K*sizeof( uint16_t )));
            uint16_t *B = ( uint16_t * ) ptr;
            posix_memalign(&ptr, 64, (M*N*sizeof( uint16_t )));
            uint16_t *C = ( uint16_t *) ptr;
            posix_memalign(&ptr, 64, (M*N*sizeof( float )));
            float *C1 = ( float *) ptr;
            
            random_matrix(M, K, A);
            random_matrix(K, N, B);

            //transform_layout(Ac,A,M,K);

            bf16gemm_amx(C, A, B, M, N, K);
            double start = dclock();
            for(i = 0;i < loop; i++)
            {
                bf16gemm_amx(C, A, B, M, N, K);
     
            } 
       
            double cost = (dclock() -start)/loop;
          
            printf("M = %d, N = %d, K = %d, ms = %.3f, GB/s = %.3f, Tflops = %.3f\n", 
                M, N, K, cost*1000, (double)N*K*2/1024/1024/1024/cost,ops/cost);
            fprintf(fp, " %.3f", ops/cost);

/*
            cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasTrans,
                        M, N, K, 1, A, K, B, K, 0, C1, N);

            Check_result(C, C1, M, N);
*/
     

            free(A);
            free(Ac);
            free(B);
            free(C);
            free(C1);      
        }
        fprintf(fp, "\n"); 
    }

    fclose(fp);
    return 0;
}
