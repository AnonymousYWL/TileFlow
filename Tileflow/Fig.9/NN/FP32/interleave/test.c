//Fp32 output

#include "pack.h"

void Check_result(float *C, float *C1, long M, long N)
{

    int i, j, flag = 0;
    for( i= 0; i< M; i++)
    {
        for( j= 0 ;j < N; j++)
        {
            if( fabs( C[i*N + j]- C1[i*N + j]) > 0.1)
            {
                printf("i = %d, j= %d\n", i,j);
                printf("C= %lf , C1= %lf\n", C[i*N + j], C1[i*N + j]);
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


void bf16gemm_amx(float *C, uint16_t *A, uint16_t *B, long M, long N, long K)
{


    int Tm = 1;
    int Tn = NUM/Tm;

    int nb = N / Tn;
    int mb = M / Tm;
    uint8_t vpermt2d_control[] = {0, 32, 1, 33, 2, 34, 3, 35, 4, 36, 5, 37, 6, 38, 7, 39,
                                8, 40, 9, 41, 10, 42, 11, 43, 12, 44, 13, 45, 14, 46, 15, 47};
    void *ptr;
    posix_memalign(&ptr, 64, NUM*GEMM_K*64*sizeof( uint16_t ));
    uint16_t *Bc = ( uint16_t * ) ptr;


    #pragma omp parallel num_threads(NUM)
    {
        // Load tile configuration 
        __tilecfg tile_data={0};
        init_tile_config(&tile_data);
        int i,j,k,ii,jj,kk,js, is;
        int mc, nc, kc, nr, kr;
        int id = omp_get_thread_num();
        int nb = (N>>4)/Tn<<4;
        int mb = M / Tm;

        js = id%32/8*4096 + id/32*2048 + id % 8 * nb;
        is = mb * (id/Tn);

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
                        float *C0 = C + ii * N + j;
                        if(nr==32)
                        {
                            mKernel_32x32x32(C0, A0, B0, Bc + id*GEMM_K*64 + GEMM_K*(j%64),
                                            mc, kc, N<<1, K<<1, kk, vpermt2d_control, j %64);
                        }
                    }
                }
            }
        }

        _tile_release ();
    }
    free(Bc);
    
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
    if( (fp=fopen("GEMMs_TileFlow_FP32_6430.txt","w+")) == NULL )
    {
        puts("Fail to open file!");
        exit(0);
    }
       
    for(j = 0 ;j < 32; j++)
    {

        N = Ns[j];
        K = Ks[j];
        M = 32 + 32 * j;
        fprintf(fp, "%d %d %d", M,N,K);
        for(int pc =0; pc < 5; pc++)
        {
            double ops = 2.0*(double)M*N*K*1.0e-12; 

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

            bf16gemm_amx(C, A, B, M, N, K);

            double start = dclock();
            for(i = 0;i < loop; i++)
            {
                bf16gemm_amx(C, A, B, M, N, K);
     
            } 
       
            double cost = (dclock() -start)/loop;
          
            printf("M = %d, N = %d, K = %d, ms = %.3f, Tflops = %.3f\n", M, N, K, cost*1000, ops/cost);
            fprintf(fp, " %.3f", ops/cost);


            cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        M, N, K, 1, A, K, B, N, 0, C1, N);

            Check_result(C, C1, M, N);

            printf("%f %f\n", C[11], C1[11]);

     

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
