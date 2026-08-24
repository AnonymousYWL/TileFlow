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
    uint8_t vpermt2d_control[] = {0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23,
                                0,1,4,5,2,3,6,7,8,9,12,13,10,11,14,15};
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
                        uint16_t *C0 = C + ii *N + j;
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
    if( (fp=fopen("GEMMs_eagerGEMM_6430.txt","a+")) == NULL )
    {
        puts("Fail to open file!");
        exit(0);
    }
       
    for(j = 7 ;j < 8; j++)
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
            posix_memalign(&ptr, 64, (M*N*sizeof( uint16_t )));
            uint16_t *C = ( uint16_t *) ptr;
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

/*
            cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        M, N, K, 1, A, K, B, N, 0, C1, N);

            Check_result(C, C1, M, N);

  */    

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
