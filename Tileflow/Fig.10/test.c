//Fp32 output

#include "pack.h"

void Check_result(int *C, int *C1, long M, long N)
{

    int i, j, flag = 0;
    for( i= 0; i< M; i++)
    {
        for( j= 0 ;j < N; j++)
        {
            if( fabs( C[i*N + j]- C1[i*N + j]) > 0.5)
            {
                printf("i = %d, j= %d\n", i,j);
                printf("C= %d , C1= %d\n", C[i*N + j], C1[i*N + j]);
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


void bf16gemm_amx(int *C, int8_t *A, uint8_t *B, long M, long N, long K)
{


    int Tm = 1;
    int Tn = NUM/Tm;

    int nb = N / Tn;
    int mb = M / Tm;
    uint8_t vpermt2d_control[] = {0, 64, 1, 65, 2, 66, 3, 67, 4, 68, 5, 69, 6, 70, 7, 71,
                                  8, 72, 9, 73, 10,74, 11,75, 12,76, 13,77, 14,78, 15,79,
                                  16,80, 17,81, 18,82, 19,83, 20,84, 21,85, 22,86, 23,87,
                                  24,88, 25,89, 26,90, 27,91, 28,92, 29,93, 30,94, 31,95,
                                  0, 32, 1, 33, 2, 34, 3, 35, 4, 36, 5, 37, 6, 38, 7, 39,
                                  8, 40, 9, 41, 10,42, 11,43, 12,44, 13,45, 14,46, 15,47};
    void *ptr;
    posix_memalign(&ptr, 64, NUM*GEMM_K*128*sizeof( uint8_t ));
    uint8_t *Bc = ( uint8_t * ) ptr;


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

        js = id%32/16*8192 + id/32*4096 + id % 16 * nb;
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


                    int8_t *A0 = A + ii*K + kk;
                    for(j = jj; j < jj + nc; j = j + nr)
                    {
                        nr = 32;
                        if(jj+nc-j<32)
                        {
                            nr = jj+nc-j;
                        }

                        //pack B
                        uint8_t *B0 = B + kk*N + j;
                        int *C0 = C + ii * N + j;
                        if(nr==32)
                        {
                            mKernel_32x32x32(C0, A0, B0, Bc + id*kc*128 + kc*(j%128),
                                            mc, kc, N, K, kk, vpermt2d_control, j %128);
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

    int8_t oa = 0;          // A 的 zero-point
    int8_t ob = 0;          // B 的 zero-point
    int32_t co = 0;         // C 的 offset（FixOffset 模式下只需一个值）

    // Request permission to linux kernel to run AMX 
    if (!set_tiledata_use())
    exit(-1);

    long Ms[] = {1024};
    long Ks[] = {8192*2};
    long Ns[] = {256*NUM};

    FILE *fp;
    if( (fp=fopen("GEMMs_Tileflow_INT8_6430.txt","w+")) == NULL )
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

            posix_memalign(&ptr, 64, ((M*K)*sizeof( int8_t )));
            int8_t *A = ( int8_t * ) ptr;
            posix_memalign(&ptr, 8192, (N*K*sizeof( uint8_t )));
            uint8_t *B = ( uint8_t * ) ptr;
            posix_memalign(&ptr, 64, (M*N*sizeof( int )));
            int *C = ( int *) ptr;
            posix_memalign(&ptr, 64, (M*N*sizeof( int )));
            int *C1 = ( int *) ptr;
            
            random_matrix_int8(M, K, A);
            random_matrix_uint8(K, N, B);

            
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
            cblas_gemm_s8u8s32(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            CblasFixOffset,       // 统一偏移模式
            M, N, K,
            1.0f,                 // alpha 是 float！
            A, K,
            oa,                   // A 无偏移
            B, N,
            ob,                   // B 无偏移
            0.0f,                 // beta 是 float！
            C1, N,
            &co                   // C 无偏移
            );

            Check_result(C, C1, M, N);
            printf("+++++++++ %d\n", C[0]);

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
