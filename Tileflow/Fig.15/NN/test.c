// Fp32 output
// UMA测试

#include "pack.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

void Check_result(float *C, float *C1, long M, long N)
{
    int i, j, flag = 0;
    for(i = 0; i < M; i++)
    {
        for(j = 0; j < N; j++)
        {
            if(fabs(C[i*N + j] - C1[i*N + j]) > 0.5)
            {
                printf("i = %d, j = %d\n", i, j);
                printf("C = %lf, C1 = %lf\n", C[i*N + j], C1[i*N + j]);
                flag = 1;
                printf("error\n");
                break;
            }
        }
        if(flag == 1) break;
    }

    if(flag == 0)
        printf("结果正确\n");
}

void bf16gemm_amx(float *C, uint16_t *A, uint16_t *B, long M, long N, long K)
{
    int Tm = 1;
    int Tn = NUM / Tm;
    uint8_t vpermt2d_control[] = {0, 32, 1, 33, 2, 34, 3, 35, 4, 36, 5, 37, 6, 38, 7, 39,
                                8, 40, 9, 41, 10, 42, 11, 43, 12, 44, 13, 45, 14, 46, 15, 47};
    void *ptr;
    posix_memalign(&ptr, 64, NUM * GEMM_K * 64 * sizeof(uint16_t));
    uint16_t *Bc = (uint16_t *) ptr;

    int Int_M = M - M % 32;
    uint32_t tail_M = M - (M-1)/16*16;

    thread_task *tasks = (thread_task *) malloc(sizeof(thread_task) * Tn);


    // Load tile configuration 
    __tilecfg tile_data[2]={0,0};
    init_tile_config(tile_data,16);
    init_tile_config(tile_data+1, tail_M);

    #pragma omp parallel num_threads(NUM)
    {

        int i,j,k,ii,jj,kk,js,is,t,G;
        int mc, nc, kc, nr, kr;
        int id = omp_get_thread_num();
        int nb = (N>>4) / Tn << 4;
        int mb = M / Tm;

        js = nb * (id % Tn);
        is = mb * (id / Tn);

        if(id % Tn < (N>>4) % Tn)
        {
            nb += 16;
            js += (id % Tn << 4);
        }
        else
        {
            js += ((N>>4) % Tn << 4);
        }

        uint8_t N_tail = nb % 64;

        for(kk = 0; kk < K; kk += kc)
        {
            kc = GEMM_K;
            if(K - kk < GEMM_K)
                kc = K - kk;

            for(ii = is; ii < is + mb; ii += mc)
            {
                mc = GEMM_M;
                if(is + mb - ii < GEMM_M)
                    mc = is + mb - ii;

                uint16_t *A0 = A + ii*K + kk;

                G = 64;
                for(j = js; j < js + nb; j += nr)
                {
                    nr = 32;
                    if(js + nb - j < 32)
                        nr = js + nb - j;

                    uint16_t *B0 = B + kk*N + j;
                    float *C0 = C + ii*N + j;

                    int dist = (j - js) % 64;
                    if(dist == 0)
                    {
                        if(js + nb - j < 64)
                            G = N_tail;

                        pack_N(B0, Bc + id*GEMM_K*64,
                               N<<1, kc, G, vpermt2d_control);
                    }

                    /*

                    if(Int_M >= 32)
                    {
                        if(nr > 16)
                            mKernel_32x32x32(C0, A0,
                                              Bc + id*GEMM_K*64 + dist*kc,
                                              Int_M, kc, N<<1, K<<1, kk);
                        else
                            mKernel_32x16x32(C0, A0,
                                              Bc + id*GEMM_K*64 + dist*kc,
                                              Int_M, kc, N<<1, K<<1, kk);
                    }

                    if(Int_M < M)
                    {
                        if(nr > 16)
                            mKernel_16x32x32(C0 + Int_M*N, A0 + Int_M*K,
                                              Bc + id*GEMM_K*64 + dist*kc,
                                              Int_M, kc, N<<1, K<<1, kk);
                        else
                            mKernel_16x16x32(C0 + Int_M*N, A0 + Int_M*K,
                                              Bc + id*GEMM_K*64 + dist*kc,
                                              Int_M, kc, N<<1, K<<1, kk);
                    }*/

                        if(Int_M >= 32)
                        {
                            //init_tile_config(&tile_data,16);
                            _tile_loadconfig(tile_data);
                            if(nr>16)
                            {
                                mKernel_32x32x32(C0, A0,
                                                Bc + id*GEMM_K*64 + dist*kc,
                                                Int_M, kc, N<<1, K<<1, kk);
                            }
                            else
                            {
                                mKernel_32x16x32(C0, A0,
                                                Bc + id*GEMM_K*64 + dist*kc,
                                                Int_M, kc, N<<1, K<<1, kk);

                            }
                        }


                        // EGDE_M
                        if(Int_M < M)
                        {
                            //init_tile_config(&tile_data, tail_M);
                            _tile_loadconfig(tile_data+1);

                            if( M - Int_M >16)
                            {
                                if(nr>16)
                                {
                                    mKernel_32x32x32(C0 + Int_M*N, A0 + Int_M*K,
                                                    Bc + id*GEMM_K*64 + dist*kc,
                                                    M-Int_M, kc, N<<1, K<<1, kk);
                                                    
                                }
                                else
                                {

                                    mKernel_32x16x32(C0 + Int_M*N, A0 + Int_M*K,
                                                    Bc + id*GEMM_K*64 + dist*kc,
                                                    M-Int_M, kc, N<<1, K<<1, kk);
                                }

                            }
                            else
                            {
                                if(nr>16)
                                {
                                    mKernel_16x32x32(C0 + Int_M*N, A0 + Int_M*K,
                                                    Bc + id*GEMM_K*64 + dist*kc,
                                                    M-Int_M, kc, N<<1, K<<1, kk);
                                                    
                                }
                                else
                                {

                                    mKernel_16x16x32(C0 + Int_M*N, A0 + Int_M*K,
                                                    Bc + id*GEMM_K*64 + dist*kc,
                                                    M-Int_M, kc, N<<1, K<<1, kk);
                                }
                            }

                        }
                }
            }
        }

        _tile_release();
    }

    free(Bc);
    free(tasks);
}

int main()
{
    long loop = 100;
    long M, N, K;
    int i,j,k;
    void *ptr;

    if(!set_tiledata_use())
        exit(-1);

    long Ms[] = {16};
    long Ns[] = {28672};
    long Ks[] = {8192};

    FILE *fp;
    if((fp=fopen("TileFlow_NN_UMA_6430.txt","w+")) == NULL)
    {
        puts("Fail to open file!");
        exit(0);
    }

    for(k = 0; k < 1; k++)
    {
        N = Ns[k];
        K = Ks[k];

        int X10=10;
        if(j<3)
        {
            X10 = 1;
        }

        for(j = 32; j < 33; j++)
        {
            M = j;
            fprintf(fp, "%ld %ld %ld", M, N, K);

            for(int pc = 0; pc < 5; pc++)
            {
                double ops = 2.0 * (double)M*N*K*1.0e-12;

                posix_memalign(&ptr, 64, ((M*K)*sizeof(uint16_t)));
                uint16_t *A = (uint16_t *) ptr;
                posix_memalign(&ptr, 8192, (N*K*X10*sizeof(uint16_t)));
                uint16_t *B = (uint16_t *) ptr;
                posix_memalign(&ptr, 64, (M*N*X10*sizeof(float)));
                float *C = (float *) ptr;
                posix_memalign(&ptr, 64, (M*N*sizeof(float)));
                float *C1 = (float *) ptr;

                random_matrix(M, K, A);
                for(int ii = 0; ii < X10; ii++)
                    random_matrix(K, N, B + ii*N*K);

                for(int ii = 0; ii < X10; ii++)
                    bf16gemm_amx(C + ii*M*N, A, B + ii*N*K, M, N, K);

                double start = dclock();
                for(i = 0; i < loop / X10; i++)
                {
                    for(int ii = 0; ii < X10; ii++)
                        bf16gemm_amx(C + ii*M*N, A, B + ii*N*K, M, N, K);
                }

                double cost = (dclock() - start)/loop;

                printf("M = %ld, N = %ld, K = %ld, ms = %.3f, Tflops = %.3f\n",
                       M, N, K, cost*1000, ops/cost);
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
    }

    fclose(fp);
    return 0;
}