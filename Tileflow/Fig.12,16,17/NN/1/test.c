//Fp32 output

#include "pack.h"

void Check_result(float *C, float *C1, long M, long N)
{

    int i, j, flag = 0;
    for( i= 0; i< M; i++)
    {
        for( j= 0 ;j < N; j++)
        {
            if( fabs( C[i*N + j]- C1[i*N + j]) > 0.5)
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
        printf("结果正确\n");

}


void bf16gemm_amx(float *C, uint16_t *A, uint16_t *B, long M, long N, long K)
{


    int Tm = 1;
    int Tn = NUM/Tm;
    uint8_t vpermt2d_control[] = {0, 32, 1, 33, 2, 34, 3, 35, 4, 36, 5, 37, 6, 38, 7, 39,
                                8, 40, 9, 41, 10, 42, 11, 43, 12, 44, 13, 45, 14, 46, 15, 47};
    void *ptr;
    posix_memalign(&ptr, 64, NUM*GEMM_K*64*sizeof( uint16_t ));
    uint16_t *Bc = ( uint16_t * ) ptr;

    int Int_M = M - M % 32;
    uint32_t tail_M = M - (M-1)/16*16;


    thread_task *tasks =(thread_task *)malloc(sizeof(thread_task)*Tn);

    // Load tile configuration 
    __tilecfg tile_data[2]={0,0/*,0*/};
    init_tile_config(tile_data,16);
    init_tile_config(tile_data+1,tail_M);
    //init_tile_config(tile_data+2,4);

    #pragma omp parallel num_threads(NUM)
    {
        
        int i,j,k,ii,jj,kk,js, is, t;
        int mc, nc, kc, nr, kr;
        int id = omp_get_thread_num();
        int mb = M / Tm, nb=N/Tn;
        is = 0;
        schedule_thread(N, Tn, id, &tasks[id]);

        uint8_t N_tail = nb % 64;

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

                for(t = 0; t < tasks[id].nseg; t++)
                {
                    uint32_t G = 64;
                    jj = tasks[id].js[t];
                    nc = tasks[id].nb[t];

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

                        int dist = (j -jj)%64;
                        if(dist==0)
                        {
                            if(jj+nc-j < 64)
                            {
                                G = N_tail;
                            }
                            //_tile_loadconfig(tile_data+2);
                            pack_N(B0, Bc + id*GEMM_K*64,
                                N<<1, kc, G, vpermt2d_control);
                        }

                        
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
        }
        

        _tile_release ();
    }
    free(Bc);
    free(tasks);
    
}

int main()
{

    long loop = 100;
    long M, N, K;
    int i,j,k;
    void *ptr, *ptr1;

    // Request permission to linux kernel to run AMX 
    if (!set_tiledata_use())
    exit(-1);

    long Ms[] = {16};
    long Ns[] = {28672, 8192,  36864, 16384};
    long Ks[] = {8192,  28672, 4608,  6144};

    FILE *fp;
    if( (fp=fopen("TileFlow_NN_FP32_6430_exp2.txt","w+")) == NULL )
    {
        puts("Fail to open file!");
        exit(0);
    }
    
    
    for(k = 0; k < 4; k++)
    {

        int TEST_REPEAT=10;
        if(k<3)
        {
            TEST_REPEAT = 1;
        }

        N = Ns[k];
        K = Ks[k];
           
        for(j = 1; j < 257; j++)
        //for(j = 0; j < 5; j++)
        {


            M = j;//16 + 16 * j+1;
            fprintf(fp, "%d %d %d", M,N,K);
            for(int pc =0; pc < 5; pc++)
            {
                double ops = 2.0*(double)M*N*K*1.0e-12; 

                posix_memalign(&ptr, 64, (M*K*sizeof( uint16_t )));
                uint16_t *A = ( uint16_t * ) ptr;
                posix_memalign(&ptr, 8192, (N*K*TEST_REPEAT*sizeof( uint16_t )));
                uint16_t *B = ( uint16_t * ) ptr;
                posix_memalign(&ptr, 64, (M*N*TEST_REPEAT*sizeof( float )));
                float *C = ( float *) ptr;
                //posix_memalign(&ptr, 64, (M*N*sizeof( float )));
                //float *C1 = ( float *) ptr;

                double DV = 2*((double)N*K+ M*K + M*N)/1024/1024/1024;
                
                random_matrix(M, K, A);
                random_matrix(K*TEST_REPEAT, N, B);

                for(int ii = 0; ii < TEST_REPEAT; ii++)
                    bf16gemm_amx(C+ii*M*N, A, B+ii*K*N, M, N, K);
                
                double start = dclock();
                for(i = 0;i < loop/TEST_REPEAT; i++)
                {
                    for(int ii = 0; ii < TEST_REPEAT; ii++)
                        bf16gemm_amx(C+ii*M*N, A, B+ii*K*N, M, N, K);
                } 
           
                double cost = (dclock() -start)/loop;
              
                printf("M = %d, N = %d, K = %d, ms = %.3f, BW = %.3f, Tflops = %.3f\n", M, N, K, cost*1000, 
                        DV/cost, ops/cost);
                fprintf(fp, " %.3f", ops/cost);

                /*
                cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            M, N, K, 1, A, K, B, N, 0, C1, N);

                Check_result(C, C1, M, N);
*/
                free(A);
                free(B);
                free(C);
                //free(C1);      
            }
            fprintf(fp, "\n");
        } 
    }

    fclose(fp);
    return 0;
}
