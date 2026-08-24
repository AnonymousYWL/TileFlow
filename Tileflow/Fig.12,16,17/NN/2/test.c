//Fp32 output

#include "pack.h"

#define TEST_REPEAT 10

void Check_result(float *C, float *C1, long M, long N)
{

    int i, j, flag = 0;
    for( i= 0; i< M; i++)
    {
        for( j= 0 ;j < N; j++)
        {
            if( fabs( C[i*N + j]- C1[i*N + j]) >1.0e-2)
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
    posix_memalign(&ptr, 64, NUM*GEMM_K*64*sizeof( uint16_t ));  // G = 64
    uint16_t *Bc = ( uint16_t * ) ptr;
    posix_memalign(&ptr, 64, M * K*sizeof( uint16_t ));
    uint16_t *inter_A = ( uint16_t * ) ptr;

    uint16_t *A_new[]={inter_A, inter_A + M*K/2};

    thread_task *tasks =(thread_task *)malloc(sizeof(thread_task)*Tn);

    uint32_t *numa = (uint32_t*)malloc(2*K*sizeof(uint32_t));
    uint32_t *numa0= numa, *numa1= numa+K;
    uint32_t *numa_k[] ={numa0, numa1};
    uint8_t *flag = (uint8_t*)calloc(2*K,sizeof(uint8_t));
    uint8_t *flag0= flag, *flag1= flag+K;

    uint32_t thread_c0[NUM]={0};
    uint32_t thread_c1[NUM]={0};

    int Int_M = M - M % 32;
    uint32_t tail_M = M - (M-1)/16*16;

    uint32_t n0=0,n1=0;


    // Load tile configuration 
    __tilecfg tile_data[2]={0,0};
    init_tile_config(tile_data,16);
    init_tile_config(tile_data+1, tail_M);


    #pragma omp parallel num_threads(NUM)
    {
        int i,j,k,ii,jj,kk,js, is, t;
        int mc, nc, kc, nr, kr;
        int id = omp_get_thread_num();
        int mb = M / Tm;
        is = 0;
        schedule_thread(N, Tn, id, &tasks[id]);


//pack A矩阵
//---------------------------------------------------
        uint32_t start=(uint64_t)id*K/NUM;
        uint32_t end  =(uint64_t)(id+1)*K/NUM;

        /* step1: 分类 */
        classify_range(start,end,N,
                       flag0,flag1,
                       &thread_c0[id],
                       &thread_c1[id]);

        #pragma omp barrier

        #pragma omp single
        {
            for(uint32_t s=0;s<K;s++)
                if(flag0[s])
                    numa0[n0++]=s;

            for(uint32_t s=0;s<K;s++)
                if(flag1[s])
                    numa1[n1++]=s;
        }

        #pragma omp barrier

        /* step2: 拆分 A */

        start=(uint64_t)id*M/NUM;
        end  =(uint64_t)(id+1)*M/NUM;

        for(i = start; i < end; i++)
        {
            uint16_t * __restrict a_row = A + (uint64_t)i * K;
            uint16_t * __restrict dst0 = A_new[0] + (uint64_t)i * n0;
            uint16_t * __restrict dst1 = A_new[1] + (uint64_t)i * n1;

            uint32_t c;
            
            for(c = 0; c < n0; c++)
            {
                __builtin_prefetch(a_row + numa0[c + 16], 0, 1);
                dst0[c] = a_row[numa0[c]];
            }

            for(c = 0; c < n1; c++)
            {
                __builtin_prefetch(a_row + numa1[c + 16], 0, 1);
                dst1[c] = a_row[numa1[c]];
            }
        }
/*
        for(i=start; i<end; i++)
        {
            uint16_t * __restrict a_row  = A  + (uint64_t)i*K;
            uint16_t * __restrict a1row = A_new[0] + (uint64_t)i*n0;
            uint16_t * __restrict a2row = A_new[1] + (uint64_t)i*n1;

            uint32_t p0 = 0;
            uint32_t p1 = 0;

            for(k = 0; k < K; k++)
            {
                __builtin_prefetch(a_row + k + 32, 0, 1);
                uint16_t v = a_row[k];
                if( (k*N+1024)/2048%2 == 0 )
                    a1row[p0++] = v;
                else
                    a2row[p1++] = v;
            }
        }
*/

        #pragma omp barrier
//-----------------------------------------------------


        for(kk = 0; kk < 2; kk++)
        {

            kc = K /2;

            uint32_t  pos = (id+kk*32)%64;
            for(ii = is; ii < is + mb; ii = ii + mc)
            {
                mc = GEMM_M;
                if(is + mb - ii < GEMM_M)
                    mc = is + mb - ii;

                uint16_t *A0 = A_new[kk] + ii*K/2;

                for(t = 0; t < tasks[pos].nseg; t++)
                {
                    uint32_t G = 64;
                    jj = tasks[pos].js[t];
                    nc = tasks[pos].nb[t];
                    uint8_t N_tail = nc % 64;

                    for(j = jj; j < jj + nc; j = j + nr)
                    {
                        nr = 32;
                        if(jj+nc-j<32)
                        {
                            nr = jj+nc-j;
                        }
                        //pack B
                        uint16_t *B0 = B + j;
                        float *C0 = C + ii * N + j;

                        int dist = (j -jj)%64;

                        if(dist==0)
                        {
                            if(jj+nc-j < 64)
                            {
                                G = N_tail;
                            }
                            
                            pack_N(B0, Bc + id*GEMM_K*64,
                                N<<1, kc, G, vpermt2d_control, numa_k[kk]);
                        }


                        if(Int_M >= 32)
                        {
                            //init_tile_config(&tile_data,16);
                            _tile_loadconfig(tile_data);
                            if(nr>16)
                            {
                                mKernel_32x32x32(C0, A0,
                                                Bc + id*GEMM_K*64 + dist*kc,
                                                Int_M, kc, N<<1, K, kk);
                            }
                            
                            else
                            {
                                mKernel_32x16x32(C0, A0,
                                                Bc + id*GEMM_K*64 + dist*kc,
                                                Int_M, kc, N<<1, K, kk);

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
                                    mKernel_32x32x32(C0 + Int_M*N, A0 + Int_M*K/2,
                                                    Bc + id*GEMM_K*64 + dist*kc,
                                                    M - Int_M, kc, N<<1, K, kk);
                                                    
                                }
                                else
                                {

                                    mKernel_32x16x32(C0 + Int_M*N, A0 + Int_M*K/2,
                                                    Bc + id*GEMM_K*64 + dist*kc,
                                                    M - Int_M, kc, N<<1, K, kk);
                                }
                           }
                           else
                           {
                                if(nr>16)
                                {
                                    mKernel_16x32x32(C0 + Int_M*N, A0 + Int_M*K/2,
                                                    Bc + id*GEMM_K*64 + dist*kc,
                                                    M - Int_M, kc, N<<1, K, kk);
                                                    
                                }
                                else
                                {

                                    mKernel_16x16x32(C0 + Int_M*N, A0 + Int_M*K/2,
                                                    Bc + id*GEMM_K*64 + dist*kc,
                                                    M - Int_M, kc, N<<1, K, kk);
                                }

                           }

                        }
                    }
                }
            }

            //#pragma omp barrier

        }
        _tile_release ();
    }
    free(Bc);
    free(tasks);
    free(inter_A);
    free(numa);
    free(flag);
    
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
    long Ns[] = {25600, 13824, 14336};
    long Ks[] = {5120,  5120,  4096};

    FILE *fp;
    if( (fp=fopen("TileFlow_NN_FP32_6430_exp2_2.txt","w+")) == NULL )
    {
        puts("Fail to open file!");
        exit(0);
    }
    
    
    for(k = 0; k < 3; k++)
    {

         N = Ns[k];
         K = Ks[k];
        for(j = 1; j < 257; j++)
        //for(j = 0; j < 1; j++)
        {

            //M = 16 + 16 * j;
            M = j;
            fprintf(fp, "%d %d %d", M,N,K);
            for(int pc =0; pc < 5; pc++)
            {
                double ops = 2.0*(double)M*N*K*1.0e-12; 

                posix_memalign(&ptr, 64, ((M*K)*sizeof( uint16_t )));
                uint16_t *A = ( uint16_t * ) ptr;
                posix_memalign(&ptr, 8192, (N*K*TEST_REPEAT*sizeof( uint16_t )));
                uint16_t *B = ( uint16_t * ) ptr;
                posix_memalign(&ptr, 64, (M*N*TEST_REPEAT*sizeof( float )));
                float *C = ( float *) ptr;
                //posix_memalign(&ptr, 64, (M*N*sizeof( float )));
                //float *C1 = ( float *) ptr;
                

                
                random_matrix(M, K, A);
                for(int ii =0;ii < TEST_REPEAT; ii++)
                    random_matrix(K, N, B+ii*N*K);

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
             
                printf("M = %d, N = %d, K = %d, ms = %.3f, Tflops = %.3f\n", M, N, K, cost*1000, ops/cost);
                fprintf(fp, " %.3f", ops/cost);
    
                //cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                //            M, N, K, 1, A, K, B, N, 0, C1, N);

                //Check_result(C, C1, M, N);
    
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
