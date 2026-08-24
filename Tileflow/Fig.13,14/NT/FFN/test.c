// 针对 numactl -i ... 做过优化
// 对齐结果
// M <= 16
// Matrix8x7B
// 8个专家中选2个

#include "pack.h"

#define TEST_REPEAT 10

#define NUM 64

struct GateUpDown {
    uint16_t* gate;
    uint16_t* up;
    uint16_t* down;
};

void Check_result(uint16_t *C, float *C1, long M, long N)
{

    int i, j, flag = 0;
    for( i= 0; i< M; i++)
    {
        for( j= 0 ;j < N; j++)
        {
            if(fabs(   (BF16_toFP32(C[i*N+j])- C1[i*N + j]) ) > 800)  // BF16累计误差，对结果的影响可忽略
            {
                printf("i = %d, j= %d\n", i,j);
                printf("C= %lf , C1= %lf\n", BF16_toFP32(C[i*N+j]), 
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
        printf("结果正确\n");

    double sum= 0;
    float max=-100;
    long size = M*N;
    for(i = 0; i < size; i++)
    {
        sum = sum + BF16_toFP32( C[i]);
        if(BF16_toFP32( C[i]) > max)
        {
            max = BF16_toFP32( C[i]);
        }
    }

    double mean = sum/size;

    sum = 0;
    for(i = 0; i < size; i++)
    {
        sum += (BF16_toFP32(C[i]) - mean) * (BF16_toFP32(C[i]) - mean);
    }

    double var = sum/size;
    printf("+++++ mean = %f, var = %f, max = %f\n", mean, sqrt(var), max);

    sum = 0;
    for(i = 0; i < size; i++)
    {
        sum = sum + C1[i];
    }

    mean = sum/size;
    
    sum = 0;
    max=-100;
    for(i = 0; i < size; i++)
    {
        sum += (C1[i] - mean) * (C1[i] - mean);
        if(C1[i] > max)
        {
            max = C1[i];
        }
    }

    var = sum/size;
    printf("***** mean = %f, var = %f, max = %f\n", mean, sqrt(var), max);

}


void bf16gemm_kernel(float *C, uint16_t *A, uint16_t *B, long M, long K,
                    int is, int js, int mb, int nb, uint32_t*index)
{
    int jj, ii, kk, nc, mc, j, kc, nr;

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


                for(j = jj; j < jj + nc; j = j + nr)
                {
                    nr = 32;

                    float *C0 = C + j*M + ii;
                    uint16_t *B0 = B + j*K + index[0];  //offset
                    uint16_t *A0 = A + (index[0]) % K*M;
                    mKernel_16x32x32_2(C0, A0, B0,
                                    mc, index+1, M<<1, K<<1, M<<2);


                }

            }
        }
    }

}

void bf16gemm_amx(uint16_t *C, uint16_t *A, GateUpDown weight,
                    long M, long N, long K)
{

    int Tm = 1;
    int Tk = 2;
    int Tn = NUM/Tm/Tk;                               //每个token激活1个专家

    float *Buffer;
    posix_memalign((void **)&Buffer, 64, 4*M*N*sizeof(float));
    uint16_t *Inter_data;
    posix_memalign((void **)&Inter_data, 64, 2*M*N*sizeof(uint16_t));

    int M_blocks = (M + 15) / 16;
    int K_blocks = (K + 255) / 256;

    #pragma omp parallel num_threads(NUM)
    {
        // Load tile configuration 
        __tilecfg tile_data={0};
        init_tile_config(&tile_data, M);
        int i,j, js, is, ks, ij, ii, kk, per_thread;
        int mr, mc, nc, kc, ke, nr;

        int Tn_1 = Tn>>1;
        int id = omp_get_thread_num();
        int nb = (N>>4)/Tn_1<<4;
        int mb = M / Tm;


        uint16_t *Ac = A;
        if(M>1)
        {
            Ac = Inter_data;
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

         
        int vid = id / 32 * Tn_1 + id % Tn_1;
        js = nb*(id%Tn_1);
        is = mb * (id/Tn/Tk);      //两个专家 + up and gate

        uint32_t*index_numa = mydata::index.data() + id/32*4;
        uint16_t*temp_B = weight.gate;
        if(id%Tn>=Tn_1)
            temp_B = weight.up;

        float*temp_C = Buffer + id%Tn/Tn_1*M*N*2 + id/32*M*N;
        bf16gemm_kernel(temp_C, Ac, temp_B, M, K, is, js, mb, nb, index_numa);
        #pragma omp barrier

        temp_C = Buffer + (id%Tn/Tn_1*M*N<<1);
        per_thread = (2*M*N)/NUM * vid;
        uint16_t*dst = Inter_data + (id%Tn/Tn_1*M*N);
        Add_matrix(dst+per_thread, temp_C+per_thread, temp_C+M*N+per_thread, M, 2*N/NUM, M<<1,
                        mydata::vpermt2d_control);
        #pragma omp barrier
        per_thread = (M*N)/NUM;
        vid = id / 32 * Tn + id % Tn;
        SiLU(Inter_data + vid*per_thread, M*N<<1, mydata::Va, per_thread);
        #pragma omp barrier

        //GEMM_down
        nb = (K>>4)/Tn<<4;
        js = nb*(id%Tn);

        if( id%Tn < (K>>4)%Tn) 
        {
            nb = nb + 16;
            js = js + (id%Tn<<4);
        }
        else
        {
            js = js + ((K>>4)%Tn<<4);
        }


// GEMM three
        temp_C = Buffer + M*N;
        index_numa = mydata::index.data() + 8;
        if(id >= 32)
        {
            index_numa = mydata::index.data() + 16;
            temp_C = Buffer + M*N + M*K;
        }

        for(j = js; j < js + nb; j = j + nr)
        {
            nr = 64;

            for(ij = 0; ij < 2; ij++)
            {
            
                uint16_t *B0 = weight.down+ j*N + index_numa[ij*4];  //offset
                uint16_t *A0 = Inter_data + (index_numa[ij*4]) % N*M;
                mKernel_16x32x32_2(temp_C + (j+ij)*M, A0, B0,
                            M, index_numa+ij*4+1, M<<1, N<<2, M<<3);
            }
            
        }

        #pragma omp barrier
        per_thread = vid*(M*K/NUM);
        Add_Transpose_matrix(C + vid*(K/NUM), Buffer+M*N+per_thread, Buffer+M*N+M*K+per_thread, M, K/NUM, 
                            M<<2, K<<1, mydata::vpermt2d_control);
        _tile_release();
    }
    
    free(Inter_data);
    free(Buffer);
}


int main()
{

    long loop = 100;
    long M, N, K;
    int i,j,ij;
    void *ptr, *ptr1;

    // Request permission to linux kernel to run AMX 
    if (!set_tiledata_use())
    exit(-1);

    long Ms[] = {1};
    long Ks[] = {4096};
    long Ns[] = {14336};

    FILE *fp;
    if( (fp=fopen("FFN_Llama3-8B_6430.txt","w")) == NULL )
    {
        puts("Fail to open file!");
        exit(0);
    }
       
    for(j = 1; j < 17; j++)
    {

        N = Ns[0];
        K = Ks[0];
        M = j;
        fprintf(fp, "%d %d %d", M,N,K);
        for(int pc = 0; pc < 5; pc++)
        {
            double ops = 2.0*(double)M*N*K*3*1.0e-12;
            double DV = 2*(3*M*K + (double)N*K*3)/1024/1024/1024;   //GB

            posix_memalign(&ptr, 64, ((M*K*(2+TEST_REPEAT) )*sizeof( uint16_t )));
            uint16_t *A = ( uint16_t * ) ptr;
            uint16_t *Ac = A + M*K;
            posix_memalign(&ptr, 4096*2, ((N*K*3)*TEST_REPEAT*sizeof( uint16_t )));
            uint16_t *B = ( uint16_t * ) ptr;
            uint16_t *C = A + M*K*2;
            posix_memalign(&ptr, 64, (2*M*N+M*K)*sizeof( float ));
            float *C1 = ( float *) ptr;

            posix_memalign(&ptr, 64, ((M*N)*sizeof( uint16_t )));
            uint16_t *A1 = ( uint16_t * ) ptr;
            
            random_matrix(M, K, A);
            random_matrix(N, K*3*TEST_REPEAT, B);

            // gate, up, down在这里初始化即可
            GateUpDown ffn_proj[10];

            for(int ij = 0; ij <TEST_REPEAT; ij++)
            {
                ffn_proj[ij].gate = B+3*ij*N*K;  
                ffn_proj[ij].up = B + 2*K*N+3*ij*N*K;
                ffn_proj[ij].down = B + K*N+3*ij*N*K;
            }
            
            for(int ij = 0; ij <TEST_REPEAT; ij++)
                bf16gemm_amx(C+ij*M*K, A, ffn_proj[ij], M, N, K);

            double start = dclock();
            for(i = 0;i < loop/TEST_REPEAT; i++)
            {
                for(int ij = 0; ij <TEST_REPEAT; ij++)
                    bf16gemm_amx(C+ij*M*K, A, ffn_proj[ij], M, N, K);
     
            } 
            double cost = (dclock() -start)/loop;
           
            printf("M = %d, N = %d, K = %d, %% of BW= %.1f, cost (ms) = %.3f, GB/s = %.3f, Tflops = %.3f\n", 
                                M, N, K, DV/cost/460.2*100, cost *1000, DV/cost, ops/cost);
            fprintf(fp, " %.3f", cost*1000);


            for(ij = 0; ij < M*K; ij++)
            {
                C1[2*M*N + ij] = 0; 
            }

            for(ij = 0; ij < 1; ij++)
            {
                cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasTrans,
                            M, N, K, 1, A, K, B + ij*3*N*K, K, 0, C1, N);
                cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasTrans,
                            M, N, K, 1, A, K, B + 2*N*K + ij*3*N*K, K, 0, C1 + M*N, N);

                for(i = 0; i < M*N; i++)
                {
                    A1[i] = FP32_toBF16( C1[i+M*N]*(C1[i]/(1+exp(0-C1[i]))) ); 
                }

                // softmax为各个专家的权重
                cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasTrans,
                            M, K, N, 1, A1, N, B + N*K + ij*3*N*K, N, 0, C1+2*M*N, K);
            }


            Check_result(C, C1+2*M*N, M, K);
            printf("##### C = %f, C1 = %f\n",BF16_toFP32(C[17]), C1[2*M*N + 17]);

            free(A);
            free(B);
            free(A1);
            free(C1);      
        }
        fprintf(fp, "\n"); 
    }

    fclose(fp);
    return 0;
}
