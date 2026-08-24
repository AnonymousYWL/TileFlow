//Fp32 output

#include "pack.h"

void Check_result(uint16_t *C, float *C1, long M, long N)
{

    int i, j, flag = 0;
    for( i= 0; i< M; i++)
    {
        for( j= 0 ;j < N; j++)
        {
            if(fabs((BF16_toFP32(C[i*N+j])- C1[i*N + j]))  > 600)  // BF16累计误差，对结果的影响可忽略
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
        printf("result correct\n");

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

    double var = sqrt(sum)/size;
    printf("+++++ mean = %f, var = %f, max = %f\n", mean, var, max);

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

    var = sqrt(sum)/size;
    printf("***** mean = %f, var = %f, max = %f\n", mean, var, max);

}

struct GateUpDown {
    uint16_t* gate;
    uint16_t* up;
    uint16_t* down;
};


void bf16gemm_amx(uint16_t *C, uint16_t *A, GateUpDown B, long M, long N, long K)
{
    const int Tm = 1;
    const int Tk = 2;
    const int Tn = NUM / Tk;

    float *Buffer;
    posix_memalign((void **)&Buffer, 64, 4*M*N*sizeof(float));

    uint16_t *Inter_data;
    posix_memalign((void **)&Inter_data, 64, M*N*sizeof(uint16_t));

    #pragma omp parallel num_threads(NUM)
    {
        __tilecfg tile_data = {0};
        init_tile_config(&tile_data, M);

        int id = omp_get_thread_num();

        int Tn_1   = Tn >> 1,nr;
        int n_tid  = id % Tn_1;
        int t_id   = (id % Tn) / Tn_1;
        int NUMA_id = id / Tn;

        int nb = ((N >> 4) / Tn_1) << 4;
        int js = nb * n_tid;

        uint16_t *temp_B = (t_id==0) ? B.gate : B.up;
        float *Cc = Buffer + NUMA_id * M * N + t_id * 2 * M * N;

        uint16_t *A0 = A + NUMA_id * 128;
        for (int j = js; j < js + nb; j += nr)
        {
            nr = (js + nb - j < 32) ? (js + nb - j) : 32;
            uint16_t *B0 = temp_B + j * K + NUMA_id * 2048;
            float *C0    = Cc + j;

            mKernel_16x32x32(C0, A0, B0, M, K / Tk, N << 2, K << 1);
        }

        #pragma omp barrier

        long chunk = (2 * M * N) / NUM;
        long start = (NUMA_id * Tn_1 + n_tid) * chunk;
        long end   = start + chunk;

        float *buf_base = Buffer + t_id * 2 * M * N;
        float *buf_base2 = buf_base + M * N;

        for (long i = start; i < end; i += 16)
        {
            __m512 v1 = _mm512_loadu_ps(buf_base + i);
            __m512 v2 = _mm512_loadu_ps(buf_base2 + i);
            _mm512_storeu_ps(buf_base + i, _mm512_add_ps(v1, v2));
        }
        #pragma omp barrier
        
        chunk = M*N/NUM;
        SiLU(Inter_data + id*chunk, Buffer + id*chunk, M*N<<3, mydata::Va, chunk);


        #pragma omp barrier
//------------------------------------------
        nb = ((K >> 4) / Tn) << 4;
        n_tid  = id % Tn;
        js = nb * n_tid;

        Cc = Buffer + NUMA_id * M * K;
        A0 = Inter_data + NUMA_id * 128;
        for (int j = js; j < js + nb; j += nr)
        {
            nr = (js + nb - j < 32) ? (js + nb - j) : 32;

            uint16_t *B0 = B.down + j*N + NUMA_id*2048;  //
            float *C0    = Cc + j;

            mKernel_16x32x32(C0, A0, B0, M, N / Tk, K << 2, N << 1);
        }

        chunk = (M * K) / NUM;
        start = id * chunk;
        end   = start + chunk;

        #pragma omp barrier

        for (long i = start; i < end; i += 16)
        {
            __m512 v1 = _mm512_loadu_ps(Buffer + i);
            __m512 v2 = _mm512_loadu_ps(Buffer + M*K + i);
            _mm256_storeu_si256((__m256i*)(C + i), (__m256i)_mm512_cvtneps_pbh
                                (_mm512_add_ps(v1, v2)) );
        }

        _tile_release();
    }

    free(Buffer);
    free(Inter_data);
}

int main()
{

    long loop = 200;
    long M, N, K;
    int i,j;
    void *ptr, *ptr1;

    // Request permission to linux kernel to run AMX 
    if (!set_tiledata_use())
    exit(-1);

    long Ms[] = {1024};
    long Ns[] = {14336/*16384*/};
    long Ks[] = {4096/*14336*/};

    FILE *fp;
    if( (fp=fopen("MeGEMM_layout_FP32_6430.txt","w+")) == NULL )
    {
        puts("Fail to open file!");
        exit(0);
    }
       
    for(j = 1 ;j < 17; j++)
    {

        N = Ns[0];
        K = Ks[0];
        M = j;
        fprintf(fp, "%d %d %d", M,N,K);
        for(int pc =0; pc < 1; pc++)
        {
            double ops = 2.0*(double)M*N*K*3*1.0e-12; 
            double DV = (double)(N*K*3 + M*K*3)*2/1024/1024/1024;
            double start=0, cost = 0;

            posix_memalign(&ptr, 64, ((M*K)*sizeof( uint16_t )));
            uint16_t *A = ( uint16_t * ) ptr;
            posix_memalign(&ptr, 8192, (3*N*K*sizeof( uint16_t )));
            uint16_t *B = ( uint16_t * ) ptr;
            posix_memalign(&ptr, 8192, (3*N*K*sizeof( uint16_t )));
            uint16_t *Bc = ( uint16_t * ) ptr;
            posix_memalign(&ptr, 64, (M*K*sizeof( uint16_t )));
            uint16_t *C = ( uint16_t *) ptr;
            posix_memalign(&ptr, 64, (2*M*N*sizeof( float )));
            float *C1 = ( float *) ptr;
            posix_memalign(&ptr, 64, (M*N*sizeof( uint16_t )));
            uint16_t *C2 = ( uint16_t *) ptr;
            
            random_matrix(M, K, A);
            random_matrix(K, 2*N, B);
            random_matrix(K, N, B+2*N*K);

            // ipex默认会将权重矩阵转换布局
            // layout example [14336, 4096] -> [896,64,32,16,2]
            transform_layout(Bc, B, N, K);
            transform_layout(Bc + N*K, B + N*K, N, K);
            transform_layout(Bc + 2*N*K, B + 2*N*K, K, N);


            // gate, up, down在这里初始化即可
            GateUpDown ffn_proj;
            ffn_proj.gate = Bc;  
            ffn_proj.up = Bc + K*N;
            ffn_proj.down = Bc + 2*K*N;

            // warm up
            bf16gemm_amx(C, A, ffn_proj, M, N, K);

            start = dclock();
            for(i = 0; i < loop; i++)
            {   
                bf16gemm_amx(C, A, ffn_proj, M, N, K);
            } 
            cost = (dclock() -start)/loop;
          
            printf("M = %d, N = %d, K = %d, ms = %.3f, GB/s = %.3f, Tflops = %.3f\n", 
                M, N, K, cost*1000,DV/cost,ops/cost);
            fprintf(fp, " %.3f", ops/cost);


            cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasTrans,
                        M, N, K, 1, A, K, B, K, 0, C1, N);
            cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasTrans,
                        M, N, K, 1, A, K, B+N*K, K, 0, C1+M*N, N);

            for(i = 0; i < M*N; i++)
            {
                C2[i] = 
                FP32_toBF16( C1[i+M*N]*(C1[i]/(1+exp(0-C1[i])))); 
            }
            
            cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasTrans,
                        M, K, N, 1, C2, N, B+2*N*K, N, 0, C1, K);

            Check_result(C, C1, M, K);

            printf("%f %f\n", BF16_toFP32 (C[0]), C1[0]);

    

            free(A);
            free(B);
            free(Bc);
            free(C);
            free(C1);
            free(C2); 
        }
        fprintf(fp, "\n"); 
    }

    fclose(fp);
    return 0;
}
