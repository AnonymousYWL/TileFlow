//Fp32 output

#include "pack.h"

void Check_result(uint16_t *C, float *C1, long M, long N)
{

    int i, j, flag = 0;
    for( i= 0; i< M; i++)
    {
        for( j= 0 ;j < N; j++)
        {
            if( fabs( BF16_toFP32( C[i*N + j])- C1[i*N + j]) > 0.5)
            {
                printf("i = %d, j= %d\n", i,j);
                printf("C= %lf , C1= %lf\n", BF16_toFP32 (C[i*N + j]), C1[i*N + j]);
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

void bf16gemm_amx(uint16_t *C, uint16_t *A, uint16_t *B, long M, long N, long K)
{


    int Tm = 1;
    int Tk = 2;
    int Tn = NUM /Tk;
    void *ptr;
    posix_memalign(&ptr, 64, 2*M*N*sizeof( float ));
    float *Buffer = ( float * ) ptr;


    #pragma omp parallel num_threads(NUM)
    {
        // Load tile configuration 
        __tilecfg tile_data = {0};
        init_tile_config(&tile_data, M);
        
        int i,j,js,nr;
        int nb = (N>>4)/Tn<<4;
        int id = omp_get_thread_num();
        int n_tid = id % Tn;
        int NUMA_id = id/Tn;

        js = nb*n_tid;
        uint16_t*temp_B = B;


        float* Cc = Buffer + NUMA_id*M*N;
        for(j = js; j < js + nb; j = j + nr)
        {
            nr = 32;
            if(js + nb - j < 32)
            {
                nr = js + nb - j;
            }
            uint16_t *A0 = A + NUMA_id * 128;    //加上NUMA的偏移量
            uint16_t *B0 = temp_B + j * K + NUMA_id * 2048;    //加上NUMA的偏移量
            float *C0 = Cc + j; 

            mKernel_16x32x32(C0, A0, B0, M, K/Tk, N<<2, K<<1);   


        }
        _tile_release();

        #pragma omp barrier

        long chunk = M*N / NUM;
        long start = id * chunk;
        long end   = start + chunk; 
        for(i = start; i < end; i += 16)
        {
            __m512 vc1 = _mm512_loadu_ps(Buffer+ i);
            __m512 vc2 = _mm512_loadu_ps(Buffer+M*N+ i);       
            _mm256_storeu_si256((__m256i*)(C + i),
                (__m256i)_mm512_cvtneps_pbh(_mm512_add_ps(vc1, vc2)));
        }
    }
    free(Buffer);
    
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
    long Ns[] = {4096 + 1024 + 1024};
    long Ks[] = {4096};

    FILE *fp;
    if( (fp=fopen("MeGEMM_layout_FP32_6430.txt","w+")) == NULL )
    {
        puts("Fail to open file!");
        exit(0);
    }
       
    for(j = 0 ;j < 1; j++)
    {

        N = Ns[0];
        K = Ks[0];
        M = 16;
        fprintf(fp, "%d %d %d", M,N,K);
        for(int pc =0; pc < 1; pc++)
        {
            double ops = 2.0*(double)M*N*K*1.0e-12; 
            double DV = (double)N*K*2/1024/1024/1024;
            double start=0, cost = 0;

            posix_memalign(&ptr, 64, ((M*K)*sizeof( uint16_t )));
            uint16_t *A = ( uint16_t * ) ptr;
            posix_memalign(&ptr, 8192, (N*K*sizeof( uint16_t )));
            uint16_t *B = ( uint16_t * ) ptr;
            posix_memalign(&ptr, 8192, (N*K*sizeof( uint16_t )));
            uint16_t *Bc = ( uint16_t * ) ptr;
            posix_memalign(&ptr, 64, (2*M*N*sizeof( uint16_t )));
            uint16_t *C = ( uint16_t *) ptr;
            posix_memalign(&ptr, 64, (2*M*N*sizeof( float )));
            float *C1 = ( float *) ptr;
            
            random_matrix(M, K, A);
            random_matrix(K, N, B);  // Q+K+V

            transform_layout(Bc, B, N, K);

            // warm up
            for(i = 0; i < 3; i++)
                bf16gemm_amx(C, A, Bc, M, N, K);

            start = dclock();
            for(i = 0; i < loop; i++)
            {   
                bf16gemm_amx(C, A, Bc, M, N, K);
            } 
            cost = (dclock() -start)/loop;
          
            printf("M = %d, N = %d, K = %d, ms = %.3f, GB/s = %.3f, Tflops = %.3f\n", 
                M, N, K, cost*1000,DV/cost,ops/cost);
            fprintf(fp, " %.3f", ops/cost);


            cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasTrans,
                        M, N, K, 1, A, K, B, K, 0, C1, N);
            cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasTrans,
                        M, N, K, 1, A, K, B+N*K, K, 0, C1+M*N, N);

            Check_result(C, C1, M, N);

     

            free(A);
            free(B);
            free(Bc);
            free(C);
            free(C1); 
        }
        fprintf(fp, "\n"); 
    }

    fclose(fp);
    return 0;
}
