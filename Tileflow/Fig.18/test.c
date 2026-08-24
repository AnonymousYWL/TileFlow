#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <Accelerate/Accelerate.h>
#include "megemm.h"
#include "utils.h"

#define num 1
#define DATA_TYPE float

void random_matrix( int m, int n, DATA_TYPE *a)
{
    double drand48();

    //#pragma omp parallel for num_threads(num)
    for(int i =0 ; i < m * n; i ++)
        a[i]= 2.0 * (DATA_TYPE)drand48( ) - 1.0 + 0.0000001 * (i/m);

}


int main()
{
    int i,j,k,pc;

    int loop=50;

    DATA_TYPE *A = NULL,*B = NULL, *C = NULL , *C1 = NULL , *flush = NULL;
    double start, cost=0;
    double ops;
    int lda, ldb, ldc, flag;
    int M, N, K;
    double best=0.0;

    FILE *fp;
    if( (fp=fopen("TileFlow-multiple.txt","a+")) == NULL )
    { 
        puts("Fail to open file!");
        exit(0);
    }

    for (j = 32 ; j <= 1024; j+=32)
    {

        M = j;
        N = 16384;
        K = 16384;

        lda = K;
        ldb = N;
        ldc = N;

        ops = (double )M  * N  * K * 1.0e-09 * 2;
        fprintf(fp, "M: %d,N: %d,K: %d,reps: %d", M, N, K, loop);
        for(pc = 0; pc < 3; pc++)
        {
            posix_memalign( (void**)&A,    128, M * lda * sizeof(float) );
            posix_memalign( (void**)&B,    128, K * ldb * sizeof(float) );
            posix_memalign( (void**)&C,    128, M * ldc * sizeof(float) );

            random_matrix(M,K,A);
            random_matrix(K,N,B);
    
            for( i= 0; i< loop ;i++)
            {
                start = dClock();
                dispatch_row_sgemm(M, N, K, A, B, C);
                cost = dClock()-start;
                best = min(best, cost);
            }
            
            printf("GEMM:  M= %d N=%d K=%d duration=%lf cost=%lf Gflops = %lf\n",M, N, K, cost * loop ,cost, ops/cost);
            fprintf(fp, " %lf", ops/cost);

            free(A);
            free(B);
            free(C);
            free(flush);

        }
        //sleep(5);
        fprintf(fp, "\n");
    }

    fclose(fp);
    return 0;

}