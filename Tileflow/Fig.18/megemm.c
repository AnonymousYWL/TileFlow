#include <math.h>
#include <string.h>
#include <assert.h>
#include <dispatch/dispatch.h>
#include "utils.h"
#include "megemm.h"

void row_sgemm(
         int    m,
         int    n,
         int    k,
         float *XA,
         float *XB,
         float *XC
         )
{   //lda, ldb, ldc are assumed to be equal to k, n, n respectively
    float * restrict btilde_sys;
    float * restrict atilde_sys;
    posix_memalign((void **)&btilde_sys, SME_CACHELINE_SIZE, ROW_KC * ROW_NR * 2 * sizeof(float));
    posix_memalign((void **)&atilde_sys, SME_CACHELINE_SIZE, ROW_MC * ROW_KC * sizeof(float));

    for (int jj=0; jj<n; jj+=ROW_NC){
        int jb = min(ROW_NC, n - jj);
        for (int kk = 0; kk < k; kk += ROW_KC) {
            int pb = min(ROW_KC, k - kk);
            for (int ii = 0; ii < m; ii += ROW_MC) {
                int ib = min(ROW_MC, m - ii);

                row_packa_opt(ib, pb, &XA[ii * k + kk], k, atilde_sys);

                row_kernel_32x32_opt(ib, pb, jb, &XB[kk * n + jj], atilde_sys, btilde_sys, &XC[ii * n + jj], ROW_KC, n, kk);
            }
        }
    }

    free(btilde_sys);
    free(atilde_sys);
}



void dispatch_row_sgemm(int m, int n, int k,
                       float* XA,
                       float* XB,
                       float* XC)
{
    int mt = (m + ROW_MC - 1) / ROW_MC;
    int nt = (n + ROW_NC - 1) / ROW_NC;
    int total_tiles = mt * nt;

    if(total_tiles == 1){
        //small gemm
        row_sgemm(m, n, k, XA, XB, XC);
        return;
    }

    const int max_tiles = 4;

    int actual_tiles = min(total_tiles, max_tiles);

    dispatch_queue_t queue = dispatch_queue_create("rowsgemm", QUEUE_QOS(QOS_CLASS_USER_INTERACTIVE));

    dispatch_apply(actual_tiles, queue, ^(size_t tid) {
        float *btilde_sys = NULL;
        float *atilde_sys = NULL;
        posix_memalign((void**)&btilde_sys, SME_CACHELINE_SIZE, ROW_KC * ROW_NR * 2 * sizeof(float));
        posix_memalign((void**)&atilde_sys, SME_CACHELINE_SIZE, ROW_MC * ROW_KC * sizeof(float));

        for (int tile_id = tid; tile_id < total_tiles; tile_id += actual_tiles) {
            int nt_id = tile_id % nt;
            int mt_id = tile_id / nt;

            int jc = nt_id * ROW_NC;
            int ic = mt_id * ROW_MC;

            int jb = min(ROW_NC, n - jc);
            int ib = min(ROW_MC, m - ic);

            int NC_aligned = (jb / ROW_NR) * ROW_NR;
            int jbr = jb - NC_aligned;

            for (int kk = 0; kk < k; kk += ROW_KC) {
                int pb = min(ROW_KC, k - kk);
                
                row_packa_opt(ib, pb, &XA[ic * k + kk], k, atilde_sys);

                row_kernel_32x32_opt(ib, pb, jb, &XB[kk * n + jc], atilde_sys, btilde_sys, &XC[ic * n + jc], ROW_KC, n, kk);
            
            }
        }

        free(btilde_sys);
        free(atilde_sys);
    });

}