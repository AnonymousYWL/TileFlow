#ifndef SGEMM_H
#define SGEMM_H

#include "utils.h"

#ifdef __cplusplus
	extern "C" {
#endif

#define SME_CACHELINE_SIZE 128

#define ROW_MC 512  
#define ROW_KC 1024
#define ROW_NC 512

#define ROW_MR 32
#define ROW_NR 32

extern void row_packa_opt(
	int    mc,
	int    kc,
	float *a,
	int    lda,
	float *packA
);

extern void row_kernel_32x32_opt(
	int		mc, 
	int 	kc,
	int 	nc,
	float  *XB, 
	float  *packa, 
	float  *packb, 
	float  *c, 
	int 	KC,
	int 	ldc,
	int 	kk
);

void row_sgemm(int m,int n,int k,float *XA,float *XB,float *XC);
void dispatch_row_sgemm(int m,int n,int k,float *XA,float *XB,float *XC);

#ifdef __cplusplus
	}
#endif

#endif
