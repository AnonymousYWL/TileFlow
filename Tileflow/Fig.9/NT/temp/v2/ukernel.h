void mKernel_32x32x32(float *C, uint16_t* A, uint16_t* B, uint16_t* Bc,
                    long M, long K, long stride_N,  long stride_K, long kk)
{

    asm volatile(

    ".macro PACK_FMA                                \n"

    "   mov     $2, %%rsi                           \n"

    "   tileloaddt1 (%%rax, %%r11, 1), %%tmm0       \n"
    "   addq        $64, %%rax                      \n"
    "   tileloaddt1   (%%rdi, %%r11, 1), %%tmm1     \n"
    "   addq        $64, %%rdi                      \n"

    "   mov         %%rbx, %%rbp                    \n"

//-----------------------------------------------------
    "BEGIN_PACK_K16:                                \n"


    "   vmovntdqa      (%%rbp), %%zmm0              \n"
    "   vmovntdqa      (%%rbp, %%r11), %%zmm1       \n"
    "   vmovntdqa      (%%rbp, %%r11, 2), %%zmm2    \n"
    "   leaq           (%%rbp, %%r11, 2), %%rbp     \n"
    "   vmovntdqa      (%%rbp, %%r11), %%zmm3       \n"
    "   vmovntdqa      (%%rbp, %%r11, 2), %%zmm4    \n"
    "   leaq         (%%rbp, %%r11, 2), %%rbp       \n"
    "   vmovntdqa      (%%rbp, %%r11), %%zmm5       \n"
    "   vmovntdqa      (%%rbp, %%r11, 2), %%zmm6    \n"
    "   leaq         (%%rbp, %%r11, 2), %%rbp       \n"
    "   vmovntdqa      (%%rbp, %%r11), %%zmm7       \n"
    "   vmovntdqa      (%%rbp, %%r11, 2), %%zmm8    \n"
    "   leaq         (%%rbp, %%r11, 2), %%rbp       \n"
    "   vmovntdqa      (%%rbp, %%r11), %%zmm9       \n"
    "   vmovntdqa      (%%rbp, %%r11, 2), %%zmm10   \n"
    "   leaq         (%%rbp, %%r11, 2), %%rbp       \n"
    "   vmovntdqa      (%%rbp, %%r11), %%zmm11      \n"
    "   vmovntdqa      (%%rbp, %%r11, 2), %%zmm12   \n"
    "   leaq         (%%rbp, %%r11, 2), %%rbp       \n"
    "   vmovntdqa      (%%rbp, %%r11), %%zmm13      \n"
    "   vmovntdqa      (%%rbp, %%r11, 2), %%zmm14   \n"
    "   leaq         (%%rbp, %%r11, 2), %%rbp       \n"
    "   vmovntdqa      (%%rbp, %%r11), %%zmm15      \n"
    "   leaq         (%%rbp, %%r11, 2), %%rbp       \n"


    "   vunpcklps %%zmm1, %%zmm0, %%zmm16           \n"
    "   vunpcklps %%zmm3, %%zmm2, %%zmm17           \n"
    "   vunpcklpd %%zmm17, %%zmm16, %%zmm18         \n"

    "   vunpcklps %%zmm5, %%zmm4, %%zmm16           \n"
    "   vunpcklps %%zmm7, %%zmm6, %%zmm17           \n"
    "   vunpcklpd %%zmm17, %%zmm16, %%zmm19         \n"

    "   vunpcklps %%zmm9, %%zmm8, %%zmm16           \n"
    "   vunpcklps %%zmm11, %%zmm10, %%zmm17         \n"
    "   vunpcklpd %%zmm17, %%zmm16, %%zmm20         \n"

    "   vunpcklps %%zmm13, %%zmm12, %%zmm16         \n"
    "   vunpcklps %%zmm15, %%zmm14, %%zmm17         \n"
    "   vunpcklpd %%zmm17, %%zmm16, %%zmm21         \n"


    "   vinsertf32x4    $1, %%xmm19, %%zmm18, %%zmm22\n"
    "   vinsertf32x4    $1, %%xmm21, %%zmm20, %%zmm23\n"
    "   vinsertf32x8    $1, %%ymm23, %%zmm22, %%zmm24\n"

    "   vextractf32x4   $1, %%zmm18, %%xmm22         \n"
    "   vinsertf32x4    $0, %%xmm22, %%zmm19, %%zmm22\n"
    "   vextractf32x4   $1, %%zmm20, %%xmm23         \n"
    "   vinsertf32x4    $0, %%xmm23, %%zmm21, %%zmm23\n"
    "   vinsertf32x8    $1, %%ymm23, %%zmm22, %%zmm25\n"

    "   vextractf32x4   $2, %%zmm18, %%xmm22         \n"
    "   vextractf32x4   $2, %%zmm19, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm26\n"
    "   vextractf32x4   $2, %%zmm20, %%xmm22         \n"
    "   vextractf32x4   $2, %%zmm21, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm22\n"
    "   vinsertf32x8    $1, %%ymm22, %%zmm26, %%zmm26\n"

    "   vextractf32x4   $3, %%zmm18, %%xmm22         \n"
    "   vextractf32x4   $3, %%zmm19, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm27\n"
    "   vextractf32x4   $3, %%zmm20, %%xmm22         \n"
    "   vextractf32x4   $3, %%zmm21, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm22\n"
    "   vinsertf32x8    $1, %%ymm22, %%zmm27, %%zmm27\n"


    "   vmovups         %%zmm24, (%%r14)             \n"
    "   vmovups         %%zmm25, 256(%%r14)          \n"
    "   vmovups         %%zmm26, 512(%%r14)          \n"
    "   vmovups         %%zmm27, 768(%%r14)          \n"

    //----------------------------------------------------
    "   vunpcklps %%zmm1, %%zmm0, %%zmm16           \n"
    "   vunpcklps %%zmm3, %%zmm2, %%zmm17           \n"
    "   vunpckhpd %%zmm17, %%zmm16, %%zmm18         \n"

    "   vunpcklps %%zmm5, %%zmm4, %%zmm16           \n"
    "   vunpcklps %%zmm7, %%zmm6, %%zmm17           \n"
    "   vunpckhpd %%zmm17, %%zmm16, %%zmm19         \n"

    "   vunpcklps %%zmm9, %%zmm8, %%zmm16           \n"
    "   vunpcklps %%zmm11, %%zmm10, %%zmm17         \n"
    "   vunpckhpd %%zmm17, %%zmm16, %%zmm20         \n"

    "   vunpcklps %%zmm13, %%zmm12, %%zmm16         \n"
    "   vunpcklps %%zmm15, %%zmm14, %%zmm17         \n"
    "   vunpckhpd %%zmm17, %%zmm16, %%zmm21         \n"

    "   vinsertf32x4    $1, %%xmm19, %%zmm18, %%zmm22\n"
    "   vinsertf32x4    $1, %%xmm21, %%zmm20, %%zmm23\n"
    "   vinsertf32x8    $1, %%ymm23, %%zmm22, %%zmm24\n"

    "   vextractf32x4   $1, %%zmm18, %%xmm22         \n"
    "   vinsertf32x4    $0, %%xmm22, %%zmm19, %%zmm22\n"
    "   vextractf32x4   $1, %%zmm20, %%xmm23         \n"
    "   vinsertf32x4    $0, %%xmm23, %%zmm21, %%zmm23\n"
    "   vinsertf32x8    $1, %%ymm23, %%zmm22, %%zmm25\n"

    "   vextractf32x4   $2, %%zmm18, %%xmm22         \n"
    "   vextractf32x4   $2, %%zmm19, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm26\n"
    "   vextractf32x4   $2, %%zmm20, %%xmm22         \n"
    "   vextractf32x4   $2, %%zmm21, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm22\n"
    "   vinsertf32x8    $1, %%ymm22, %%zmm26, %%zmm26\n"

    "   vextractf32x4   $3, %%zmm18, %%xmm22         \n"
    "   vextractf32x4   $3, %%zmm19, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm27\n"
    "   vextractf32x4   $3, %%zmm20, %%xmm22         \n"
    "   vextractf32x4   $3, %%zmm21, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm22\n"
    "   vinsertf32x8    $1, %%ymm22, %%zmm27, %%zmm27\n"


    "   vmovups         %%zmm24, 64(%%r14)           \n"
    "   vmovups         %%zmm25, 320(%%r14)          \n"
    "   vmovups         %%zmm26, 576(%%r14)          \n"
    "   vmovups         %%zmm27, 832(%%r14)          \n"

    
    "   vunpckhps %%zmm1, %%zmm0, %%zmm16           \n"
    "   vunpckhps %%zmm3, %%zmm2, %%zmm17           \n"
    "   vunpcklpd %%zmm17, %%zmm16, %%zmm18         \n"

    "   vunpckhps %%zmm5, %%zmm4, %%zmm16           \n"
    "   vunpckhps %%zmm7, %%zmm6, %%zmm17           \n"
    "   vunpcklpd %%zmm17, %%zmm16, %%zmm19         \n"

    "   vunpckhps %%zmm9, %%zmm8, %%zmm16           \n"
    "   vunpckhps %%zmm11, %%zmm10, %%zmm17         \n"
    "   vunpcklpd %%zmm17, %%zmm16, %%zmm20         \n"

    "   vunpckhps %%zmm13, %%zmm12, %%zmm16         \n"
    "   vunpckhps %%zmm15, %%zmm14, %%zmm17         \n"
    "   vunpcklpd %%zmm17, %%zmm16, %%zmm21         \n"


    "   vinsertf32x4    $1, %%xmm19, %%zmm18, %%zmm22\n"
    "   vinsertf32x4    $1, %%xmm21, %%zmm20, %%zmm23\n"
    "   vinsertf32x8    $1, %%ymm23, %%zmm22, %%zmm24\n"

    "   vextractf32x4   $1, %%zmm18, %%xmm22         \n"
    "   vinsertf32x4    $0, %%xmm22, %%zmm19, %%zmm22\n"
    "   vextractf32x4   $1, %%zmm20, %%xmm23         \n"
    "   vinsertf32x4    $0, %%xmm23, %%zmm21, %%zmm23\n"
    "   vinsertf32x8    $1, %%ymm23, %%zmm22, %%zmm25\n"

    "   vextractf32x4   $2, %%zmm18, %%xmm22         \n"
    "   vextractf32x4   $2, %%zmm19, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm26\n"
    "   vextractf32x4   $2, %%zmm20, %%xmm22         \n"
    "   vextractf32x4   $2, %%zmm21, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm22\n"
    "   vinsertf32x8    $1, %%ymm22, %%zmm26, %%zmm26\n"

    "   vextractf32x4   $3, %%zmm18, %%xmm22         \n"
    "   vextractf32x4   $3, %%zmm19, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm27\n"
    "   vextractf32x4   $3, %%zmm20, %%xmm22         \n"
    "   vextractf32x4   $3, %%zmm21, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm22\n"
    "   vinsertf32x8    $1, %%ymm22, %%zmm27, %%zmm27\n"


    "   vmovups         %%zmm24, 128(%%r14)          \n"
    "   vmovups         %%zmm25, 384(%%r14)          \n"
    "   vmovups         %%zmm26, 640(%%r14)          \n"
    "   vmovups         %%zmm27, 896(%%r14)          \n"

    //----------------------------------------------------
    "   vunpckhps %%zmm1, %%zmm0, %%zmm16           \n"
    "   vunpckhps %%zmm3, %%zmm2, %%zmm17           \n"
    "   vunpckhpd %%zmm17, %%zmm16, %%zmm18         \n"

    "   vunpckhps %%zmm5, %%zmm4, %%zmm16           \n"
    "   vunpckhps %%zmm7, %%zmm6, %%zmm17           \n"
    "   vunpckhpd %%zmm17, %%zmm16, %%zmm19         \n"

    "   vunpckhps %%zmm9, %%zmm8, %%zmm16           \n"
    "   vunpckhps %%zmm11, %%zmm10, %%zmm17         \n"
    "   vunpckhpd %%zmm17, %%zmm16, %%zmm20         \n"

    "   vunpckhps %%zmm13, %%zmm12, %%zmm16         \n"
    "   vunpckhps %%zmm15, %%zmm14, %%zmm17         \n"
    "   vunpckhpd %%zmm17, %%zmm16, %%zmm21         \n"


    "   vinsertf32x4    $1, %%xmm19, %%zmm18, %%zmm22\n"
    "   vinsertf32x4    $1, %%xmm21, %%zmm20, %%zmm23\n"
    "   vinsertf32x8    $1, %%ymm23, %%zmm22, %%zmm24\n"

    "   vextractf32x4   $1, %%zmm18, %%xmm22         \n"
    "   vinsertf32x4    $0, %%xmm22, %%zmm19, %%zmm22\n"
    "   vextractf32x4   $1, %%zmm20, %%xmm23         \n"
    "   vinsertf32x4    $0, %%xmm23, %%zmm21, %%zmm23\n"
    "   vinsertf32x8    $1, %%ymm23, %%zmm22, %%zmm25\n"

    "   vextractf32x4   $2, %%zmm18, %%xmm22         \n"
    "   vextractf32x4   $2, %%zmm19, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm26\n"
    "   vextractf32x4   $2, %%zmm20, %%xmm22         \n"
    "   vextractf32x4   $2, %%zmm21, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm22\n"
    "   vinsertf32x8    $1, %%ymm22, %%zmm26, %%zmm26\n"

    "   vextractf32x4   $3, %%zmm18, %%xmm22         \n"
    "   vextractf32x4   $3, %%zmm19, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm27\n"
    "   vextractf32x4   $3, %%zmm20, %%xmm22         \n"
    "   vextractf32x4   $3, %%zmm21, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm22\n"
    "   vinsertf32x8    $1, %%ymm22, %%zmm27, %%zmm27\n"


    "   vmovups         %%zmm24, 192(%%r14)          \n"
    "   vmovups         %%zmm25, 448(%%r14)          \n"
    "   vmovups         %%zmm26, 704(%%r14)          \n"
    "   vmovups         %%zmm27, 960(%%r14)          \n"


    "   addq        $1024, %%r14                    \n"

    "   subq        $1, %%rsi                       \n"
    "   cmp         $0, %%rsi                       \n"
    "   jg          BEGIN_PACK_K16                  \n"


    "   tileloadd   -2048(%%r14, %%rdx, 1), %%tmm2  \n"
    "   tdpbf16ps   %%tmm2, %%tmm0, %%tmm4          \n"
    "   tdpbf16ps   %%tmm2, %%tmm1, %%tmm6          \n"
    "   tileloadd   -1024(%%r14, %%rdx, 1), %%tmm3  \n"
    "   tdpbf16ps   %%tmm3, %%tmm0, %%tmm5          \n"
    "   tdpbf16ps   %%tmm3, %%tmm1, %%tmm7          \n"
    "   addq        $64, %%rbx                      \n"
    
    ".endm                                          \n"


    "BEGIN_KERNEKL_32x32x32:                        \n"

    "   mov       %[A], %%r13                       \n"
    "   mov       %[B], %%rbx                       \n"
    "   mov       %[Bc], %%r10                      \n"
    "   mov       %[C], %%rcx                       \n"
    "   mov       %[M], %%r8                        \n"
    "   mov       %[K], %%r12                       \n"
    "   kmovq     %%r12, %%k1                       \n"
    "   mov       %[stride_K], %%r11                \n"
    "   mov       %[stride_N], %%rsi                \n"
    "   mov       %[kk], %%r15                      \n"
    "   shl       $5, %%r12                         \n"
//栈空间
    "   mov      %%rsp, %%r9                        \n"
    "   and      $63, %%r9                          \n"
    "   addq     $1024, %%r9                        \n"
    "   subq     %%r9, %%rsp                        \n"
    "   kmovq    %%r9, %%k3                         \n"


    "   mov         $64, %%rdx                      \n"


    "   tilezero    %%tmm4                          \n"   
    "   tilezero    %%tmm5                          \n"
    "   tilezero    %%tmm6                          \n"
    "   tilezero    %%tmm7                          \n"

    "   kmovq    %%rsi, %%k2                        \n"
    "   kshiftlq $4, %%k2, %%k2                     \n"

    "BEGIN_PACK:                                    \n"

    "   mov         %%r10, %%r14                    \n"
    "   mov         %%r13, %%rax                    \n"
    "   leaq        (%%rax, %%r11, 8), %%rdi        \n"
    "   leaq        (%%rdi, %%r11, 8), %%rdi        \n"
    
    "   kmovq       %%k1, %%r9                      \n"    //K

   "BEGIN_PACK_K:                                   \n"

    "   subq        $32, %%r9                       \n"
    "   PACK_FMA                                    \n"
    "   cmpq        $0, %%r9                        \n"
    "   jg          BEGIN_PACK_K                    \n"

    "   jmp         Store32x32                      \n"
    

    "BEGIN_M:                                       \n"

    "   mov         %%r13, %%rax                    \n"
    "   leaq        (%%rax, %%r11, 8), %%rdi        \n"
    "   leaq        (%%rdi, %%r11, 8), %%rdi        \n"

    "   kmovq       %%k1, %%r9                      \n"    //K
    "   mov         %%r10, %%rbx                    \n"
    
//-----------------------------------------------------------
    "BEGIN_K:                                       \n"

    "   subq        $32, %%r9                       \n"
    "   tileloadd   (%%rax, %%r11, 1), %%tmm0       \n"
    "   addq        $64, %%rax                      \n"
    "   tileloadd   (%%rbx, %%rdx, 1), %%tmm2       \n"
    "   tdpbf16ps   %%tmm0, %%tmm2, %%tmm4          \n"
    "   tileloadd   1024(%%rbx, %%rdx, 1), %%tmm3   \n"
    "   addq        $2048, %%rbx                    \n"
    "   tdpbf16ps   %%tmm0, %%tmm3, %%tmm5          \n"
    "   tileloadd   (%%rdi, %%r11, 1), %%tmm1       \n"
    "   addq        $64, %%rdi                      \n"
    "   tdpbf16ps   %%tmm1, %%tmm2, %%tmm6          \n"
    "   tdpbf16ps   %%tmm1, %%tmm3, %%tmm7          \n"

    "   cmpq        $0, %%r9                        \n"
    "   jg          BEGIN_K                         \n"

    "Store32x32:                                    \n"

   // "   Transpose_Store                            \n"
//------------------------------------------------------

    "   leaq        (%%r13, %%r11, 8), %%r13        \n"
    "   leaq        (%%r13, %%r11, 8), %%r13        \n"
    "   leaq        (%%r13, %%r11, 8), %%r13        \n"
    "   leaq        (%%r13, %%r11, 8), %%r13        \n"

    
    "   subq        $32, %%r8                       \n"
    "   cmp         $0, %%r8                        \n"
    "   jg          BEGIN_M                         \n"

    "   kmovq       %%k3, %%r9                      \n"
    "   addq        %%r9, %%rsp                     \n"

    
    :
    :
      [A] "m" (A),
      [B] "m" (B),
      [Bc] "m" (Bc),
      [C] "m" (C),
      [M] "m" (M),
      [K] "m" (K),
      [stride_N] "m" (stride_N),
      [stride_K] "m" (stride_K),
      [kk] "m" (kk)
    :"rax", "rbx", "rcx", "rdx", "rdi", "rsi","rbp","r8", "r9", "r10", "r11", "r12",
     "r13", "r14", "r15","k1","k2","k3","zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
     "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
     "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
     "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
     "zmm30", "zmm31", "memory"
    );

}

