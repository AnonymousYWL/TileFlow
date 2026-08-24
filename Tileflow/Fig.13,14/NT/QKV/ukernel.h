void mKernel_16x32x32(float *C, uint16_t* A, uint16_t* B,
                    long M, long K, long stride_N,  long stride_K, long kk)
{

    asm volatile(

    ".macro Transpose_Store                         \n"

    "   mov          $0, %%rax                      \n"
//----------------------------------------------------------
    "   tilestored   %%tmm6, (%%rsp, %%rdx, 4)      \n"

    "0:                                             \n"
//------------------------------------------------------
    "   vmovups      (%%rsp), %%zmm0%{%%k2}          \n"
    "   vmovups      (%%rsp, %%rdx, 4), %%zmm1%{%%k2}\n"
    "   vmovups      (%%rsp, %%rdx, 8), %%zmm2%{%%k2}\n"
    "   leaq         (%%rsp, %%rdx, 8), %%rbx        \n"

    "   vmovups      (%%rbx, %%rdx, 4), %%zmm3%{%%k2}\n"
    "   vmovups      (%%rbx, %%rdx, 8), %%zmm4%{%%k2}\n"
    "   leaq         (%%rbx, %%rdx, 8), %%rbx       \n"

    "   vmovups      (%%rbx, %%rdx, 4), %%zmm5%{%%k2}\n"
    "   vmovups      (%%rbx, %%rdx, 8), %%zmm6%{%%k2}\n"
    "   leaq         (%%rbx, %%rdx, 8), %%rbx       \n"

    "   vmovups      (%%rbx, %%rdx, 4), %%zmm7%{%%k2}\n"
    "   vmovups      (%%rbx, %%rdx, 8), %%zmm8%{%%k2}\n"
    "   leaq         (%%rbx, %%rdx, 8), %%rbx       \n"


    "   vmovups      (%%rbx, %%rdx, 4), %%zmm9%{%%k2}\n"
    "   vmovups      (%%rbx, %%rdx, 8), %%zmm10%{%%k2}\n"
    "   leaq         (%%rbx, %%rdx, 8), %%rbx       \n"

    "   vmovups      (%%rbx, %%rdx, 4), %%zmm11%{%%k2}\n"
    "   vmovups      (%%rbx, %%rdx, 8), %%zmm12%{%%k2}\n"
    "   leaq         (%%rbx, %%rdx, 8), %%rbx       \n"

    "   vmovups      (%%rbx, %%rdx, 4), %%zmm13%{%%k2}\n"
    "   vmovups      (%%rbx, %%rdx, 8), %%zmm14%{%%k2}\n"
    "   leaq         (%%rbx, %%rdx, 8), %%rbx       \n"
    "   vmovups      (%%rbx, %%rdx, 4), %%zmm15%{%%k2}\n"

    // Transpose        
//********************************************************************
    "   vunpcklps %%zmm1, %%zmm0, %%zmm16           \n"
    "   vunpcklps %%zmm3, %%zmm2, %%zmm17           \n"
    "   vunpcklpd %%zmm17, %%zmm16, %%zmm18         \n"
    "   vunpckhpd %%zmm17, %%zmm16, %%zmm28         \n"

    "   vunpcklps %%zmm5, %%zmm4, %%zmm16           \n"
    "   vunpcklps %%zmm7, %%zmm6, %%zmm17           \n"
    "   vunpcklpd %%zmm17, %%zmm16, %%zmm19         \n"
    "   vunpckhpd %%zmm17, %%zmm16, %%zmm29         \n"

    "   vunpcklps %%zmm9, %%zmm8, %%zmm16           \n"
    "   vunpcklps %%zmm11, %%zmm10, %%zmm17         \n"
    "   vunpcklpd %%zmm17, %%zmm16, %%zmm20         \n"
    "   vunpckhpd %%zmm17, %%zmm16, %%zmm30         \n"

    "   vunpcklps %%zmm13, %%zmm12, %%zmm16         \n"
    "   vunpcklps %%zmm15, %%zmm14, %%zmm17         \n"
    "   vunpcklpd %%zmm17, %%zmm16, %%zmm21         \n"
    "   vunpckhpd %%zmm17, %%zmm16, %%zmm31         \n"


    "   vinsertf32x4    $1, %%xmm19, %%zmm18, %%zmm22\n"
    "   vinsertf32x4    $1, %%xmm21, %%zmm20, %%zmm23\n"
    "   vinsertf32x8    $1, %%ymm23, %%zmm22, %%zmm24\n"

    "   vmovups         %%zmm24, (%%rcx)             \n"
    "   cmp             $5, %%rdx                    \n"
    "   jl              Tanspose_2_%=                \n"

    "   vextractf32x4   $1, %%zmm18, %%xmm22         \n"
    "   vinsertf32x4    $0, %%xmm22, %%zmm19, %%zmm22\n"
    "   vextractf32x4   $1, %%zmm20, %%xmm23         \n"
    "   vinsertf32x4    $0, %%xmm23, %%zmm21, %%zmm23\n"
    "   vinsertf32x8    $1, %%ymm23, %%zmm22, %%zmm25\n"


    "   vmovups         %%zmm25, (%%rcx, %%rsi, 4)   \n"
    "   cmp             $9, %%rdx                    \n"
    "   jl              Tanspose_2_%=                \n"

    "   vextractf32x4   $2, %%zmm18, %%xmm22         \n"
    "   vextractf32x4   $2, %%zmm19, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm26\n"
    "   vextractf32x4   $2, %%zmm20, %%xmm22         \n"
    "   vextractf32x4   $2, %%zmm21, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm22\n"
    "   vinsertf32x8    $1, %%ymm22, %%zmm26, %%zmm26\n"

    "   vmovups         %%zmm26, (%%rcx, %%rsi, 8)   \n"
    "   cmp             $13, %%rdx                   \n"
    "   jl              Tanspose_2_%=                \n"

    "   vextractf32x4   $3, %%zmm18, %%xmm22         \n"
    "   vextractf32x4   $3, %%zmm19, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm27\n"
    "   vextractf32x4   $3, %%zmm20, %%xmm22         \n"
    "   vextractf32x4   $3, %%zmm21, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm22\n"
    "   vinsertf32x8    $1, %%ymm22, %%zmm27, %%zmm27\n"

    "   leaq            (%%rcx, %%rsi, 8), %%r14     \n"
    "   vmovups         %%zmm27, (%%r14, %%rsi, 4)   \n"

    "Tanspose_2_%=:                                  \n"

    "   cmp             $2, %%rdx                    \n"
    "   jl              1f                           \n"

//-----------------------------------------------------

    "   vinsertf32x4    $1, %%xmm29, %%zmm28, %%zmm22\n"
    "   vinsertf32x4    $1, %%xmm31, %%zmm30, %%zmm23\n"
    "   vinsertf32x8    $1, %%ymm23, %%zmm22, %%zmm24\n"

    "   vmovups         %%zmm24, (%%rcx, %%rsi)      \n"
    "   cmp             $6, %%rdx                    \n"
    "   jl              Tanspose_3_%=                \n"

    "   vextractf32x4   $1, %%zmm28, %%xmm22         \n"
    "   vinsertf32x4    $0, %%xmm22, %%zmm29, %%zmm22\n"
    "   vextractf32x4   $1, %%zmm30, %%xmm23         \n"
    "   vinsertf32x4    $0, %%xmm23, %%zmm31, %%zmm23\n"
    "   vinsertf32x8    $1, %%ymm23, %%zmm22, %%zmm25\n"

    "   leaq            (%%rcx, %%rsi), %%r14        \n"
    "   vmovups         %%zmm25, (%%r14, %%rsi, 4)   \n"
    "   cmp             $10, %%rdx                   \n"
    "   jl              Tanspose_3_%=                \n"

    "   vextractf32x4   $2, %%zmm28, %%xmm22         \n"
    "   vextractf32x4   $2, %%zmm29, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm26\n"
    "   vextractf32x4   $2, %%zmm30, %%xmm22         \n"
    "   vextractf32x4   $2, %%zmm31, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm22\n"
    "   vinsertf32x8    $1, %%ymm22, %%zmm26, %%zmm26\n"

    "   vmovups         %%zmm26, (%%r14, %%rsi, 8)   \n"
    "   cmp             $14, %%rdx                   \n"
    "   jl              Tanspose_3_%=                \n"

    "   vextractf32x4   $3, %%zmm28, %%xmm22         \n"
    "   vextractf32x4   $3, %%zmm29, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm27\n"
    "   vextractf32x4   $3, %%zmm30, %%xmm22         \n"
    "   vextractf32x4   $3, %%zmm31, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm22\n"
    "   vinsertf32x8    $1, %%ymm22, %%zmm27, %%zmm27\n"

    "   leaq            (%%r14, %%rsi, 8), %%r14     \n"
    "   vmovups         %%zmm27, (%%r14, %%rsi, 4)   \n"


    "Tanspose_3_%=:                                  \n"

    "   cmp             $3, %%rdx                    \n"
    "   jl              1f                           \n"

//----------------------------------------------------
    "   vunpckhps %%zmm1, %%zmm0, %%zmm16           \n"
    "   vunpckhps %%zmm3, %%zmm2, %%zmm17           \n"
    "   vunpcklpd %%zmm17, %%zmm16, %%zmm18         \n"
    "   vunpckhpd %%zmm17, %%zmm16, %%zmm28         \n"

    "   vunpckhps %%zmm5, %%zmm4, %%zmm16           \n"
    "   vunpckhps %%zmm7, %%zmm6, %%zmm17           \n"
    "   vunpcklpd %%zmm17, %%zmm16, %%zmm19         \n"
    "   vunpckhpd %%zmm17, %%zmm16, %%zmm29         \n"

    "   vunpckhps %%zmm9, %%zmm8, %%zmm16           \n"
    "   vunpckhps %%zmm11, %%zmm10, %%zmm17         \n"
    "   vunpcklpd %%zmm17, %%zmm16, %%zmm20         \n"
    "   vunpckhpd %%zmm17, %%zmm16, %%zmm30         \n"

    "   vunpckhps %%zmm13, %%zmm12, %%zmm16         \n"
    "   vunpckhps %%zmm15, %%zmm14, %%zmm17         \n"
    "   vunpcklpd %%zmm17, %%zmm16, %%zmm21         \n"
    "   vunpckhpd %%zmm17, %%zmm16, %%zmm31         \n"


    "   vinsertf32x4    $1, %%xmm19, %%zmm18, %%zmm22\n"
    "   vinsertf32x4    $1, %%xmm21, %%zmm20, %%zmm23\n"
    "   vinsertf32x8    $1, %%ymm23, %%zmm22, %%zmm24\n"

    "   vmovups         %%zmm24, (%%rcx, %%rsi, 2)   \n"
    "   cmp             $7, %%rdx                    \n"
    "   jl              Tanspose_4_%=                \n"

    "   vextractf32x4   $1, %%zmm18, %%xmm22         \n"
    "   vinsertf32x4    $0, %%xmm22, %%zmm19, %%zmm22\n"
    "   vextractf32x4   $1, %%zmm20, %%xmm23         \n"
    "   vinsertf32x4    $0, %%xmm23, %%zmm21, %%zmm23\n"
    "   vinsertf32x8    $1, %%ymm23, %%zmm22, %%zmm25\n"

    "   leaq            (%%rcx, %%rsi, 2), %%r14     \n"
    "   vmovups         %%zmm25, (%%r14, %%rsi, 4)   \n"
    "   cmp             $11, %%rdx                   \n"
    "   jl              Tanspose_4_%=                \n"

    "   vextractf32x4   $2, %%zmm18, %%xmm22         \n"
    "   vextractf32x4   $2, %%zmm19, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm26\n"
    "   vextractf32x4   $2, %%zmm20, %%xmm22         \n"
    "   vextractf32x4   $2, %%zmm21, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm22\n"
    "   vinsertf32x8    $1, %%ymm22, %%zmm26, %%zmm26\n"

    "   vmovups         %%zmm26, (%%r14, %%rsi, 8)   \n"
    "   cmp             $15, %%rdx                    \n"
    "   jl              Tanspose_4_%=                \n"

    "   vextractf32x4   $3, %%zmm18, %%xmm22         \n"
    "   vextractf32x4   $3, %%zmm19, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm27\n"
    "   vextractf32x4   $3, %%zmm20, %%xmm22         \n"
    "   vextractf32x4   $3, %%zmm21, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm22\n"
    "   vinsertf32x8    $1, %%ymm22, %%zmm27, %%zmm27\n"

    "   leaq            (%%r14, %%rsi, 8), %%r14     \n"
    "   vmovups         %%zmm27, (%%r14, %%rsi, 4)   \n"

    "Tanspose_4_%=:                                  \n"

    "   cmp             $4, %%rdx                    \n"
    "   jl              1f                           \n"

//------------------------------------------------------

    "   vinsertf32x4    $1, %%xmm29, %%zmm28, %%zmm22\n"
    "   vinsertf32x4    $1, %%xmm31, %%zmm30, %%zmm23\n"
    "   vinsertf32x8    $1, %%ymm23, %%zmm22, %%zmm24\n"

    "   leaq            (%%rcx, %%rsi, 2), %%r14     \n"
    "   leaq            (%%r14, %%rsi), %%r14        \n"
    "   vmovups         %%zmm24, (%%r14)             \n"
    "   cmp             $8, %%rdx                    \n"
    "   jl              1f                           \n"

    "   vextractf32x4   $1, %%zmm28, %%xmm22         \n"
    "   vinsertf32x4    $0, %%xmm22, %%zmm29, %%zmm22\n"
    "   vextractf32x4   $1, %%zmm30, %%xmm23         \n"
    "   vinsertf32x4    $0, %%xmm23, %%zmm31, %%zmm23\n"
    "   vinsertf32x8    $1, %%ymm23, %%zmm22, %%zmm25\n"

    "   vmovups         %%zmm25, (%%r14, %%rsi, 4)   \n"
    "   cmp             $12, %%rdx                   \n"
    "   jl              1f                           \n"

    "   vextractf32x4   $2, %%zmm28, %%xmm22         \n"
    "   vextractf32x4   $2, %%zmm29, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm26\n"
    "   vextractf32x4   $2, %%zmm30, %%xmm22         \n"
    "   vextractf32x4   $2, %%zmm31, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm22\n"
    "   vinsertf32x8    $1, %%ymm22, %%zmm26, %%zmm26\n"

    "   vmovups         %%zmm26, (%%r14, %%rsi, 8)   \n"
    "   cmp             $16, %%rdx                   \n"
    "   jl              1f                           \n"

    "   vextractf32x4   $3, %%zmm28, %%xmm22         \n"
    "   vextractf32x4   $3, %%zmm29, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm27\n"
    "   vextractf32x4   $3, %%zmm30, %%xmm22         \n"
    "   vextractf32x4   $3, %%zmm31, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm22\n"
    "   vinsertf32x8    $1, %%ymm22, %%zmm27, %%zmm27\n"

    "   leaq            (%%r14, %%rsi, 8), %%r14     \n"
    "   vmovups         %%zmm27, (%%r14, %%rsi, 4)   \n"

    "1:                                             \n"

    "   addq         $1, %%rax                      \n"
    "   cmp          $2, %%rax                      \n"
    "   je           3f                             \n"

    "2:                                             \n"
    "   tilestored   %%tmm7, (%%rsp, %%rdx, 4)      \n"
    "   addq        $64, %%rcx                      \n"
    "   jmp         0b                              \n"
 
    "3:                                             \n"

    ".endm                                          \n"


    "BEGIN_KERNEKL_16x32x32:                        \n"

    "   mov       %[A], %%rax                       \n"
    "   mov       %[B], %%rbx                       \n"
    "   mov       %[C], %%rcx                       \n"
    "   mov       %[M], %%rdx                       \n"
    "   mov       %[K], %%r12                       \n"
    "   kmovq     %%r12, %%k1                       \n"
    "   mov       %[stride_K], %%r11                \n"
    "   mov       %[stride_N], %%rsi                \n"
    "   mov       %[kk], %%r15                      \n"
    "   shl       $5, %%r12                         \n"

    "   mov       %%rdx, %%r8                       \n"
    "   shl       $6, %%r8                          \n"

//栈空间
    "   mov      %%rsp, %%r9                        \n"
    "   and      $63, %%r9                          \n"
    "   addq     $1024, %%r9                        \n"
    "   subq     %%r9, %%rsp                        \n"
    "   kmovq    %%r9, %%k3                         \n"

    "   mov     $1, %%r13                           \n"
    "   shlx    %%rdx, %%r13, %%r13                 \n"
    "   sub     $1, %%r13                           \n"
    "   kmovq   %%r13, %%k2                         \n"//mask


    "   tilezero    %%tmm6                          \n"
    "   tilezero    %%tmm7                          \n"

    "   leaq        (%%rbx, %%r11, 8), %%rbp        \n"
    "   leaq        (%%rbp, %%r11, 8), %%rbp        \n"

    "   kmovq       %%k1, %%r9                      \n"    //K
//-----------------------------------------------------------
    "BEGIN_K:                                       \n"

    "   subq        $32, %%r9                       \n"
    "   tileloadd   (%%rax, %%rdx, 4), %%tmm1       \n"
    "   addq        %%r8, %%rax                    \n"
    "   tileloaddt1   (%%rbx, %%r11, 1), %%tmm2     \n"
    "   addq        $64, %%rbx                      \n"
    "   tdpbf16ps   %%tmm1, %%tmm2, %%tmm6          \n"
    "   tileloaddt1   (%%rbp, %%r11, 1), %%tmm3     \n"
    "   addq        $64, %%rbp                      \n"
    "   tdpbf16ps   %%tmm1, %%tmm3, %%tmm7          \n"

    "   cmpq        $0, %%r9                        \n"
    "   jg          BEGIN_K                         \n"

    "Store32x32:                                    \n"

    "   Transpose_Store                            \n"
//------------------------------------------------------

    "   kmovq       %%k3, %%r9                      \n"
    "   addq        %%r9, %%rsp                     \n"

    
    :
    :
      [A] "m" (A),
      [B] "m" (B),
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