void mKernel_32x32x32(float *C, uint16_t* A, uint16_t* B,
                    long M, long K, long stride_N,  long stride_K, long kk)
{

    asm volatile(

    ".macro Transpose_Store                         \n"

    "   mov          $0, %%rax                      \n"
//----------------------------------------------------------
    "   tilestored   %%tmm4, (%%rsp, %%rdx, 1)      \n"
    "   tilezero    %%tmm4                          \n"

    "0:                                             \n"
//------------------------------------------------------

    "   vmovups      (%%rsp), %%zmm0                \n"
    "   vmovups      64(%%rsp), %%zmm1              \n"
    "   vmovups      128(%%rsp), %%zmm2             \n"
    "   vmovups      192(%%rsp), %%zmm3             \n"
    "   vmovups      256(%%rsp), %%zmm4             \n"
    "   vmovups      320(%%rsp), %%zmm5             \n"
    "   vmovups      384(%%rsp), %%zmm6             \n"
    "   vmovups      448(%%rsp), %%zmm7             \n"
    "   vmovups      512(%%rsp), %%zmm8             \n"
    "   vmovups      576(%%rsp), %%zmm9             \n"
    "   vmovups      640(%%rsp), %%zmm10            \n"
    "   vmovups      704(%%rsp), %%zmm11            \n"
    "   vmovups      768(%%rsp), %%zmm12            \n"
    "   vmovups      832(%%rsp), %%zmm13            \n"
    "   vmovups      896(%%rsp), %%zmm14            \n"
    "   vmovups      960(%%rsp), %%zmm15            \n"

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


    "   cmp             $0, %%r15                    \n"
    "   je              5f                           \n"
    "   vmovups         (%%rcx), %%zmm18             \n"
    "   vmovups         (%%rcx, %%rsi, 4), %%zmm19   \n"
    "   vaddps          %%zmm18, %%zmm24, %%zmm24    \n"
    "   vaddps          %%zmm19, %%zmm25, %%zmm25    \n"
    "   vmovups         (%%rcx, %%rsi, 8), %%zmm18   \n"
    "   leaq            (%%rcx, %%rsi, 8), %%r14     \n"
    "   vmovups         (%%r14, %%rsi, 4), %%zmm19   \n"
    "   vaddps          %%zmm18, %%zmm26, %%zmm26    \n"
    "   vaddps          %%zmm19, %%zmm27, %%zmm27    \n"

    "5:                                              \n"
    "   vmovups         %%zmm24, (%%rcx)             \n"
    "   vmovups         %%zmm25, (%%rcx, %%rsi, 4)   \n"
    "   vmovups         %%zmm26, (%%rcx, %%rsi, 8)   \n"
    "   leaq            (%%rcx, %%rsi, 8), %%r14     \n"
    "   vmovups         %%zmm27, (%%r14, %%rsi, 4)   \n"

//----------------------------------------------------

    "   vinsertf32x4    $1, %%xmm29, %%zmm28, %%zmm22\n"
    "   vinsertf32x4    $1, %%xmm31, %%zmm30, %%zmm23\n"
    "   vinsertf32x8    $1, %%ymm23, %%zmm22, %%zmm24\n"

    "   vextractf32x4   $1, %%zmm28, %%xmm22         \n"
    "   vinsertf32x4    $0, %%xmm22, %%zmm29, %%zmm22\n"
    "   vextractf32x4   $1, %%zmm30, %%xmm23         \n"
    "   vinsertf32x4    $0, %%xmm23, %%zmm31, %%zmm23\n"
    "   vinsertf32x8    $1, %%ymm23, %%zmm22, %%zmm25\n"

    "   vextractf32x4   $2, %%zmm28, %%xmm22         \n"
    "   vextractf32x4   $2, %%zmm29, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm26\n"
    "   vextractf32x4   $2, %%zmm30, %%xmm22         \n"
    "   vextractf32x4   $2, %%zmm31, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm22\n"
    "   vinsertf32x8    $1, %%ymm22, %%zmm26, %%zmm26\n"

    "   vextractf32x4   $3, %%zmm28, %%xmm22         \n"
    "   vextractf32x4   $3, %%zmm29, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm27\n"
    "   vextractf32x4   $3, %%zmm30, %%xmm22         \n"
    "   vextractf32x4   $3, %%zmm31, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm22\n"
    "   vinsertf32x8    $1, %%ymm22, %%zmm27, %%zmm27\n"


    "   cmp             $0, %%r15                    \n"
    "   je              6f                           \n"
    "   vmovups         (%%rcx, %%rsi), %%zmm18      \n"
    "   leaq            (%%rcx, %%rsi), %%r14        \n"
    "   vmovups         (%%r14, %%rsi, 4), %%zmm19   \n"
    "   vaddps          %%zmm18, %%zmm24, %%zmm24    \n"
    "   vaddps          %%zmm19, %%zmm25, %%zmm25    \n"
    "   vmovups         (%%r14, %%rsi, 8), %%zmm18   \n"
    "   leaq            (%%r14, %%rsi, 8), %%r14     \n"
    "   vmovups         (%%r14, %%rsi, 4), %%zmm19   \n"
    "   vaddps          %%zmm18, %%zmm26, %%zmm26    \n"
    "   vaddps          %%zmm19, %%zmm27, %%zmm27    \n"

    "6:                                              \n"

    "   vmovups         %%zmm24, (%%rcx, %%rsi)      \n"
    "   leaq            (%%rcx, %%rsi), %%r14        \n"
    "   vmovups         %%zmm25, (%%r14, %%rsi, 4)   \n"
    "   vmovups         %%zmm26, (%%r14, %%rsi, 8)   \n"
    "   leaq            (%%r14, %%rsi, 8), %%r14     \n"
    "   vmovups         %%zmm27, (%%r14, %%rsi, 4)   \n"


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


    "   cmp             $0, %%r15                    \n"
    "   je              7f                           \n"
    "   vmovups         (%%rcx, %%rsi, 2), %%zmm18   \n"
    "   leaq            (%%rcx, %%rsi, 2), %%r14     \n"
    "   vmovups         (%%r14, %%rsi, 4), %%zmm19   \n"
    "   vaddps          %%zmm18, %%zmm24, %%zmm24    \n"
    "   vaddps          %%zmm19, %%zmm25, %%zmm25    \n"
    "   vmovups         (%%r14, %%rsi, 8), %%zmm18   \n"
    "   leaq            (%%r14, %%rsi, 8), %%r14     \n"
    "   vmovups         (%%r14, %%rsi, 4), %%zmm19   \n"
    "   vaddps          %%zmm18, %%zmm26, %%zmm26    \n"
    "   vaddps          %%zmm19, %%zmm27, %%zmm27    \n"

    "7:                                              \n"

    "   vmovups         %%zmm24, (%%rcx, %%rsi, 2)   \n"
    "   leaq            (%%rcx, %%rsi, 2), %%r14     \n"
    "   vmovups         %%zmm25, (%%r14, %%rsi, 4)   \n"
    "   vmovups         %%zmm26, (%%r14, %%rsi, 8)   \n"
    "   leaq            (%%r14, %%rsi, 8), %%r14     \n"
    "   vmovups         %%zmm27, (%%r14, %%rsi, 4)   \n"

//----------------------------------------------------

    "   vinsertf32x4    $1, %%xmm29, %%zmm28, %%zmm22\n"
    "   vinsertf32x4    $1, %%xmm31, %%zmm30, %%zmm23\n"
    "   vinsertf32x8    $1, %%ymm23, %%zmm22, %%zmm24\n"

    "   vextractf32x4   $1, %%zmm28, %%xmm22         \n"
    "   vinsertf32x4    $0, %%xmm22, %%zmm29, %%zmm22\n"
    "   vextractf32x4   $1, %%zmm30, %%xmm23         \n"
    "   vinsertf32x4    $0, %%xmm23, %%zmm31, %%zmm23\n"
    "   vinsertf32x8    $1, %%ymm23, %%zmm22, %%zmm25\n"

    "   vextractf32x4   $2, %%zmm28, %%xmm22         \n"
    "   vextractf32x4   $2, %%zmm29, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm26\n"
    "   vextractf32x4   $2, %%zmm30, %%xmm22         \n"
    "   vextractf32x4   $2, %%zmm31, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm22\n"
    "   vinsertf32x8    $1, %%ymm22, %%zmm26, %%zmm26\n"

    "   vextractf32x4   $3, %%zmm28, %%xmm22         \n"
    "   vextractf32x4   $3, %%zmm29, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm27\n"
    "   vextractf32x4   $3, %%zmm30, %%xmm22         \n"
    "   vextractf32x4   $3, %%zmm31, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm22\n"
    "   vinsertf32x8    $1, %%ymm22, %%zmm27, %%zmm27\n"


    "   cmp             $0, %%r15                    \n"
    "   je              8f                           \n"

    "   leaq            (%%rcx, %%rsi, 2), %%r14     \n"
    "   leaq            (%%r14, %%rsi), %%r14        \n"
    "   vmovups         (%%r14), %%zmm18             \n"
    "   vmovups         (%%r14, %%rsi, 4), %%zmm19   \n"
    "   vaddps          %%zmm18, %%zmm24, %%zmm24    \n"
    "   vaddps          %%zmm19, %%zmm25, %%zmm25    \n"
    "   vmovups         (%%r14, %%rsi, 8), %%zmm18   \n"
    "   leaq            (%%r14, %%rsi, 8), %%r14     \n"
    "   vmovups         (%%r14, %%rsi, 4), %%zmm19   \n"
    "   vaddps          %%zmm18, %%zmm26, %%zmm26    \n"
    "   vaddps          %%zmm19, %%zmm27, %%zmm27    \n"

    "8:                                              \n"

    "   leaq            (%%rcx, %%rsi, 2), %%r14     \n"
    "   leaq            (%%r14, %%rsi), %%r14        \n"
    "   vmovups         %%zmm24, (%%r14)             \n"
    "   vmovups         %%zmm25, (%%r14, %%rsi, 4)   \n"
    "   vmovups         %%zmm26, (%%r14, %%rsi, 8)   \n"
    "   leaq            (%%r14, %%rsi, 8), %%r14     \n"
    "   vmovups         %%zmm27, (%%r14, %%rsi, 4)   \n"


    "   addq         $1, %%rax                      \n"
    "   cmp          $4, %%rax                      \n"
    "   je           4f                             \n"

    "1:                                             \n"
    "   cmp          $1, %%rax                      \n"
    "   jg           2f                             \n"
    "   tilestored   %%tmm5, (%%rsp, %%rdx, 1)      \n"
    "   tilezero    %%tmm5                          \n"
    "   addq        $64, %%rcx                      \n"
    "   jmp         0b                              \n"


    "2:                                             \n"
    "   cmp         $2, %%rax                       \n"
    "   jg          3f                              \n"
    "   tilestored   %%tmm6, (%%rsp, %%rdx, 1)      \n"
    "   tilezero    %%tmm6                          \n"
    "   subq        $64, %%rcx                      \n"
    "   kmovq       %%k2, %%r14                     \n"
    "   leaq        (%%rcx, %%r14), %%rcx           \n"
    "   jmp         0b                              \n"

    "3:                                             \n"
    "   tilestored   %%tmm7, (%%rsp, %%rdx, 1)      \n"
    "   tilezero     %%tmm7                         \n"
    "   addq        $64, %%rcx                      \n"
    "   jmp         0b                              \n"

    "4:                                             \n"
    "   subq        $64, %%rcx                      \n"
    "   kmovq       %%k2, %%r14                     \n"
    "   leaq        (%%rcx, %%r14), %%rcx           \n"

    ".endm                                          \n"


    "BEGIN_KERNEKL_32x32x32:                        \n"

    "   mov       %[A], %%r13                       \n"
    "   mov       %[B], %%r10                       \n"
    "   mov       %[C], %%rcx                       \n"
    "   mov       %[M], %%r8                        \n"
    "   mov       %[K], %%r12                       \n"
    "   kmovq     %%r12, %%k1                       \n"
    "   mov       %[stride_K], %%r11                \n"
    "   mov       %[stride_N], %%rsi                \n"
    "   mov       %[kk], %%r15                      \n"
    "   shl       $5, %%r12                         \n"

    "   mov       $64, %%rdx                        \n"
//栈空间
    "   mov      %%rsp, %%r9                        \n"
    "   and      $63, %%r9                          \n"
    "   addq     $1024, %%r9                        \n"
    "   subq     %%r9, %%rsp                        \n"
    "   kmovq    %%r9, %%k3                         \n"


    "   tilezero    %%tmm4                          \n"   
    "   tilezero    %%tmm5                          \n"
    "   tilezero    %%tmm6                          \n"
    "   tilezero    %%tmm7                          \n"

    "   kmovq    %%rsi, %%k2                        \n"
    "   kshiftlq $4, %%k2, %%k2                     \n"
    

    "BEGIN_M:                                       \n"
    
    "   mov         %%r10, %%rbx                    \n" //B*
    "   leaq        (%%rbx, %%r11, 8), %%rbp        \n"
    "   leaq        (%%rbp, %%r11, 8), %%rbp        \n"

    "   mov         %%r13, %%rax                    \n" //A*
    "   leaq        (%%rax, %%r12), %%rdi           \n"

    "   kmovq       %%k1, %%r9                      \n"    //K
    
//-----------------------------------------------------------
    "BEGIN_K:                                       \n"

    "   subq        $32, %%r9                       \n"
    "   tileloadd   (%%rax, %%rdx, 1), %%tmm0     \n"
    "   addq        $1024, %%rax                    \n"
    "   tileloadd   (%%rbx, %%r11, 1), %%tmm2       \n"
    "   addq        $64, %%rbx                      \n"
    "   tdpbf16ps   %%tmm0, %%tmm2, %%tmm4          \n"
    "   tileloadd   (%%rbp, %%r11, 1), %%tmm3       \n"
    "   addq        $64, %%rbp                      \n"
    "   tdpbf16ps   %%tmm0, %%tmm3, %%tmm5          \n"
    "   tileloadd   (%%rdi, %%rdx, 1), %%tmm1     \n"
    "   addq        $1024, %%rdi                    \n"
    "   tdpbf16ps   %%tmm1, %%tmm2, %%tmm6          \n"
    "   tdpbf16ps   %%tmm1, %%tmm3, %%tmm7          \n"

    "   cmpq        $0, %%r9                        \n"
    "   jg          BEGIN_K                         \n"

    "Store32x32:                                    \n"

    "   Transpose_Store                            \n"
//------------------------------------------------------

    "   leaq        (%%r13, %%r12, 2), %%r13        \n"

    
    "   subq        $32, %%r8                       \n"
    "   cmp         $0, %%r8                        \n"
    "   jg          BEGIN_M                         \n"

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


void mKernel_16x32x32(float *C, uint16_t* A, uint16_t* B,
                    uint32_t M, long K, long stride_N,  long stride_K, long kk)
{

    asm volatile(

    ".macro Transpose_Store_16x32                   \n"

    "   mov          $0, %%rax                      \n"
//----------------------------------------------------------
    "   tilestored   %%tmm6, (%%rsp, %%rdx, 4)      \n"
    "   tilezero    %%tmm6                          \n"

    "0:                                             \n"
//------------------------------------------------------

    "   vmovups      (%%rsp), %%zmm0%{%%k2}         \n"
    "   vmovups      (%%rsp, %%rdx, 4), %%zmm1%{%%k2} \n"
    "   vmovups      (%%rsp, %%rdx, 8), %%zmm2%{%%k2} \n"
    "   leaq         (%%rsp, %%rdx, 8), %%r9     \n"
    "   vmovups      (%%r9, %%rdx, 4), %%zmm3%{%%k2} \n"
    "   vmovups      (%%r9, %%rdx, 8), %%zmm4%{%%k2} \n"
    "   leaq         (%%r9, %%rdx, 8), %%r9     \n"
    "   vmovups      (%%r9, %%rdx, 4), %%zmm5%{%%k2} \n"
    "   vmovups      (%%r9, %%rdx, 8), %%zmm6%{%%k2} \n"
    "   leaq         (%%r9, %%rdx, 8), %%r9     \n"
    "   vmovups      (%%r9, %%rdx, 4), %%zmm7%{%%k2}      \n"
    "   vmovups      (%%r9, %%rdx, 8), %%zmm8%{%%k2}      \n"
    "   leaq         (%%r9, %%rdx, 8), %%r9     \n"
    "   vmovups      (%%r9, %%rdx, 4), %%zmm9%{%%k2}      \n"
    "   vmovups      (%%r9, %%rdx, 8), %%zmm10%{%%k2}     \n"
    "   leaq         (%%r9, %%rdx, 8), %%r9     \n"
    "   vmovups      (%%r9, %%rdx, 4), %%zmm11%{%%k2}     \n"
    "   vmovups      (%%r9, %%rdx, 8), %%zmm12%{%%k2}     \n"
    "   leaq         (%%r9, %%rdx, 8), %%r9     \n"
    "   vmovups      (%%r9, %%rdx, 4), %%zmm13%{%%k2}     \n"
    "   vmovups      (%%r9, %%rdx, 8), %%zmm14%{%%k2}     \n"
    "   leaq         (%%r9, %%rdx, 8), %%r9     \n"
    "   vmovups      (%%r9, %%rdx, 4), %%zmm15%{%%k2}     \n"

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


    "   cmp             $0, %%r15                    \n"
    "   je              5f                           \n"
    "   vmovups         (%%rcx), %%zmm18             \n"
    "   vaddps          %%zmm18, %%zmm24, %%zmm24    \n"
    "   cmp             $5, %%rdx                    \n"
    "   jl              5f                           \n"
    "   vmovups         (%%rcx, %%rsi, 4), %%zmm19   \n"
    "   vaddps          %%zmm19, %%zmm25, %%zmm25    \n"
    "   cmp             $9, %%rdx                    \n"
    "   jl              5f                           \n"
    "   vmovups         (%%rcx, %%rsi, 8), %%zmm18   \n"
    "   vaddps          %%zmm18, %%zmm26, %%zmm26    \n"
    "   cmp             $13, %%rdx                   \n"
    "   jl              5f                           \n"
    "   leaq            (%%rcx, %%rsi, 8), %%r14     \n"
    "   vmovups         (%%r14, %%rsi, 4), %%zmm19   \n"
    "   vaddps          %%zmm19, %%zmm27, %%zmm27    \n"

    "5:                                              \n"
    "   vmovups         %%zmm24, (%%rcx)             \n"
    "   cmp             $5, %%rdx                    \n"
    "   jl              Tanspose_2_%=                \n"
    "   vmovups         %%zmm25, (%%rcx, %%rsi, 4)   \n"
    "   cmp             $9, %%rdx                    \n"
    "   jl              Tanspose_2_%=                \n"
    "   vmovups         %%zmm26, (%%rcx, %%rsi, 8)   \n"
    "   cmp             $13, %%rdx                   \n"
    "   jl              Tanspose_2_%=                \n"
    "   leaq            (%%rcx, %%rsi, 8), %%r14     \n"
    "   vmovups         %%zmm27, (%%r14, %%rsi, 4)   \n"


    "Tanspose_2_%=:                                  \n"
//----------------------------------------------------

    "   cmp             $2, %%rdx                   \n"
    "   jl              1f                          \n"
    
    "   vinsertf32x4    $1, %%xmm29, %%zmm28, %%zmm22\n"
    "   vinsertf32x4    $1, %%xmm31, %%zmm30, %%zmm23\n"
    "   vinsertf32x8    $1, %%ymm23, %%zmm22, %%zmm24\n"

    "   vextractf32x4   $1, %%zmm28, %%xmm22         \n"
    "   vinsertf32x4    $0, %%xmm22, %%zmm29, %%zmm22\n"
    "   vextractf32x4   $1, %%zmm30, %%xmm23         \n"
    "   vinsertf32x4    $0, %%xmm23, %%zmm31, %%zmm23\n"
    "   vinsertf32x8    $1, %%ymm23, %%zmm22, %%zmm25\n"

    "   vextractf32x4   $2, %%zmm28, %%xmm22         \n"
    "   vextractf32x4   $2, %%zmm29, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm26\n"
    "   vextractf32x4   $2, %%zmm30, %%xmm22         \n"
    "   vextractf32x4   $2, %%zmm31, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm22\n"
    "   vinsertf32x8    $1, %%ymm22, %%zmm26, %%zmm26\n"

    "   vextractf32x4   $3, %%zmm28, %%xmm22         \n"
    "   vextractf32x4   $3, %%zmm29, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm27\n"
    "   vextractf32x4   $3, %%zmm30, %%xmm22         \n"
    "   vextractf32x4   $3, %%zmm31, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm22\n"
    "   vinsertf32x8    $1, %%ymm22, %%zmm27, %%zmm27\n"

    "   cmp             $0, %%r15                    \n"
    "   je              6f                           \n"
    "   vmovups         (%%rcx, %%rsi), %%zmm18      \n"
    "   vaddps          %%zmm18, %%zmm24, %%zmm24    \n"
    "   cmp             $6, %%rdx                    \n"
    "   jl              6f                           \n"
    "   leaq            (%%rcx, %%rsi), %%r14        \n"
    "   vmovups         (%%r14, %%rsi, 4), %%zmm19   \n"
    "   vaddps          %%zmm19, %%zmm25, %%zmm25    \n"
    "   cmp             $10, %%rdx                   \n"
    "   jl              6f                           \n"
    "   vmovups         (%%r14, %%rsi, 8), %%zmm18   \n"
    "   vaddps          %%zmm18, %%zmm26, %%zmm26    \n"
    "   cmp             $14, %%rdx                   \n"
    "   jl              6f                           \n"
    "   leaq            (%%r14, %%rsi, 8), %%r14     \n"
    "   vmovups         (%%r14, %%rsi, 4), %%zmm19   \n"
    "   vaddps          %%zmm19, %%zmm27, %%zmm27    \n"

    "6:                                              \n"

    "   vmovups         %%zmm24, (%%rcx, %%rsi)      \n"
    "   cmp             $6, %%rdx                    \n"
    "   jl              Tanspose_3_%=                \n"
    "   leaq            (%%rcx, %%rsi), %%r14        \n"
    "   vmovups         %%zmm25, (%%r14, %%rsi, 4)   \n"
    "   cmp             $10, %%rdx                   \n"
    "   jl              Tanspose_3_%=                \n"
    "   vmovups         %%zmm26, (%%r14, %%rsi, 8)   \n"
    "   cmp             $14, %%rdx                   \n"
    "   jl              Tanspose_3_%=                \n"
    "   leaq            (%%r14, %%rsi, 8), %%r14     \n"
    "   vmovups         %%zmm27, (%%r14, %%rsi, 4)   \n"

    "Tanspose_3_%=:                                  \n"
//----------------------------------------------------

    "   cmp             $3, %%rdx                   \n"
    "   jl              1f                          \n"

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


    "   cmp             $0, %%r15                    \n"
    "   je              7f                           \n"
    "   vmovups         (%%rcx, %%rsi, 2), %%zmm18   \n"
    "   vaddps          %%zmm18, %%zmm24, %%zmm24    \n"
    "   cmp             $7, %%rdx                    \n"
    "   jl              7f                           \n"
    "   leaq            (%%rcx, %%rsi, 2), %%r14     \n"
    "   vmovups         (%%r14, %%rsi, 4), %%zmm19   \n"
    "   vaddps          %%zmm19, %%zmm25, %%zmm25    \n"
    "   cmp             $11, %%rdx                   \n"
    "   jl              7f                           \n"
    "   vmovups         (%%r14, %%rsi, 8), %%zmm18   \n"
    "   vaddps          %%zmm18, %%zmm26, %%zmm26    \n"
    "   cmp             $15, %%rdx                   \n"
    "   jl              7f                           \n"
    "   leaq            (%%r14, %%rsi, 8), %%r14     \n"
    "   vmovups         (%%r14, %%rsi, 4), %%zmm19   \n"
    "   vaddps          %%zmm19, %%zmm27, %%zmm27    \n"

    "7:                                              \n"

    "   vmovups         %%zmm24, (%%rcx, %%rsi, 2)   \n"
    "   cmp             $7, %%rdx                    \n"
    "   jl              Tanspose_4_%=                \n"
    "   leaq            (%%rcx, %%rsi, 2), %%r14     \n"
    "   vmovups         %%zmm25, (%%r14, %%rsi, 4)   \n"
    "   cmp             $11, %%rdx                   \n"
    "   jl              Tanspose_4_%=                \n"
    "   vmovups         %%zmm26, (%%r14, %%rsi, 8)   \n"
    "   cmp             $15, %%rdx                    \n"
    "   jl              Tanspose_4_%=                \n"
    "   leaq            (%%r14, %%rsi, 8), %%r14     \n"
    "   vmovups         %%zmm27, (%%r14, %%rsi, 4)   \n"

    "Tanspose_4_%=:                                  \n"
//----------------------------------------------------

    "   cmp             $4, %%rdx                   \n"
    "   jl              1f                          \n"

    "   vinsertf32x4    $1, %%xmm29, %%zmm28, %%zmm22\n"
    "   vinsertf32x4    $1, %%xmm31, %%zmm30, %%zmm23\n"
    "   vinsertf32x8    $1, %%ymm23, %%zmm22, %%zmm24\n"

    "   vextractf32x4   $1, %%zmm28, %%xmm22         \n"
    "   vinsertf32x4    $0, %%xmm22, %%zmm29, %%zmm22\n"
    "   vextractf32x4   $1, %%zmm30, %%xmm23         \n"
    "   vinsertf32x4    $0, %%xmm23, %%zmm31, %%zmm23\n"
    "   vinsertf32x8    $1, %%ymm23, %%zmm22, %%zmm25\n"

    "   vextractf32x4   $2, %%zmm28, %%xmm22         \n"
    "   vextractf32x4   $2, %%zmm29, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm26\n"
    "   vextractf32x4   $2, %%zmm30, %%xmm22         \n"
    "   vextractf32x4   $2, %%zmm31, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm22\n"
    "   vinsertf32x8    $1, %%ymm22, %%zmm26, %%zmm26\n"

    "   vextractf32x4   $3, %%zmm28, %%xmm22         \n"
    "   vextractf32x4   $3, %%zmm29, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm27\n"
    "   vextractf32x4   $3, %%zmm30, %%xmm22         \n"
    "   vextractf32x4   $3, %%zmm31, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm22\n"
    "   vinsertf32x8    $1, %%ymm22, %%zmm27, %%zmm27\n"


    "   cmp             $0, %%r15                    \n"
    "   je              8f                           \n"

    "   leaq            (%%rcx, %%rsi, 2), %%r14     \n"
    "   leaq            (%%r14, %%rsi), %%r14        \n"
    "   vmovups         (%%r14), %%zmm18             \n"
    "   vaddps          %%zmm18, %%zmm24, %%zmm24    \n"
    "   cmp             $8, %%rdx                    \n"
    "   jl              8f                           \n"
    "   vmovups         (%%r14, %%rsi, 4), %%zmm19   \n"
    "   vaddps          %%zmm19, %%zmm25, %%zmm25    \n"
    "   cmp             $12, %%rdx                   \n"
    "   jl              8f                           \n"
    "   vmovups         (%%r14, %%rsi, 8), %%zmm18   \n"
    "   vaddps          %%zmm18, %%zmm26, %%zmm26    \n"
    "   cmp             $16, %%rdx                   \n"
    "   jl              8f                           \n"
    "   leaq            (%%r14, %%rsi, 8), %%r14     \n"
    "   vmovups         (%%r14, %%rsi, 4), %%zmm19   \n"
    "   vaddps          %%zmm19, %%zmm27, %%zmm27    \n"

    "8:                                              \n"

    "   leaq            (%%rcx, %%rsi, 2), %%r14     \n"
    "   leaq            (%%r14, %%rsi), %%r14        \n"
    "   vmovups         %%zmm24, (%%r14)             \n"
    "   cmp             $8, %%rdx                    \n"
    "   jl              1f                           \n"
    "   vmovups         %%zmm25, (%%r14, %%rsi, 4)   \n"
    "   cmp             $12, %%rdx                   \n"
    "   jl              1f                           \n"
    "   vmovups         %%zmm26, (%%r14, %%rsi, 8)   \n"
    "   cmp             $16, %%rdx                   \n"
    "   jl              1f                           \n"
    "   leaq            (%%r14, %%rsi, 8), %%r14     \n"
    "   vmovups         %%zmm27, (%%r14, %%rsi, 4)   \n"

    "1:                                             \n"
    "   addq         $1, %%rax                      \n"
    "   cmp          $2, %%rax                      \n"
    "   je           2f                             \n"
    "   tilestored   %%tmm7, (%%rsp, %%rdx, 4)      \n"
    "   tilezero    %%tmm7                          \n"
    "   addq        $64, %%rcx                      \n"
    "   jmp         0b                              \n"


    "2:                                             \n"

    ".endm                                          \n"


    "BEGIN_KERNEKL_16x32x32:                        \n"

    "   mov       %[A], %%rax                       \n"
    "   mov       %[B], %%rbx                       \n"
    "   mov       %[C], %%rcx                       \n"
    "   mov       %[K], %%r12                       \n"
    "   kmovq     %%r12, %%k1                       \n"
    "   mov       %[stride_K], %%r11                \n"
    "   mov       %[stride_N], %%rsi                \n"
    "   mov       %[kk], %%r15                      \n"
    "   shl       $5, %%r12                         \n"

    "   movl      %[M], %%edx                       \n"
    "   mov       %%rdx, %%r8                      \n"
    "   shl       $6, %%r8                         \n"


    "   mov       $1, %%r14                         \n"
    "   shlx      %%rdx, %%r14, %%r14               \n"
    "   subq      $1, %%r14                         \n"
    "   kmovq     %%r14, %%k2                       \n"


//栈空间
    "   mov      %%rsp, %%r9                        \n"
    "   and      $63, %%r9                          \n"
    "   addq     $1024, %%r9                        \n"
    "   subq     %%r9, %%rsp                        \n"
    "   kmovq    %%r9, %%k3                         \n"


    "   tilezero    %%tmm6                          \n"   
    "   tilezero    %%tmm7                          \n"

    "BEGIN_M_16x32:                                 \n"
    
    "   leaq        (%%rbx, %%r11, 8), %%rbp        \n"
    "   leaq        (%%rbp, %%r11, 8), %%rbp        \n"

    "   kmovq       %%k1, %%r9                      \n"    //K
    
//-----------------------------------------------------------
    "BEGIN_K_16x32:                                 \n"

    "   subq        $32, %%r9                       \n"
    "   tileloadd   (%%rax, %%rdx, 4), %%tmm1       \n"
    "   addq        %%r8, %%rax                     \n"
    "   tileloadd   (%%rbx, %%r11, 1), %%tmm2       \n"
    "   addq        $64, %%rbx                      \n"
    "   tdpbf16ps   %%tmm1, %%tmm2, %%tmm6          \n"
    "   tileloadd   (%%rbp, %%r11, 1), %%tmm3       \n"
    "   addq        $64, %%rbp                      \n"
    "   tdpbf16ps   %%tmm1, %%tmm3, %%tmm7          \n"

    "   cmpq        $0, %%r9                        \n"
    "   jg          BEGIN_K_16x32                   \n"

    "Store16x32:                                    \n"

    "   Transpose_Store_16x32                       \n"
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

void mKernel_32x16x32(float *C, uint16_t* A, uint16_t* B,
                    long M, long K, long stride_N,  long stride_K, long kk)
{

    asm volatile(

    ".macro Transpose_Store_N16                     \n"

    "   mov          $0, %%rax                      \n"
//----------------------------------------------------------
    "   tilestored   %%tmm5, (%%rsp, %%rdx, 1)      \n"
    "   tilezero    %%tmm5                          \n"

    "0:                                             \n"
//------------------------------------------------------

    "   vmovups      (%%rsp), %%zmm0                \n"
    "   vmovups      64(%%rsp), %%zmm1              \n"
    "   vmovups      128(%%rsp), %%zmm2             \n"
    "   vmovups      192(%%rsp), %%zmm3             \n"
    "   vmovups      256(%%rsp), %%zmm4             \n"
    "   vmovups      320(%%rsp), %%zmm5             \n"
    "   vmovups      384(%%rsp), %%zmm6             \n"
    "   vmovups      448(%%rsp), %%zmm7             \n"
    "   vmovups      512(%%rsp), %%zmm8             \n"
    "   vmovups      576(%%rsp), %%zmm9             \n"
    "   vmovups      640(%%rsp), %%zmm10            \n"
    "   vmovups      704(%%rsp), %%zmm11            \n"
    "   vmovups      768(%%rsp), %%zmm12            \n"
    "   vmovups      832(%%rsp), %%zmm13            \n"
    "   vmovups      896(%%rsp), %%zmm14            \n"
    "   vmovups      960(%%rsp), %%zmm15            \n"

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


    "   cmp             $0, %%r15                    \n"
    "   je              5f                           \n"
    "   vmovups         (%%rcx), %%zmm18             \n"
    "   vmovups         (%%rcx, %%rsi, 4), %%zmm19   \n"
    "   vaddps          %%zmm18, %%zmm24, %%zmm24    \n"
    "   vaddps          %%zmm19, %%zmm25, %%zmm25    \n"
    "   vmovups         (%%rcx, %%rsi, 8), %%zmm18   \n"
    "   leaq            (%%rcx, %%rsi, 8), %%r14     \n"
    "   vmovups         (%%r14, %%rsi, 4), %%zmm19   \n"
    "   vaddps          %%zmm18, %%zmm26, %%zmm26    \n"
    "   vaddps          %%zmm19, %%zmm27, %%zmm27    \n"

    "5:                                              \n"
    "   vmovups         %%zmm24, (%%rcx)             \n"
    "   vmovups         %%zmm25, (%%rcx, %%rsi, 4)   \n"
    "   vmovups         %%zmm26, (%%rcx, %%rsi, 8)   \n"
    "   leaq            (%%rcx, %%rsi, 8), %%r14     \n"
    "   vmovups         %%zmm27, (%%r14, %%rsi, 4)   \n"

//----------------------------------------------------

    "   vinsertf32x4    $1, %%xmm29, %%zmm28, %%zmm22\n"
    "   vinsertf32x4    $1, %%xmm31, %%zmm30, %%zmm23\n"
    "   vinsertf32x8    $1, %%ymm23, %%zmm22, %%zmm24\n"

    "   vextractf32x4   $1, %%zmm28, %%xmm22         \n"
    "   vinsertf32x4    $0, %%xmm22, %%zmm29, %%zmm22\n"
    "   vextractf32x4   $1, %%zmm30, %%xmm23         \n"
    "   vinsertf32x4    $0, %%xmm23, %%zmm31, %%zmm23\n"
    "   vinsertf32x8    $1, %%ymm23, %%zmm22, %%zmm25\n"

    "   vextractf32x4   $2, %%zmm28, %%xmm22         \n"
    "   vextractf32x4   $2, %%zmm29, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm26\n"
    "   vextractf32x4   $2, %%zmm30, %%xmm22         \n"
    "   vextractf32x4   $2, %%zmm31, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm22\n"
    "   vinsertf32x8    $1, %%ymm22, %%zmm26, %%zmm26\n"

    "   vextractf32x4   $3, %%zmm28, %%xmm22         \n"
    "   vextractf32x4   $3, %%zmm29, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm27\n"
    "   vextractf32x4   $3, %%zmm30, %%xmm22         \n"
    "   vextractf32x4   $3, %%zmm31, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm22\n"
    "   vinsertf32x8    $1, %%ymm22, %%zmm27, %%zmm27\n"


    "   cmp             $0, %%r15                    \n"
    "   je              6f                           \n"
    "   vmovups         (%%rcx, %%rsi), %%zmm18      \n"
    "   leaq            (%%rcx, %%rsi), %%r14        \n"
    "   vmovups         (%%r14, %%rsi, 4), %%zmm19   \n"
    "   vaddps          %%zmm18, %%zmm24, %%zmm24    \n"
    "   vaddps          %%zmm19, %%zmm25, %%zmm25    \n"
    "   vmovups         (%%r14, %%rsi, 8), %%zmm18   \n"
    "   leaq            (%%r14, %%rsi, 8), %%r14     \n"
    "   vmovups         (%%r14, %%rsi, 4), %%zmm19   \n"
    "   vaddps          %%zmm18, %%zmm26, %%zmm26    \n"
    "   vaddps          %%zmm19, %%zmm27, %%zmm27    \n"

    "6:                                              \n"

    "   vmovups         %%zmm24, (%%rcx, %%rsi)      \n"
    "   leaq            (%%rcx, %%rsi), %%r14        \n"
    "   vmovups         %%zmm25, (%%r14, %%rsi, 4)   \n"
    "   vmovups         %%zmm26, (%%r14, %%rsi, 8)   \n"
    "   leaq            (%%r14, %%rsi, 8), %%r14     \n"
    "   vmovups         %%zmm27, (%%r14, %%rsi, 4)   \n"


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


    "   cmp             $0, %%r15                    \n"
    "   je              7f                           \n"
    "   vmovups         (%%rcx, %%rsi, 2), %%zmm18   \n"
    "   leaq            (%%rcx, %%rsi, 2), %%r14     \n"
    "   vmovups         (%%r14, %%rsi, 4), %%zmm19   \n"
    "   vaddps          %%zmm18, %%zmm24, %%zmm24    \n"
    "   vaddps          %%zmm19, %%zmm25, %%zmm25    \n"
    "   vmovups         (%%r14, %%rsi, 8), %%zmm18   \n"
    "   leaq            (%%r14, %%rsi, 8), %%r14     \n"
    "   vmovups         (%%r14, %%rsi, 4), %%zmm19   \n"
    "   vaddps          %%zmm18, %%zmm26, %%zmm26    \n"
    "   vaddps          %%zmm19, %%zmm27, %%zmm27    \n"

    "7:                                              \n"

    "   vmovups         %%zmm24, (%%rcx, %%rsi, 2)   \n"
    "   leaq            (%%rcx, %%rsi, 2), %%r14     \n"
    "   vmovups         %%zmm25, (%%r14, %%rsi, 4)   \n"
    "   vmovups         %%zmm26, (%%r14, %%rsi, 8)   \n"
    "   leaq            (%%r14, %%rsi, 8), %%r14     \n"
    "   vmovups         %%zmm27, (%%r14, %%rsi, 4)   \n"

//----------------------------------------------------

    "   vinsertf32x4    $1, %%xmm29, %%zmm28, %%zmm22\n"
    "   vinsertf32x4    $1, %%xmm31, %%zmm30, %%zmm23\n"
    "   vinsertf32x8    $1, %%ymm23, %%zmm22, %%zmm24\n"

    "   vextractf32x4   $1, %%zmm28, %%xmm22         \n"
    "   vinsertf32x4    $0, %%xmm22, %%zmm29, %%zmm22\n"
    "   vextractf32x4   $1, %%zmm30, %%xmm23         \n"
    "   vinsertf32x4    $0, %%xmm23, %%zmm31, %%zmm23\n"
    "   vinsertf32x8    $1, %%ymm23, %%zmm22, %%zmm25\n"

    "   vextractf32x4   $2, %%zmm28, %%xmm22         \n"
    "   vextractf32x4   $2, %%zmm29, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm26\n"
    "   vextractf32x4   $2, %%zmm30, %%xmm22         \n"
    "   vextractf32x4   $2, %%zmm31, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm22\n"
    "   vinsertf32x8    $1, %%ymm22, %%zmm26, %%zmm26\n"

    "   vextractf32x4   $3, %%zmm28, %%xmm22         \n"
    "   vextractf32x4   $3, %%zmm29, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm27\n"
    "   vextractf32x4   $3, %%zmm30, %%xmm22         \n"
    "   vextractf32x4   $3, %%zmm31, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm22\n"
    "   vinsertf32x8    $1, %%ymm22, %%zmm27, %%zmm27\n"


    "   cmp             $0, %%r15                    \n"
    "   je              8f                           \n"

    "   leaq            (%%rcx, %%rsi, 2), %%r14     \n"
    "   leaq            (%%r14, %%rsi), %%r14        \n"
    "   vmovups         (%%r14), %%zmm18             \n"
    "   vmovups         (%%r14, %%rsi, 4), %%zmm19   \n"
    "   vaddps          %%zmm18, %%zmm24, %%zmm24    \n"
    "   vaddps          %%zmm19, %%zmm25, %%zmm25    \n"
    "   vmovups         (%%r14, %%rsi, 8), %%zmm18   \n"
    "   leaq            (%%r14, %%rsi, 8), %%r14     \n"
    "   vmovups         (%%r14, %%rsi, 4), %%zmm19   \n"
    "   vaddps          %%zmm18, %%zmm26, %%zmm26    \n"
    "   vaddps          %%zmm19, %%zmm27, %%zmm27    \n"

    "8:                                              \n"

    "   leaq            (%%rcx, %%rsi, 2), %%r14     \n"
    "   leaq            (%%r14, %%rsi), %%r14        \n"
    "   vmovups         %%zmm24, (%%r14)             \n"
    "   vmovups         %%zmm25, (%%r14, %%rsi, 4)   \n"
    "   vmovups         %%zmm26, (%%r14, %%rsi, 8)   \n"
    "   leaq            (%%r14, %%rsi, 8), %%r14     \n"
    "   vmovups         %%zmm27, (%%r14, %%rsi, 4)   \n"


    "   addq         $1, %%rax                      \n"
    "   cmp          $2, %%rax                      \n"
    "   je           4f                             \n"


    "1:                                             \n"
    "   tilestored   %%tmm7, (%%rsp, %%rdx, 1)      \n"
    "   tilezero    %%tmm7                          \n"
    "   kmovq       %%k2, %%r14                     \n"
    "   leaq        (%%rcx, %%r14), %%rcx           \n"
    "   jmp         0b                              \n"


    "4:                                             \n"
    "   kmovq       %%k2, %%r14                     \n"
    "   leaq        (%%rcx, %%r14), %%rcx           \n"

    ".endm                                          \n"


    "BEGIN_KERNEKL_32x16x32:                        \n"

    "   mov       %[A], %%r13                       \n"
    "   mov       %[B], %%r10                       \n"
    "   mov       %[C], %%rcx                       \n"
    "   mov       %[M], %%r8                        \n"
    "   mov       %[K], %%r12                       \n"
    "   kmovq     %%r12, %%k1                       \n"
    "   mov       %[stride_K], %%r11                \n"
    "   mov       %[stride_N], %%rsi                \n"
    "   mov       %[kk], %%r15                      \n"
    "   shl       $5, %%r12                         \n"

    "   mov       $64, %%rdx                        \n"
//栈空间
    "   mov      %%rsp, %%r9                        \n"
    "   and      $63, %%r9                          \n"
    "   addq     $1024, %%r9                        \n"
    "   subq     %%r9, %%rsp                        \n"
    "   kmovq    %%r9, %%k3                         \n"


    "   tilezero    %%tmm5                          \n"   
    "   tilezero    %%tmm7                          \n"

    "   kmovq    %%rsi, %%k2                        \n"
    "   kshiftlq $4, %%k2, %%k2                     \n"
    

    "BEGIN_M_N16:                                   \n"
    
    "   mov         %%r10, %%rbx                    \n" //B*

    "   mov         %%r13, %%rax                    \n" //A*
    "   leaq        (%%rax, %%r12), %%rdi           \n"

    "   kmovq       %%k1, %%r9                      \n"    //K
    
//-----------------------------------------------------------
    "BEGIN_K_N16:                                   \n"

    "   subq        $32, %%r9                       \n"
    "   tileloadd   (%%rax, %%rdx, 1), %%tmm0     \n"
    "   addq        $1024, %%rax                    \n"
    "   tileloadd   (%%rbx, %%r11, 1), %%tmm3       \n"
    "   addq        $64, %%rbx                      \n"
    "   tdpbf16ps   %%tmm0, %%tmm3, %%tmm5          \n"
    "   tileloadd   (%%rdi, %%rdx, 1), %%tmm1     \n"
    "   addq        $1024, %%rdi                    \n"
    "   tdpbf16ps   %%tmm1, %%tmm3, %%tmm7          \n"

    "   cmpq        $0, %%r9                        \n"
    "   jg          BEGIN_K_N16                     \n"

    "Store32x16:                                    \n"

    "   Transpose_Store_N16                         \n"
//------------------------------------------------------

    "   leaq        (%%r13, %%r12, 2), %%r13        \n"

    
    "   subq        $32, %%r8                       \n"
    "   cmp         $0, %%r8                        \n"
    "   jg          BEGIN_M_N16                     \n"

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


void mKernel_16x16x32(float *C, uint16_t* A, uint16_t* B,
                    uint32_t M, long K, long stride_N,  long stride_K, long kk)
{

    asm volatile(

    ".macro Transpose_Store_16x16                   \n"

    "   tilestored   %%tmm7, (%%rsp, %%rdx, 4)      \n"

    "0:                                             \n"
//------------------------------------------------------

    "   vmovups      (%%rsp), %%zmm0%{%%k2}         \n"
    "   vmovups      (%%rsp, %%rdx, 4), %%zmm1%{%%k2} \n"
    "   vmovups      (%%rsp, %%rdx, 8), %%zmm2%{%%k2} \n"
    "   leaq         (%%rsp, %%rdx, 8), %%r9     \n"
    "   vmovups      (%%r9, %%rdx, 4), %%zmm3%{%%k2} \n"
    "   vmovups      (%%r9, %%rdx, 8), %%zmm4%{%%k2} \n"
    "   leaq         (%%r9, %%rdx, 8), %%r9     \n"
    "   vmovups      (%%r9, %%rdx, 4), %%zmm5%{%%k2} \n"
    "   vmovups      (%%r9, %%rdx, 8), %%zmm6%{%%k2} \n"
    "   leaq         (%%r9, %%rdx, 8), %%r9     \n"
    "   vmovups      (%%r9, %%rdx, 4), %%zmm7%{%%k2}      \n"
    "   vmovups      (%%r9, %%rdx, 8), %%zmm8%{%%k2}      \n"
    "   leaq         (%%r9, %%rdx, 8), %%r9     \n"
    "   vmovups      (%%r9, %%rdx, 4), %%zmm9%{%%k2}      \n"
    "   vmovups      (%%r9, %%rdx, 8), %%zmm10%{%%k2}     \n"
    "   leaq         (%%r9, %%rdx, 8), %%r9     \n"
    "   vmovups      (%%r9, %%rdx, 4), %%zmm11%{%%k2}     \n"
    "   vmovups      (%%r9, %%rdx, 8), %%zmm12%{%%k2}     \n"
    "   leaq         (%%r9, %%rdx, 8), %%r9     \n"
    "   vmovups      (%%r9, %%rdx, 4), %%zmm13%{%%k2}     \n"
    "   vmovups      (%%r9, %%rdx, 8), %%zmm14%{%%k2}     \n"
    "   leaq         (%%r9, %%rdx, 8), %%r9     \n"
    "   vmovups      (%%r9, %%rdx, 4), %%zmm15%{%%k2}     \n"

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


    "   cmp             $0, %%r15                    \n"
    "   je              5f                           \n"
    "   vmovups         (%%rcx), %%zmm18             \n"
    "   vaddps          %%zmm18, %%zmm24, %%zmm24    \n"
    "   cmp             $5, %%rdx                    \n"
    "   jl              5f                           \n"
    "   vmovups         (%%rcx, %%rsi, 4), %%zmm19   \n"
    "   vaddps          %%zmm19, %%zmm25, %%zmm25    \n"
    "   cmp             $9, %%rdx                    \n"
    "   jl              5f                           \n"
    "   vmovups         (%%rcx, %%rsi, 8), %%zmm18   \n"
    "   vaddps          %%zmm18, %%zmm26, %%zmm26    \n"
    "   cmp             $13, %%rdx                   \n"
    "   jl              5f                           \n"
    "   leaq            (%%rcx, %%rsi, 8), %%r14     \n"
    "   vmovups         (%%r14, %%rsi, 4), %%zmm19   \n"
    "   vaddps          %%zmm19, %%zmm27, %%zmm27    \n"

    "5:                                              \n"
    "   vmovups         %%zmm24, (%%rcx)             \n"
    "   cmp             $5, %%rdx                    \n"
    "   jl              Tanspose_2_%=                \n"
    "   vmovups         %%zmm25, (%%rcx, %%rsi, 4)   \n"
    "   cmp             $9, %%rdx                    \n"
    "   jl              Tanspose_2_%=                \n"
    "   vmovups         %%zmm26, (%%rcx, %%rsi, 8)   \n"
    "   cmp             $13, %%rdx                   \n"
    "   jl              Tanspose_2_%=                \n"
    "   leaq            (%%rcx, %%rsi, 8), %%r14     \n"
    "   vmovups         %%zmm27, (%%r14, %%rsi, 4)   \n"


    "Tanspose_2_%=:                                  \n"
//----------------------------------------------------

    "   cmp             $2, %%rdx                   \n"
    "   jl              1f                          \n"


    "   vinsertf32x4    $1, %%xmm29, %%zmm28, %%zmm22\n"
    "   vinsertf32x4    $1, %%xmm31, %%zmm30, %%zmm23\n"
    "   vinsertf32x8    $1, %%ymm23, %%zmm22, %%zmm24\n"

    "   vextractf32x4   $1, %%zmm28, %%xmm22         \n"
    "   vinsertf32x4    $0, %%xmm22, %%zmm29, %%zmm22\n"
    "   vextractf32x4   $1, %%zmm30, %%xmm23         \n"
    "   vinsertf32x4    $0, %%xmm23, %%zmm31, %%zmm23\n"
    "   vinsertf32x8    $1, %%ymm23, %%zmm22, %%zmm25\n"

    "   vextractf32x4   $2, %%zmm28, %%xmm22         \n"
    "   vextractf32x4   $2, %%zmm29, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm26\n"
    "   vextractf32x4   $2, %%zmm30, %%xmm22         \n"
    "   vextractf32x4   $2, %%zmm31, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm22\n"
    "   vinsertf32x8    $1, %%ymm22, %%zmm26, %%zmm26\n"

    "   vextractf32x4   $3, %%zmm28, %%xmm22         \n"
    "   vextractf32x4   $3, %%zmm29, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm27\n"
    "   vextractf32x4   $3, %%zmm30, %%xmm22         \n"
    "   vextractf32x4   $3, %%zmm31, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm22\n"
    "   vinsertf32x8    $1, %%ymm22, %%zmm27, %%zmm27\n"


    "   cmp             $0, %%r15                    \n"
    "   je              6f                           \n"
    "   vmovups         (%%rcx, %%rsi), %%zmm18      \n"
    "   vaddps          %%zmm18, %%zmm24, %%zmm24    \n"
    "   cmp             $6, %%rdx                    \n"
    "   jl              6f                           \n"
    "   leaq            (%%rcx, %%rsi), %%r14        \n"
    "   vmovups         (%%r14, %%rsi, 4), %%zmm19   \n"
    "   vaddps          %%zmm19, %%zmm25, %%zmm25    \n"
    "   cmp             $10, %%rdx                   \n"
    "   jl              6f                           \n"
    "   vmovups         (%%r14, %%rsi, 8), %%zmm18   \n"
    "   vaddps          %%zmm18, %%zmm26, %%zmm26    \n"
    "   cmp             $14, %%rdx                   \n"
    "   jl              6f                           \n"
    "   leaq            (%%r14, %%rsi, 8), %%r14     \n"
    "   vmovups         (%%r14, %%rsi, 4), %%zmm19   \n"
    "   vaddps          %%zmm19, %%zmm27, %%zmm27    \n"

    "6:                                              \n"

    "   vmovups         %%zmm24, (%%rcx, %%rsi)      \n"
    "   cmp             $6, %%rdx                    \n"
    "   jl              Tanspose_3_%=                \n"
    "   leaq            (%%rcx, %%rsi), %%r14        \n"
    "   vmovups         %%zmm25, (%%r14, %%rsi, 4)   \n"
    "   cmp             $10, %%rdx                   \n"
    "   jl              Tanspose_3_%=                \n"
    "   vmovups         %%zmm26, (%%r14, %%rsi, 8)   \n"
    "   cmp             $14, %%rdx                   \n"
    "   jl              Tanspose_3_%=                \n"
    "   leaq            (%%r14, %%rsi, 8), %%r14     \n"
    "   vmovups         %%zmm27, (%%r14, %%rsi, 4)   \n"

    "Tanspose_3_%=:                                  \n"
//----------------------------------------------------

    "   cmp             $3, %%rdx                   \n"
    "   jl              1f                          \n"

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


    "   cmp             $0, %%r15                    \n"
    "   je              7f                           \n"
    "   vmovups         (%%rcx, %%rsi, 2), %%zmm18   \n"
    "   vaddps          %%zmm18, %%zmm24, %%zmm24    \n"
    "   cmp             $7, %%rdx                    \n"
    "   jl              7f                           \n"
    "   leaq            (%%rcx, %%rsi, 2), %%r14     \n"
    "   vmovups         (%%r14, %%rsi, 4), %%zmm19   \n"
    "   vaddps          %%zmm19, %%zmm25, %%zmm25    \n"
    "   cmp             $11, %%rdx                   \n"
    "   jl              7f                           \n"
    "   vmovups         (%%r14, %%rsi, 8), %%zmm18   \n"
    "   vaddps          %%zmm18, %%zmm26, %%zmm26    \n"
    "   cmp             $15, %%rdx                   \n"
    "   jl              7f                           \n"
    "   leaq            (%%r14, %%rsi, 8), %%r14     \n"
    "   vmovups         (%%r14, %%rsi, 4), %%zmm19   \n"
    "   vaddps          %%zmm19, %%zmm27, %%zmm27    \n"

    "7:                                              \n"

    "   vmovups         %%zmm24, (%%rcx, %%rsi, 2)   \n"
    "   cmp             $7, %%rdx                    \n"
    "   jl              Tanspose_4_%=                \n"
    "   leaq            (%%rcx, %%rsi, 2), %%r14     \n"
    "   vmovups         %%zmm25, (%%r14, %%rsi, 4)   \n"
    "   cmp             $11, %%rdx                   \n"
    "   jl              Tanspose_4_%=                \n"
    "   vmovups         %%zmm26, (%%r14, %%rsi, 8)   \n"
    "   cmp             $15, %%rdx                    \n"
    "   jl              Tanspose_4_%=                \n"
    "   leaq            (%%r14, %%rsi, 8), %%r14     \n"
    "   vmovups         %%zmm27, (%%r14, %%rsi, 4)   \n"

    "Tanspose_4_%=:                                  \n"
//----------------------------------------------------

    "   cmp             $4, %%rdx                   \n"
    "   jl              1f                          \n"


    "   vinsertf32x4    $1, %%xmm29, %%zmm28, %%zmm22\n"
    "   vinsertf32x4    $1, %%xmm31, %%zmm30, %%zmm23\n"
    "   vinsertf32x8    $1, %%ymm23, %%zmm22, %%zmm24\n"

    "   vextractf32x4   $1, %%zmm28, %%xmm22         \n"
    "   vinsertf32x4    $0, %%xmm22, %%zmm29, %%zmm22\n"
    "   vextractf32x4   $1, %%zmm30, %%xmm23         \n"
    "   vinsertf32x4    $0, %%xmm23, %%zmm31, %%zmm23\n"
    "   vinsertf32x8    $1, %%ymm23, %%zmm22, %%zmm25\n"

    "   vextractf32x4   $2, %%zmm28, %%xmm22         \n"
    "   vextractf32x4   $2, %%zmm29, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm26\n"
    "   vextractf32x4   $2, %%zmm30, %%xmm22         \n"
    "   vextractf32x4   $2, %%zmm31, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm22\n"
    "   vinsertf32x8    $1, %%ymm22, %%zmm26, %%zmm26\n"

    "   vextractf32x4   $3, %%zmm28, %%xmm22         \n"
    "   vextractf32x4   $3, %%zmm29, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm27\n"
    "   vextractf32x4   $3, %%zmm30, %%xmm22         \n"
    "   vextractf32x4   $3, %%zmm31, %%xmm23         \n"
    "   vinsertf32x4    $1, %%xmm23, %%zmm22, %%zmm22\n"
    "   vinsertf32x8    $1, %%ymm22, %%zmm27, %%zmm27\n"


    "   cmp             $0, %%r15                    \n"
    "   je              8f                           \n"

    "   leaq            (%%rcx, %%rsi, 2), %%r14     \n"
    "   leaq            (%%r14, %%rsi), %%r14        \n"
    "   vmovups         (%%r14), %%zmm18             \n"
    "   vaddps          %%zmm18, %%zmm24, %%zmm24    \n"
    "   cmp             $8, %%rdx                    \n"
    "   jl              8f                           \n"
    "   vmovups         (%%r14, %%rsi, 4), %%zmm19   \n"
    "   vaddps          %%zmm19, %%zmm25, %%zmm25    \n"
    "   cmp             $12, %%rdx                   \n"
    "   jl              8f                           \n"
    "   vmovups         (%%r14, %%rsi, 8), %%zmm18   \n"
    "   vaddps          %%zmm18, %%zmm26, %%zmm26    \n"
    "   cmp             $16, %%rdx                   \n"
    "   jl              8f                           \n"
    "   leaq            (%%r14, %%rsi, 8), %%r14     \n"
    "   vmovups         (%%r14, %%rsi, 4), %%zmm19   \n"
    "   vaddps          %%zmm19, %%zmm27, %%zmm27    \n"

    "8:                                              \n"

    "   leaq            (%%rcx, %%rsi, 2), %%r14     \n"
    "   leaq            (%%r14, %%rsi), %%r14        \n"
    "   vmovups         %%zmm24, (%%r14)             \n"
    "   cmp             $8, %%rdx                    \n"
    "   jl              1f                           \n"
    "   vmovups         %%zmm25, (%%r14, %%rsi, 4)   \n"
    "   cmp             $12, %%rdx                   \n"
    "   jl              1f                           \n"
    "   vmovups         %%zmm26, (%%r14, %%rsi, 8)   \n"
    "   cmp             $16, %%rdx                   \n"
    "   jl              1f                           \n"
    "   leaq            (%%r14, %%rsi, 8), %%r14     \n"
    "   vmovups         %%zmm27, (%%r14, %%rsi, 4)   \n"

    "1:                                             \n"

    ".endm                                          \n"


    "BEGIN_KERNEKL_16x16x32:                        \n"

    "   mov       %[A], %%rax                       \n"
    "   mov       %[B], %%rbx                       \n"
    "   mov       %[C], %%rcx                       \n"
    "   mov       %[K], %%r12                       \n"
    "   kmovq     %%r12, %%k1                       \n"
    "   mov       %[stride_K], %%r11                \n"
    "   mov       %[stride_N], %%rsi                \n"
    "   mov       %[kk], %%r15                      \n"
    "   shl       $5, %%r12                         \n"

    "   movl      %[M], %%edx                       \n"
    "   mov       %%rdx, %%r8                      \n"
    "   shl       $6, %%r8                         \n"


    "   mov       $1, %%r14                         \n"
    "   shlx      %%rdx, %%r14, %%r14               \n"
    "   subq      $1, %%r14                         \n"
    "   kmovq     %%r14, %%k2                       \n"
//栈空间
    "   mov      %%rsp, %%r9                        \n"
    "   and      $63, %%r9                          \n"
    "   addq     $1024, %%r9                        \n"
    "   subq     %%r9, %%rsp                        \n"
    "   kmovq    %%r9, %%k3                         \n"


    "   tilezero    %%tmm7                          \n"   

    "BEGIN_M_16x16:                                 \n"
    
    "   kmovq       %%k1, %%r9                      \n"    //K
    
//-----------------------------------------------------------
    "BEGIN_K_16x16:                                 \n"

    "   subq        $32, %%r9                       \n"
    "   tileloadd   (%%rax, %%rdx, 4), %%tmm1     \n"
    "   addq        %%r8, %%rax                    \n"
    "   tileloadd   (%%rbx, %%r11, 1), %%tmm3       \n"
    "   addq        $64, %%rbx                      \n"
    "   tdpbf16ps   %%tmm1, %%tmm3, %%tmm7          \n"
    "   cmpq        $0, %%r9                        \n"
    "   jg          BEGIN_K_16x16                   \n"

    "Store16x16:                                    \n"

    "   Transpose_Store_16x16                       \n"


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
     "r13", "r14", "r15","k1","k3","zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
     "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
     "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
     "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
     "zmm30", "zmm31", "memory"
    );

}