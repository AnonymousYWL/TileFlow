void mKernel_32x32x32(int *C, int8_t* A, uint8_t* B, uint8_t* Bc,
                    long M, long K, long stride_N,  long stride_K, long kk,
                    uint8_t *vpermt2d_control, long tag)
{

    asm volatile(


    ".macro Interleave_Store                        \n"

//----------------------------------------------------------
    "   tilestored   %%tmm4, (%%rcx, %%rsi, 4)      \n"
    
    "   tilezero    %%tmm4                          \n"
    "   tilestored   %%tmm5, 64(%%rcx, %%rsi, 4)    \n"
    "   tilezero    %%tmm5                          \n"

    "   kmovq       %%k2, %%r14                     \n"
    "   addq        %%r14, %%rcx                    \n"

    "   tilestored   %%tmm6, (%%rcx, %%rsi, 4)      \n"
    "   tilezero    %%tmm6                          \n"
    "   tilestored   %%tmm7, 64(%%rcx, %%rsi, 4)    \n"
    "   tilezero    %%tmm7                          \n"

    "   addq        %%r14, %%rcx                    \n"

    ".endm                                          \n"



    ".macro PACK_FMA                                \n"

    "   mov         $16, %%r14                      \n"

    "   tileloaddt1 (%%rax, %%r11, 1), %%tmm0       \n"
    "   addq        $64, %%rax                      \n"
    "   tileloaddt1   (%%rdi, %%r11, 1), %%tmm1     \n"
    "   addq        $64, %%rdi                      \n"

//-----------------------------------------------------
    "BEGIN_PACK_K16:                                \n"

    "   vmovdqu8   (%%rbx), %%zmm0                  \n"
    "   vmovdqu8   64(%%rbx), %%zmm4                \n"
    "   vmovdqu8   (%%rbx, %%rsi, 1), %%zmm1        \n"
    "   vmovdqu8   64(%%rbx, %%rsi, 1), %%zmm5      \n"
    "   vmovdqu8   (%%rbx, %%rsi, 2), %%zmm2        \n"
    "   vmovdqu8   64(%%rbx, %%rsi, 2), %%zmm6      \n"
    "   leaq        (%%rbx, %%rsi, 2), %%rbx        \n"
    "   vmovdqu8   (%%rbx, %%rsi, 1), %%zmm3        \n"
    "   vmovdqu8   64(%%rbx, %%rsi, 1), %%zmm7      \n"
    "   leaq        (%%rbx, %%rsi, 2), %%rbx        \n"

//--------------------------------------------------------
    "   vextractf64x4   $1, %%zmm0, %%ymm8          \n"
    "   vextractf64x4   $1, %%zmm1, %%ymm9          \n"
    "   vextractf64x4   $1, %%zmm2, %%ymm10         \n"
    "   vextractf64x4   $1, %%zmm3, %%ymm11         \n"

    "   vpermt2b    %%zmm1, %%zmm26, %%zmm0         \n"
    "   vpermt2b    %%zmm3, %%zmm26, %%zmm2         \n"

    "   vextractf64x4   $1, %%zmm0, %%ymm12         \n"
    "   vextractf64x4   $1, %%zmm2, %%ymm13         \n"

    "   vpermt2w    %%zmm2, %%zmm27, %%zmm0         \n"
    "   vpermt2w    %%zmm13, %%zmm27, %%zmm12       \n"

    "   vmovdqu8    %%zmm0, (%%r13)                 \n"
    "   vmovdqu8    %%zmm12, 1024(%%r13)            \n"
//--------------------------------------------------------

    "   vpermt2b    %%zmm9, %%zmm26, %%zmm8         \n"
    "   vpermt2b    %%zmm11, %%zmm26, %%zmm10       \n"

    "   vextractf64x4   $1, %%zmm8, %%ymm12         \n"
    "   vextractf64x4   $1, %%zmm10, %%ymm13        \n"

    "   vpermt2w    %%zmm10, %%zmm27, %%zmm8        \n"
    "   vpermt2w    %%zmm13, %%zmm27, %%zmm12       \n"

    "   vmovdqu8    %%zmm8, (%%r13,%%r12)           \n"
    "   vmovdqu8    %%zmm12, 1024(%%r13,%%r12)      \n"
//--------------------------------------------------------

    "   vextractf64x4   $1, %%zmm4, %%ymm8          \n"
    "   vextractf64x4   $1, %%zmm5, %%ymm9          \n"
    "   vextractf64x4   $1, %%zmm6, %%ymm10         \n"
    "   vextractf64x4   $1, %%zmm7, %%ymm11         \n"

    "   vpermt2b    %%zmm5, %%zmm26, %%zmm4         \n"
    "   vpermt2b    %%zmm7, %%zmm26, %%zmm6         \n"

    "   vextractf64x4   $1, %%zmm4, %%ymm12         \n"
    "   vextractf64x4   $1, %%zmm6, %%ymm13         \n"

    "   vpermt2w    %%zmm6, %%zmm27, %%zmm4         \n"
    "   vpermt2w    %%zmm13, %%zmm27, %%zmm12       \n"

    "   vmovdqu8    %%zmm4, (%%r13,%%r12,2)         \n"
    "   vmovdqu8    %%zmm12, 1024(%%r13,%%r12,2)    \n"
//--------------------------------------------------------

    "   vpermt2b    %%zmm9, %%zmm26, %%zmm8         \n"
    "   vpermt2b    %%zmm11, %%zmm26, %%zmm10       \n"

    "   vextractf64x4   $1, %%zmm8, %%ymm12         \n"
    "   vextractf64x4   $1, %%zmm10, %%ymm13        \n"

    "   vpermt2w    %%zmm10, %%zmm27, %%zmm8        \n"
    "   vpermt2w    %%zmm13, %%zmm27, %%zmm12       \n"
    "   addq         %%r12, %%r13                   \n"
    "   vmovdqu8    %%zmm8, (%%r13,%%r12,2)         \n"
    "   vmovdqu8    %%zmm12, 1024(%%r13,%%r12,2)    \n"
    "   subq         %%r12, %%r13                   \n"
//--------------------------------------------------------

    "   addq        $64, %%r13                      \n"

    "   subq        $1, %%r14                       \n"
    "   cmp         $0, %%r14                       \n"
    "   jg          BEGIN_PACK_K16                  \n"


    "   tileloadd   -1024(%%r13, %%rdx, 1), %%tmm2  \n"
    "   tdpbsud   %%tmm2, %%tmm0, %%tmm4            \n"
    "   tdpbsud   %%tmm2, %%tmm1, %%tmm6            \n"
    "   tileloadd   (%%r13, %%rdx, 1), %%tmm3       \n"
    "   tdpbsud   %%tmm3, %%tmm0, %%tmm5            \n"
    "   tdpbsud   %%tmm3, %%tmm1, %%tmm7            \n"
    "   addq        $1024, %%r13                    \n"
    
    ".endm                                          \n"



    "BEGIN_KERNEKL_32x32x32:                        \n"

    "   mov       %[A], %%rbp                       \n"
    "   mov       %[B], %%rbx                       \n"
    "   mov       %[C], %%rcx                       \n"
    "   mov       %[M], %%r8                        \n"
    "   mov       %[K], %%r12                       \n"
    "   kmovq     %%r12, %%k1                       \n"
    "   mov       %[stride_K], %%r11                \n"
    "   mov       %[stride_N], %%rsi                \n"
    "   mov       %[vpermt2d_control], %%r14        \n"
    "   mov       %[kk], %%r15                      \n"
    "   mov       %[Bc], %%r10                      \n"
    "   shl       $5, %%r12                         \n"

    "   vmovdqu8   (%%r14), %%zmm26                 \n"
    "   vpmovzxbw   64(%%r14), %%zmm27              \n"

    "   mov       %[tag], %%r14                     \n"

    "   mov       $64, %%rdx                        \n"

    "   tilezero    %%tmm4                          \n"   
    "   tilezero    %%tmm5                          \n"
    "   tilezero    %%tmm6                          \n"
    "   tilezero    %%tmm7                          \n"

    "   kmovq    %%rsi, %%k2                        \n"
    "   kshiftlq $6, %%k2, %%k2                     \n"

    "   cmp     $0, %%r14                           \n"
    "   jg      BEGIN_M                             \n"

    "BEGIN_PACK:                                    \n"

    "   mov         %%r10, %%r13                    \n"
    "   mov         %%rbp, %%rax                    \n"
    "   leaq        (%%rax, %%r11, 8), %%rdi        \n"
    "   leaq        (%%rdi, %%r11, 8), %%rdi        \n"
    
    "   kmovq       %%k1, %%r9                      \n"    //K

    "   cmp         $0, %%r15                       \n"
    "   je          BEGIN_PACK_K                    \n"
    "   tileloadd   (%%rcx, %%rsi, 4), %%tmm4       \n"
    "   tileloadd   64(%%rcx, %%rsi, 4), %%tmm5     \n"
    "   kmovq       %%k2, %%r14                     \n"
    "   addq        %%r14, %%rcx                    \n"
    "   tileloadd   (%%rcx, %%rsi, 4), %%tmm6       \n"
    "   tileloadd   64(%%rcx, %%rsi, 4), %%tmm7     \n"
    "   subq        %%r14, %%rcx                    \n"

   "BEGIN_PACK_K:                                   \n"

    "   subq        $64, %%r9                       \n"
    "   PACK_FMA                                    \n"
    "   cmpq        $0, %%r9                        \n"
    "   jg          BEGIN_PACK_K                    \n"

    "   jmp         Store32x32                      \n"
    

    "BEGIN_M:                                       \n"
    
    "   mov         %%r10, %%rbx                    \n"

    "   mov         %%rbp, %%rax                    \n"
    "   leaq        (%%rax, %%r11, 8), %%rdi        \n"
    "   leaq        (%%rdi, %%r11, 8), %%rdi        \n"

    "   kmovq       %%k1, %%r9                      \n"    //K

    "   cmp         $0, %%r15                       \n"
    "   je          BEGIN_K                         \n"

    "   tileloadd   (%%rcx, %%rsi, 4), %%tmm4       \n"
    "   tileloadd   64(%%rcx, %%rsi, 4), %%tmm5     \n"
    "   kmovq       %%k2, %%r14                     \n"
    "   addq        %%r14, %%rcx                    \n"
    "   tileloadd   (%%rcx, %%rsi, 4), %%tmm6       \n"
    "   tileloadd   64(%%rcx, %%rsi, 4), %%tmm7     \n"
    "   subq        %%r14, %%rcx                    \n"
    
//-----------------------------------------------------------
    "BEGIN_K:                                       \n"

    "   subq        $64, %%r9                       \n"
    "   tileloaddt1   (%%rax, %%r11, 1), %%tmm0     \n"
    "   addq        $64, %%rax                      \n"
    "   tileloadd   (%%rbx, %%rdx, 1), %%tmm2       \n"
    "   tdpbsud     %%tmm2, %%tmm0, %%tmm4          \n"
    "   tileloadd   1024(%%rbx, %%rdx, 1), %%tmm3   \n"
    "   addq        $2048, %%rbx                    \n"
    "   tdpbsud     %%tmm3, %%tmm0, %%tmm5          \n"
    "   tileloaddt1   (%%rdi, %%r11, 1), %%tmm1     \n"
    "   addq        $64, %%rdi                      \n"
    "   tdpbsud     %%tmm2, %%tmm1, %%tmm6          \n"
    "   tdpbsud     %%tmm3, %%tmm1, %%tmm7          \n"

    "   cmpq        $0, %%r9                        \n"
    "   jg          BEGIN_K                         \n"

    "Store32x32:                                    \n"

    "   Interleave_Store                            \n"
//------------------------------------------------------

    "   leaq        (%%rbp, %%r11, 8), %%rbp        \n"
    "   leaq        (%%rbp, %%r11, 8), %%rbp        \n"
    "   leaq        (%%rbp, %%r11, 8), %%rbp        \n"
    "   leaq        (%%rbp, %%r11, 8), %%rbp        \n"
    
    "   subq    $32, %%r8                           \n"
    "   cmp     $0, %%r8                            \n"
    "   jg      BEGIN_M                             \n"

    
    :
    :
      [A] "m" (A),
      [B] "m" (B),
      [C] "m" (C),
      [Bc] "m" (Bc),
      [M] "m" (M),
      [K] "m" (K),
      [stride_N] "m" (stride_N),
      [stride_K] "m" (stride_K),
      [kk] "m" (kk),
      [vpermt2d_control] "m" (vpermt2d_control),
      [tag] "m" (tag)
    :"rax", "rbx", "rcx", "rdx", "rdi", "rsi","rbp","r8", "r9", "r10", "r11", "r12",
     "r13", "r14", "r15","k1","k2","zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
     "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
     "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
     "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
     "zmm30", "zmm31", "memory"
    );

}
