void mKernel_32x32x32(float *C, uint16_t* A, uint16_t* B, uint16_t* Bc,
                    long M, long K, long stride_N,  long stride_K, long kk,
                    uint8_t *vpermt2d_control, long tag)
{

    asm volatile(


    ".macro Interleave_Store                        \n"

//----------------------------------------------------------
    "   tilestored   %%tmm4, (%%rcx, %%rsi, 2)      \n"
    
    "   tilezero    %%tmm4                          \n"
    "   tilestored   %%tmm5, 64(%%rcx, %%rsi, 2)    \n"
    "   tilezero    %%tmm5                          \n"

    "   kmovq       %%k2, %%r14                     \n"
    "   addq        %%r14, %%rcx                    \n"

    "   tilestored   %%tmm6, (%%rcx, %%rsi, 2)      \n"
    "   tilezero    %%tmm6                          \n"
    "   tilestored   %%tmm7, 64(%%rcx, %%rsi, 2)    \n"
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

    "   vmovdqu16   (%%rbx), %%zmm0                 \n"
    "   vmovdqu16   64(%%rbx), %%zmm4               \n"
    "   vmovdqu16   (%%rbx, %%rsi, 1), %%zmm1       \n"
    "   vmovdqu16   64(%%rbx, %%rsi, 1), %%zmm5     \n"

    "   vmovdqu16    %%zmm0, %%zmm2                 \n"
    "   vpermt2w    %%zmm1, %%zmm26, %%zmm0         \n"
    "   vmovdqu16    %%zmm0, (%%r13)                \n"
    "   vpermt2w    %%zmm1, %%zmm27, %%zmm2         \n"
    "   vmovdqu16    %%zmm2, 1024(%%r13)            \n"

    "   vmovdqu16    %%zmm4, %%zmm2                 \n"
    "   vpermt2w    %%zmm5, %%zmm26, %%zmm4         \n"
    "   vmovdqu16    %%zmm4, (%%r13, %%r12)         \n"
    "   vpermt2w    %%zmm5, %%zmm27, %%zmm2         \n"
    "   vmovdqu16    %%zmm2, 1024(%%r13, %%r12)     \n"

    "   addq        $64, %%r13                      \n"
    "   leaq        (%%rbx, %%rsi, 2), %%rbx        \n"

    "   subq        $1, %%r14                       \n"
    "   cmp         $0, %%r14                       \n"
    "   jg          BEGIN_PACK_K16                  \n"


    "   tileloadd   -1024(%%r13, %%rdx, 1), %%tmm2  \n"
    "   tdpbf16ps   %%tmm2, %%tmm0, %%tmm4          \n"
    "   tdpbf16ps   %%tmm2, %%tmm1, %%tmm6          \n"
    "   tileloadd   (%%r13, %%rdx, 1), %%tmm3       \n"
    "   tdpbf16ps   %%tmm3, %%tmm0, %%tmm5          \n"
    "   tdpbf16ps   %%tmm3, %%tmm1, %%tmm7          \n"
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
    "   shl       $6, %%r12                         \n"

    "   mov       $0x10, %%eax                      \n" 
    "   vpbroadcastw    %%eax, %%zmm27              \n"
    "   vpmovzxbw   (%%r14), %%zmm26                \n"
    "   vpaddw    %%zmm26, %%zmm27, %%zmm27         \n"

    "   mov       %[tag], %%r14                     \n"

    "   mov       $64, %%rdx                        \n"

    "   tilezero    %%tmm4                          \n"   
    "   tilezero    %%tmm5                          \n"
    "   tilezero    %%tmm6                          \n"
    "   tilezero    %%tmm7                          \n"

    "   kmovq    %%rsi, %%k2                        \n"
    "   kshiftlq $5, %%k2, %%k2                     \n"

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
    "   tileloadd   (%%rcx, %%rsi, 2), %%tmm4       \n"
    "   tileloadd   64(%%rcx, %%rsi, 2), %%tmm5     \n"
    "   kmovq       %%k2, %%r14                     \n"
    "   addq        %%r14, %%rcx                    \n"
    "   tileloadd   (%%rcx, %%rsi, 2), %%tmm6       \n"
    "   tileloadd   64(%%rcx, %%rsi, 2), %%tmm7     \n"
    "   subq        %%r14, %%rcx                    \n"

   "BEGIN_PACK_K:                                   \n"

    "   subq        $32, %%r9                       \n"
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

    "   tileloadd   (%%rcx, %%rsi, 2), %%tmm4       \n"
    "   tileloadd   64(%%rcx, %%rsi, 2), %%tmm5     \n"
    "   kmovq       %%k2, %%r14                     \n"
    "   addq        %%r14, %%rcx                    \n"
    "   tileloadd   (%%rcx, %%rsi, 2), %%tmm6       \n"
    "   tileloadd   64(%%rcx, %%rsi, 2), %%tmm7     \n"
    "   subq        %%r14, %%rcx                    \n"
    
//-----------------------------------------------------------
    "BEGIN_K:                                       \n"

    "   subq        $32, %%r9                       \n"
    "   tileloaddt1   (%%rax, %%r11, 1), %%tmm0     \n"
    "   addq        $64, %%rax                      \n"
    "   tileloadd   (%%rbx, %%rdx, 1), %%tmm2       \n"
    "   tdpbf16ps   %%tmm2, %%tmm0, %%tmm4          \n"
    "   tileloadd   1024(%%rbx, %%rdx, 1), %%tmm3   \n"
    "   addq        $2048, %%rbx                    \n"
    "   tdpbf16ps   %%tmm3, %%tmm0, %%tmm5          \n"
    "   tileloaddt1   (%%rdi, %%r11, 1), %%tmm1     \n"
    "   addq        $64, %%rdi                      \n"
    "   tdpbf16ps   %%tmm2, %%tmm1, %%tmm6          \n"
    "   tdpbf16ps   %%tmm3, %%tmm1, %%tmm7          \n"

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
