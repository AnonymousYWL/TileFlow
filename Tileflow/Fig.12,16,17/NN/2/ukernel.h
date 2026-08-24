void pack_N(uint16_t* B, uint16_t* Bc, long stride_N, long K,
            uint32_t G, uint8_t *vpermt2d_control, uint32_t *numa_k)
{

    asm volatile(

    "   mov       %[B], %%rbx                       \n"
    "   mov       %[K], %%r9                        \n"
    "   mov       %[stride_N], %%rsi                \n"
    "   mov       %[vpermt2d_control], %%r14        \n"
    "   mov       %[Bc], %%r13                      \n"
    "   movl      %[G], %%r8d                       \n"
    "   mov       %[numa_k], %%r15                  \n"
    "   mov       %%r9, %%r12                       \n"
    "   shl       $6, %%r12                         \n"

    "   mov       $0x10, %%eax                      \n" 
    "   vpbroadcastw    %%eax, %%zmm27              \n"
    "   vpmovzxbw   (%%r14), %%zmm26                \n"
    "   vpaddw    %%zmm26, %%zmm27, %%zmm27         \n"


    "   cmp       $64, %%r8                         \n"
    "   jl        G48                               \n"

    "0:                                             \n"

    "   mov      $32, %%rax                         \n"

    "1:                                             \n"

    "   movl         (%%r15), %%r10d                \n"
    "   mov         %%rsi, %%r11                    \n"
    "   imul        %%r10, %%r11                    \n"


    "   vmovdqu16   (%%rbx, %%r11), %%zmm0          \n"
    "   vmovdqu16   64(%%rbx, %%r11), %%zmm4        \n"

    "   movl         4(%%r15), %%r10d               \n"
    "   mov         %%rsi, %%r11                    \n"
    "   imul        %%r10, %%r11                    \n"
    "   vmovdqu16   (%%rbx, %%r11), %%zmm1          \n"
    "   vmovdqu16   64(%%rbx, %%r11), %%zmm5        \n"


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


    "   addq        $8, %%r15                       \n"
    "   addq        $64, %%r13                      \n"
    "   sub      $2, %%rax                          \n"
    "   cmp      $0, %%rax                          \n"
    "   jg       1b                                 \n"
    "   addq     $1024, %%r13                       \n"
    "   sub      $32, %%r9                          \n"
    "   cmp      $0, %%r9                           \n"
    "   jg       0b                                 \n"

    "   jmp      END_PACK                           \n"


//--------------------------------------------------------
    "G48:                                           \n"
  
    "   cmp      $48, %%r8                          \n"
    "   jl       G32                                \n"
    "   leaq     (%%r13, %%r12), %%rdx              \n"

    "2:                                             \n"

    "   mov      $32, %%rax                         \n"

    "3:                                             \n"

    "   movl         (%%r15), %%r10d                \n"
    "   mov         %%rsi, %%r11                    \n"
    "   imul        %%r10, %%r11                    \n"
    "   vmovdqu16   (%%rbx, %%r11), %%zmm0          \n"
    "   vmovdqu16   64(%%rbx, %%r11), %%ymm4        \n"

    "   movl         4(%%r15), %%r10d               \n"
    "   mov         %%rsi, %%r11                    \n"
    "   imul        %%r10, %%r11                    \n"
    "   vmovdqu16   (%%rbx, %%r11), %%zmm1          \n"
    "   vmovdqu16   64(%%rbx, %%r11), %%ymm5        \n"

    /*
    "   vextractf64x4   $1, %%zmm0, %%ymm2          \n"
    "   vextractf64x4   $1, %%zmm1, %%ymm3          \n"
    "   vpermt2d    %%zmm1, %%zmm26, %%zmm0         \n"
    "   vpshufb     %%zmm27, %%zmm0, %%zmm0         \n"
    "   vmovdqu16    %%zmm0, (%%r13)                \n"
    "   vpermt2d    %%zmm3, %%zmm26, %%zmm2         \n"
    "   vpshufb     %%zmm27, %%zmm2, %%zmm2         \n"
    "   vmovdqu16    %%zmm2, 1024(%%r13)            \n"

    "   vpermt2d    %%zmm5, %%zmm26, %%zmm4         \n"
    "   vpshufb     %%zmm27, %%zmm4, %%zmm4         \n"
    "   vmovdqu16    %%zmm4, (%%rdx)                \n"*/


    "   vmovdqu16    %%zmm0, %%zmm2                 \n"
    "   vpermt2w    %%zmm1, %%zmm26, %%zmm0         \n"
    "   vmovdqu16    %%zmm0, (%%r13)                \n"
    "   vpermt2w    %%zmm1, %%zmm27, %%zmm2         \n"
    "   vmovdqu16    %%zmm2, 1024(%%r13)            \n"

    "   vpermt2w    %%zmm5, %%zmm26, %%zmm4         \n"
    "   vmovdqu16    %%zmm4, (%%rdx)                \n"

    
    "   addq        $64, %%r13                      \n"
    "   addq        $64, %%rdx                      \n"
    "   addq        $8, %%r15                       \n"

    "   sub      $2, %%rax                          \n"
    "   cmp      $0, %%rax                          \n"
    "   jg       3b                                 \n"
    "   addq     $1024, %%r13                       \n"
    "   sub      $32, %%r9                          \n"
    "   cmp      $0, %%r9                           \n"
    "   jg       2b                                 \n"

    "   jmp      END_PACK                           \n"
    
//--------------------------------------------------------
    "G32:                                           \n"
  
    "   cmp      $32, %%r8                          \n"
    "   jl       G16                                \n"

    "4:                                             \n"

    "   mov      $32, %%rax                         \n"

    "5:                                             \n"


    "   movl         (%%r15), %%r10d                \n"
    "   mov         %%rsi, %%r11                    \n"
    "   imul        %%r10, %%r11                    \n"
    "   vmovdqu16   (%%rbx, %%r11), %%zmm0          \n"

    "   movl         4(%%r15), %%r10d               \n"
    "   mov         %%rsi, %%r11                    \n"
    "   imul        %%r10, %%r11                    \n"
    "   vmovdqu16   (%%rbx, %%r11), %%zmm1          \n"


    "   vmovdqu16    %%zmm0, %%zmm2                 \n"
    "   vpermt2w    %%zmm1, %%zmm26, %%zmm0         \n"
    "   vmovdqu16    %%zmm0, (%%r13)                \n"
    "   vpermt2w    %%zmm1, %%zmm27, %%zmm2         \n"
    "   vmovdqu16    %%zmm2, 1024(%%r13)            \n"

    /*
    "   vextractf64x4   $1, %%zmm0, %%ymm2          \n"
    "   vextractf64x4   $1, %%zmm1, %%ymm3          \n"
    "   vpermt2d    %%zmm1, %%zmm26, %%zmm0         \n"
    "   vpshufb     %%zmm27, %%zmm0, %%zmm0         \n"
    "   vmovdqu16    %%zmm0, (%%r13)                \n"
    "   vpermt2d    %%zmm3, %%zmm26, %%zmm2         \n"
    "   vpshufb     %%zmm27, %%zmm2, %%zmm2         \n"
    "   vmovdqu16    %%zmm2, 1024(%%r13)            \n"*/
    
    "   addq        $64, %%r13                      \n"
    "   addq        $8, %%r15                       \n"

    "   sub      $2, %%rax                          \n"
    "   cmp      $0, %%rax                          \n"
    "   jg       5b                                 \n"
    "   addq     $1024, %%r13                       \n"
    "   sub      $32, %%r9                          \n"
    "   cmp      $0, %%r9                           \n"
    "   jg       4b                                 \n"

    "   jmp      END_PACK                           \n"

//--------------------------------------------------------
    "G16:                                           \n"


    "6:                                             \n"

    "   movl         (%%r15), %%r10d                \n"
    "   mov         %%rsi, %%r11                    \n"
    "   imul        %%r10, %%r11                    \n"
    "   vmovdqu16   (%%rbx, %%r11), %%ymm0          \n"

    "   movl         4(%%r15), %%r10d               \n"
    "   mov         %%rsi, %%r11                    \n"
    "   imul        %%r10, %%r11                    \n"
    "   vmovdqu16   (%%rbx, %%r11), %%ymm1          \n"


    "   vpermt2w    %%zmm1, %%zmm26, %%zmm0         \n"
    "   vmovdqu16    %%zmm0, (%%r13)                \n"

    "   addq        $8, %%r15                       \n"
    "   addq        $64, %%r13                      \n"

    "   sub      $2, %%r9                           \n"
    "   cmp      $0, %%r9                           \n"
    "   jg       6b                                 \n"

    "END_PACK:                                      \n"
    :
    :
      [B] "m" (B),
      [Bc] "m" (Bc),
      [K] "m" (K),
      [stride_N] "m" (stride_N),
      [G] "m" (G),
      [vpermt2d_control] "m" (vpermt2d_control),
      [numa_k] "m" (numa_k)
    :"rax", "rbx", "rcx", "rdx", "rdi", "rsi","r8", "r9", "r10", "r11", "r12",
     "r13", "r14", "r15",
     "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
     "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
     "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
     "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
     "zmm30", "zmm31", "memory"
    );

}



void mKernel_32x32x32(float *C, uint16_t* A, uint16_t* B,
                    long M, long K, long stride_N, long stride_K, long kk)
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


    "BEGIN_KERNEKL_32x32x32:                        \n"

    "   mov       %[A], %%rbp                       \n"
    "   mov       %[C], %%rcx                       \n"
    "   mov       %[M], %%r8                        \n"
    "   mov       %[K], %%r12                       \n"
    "   kmovq     %%r12, %%k1                       \n"
    "   mov       %[stride_K], %%r11                \n"
    "   mov       %[stride_N], %%rsi                \n"
    "   mov       %[kk], %%r15                      \n"
    "   mov       %[B], %%r10                      \n"
    "   shl       $6, %%r12                         \n"

    "   mov       $64, %%rdx                        \n"

    "   tilezero    %%tmm4                          \n"   
    "   tilezero    %%tmm5                          \n"
    "   tilezero    %%tmm6                          \n"
    "   tilezero    %%tmm7                          \n"

    "   kmovq    %%rsi, %%k2                        \n"
    "   kshiftlq $5, %%k2, %%k2                     \n"
    
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
    "   tileloadd   (%%rax, %%r11, 1), %%tmm0     \n"
    "   addq        $64, %%rax                      \n"
    "   tileloadd   (%%rbx, %%rdx, 1), %%tmm2       \n"
    "   tdpbf16ps   %%tmm2, %%tmm0, %%tmm4          \n"
    "   tileloadd   1024(%%rbx, %%rdx, 1), %%tmm3   \n"
    "   addq        $2048, %%rbx                    \n"
    "   tdpbf16ps   %%tmm3, %%tmm0, %%tmm5          \n"
    "   tileloadd   (%%rdi, %%r11, 1), %%tmm1     \n"
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
      [C] "m" (C),
      [B] "m" (B),
      [M] "m" (M),
      [K] "m" (K),
      [stride_N] "m" (stride_N),
      [stride_K] "m" (stride_K),
      [kk] "m" (kk)
    :"rax", "rbx", "rcx", "rdx", "rdi", "rsi","rbp","r8", "r9", "r10", "r11", "r12",
     "r13", "r14", "r15",
     "k1","k2", "k3", "k4",
     "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
     "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
     "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
     "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
     "zmm30", "zmm31", "memory"
    );

}


void mKernel_32x16x32(float *C, uint16_t* A, uint16_t* B,
                    long M, long K, long stride_N, long stride_K, long kk)
{

    asm volatile(


    ".macro Interleave_Store_32x16                  \n"

//----------------------------------------------------------
    "   tilestored   %%tmm4, (%%rcx, %%rsi, 2)      \n"
    "   tilezero    %%tmm4                          \n"

    "   kmovq       %%k2, %%r14                     \n"
    "   addq        %%r14, %%rcx                    \n"

    "   tilestored   %%tmm6, (%%rcx, %%rsi, 2)      \n"
    "   tilezero    %%tmm6                          \n"
    "   addq        %%r14, %%rcx                    \n"

    ".endm                                          \n"


    "BEGIN_KERNEKL_32x16x32:                        \n"

    "   mov       %[A], %%rbp                       \n"
    "   mov       %[C], %%rcx                       \n"
    "   mov       %[M], %%r8                        \n"
    "   mov       %[K], %%r12                       \n"
    "   kmovq     %%r12, %%k1                       \n"
    "   mov       %[stride_K], %%r11                \n"
    "   mov       %[stride_N], %%rsi                \n"
    "   mov       %[kk], %%r15                      \n"
    "   mov       %[B], %%r10                      \n"
    "   shl       $6, %%r12                         \n"

    "   mov       $64, %%rdx                        \n"

    "   tilezero    %%tmm4                          \n"   
    "   tilezero    %%tmm6                          \n"

    "   kmovq    %%rsi, %%k2                        \n"
    "   kshiftlq $5, %%k2, %%k2                     \n"
    
    "BEGIN_M_32x16:                                 \n"
    
    "   mov         %%r10, %%rbx                    \n"

    "   mov         %%rbp, %%rax                    \n"
    "   leaq        (%%rax, %%r11, 8), %%rdi        \n"
    "   leaq        (%%rdi, %%r11, 8), %%rdi        \n"

    "   kmovq       %%k1, %%r9                      \n"    //K

    "   cmp         $0, %%r15                       \n"
    "   je          BEGIN_K_32x16                   \n"

    "   tileloadd   (%%rcx, %%rsi, 2), %%tmm4       \n"
    "   kmovq       %%k2, %%r14                     \n"
    "   addq        %%r14, %%rcx                    \n"
    "   tileloadd   (%%rcx, %%rsi, 2), %%tmm6       \n"
    "   subq        %%r14, %%rcx                    \n"
    
//-----------------------------------------------------------
    "BEGIN_K_32x16:                                 \n"

    "   subq        $32, %%r9                       \n"
    "   tileloadd   (%%rax, %%r11, 1), %%tmm0     \n"
    "   addq        $64, %%rax                      \n"
    "   tileloadd   (%%rbx, %%rdx, 1), %%tmm2       \n"
    "   tdpbf16ps   %%tmm2, %%tmm0, %%tmm4          \n"
    "   addq        $1024, %%rbx                    \n"
    "   tileloadd   (%%rdi, %%r11, 1), %%tmm1     \n"
    "   addq        $64, %%rdi                      \n"
    "   tdpbf16ps   %%tmm2, %%tmm1, %%tmm6          \n"

    "   cmpq        $0, %%r9                        \n"
    "   jg          BEGIN_K_32x16                   \n"

    "Store32x16:                                    \n"

    "   Interleave_Store_32x16                      \n"
//------------------------------------------------------

    "   leaq        (%%rbp, %%r11, 8), %%rbp        \n"
    "   leaq        (%%rbp, %%r11, 8), %%rbp        \n"
    "   leaq        (%%rbp, %%r11, 8), %%rbp        \n"
    "   leaq        (%%rbp, %%r11, 8), %%rbp        \n"
    
    "   subq    $32, %%r8                           \n"
    "   cmp     $0, %%r8                            \n"
    "   jg      BEGIN_M_32x16                       \n"

    
    :
    :
      [A] "m" (A),
      [C] "m" (C),
      [B] "m" (B),
      [M] "m" (M),
      [K] "m" (K),
      [stride_N] "m" (stride_N),
      [stride_K] "m" (stride_K),
      [kk] "m" (kk)
    :"rax", "rbx", "rcx", "rdx", "rdi", "rsi","rbp","r8", "r9", "r10", "r11", "r12",
     "r13", "r14", "r15",
     "k1","k2", "k3", "k4",
     "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
     "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
     "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
     "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
     "zmm30", "zmm31", "memory"
    );

}


void mKernel_16x32x32(float *C, uint16_t* A, uint16_t* B,
                    long M, long K, long stride_N, long stride_K, long kk)
{

    asm volatile(


    ".macro Interleave_Store_16x32                  \n"

//----------------------------------------------------------
    "   tilestored   %%tmm6, (%%rcx, %%rsi, 2)      \n"
    "   tilestored   %%tmm7, 64(%%rcx, %%rsi, 2)    \n"

    ".endm                                          \n"



    "BEGIN_KERNEKL_16x32x32:                        \n"

    "   mov       %[A], %%rbp                       \n"
    "   mov       %[C], %%rcx                       \n"
    "   mov       %[M], %%r8                        \n"
    "   mov       %[K], %%r12                       \n"
    "   kmovq     %%r12, %%k1                       \n"
    "   mov       %[stride_K], %%r11                \n"
    "   mov       %[stride_N], %%rsi                \n"
    "   mov       %[kk], %%r15                      \n"
    "   mov       %[B], %%r10                      \n"

    "   shl       $6, %%r12                         \n"   // 32*2*kc
    "   mov       $64, %%rdx                        \n"

    "   tilezero    %%tmm6                          \n"   
    "   tilezero    %%tmm7                          \n"
    

    "BEGIN_M_16x32:                                 \n"
    
    "   mov         %%r10, %%rbx                    \n"
    "   mov         %%rbp, %%rax                    \n"

    "   kmovq       %%k1, %%r9                      \n"    //K

    "   cmp         $0, %%r15                       \n"
    "   je          2f                              \n"

    "   tileloadd   (%%rcx, %%rsi, 2), %%tmm6       \n"
    "   tileloadd   64(%%rcx, %%rsi, 2), %%tmm7     \n"
    
//-----------------------------------------------------------
    "2:                                             \n"

    "   subq        $32, %%r9                       \n"
    "   tileloadd   (%%rax, %%r11, 1), %%tmm1     \n"
    "   addq        $64, %%rax                      \n"
    "   tileloadd   (%%rbx, %%rdx, 1), %%tmm2       \n"
    "   tdpbf16ps   %%tmm2, %%tmm1, %%tmm6          \n"
    "   tileloadd   1024(%%rbx, %%rdx, 1), %%tmm3   \n"
    "   addq        $2048, %%rbx                    \n"
    "   tdpbf16ps   %%tmm3, %%tmm1, %%tmm7          \n"

    "   cmpq        $0, %%r9                        \n"
    "   jg          2b                              \n"

    "Store16x32:                                    \n"

    "   Interleave_Store_16x32                      \n"

    
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
     "r13", "r14", "r15",
     "k1", "k2", "k3",
     "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
     "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
     "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
     "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
     "zmm30", "zmm31", "memory"
    );

}


void mKernel_16x16x32(float *C, uint16_t* A, uint16_t* B,
                    long M, long K, long stride_N,  long stride_K, long kk)
{

    asm volatile(


    ".macro Interleave_Store_16x16                  \n"

//----------------------------------------------------------
    "   tilestored   %%tmm7, (%%rcx, %%rsi, 2)      \n"

    ".endm                                          \n"


    "BEGIN_KERNEKL_16x16x32:                        \n"

    "   mov       %[A], %%rbp                       \n"
    "   mov       %[C], %%rcx                       \n"
    "   mov       %[M], %%r8                        \n"
    "   mov       %[K], %%r12                       \n"
    "   kmovq     %%r12, %%k1                       \n"
    "   mov       %[stride_K], %%r11                \n"
    "   mov       %[stride_N], %%rsi                \n"
    "   mov       %[kk], %%r15                      \n"
    "   mov       %[B], %%r10                       \n"
    "   shl       $6, %%r12                         \n"

    "   mov       $64, %%rdx                        \n"
    "   tilezero    %%tmm7                          \n"

    "1:                                             \n"
    
    "   mov         %%r10, %%rbx                    \n"
    "   mov         %%rbp, %%rax                    \n"

    "   kmovq       %%k1, %%r9                      \n"    //K

    "   cmp         $0, %%r15                       \n"
    "   je          2f                              \n"

    "   tileloadd   (%%rcx, %%rsi, 2), %%tmm7       \n"
//-----------------------------------------------------------
    "2:                                             \n"

    "   subq        $32, %%r9                       \n"
    "   tileloadd   (%%rax, %%r11, 1), %%tmm1     \n"
    "   addq        $64, %%rax                      \n"
    "   tileloadd   (%%rbx, %%rdx, 1), %%tmm3       \n"
    "   addq        $1024, %%rbx                    \n"
    "   tdpbf16ps   %%tmm3, %%tmm1, %%tmm7          \n"

    "   cmpq        $0, %%r9                        \n"
    "   jg          2b                              \n"

    "Store16x16:                                    \n"

    "   Interleave_Store_16x16                      \n"

    
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
     "r13", "r14", "r15","zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
     "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
     "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
     "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
     "zmm30", "zmm31", "memory"
    );

}

