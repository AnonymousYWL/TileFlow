void mKernel_16x32x32(float *C, uint16_t* A, uint16_t* B,
                    long M, long K, long stride_N, long stride_K)
{

    asm volatile(

    ".macro Interleave_Store                        \n"

//----------------------------------------------------------
    "   tilestored   %%tmm6, (%%rcx, %%rsi)         \n"
    "   tilestored   %%tmm7, 64(%%rcx, %%rsi)       \n"

    ".endm                                          \n"


    "BEGIN_KERNEKL_16x32x32:                        \n"

    "   mov       %[A], %%rax                       \n"
    "   mov       %[B], %%rbx                       \n"
    "   mov       %[C], %%rcx                       \n"
    "   mov       %[M], %%r8                        \n"
    "   mov       %[K], %%r9                        \n"
    "   mov       %[stride_N], %%rsi                \n"
    "   mov       %[stride_K], %%r12                \n"

    "   mov       $64, %%rdx                        \n"

    "   tilezero    %%tmm6                          \n"
    "   tilezero    %%tmm7                          \n"

    "   leaq        (%%rbx, %%r12, 8), %%rbp        \n"
    "   leaq        (%%rbp, %%r12, 8), %%rbp        \n"

    "0:                                             \n"
    "   subq        $128, %%r9                      \n"

    "   tileloadd   (%%rax, %%r12, 1), %%tmm0       \n"
    "   tileloadd   64(%%rax, %%r12, 1), %%tmm1     \n"

    "   tileloaddt1   (%%rbx, %%rdx, 1), %%tmm2     \n"
    "   tileloaddt1   (%%rbp, %%rdx, 1), %%tmm3     \n"

    "   tdpbf16ps   %%tmm2, %%tmm0, %%tmm6          \n"
    "   tdpbf16ps   %%tmm3, %%tmm0, %%tmm7          \n"

    "   tileloaddt1   1024(%%rbx, %%rdx, 1), %%tmm4 \n"
    "   tileloaddt1   1024(%%rbp, %%rdx, 1), %%tmm5 \n"

    "   tdpbf16ps   %%tmm4, %%tmm1, %%tmm6          \n"
    "   tdpbf16ps   %%tmm5, %%tmm1, %%tmm7          \n"

//----------------------------------------------------------

    "   tileloadd   128(%%rax, %%r12, 1), %%tmm0    \n"
    "   tileloadd   192(%%rax, %%r12, 1), %%tmm1    \n"

    "   tileloaddt1   2048(%%rbx, %%rdx, 1), %%tmm2 \n"
    "   tileloaddt1   2048(%%rbp, %%rdx, 1), %%tmm3 \n"

    "   tdpbf16ps   %%tmm2, %%tmm0, %%tmm6          \n"
    "   tdpbf16ps   %%tmm3, %%tmm0, %%tmm7          \n"

    "   tileloaddt1   3072(%%rbx, %%rdx, 1), %%tmm4 \n"
    "   tileloaddt1   3072(%%rbp, %%rdx, 1), %%tmm5 \n"

    "   tdpbf16ps   %%tmm4, %%tmm1, %%tmm6          \n"
    "   tdpbf16ps   %%tmm5, %%tmm1, %%tmm7          \n"

    "   addq        $512, %%rax                     \n"
    "   addq        $8192, %%rbx                    \n"
    "   addq        $8192, %%rbp                    \n"

    "   cmp         $0, %%r9                        \n"
    "   jg          0b                              \n"

    "   Interleave_Store                            \n"
    :
    :
      [A] "m" (A),
      [B] "m" (B),
      [C] "m" (C),
      [M] "m" (M),
      [K] "m" (K),
      [stride_N] "m" (stride_N),
      [stride_K] "m" (stride_K)
    :"rax", "rbx", "rcx", "rdx", "rdi", "rsi","rbp","r8", "r9", "r10", "r11", "r12",
     "r13", "r14", "r15", "memory"
    );

}