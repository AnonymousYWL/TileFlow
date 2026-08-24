void mKernel_16x32x32_2(float *C, uint16_t* A, uint16_t* B,
                    long M, uint32_t* K, long stride_M,  long stride_K,
                    long stride_store)
{

    asm volatile(

    ".macro Interleave_Store                        \n"

    "   kmovq       %%k3, %%r9                      \n"
    //----------------------------------------------------------
    "   tilestored   %%tmm4, (%%rcx, %%r9)          \n"
    "   leaq        (%%rcx, %%r9, 8), %%rcx         \n"
    "   leaq        (%%rcx, %%r9, 8), %%rcx         \n"
    "   tilestored   %%tmm5, (%%rcx, %%r9)          \n"


    ".endm                                          \n"


    "BEGIN_KERNEKL_32x32x32:                        \n"

    "   mov       %[A], %%rax                       \n"
    "   mov       %[B], %%rbx                       \n"
    "   mov       %[C], %%rcx                       \n"
    "   mov       %[M], %%r8                        \n"
    "   mov       %[K], %%r12                       \n"   
    "   movl      (%%r12), %%r9d                    \n"   //K_0
    "   mov       %[stride_K], %%r11                \n"
    "   mov       %[stride_M], %%rsi                \n"
    "   mov       %[stride_store], %%r15            \n"
    "   kmovq     %%r15, %%k3                       \n"

    "   mov       %%r8, %%r13                       \n"
    "   shl       $6, %%r13                         \n"

    "   movl      4(%%r12), %%r15d                  \n"  // k_num
    "   movl      8(%%r12), %%r12d                  \n"
    "   kmovq     %%r12, %%k1                       \n"   //K_1

    "   mov       $1, %%rdi                         \n"
    "   shlx      %%r8, %%rdi, %%rdi                \n"
    "   shlx      %%r8, %%rdi, %%rdi                \n"
    "   sub       $1, %%rdi                         \n"
    "   kmovq     %%rdi, %%k2                       \n"  // mask1
    "   shrx      %%r8, %%rdi, %%rdi                \n"
    "   kmovq     %%rdi, %%k4                       \n"  // mask2

    "   kmovq     %%r11, %%k5                       \n"
    "   kshiftlq   $4, %%k5, %%k5                   \n" // + 16*stride_K

    "   mov       %%r13, %%rdx                      \n"
    "   shr       $4, %%rdx                         \n"  // 16行

    "   mov      %%rsp, %%r12                       \n"
    "   and      $63, %%r12                         \n"
    "   addq     $1024, %%r12                       \n"
    "   subq     %%r12, %%rsp                       \n"

    "   tilezero    %%tmm4                          \n"   
    "   tilezero    %%tmm5                          \n"

    "BEGIN_M:                                       \n"


    "BEGIN_OUTER_K:                                 \n"
    
    "   kmovq       %%k5, %%rdi                     \n"
    "   leaq        (%%rbx, %%rdi), %%rdi           \n"
//-----------------------------------------------------------
    "BEGIN_K:                                       \n"

    "   subq        $32, %%r9                       \n"
    "   tileloadd   (%%rax, %%rdx, 1), %%tmm0       \n"
    "   addq        %%r13, %%rax                    \n"
    "   tileloaddt1   (%%rbx, %%r11, 1), %%tmm2     \n"
    "   addq        $64, %%rbx                      \n"
    "   tdpbf16ps   %%tmm0, %%tmm2, %%tmm4          \n"
    "   tileloaddt1   (%%rdi, %%r11, 1), %%tmm3     \n"
    "   addq        $64, %%rdi                      \n"
    "   tdpbf16ps   %%tmm0, %%tmm3, %%tmm5          \n"
    "   cmpq        $0, %%r9                        \n"
    "   jg          BEGIN_K                         \n"


    "   addq        $4096, %%rbx                    \n"
    "   mov         %%rsi, %%rdi                    \n"
    "   shl         $11, %%rdi                      \n"
    "   addq        %%rdi, %%rax                    \n"
    "   sub         $1, %%r15d                      \n"

    "   mov         $2048, %%r9                     \n"
    "   cmp         $1, %%r15d                      \n"
    "   jg          BEGIN_OUTER_K                   \n"

    "   kmovq       %%k1, %%r9                      \n"
    "   je          BEGIN_OUTER_K                   \n"

    "Store16x32:                                    \n"


    "   Interleave_Store                            \n"


    "   addq        %%r12, %%rsp                    \n"

    
    :
    :
      [A] "m" (A),
      [B] "m" (B),
      [C] "m" (C),
      [M] "m" (M),
      [K] "m" (K),
      [stride_M] "m" (stride_M),
      [stride_K] "m" (stride_K),
      [stride_store] "m" (stride_store)
    : "rax", "rbx", "rcx", "rdx", "rdi", "rsi","rbp","r8", "r9", "r10", "r11", "r12",
     "r13", "r14", "r15","k1","k2","k3","k4","k5","zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
     "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
     "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
     "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
     "zmm30", "zmm31", "memory"
    );

}