void SiLU(uint16_t* dst, float* input, long stride, float* Va, long M)
{
    asm volatile(

        "   mov       %[input], %%rax                         \n"
        "   mov       %[dst], %%rdi                           \n"
        "   mov       %[Va], %%rcx                            \n"
        "   mov       %[M], %%r12                             \n"
        "   mov       %[stride], %%rdx                        \n"

        "   vbroadcastss  (%%rcx), %%zmm0                     \n"
        "   vbroadcastss  4(%%rcx), %%zmm1                    \n"
        "   vbroadcastss  8(%%rcx), %%zmm2                    \n"
        "   vbroadcastss  12(%%rcx), %%zmm23                  \n"
        "   vbroadcastss  16(%%rcx), %%zmm24                  \n"
        "   vbroadcastss  20(%%rcx), %%zmm25                  \n"
        "   vbroadcastss  24(%%rcx), %%zmm26                  \n"
        "   vbroadcastss  28(%%rcx), %%zmm27                  \n"

        "   mov     $0xC2FC0000, %%r9d                        \n"
        "   vpbroadcastd %%r9d, %%zmm28                       \n"

        "   mov     $0x42FE0000, %%r9d                        \n"
        "   vpbroadcastd %%r9d, %%zmm29                       \n"

        "   mov     $0x3F800000, %%r9d                        \n"
        "   vpbroadcastd %%r9d, %%zmm11                       \n"   // 1.0

        "   vpxord      %%zmm31, %%zmm31, %%zmm31             \n"

        "BEGIN_SiLU:                                          \n"

        "   vmovaps       %%zmm23, %%zmm3                     \n"
        "   vmovaps       %%zmm24, %%zmm4                     \n"
        "   vmovaps       %%zmm25, %%zmm5                     \n"
        "   vmovaps       %%zmm26, %%zmm6                     \n"
        "   vmovaps       %%zmm27, %%zmm7                     \n"

        // load x
        "   vmovups       (%%rax), %%zmm30                    \n"

        // load y (stride 已经在外部修正)
        "   vmovups       (%%rax, %%rdx), %%zmm12             \n"

        // -x
        "   vsubps        %%zmm30, %%zmm31, %%zmm8            \n"

        // exp polynomial approximation
        "   vmulps        %%zmm0, %%zmm8, %%zmm9              \n"
        "   vrndscaleps   $0, %%zmm9, %%zmm9                  \n"
        "   vfmadd231ps   %%zmm1, %%zmm9, %%zmm8              \n"
        "   vfmadd231ps   %%zmm2, %%zmm8, %%zmm3              \n"
        "   vfmadd231ps   %%zmm3, %%zmm8, %%zmm4              \n"
        "   vfmadd231ps   %%zmm4, %%zmm8, %%zmm5              \n"
        "   vfmadd231ps   %%zmm5, %%zmm8, %%zmm6              \n"
        "   vfmadd231ps   %%zmm6, %%zmm8, %%zmm7              \n"

        // exponent clamp
        "   vcmpltps    %%zmm9, %%zmm28, %%k1                 \n"
        "   vcmpltps    %%zmm29, %%zmm9, %%k2                 \n"
        "   kandw      %%k1, %%k2, %%k1                       \n"

        // exp reconstruction
        "   vscalefps   %%zmm9, %%zmm7, %%zmm9%{%%k1}%{z}     \n"

        // 1 + exp(-x)
        "   vaddps   %%zmm11, %%zmm9, %%zmm9                  \n"

        // x / (1 + exp(-x))
        "   vdivps   %%zmm9, %%zmm30, %%zmm9%{%%k2}%{z}       \n"

        // y * SiLU(x)
        "   vmulps        %%zmm9, %%zmm12, %%zmm9             \n"

        // store result
        "   vcvtneps2bf16     %%zmm9, %%ymm9                  \n"
        "   vmovdqu16     %%ymm9,  (%%rdi)                    \n"

        "   addq        $64, %%rax                            \n"
        "   addq        $32, %%rdi                            \n"
        "   subq        $16, %%r12                            \n"
        "   cmp         $0, %%r12                             \n"
        "   jg          BEGIN_SiLU                            \n"

        :
        :
        [dst]"m"(dst),
        [input] "m" (input),
        [Va] "m" (Va),
        [M] "m" (M),
        [stride]"m" (stride)
        :"rax", "rbx", "rcx", "rdx", "rdi","rbp", "rsi","r8", "r9", "r10", "r11", "r12",
        "r13", "r14", "r15","k1","k2",
        "zmm0","zmm1","zmm2","zmm3","zmm4","zmm5",
        "zmm6","zmm7","zmm8","zmm9","zmm11","zmm12",
        "zmm23","zmm24","zmm25","zmm26","zmm27",
        "zmm28","zmm29","zmm30","zmm31","memory"
    );
}