void SiLU(uint16_t* input, long stride, float* Va, long M)
{
    asm volatile(

        "   mov       %[input], %%rax                         \n"
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
        "   vpbroadcastd %%r9d, %%zmm11                       \n"   // 1

        "   vpxord      %%zmm31, %%zmm31, %%zmm31             \n"

        "BEGIN_SiLU:                                          \n"

        "   vmovaps       %%zmm23, %%zmm3                     \n"
        "   vmovaps       %%zmm24, %%zmm4                     \n"
        "   vmovaps       %%zmm25, %%zmm5                     \n"
        "   vmovaps       %%zmm26, %%zmm6                     \n"
        "   vmovaps       %%zmm27, %%zmm7                     \n"

        "   vpmovzxwd       (%%rax), %%zmm30                  \n"
        "   vpslld          $16, %%zmm30, %%zmm30             \n"  //out

        "   vpmovzxwd       (%%rax, %%rdx), %%zmm12           \n"
        "   vpslld          $16, %%zmm12, %%zmm12             \n"  // out1

        "   vsubps        %%zmm30, %%zmm31, %%zmm8            \n"
        "   vmulps        %%zmm0, %%zmm8, %%zmm9              \n"
        "   vrndscaleps   $0, %%zmm9, %%zmm9                  \n"
        "   vfmadd231ps   %%zmm1, %%zmm9, %%zmm8              \n"
        "   vfmadd231ps   %%zmm2, %%zmm8, %%zmm3              \n"
        "   vfmadd231ps   %%zmm3, %%zmm8, %%zmm4              \n"
        "   vfmadd231ps   %%zmm4, %%zmm8, %%zmm5              \n"
        "   vfmadd231ps   %%zmm5, %%zmm8, %%zmm6              \n"
        "   vfmadd231ps   %%zmm6, %%zmm8, %%zmm7              \n"

        "   vcmpltps    %%zmm9, %%zmm28, %%k1                 \n"
        "   vcmpltps    %%zmm29, %%zmm9, %%k2                 \n"

        "   kandw      %%k1, %%k2, %%k1                       \n"
        
        "   vscalefps   %%zmm9, %%zmm7, %%zmm9%{%%k1}%{z}     \n"
        "   vaddps   %%zmm11, %%zmm9, %%zmm9                  \n"
        "   vdivps   %%zmm9, %%zmm30, %%zmm9%{%%k2}%{z}       \n"

        "   vmulps        %%zmm9, %%zmm12, %%zmm9             \n"
        "   vcvtneps2bf16     %%zmm9, %%ymm9                  \n"
        "   vmovdqu16     %%ymm9,  (%%rax)                    \n"

        "   addq        $32, %%rax                            \n"
        "   subq        $16, %%r12                            \n"
        "   cmp         $0, %%r12                             \n"
        "   jg          BEGIN_SiLU                            \n"
        
        :
        :
        [input] "m" (input),
        [Va] "m" (Va),
        [M] "m" (M),
        [stride]"m" (stride)
        :"rax", "rbx", "rcx", "rdx", "rdi","rbp", "rsi","r8", "r9", "r10", "r11", "r12",
        "r13", "r14", "r15","k1","zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
        "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
        "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
        "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
        "zmm30", "zmm31", "memory"
    );

}


void Add_matrix(uint16_t* dst, float* input0, float* input1, long M, long K, long stride_M,
                uint8_t *vpermt2d_control)
{

        asm volatile(

        "   mov       %[dst], %%rdx                           \n"
        "   mov       %[input0], %%rax                        \n"
        "   mov       %[input1], %%rbx                        \n"
        "   mov       %[M], %%r12                             \n"
        "   mov       %[K], %%r13                             \n"
        "   mov       %[vpermt2d_control], %%r14              \n"
        "   mov       %[stride_M], %%rsi                      \n"

        "   movl      $1, %%r8d                               \n"
        "   shlx      %%r12, %%r8, %%r8                       \n"
        "   subq      $1, %%r8                                \n"
        "   kmovq     %%r8, %%k2                              \n"

        "   vpmovzxbw   (%%r14), %%zmm31                      \n"

        "Add_matrix_begin_K:                                  \n"

        "   vmovups     (%%rax), %%zmm0%{%%k2}                \n"
        "   vmovups     (%%rax, %%rsi, 2), %%zmm1%{%%k2}      \n"

        "   vmovups     (%%rbx), %%zmm2%{%%k2}                \n"
        "   vmovups     (%%rbx, %%rsi, 2), %%zmm3%{%%k2}      \n"

        "   vaddps       %%zmm0, %%zmm2, %%zmm0               \n"
        "   vaddps       %%zmm1, %%zmm3, %%zmm1               \n"
        "   vcvtne2ps2bf16  %%zmm0, %%zmm1, %%zmm0            \n"
        "   vpermt2w    %%zmm0, %%zmm31, %%zmm0               \n"   
        "   vmovups    %%zmm0, (%%rdx)%{%%k2}                 \n"

        "   leaq        (%%rax, %%rsi, 4), %%rax              \n"
        "   leaq        (%%rbx, %%rsi, 4), %%rbx              \n"
        "   leaq        (%%rdx, %%rsi, 2), %%rdx              \n"

        "   subq    $2, %%r13                                 \n"
        "   cmp     $2, %%r13                                 \n"
        "   jge     Add_matrix_begin_K                        \n"
        
        :
        :
        [dst] "m" (dst),
        [input0] "m" (input0),
        [input1] "m" (input1),
        [M] "m" (M),
        [K]"m" (K),
        [stride_M] "m" (stride_M),
        [vpermt2d_control] "m" (vpermt2d_control)
        :"rax", "rbx", "rcx", "rdx", "rsi","r8", "r9", "r10", "r11", "r12",
        "r13", "r14", "r15","k2","zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
        "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
        "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
        "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
        "zmm30", "zmm31", "memory"
    );


}


void Add_Transpose_matrix(uint16_t* dst, float* input0, float* input1, long M, long K, 
                            long stride_M, long stride_N,
                            uint8_t *vpermt2d_control)
{

        asm volatile(

        "   mov       %[input0], %%rax                          \n"
        "   mov       %[input1], %%rbx                          \n"
        "   mov       %[dst], %%rdx                             \n"
        "   mov       %[M], %%r12                               \n"
        "   mov       %[K], %%r13                               \n"
        "   mov       %[stride_M], %%rsi                        \n"
        "   mov       %[stride_N], %%rdi                        \n"

        "   mov       %[vpermt2d_control], %%r14                \n"
        "   vpmovzxbw   (%%r14), %%zmm31                        \n"

        "   movl     $1, %%r8d                                  \n"
        "   shlx     %%r12, %%r8, %%r8                          \n"
        "   subq     $1, %%r8                                   \n"
        "   kmovq    %%r8, %%k2                                 \n"

        "Add_Transpose_matrix_begin_K:                          \n"


        // input0: 16 rows of FP32, masked by k2
        "   vmovups    (%%rax), %%zmm0%{%%k2}                \n"
        "   vmovups    (%%rax, %%rsi), %%zmm1%{%%k2}         \n"
        "   vmovups    (%%rax, %%rsi, 2), %%zmm2%{%%k2}      \n"
        "   leaq       (%%rax, %%rsi, 2), %%rax                 \n"
        "   vmovups    (%%rax, %%rsi), %%zmm3%{%%k2}         \n"
        "   vmovups    (%%rax, %%rsi, 2), %%zmm4%{%%k2}      \n"
        "   leaq       (%%rax, %%rsi, 2), %%rax                 \n"
        "   vmovups    (%%rax, %%rsi), %%zmm5%{%%k2}         \n"
        "   vmovups    (%%rax, %%rsi, 2), %%zmm6%{%%k2}      \n"
        "   leaq       (%%rax, %%rsi, 2), %%rax                 \n"
        "   vmovups    (%%rax, %%rsi), %%zmm7%{%%k2}         \n"
        "   vmovups    (%%rax, %%rsi, 2), %%zmm8%{%%k2}      \n"
        "   leaq       (%%rax, %%rsi, 2), %%rax                 \n"
        "   vmovups    (%%rax, %%rsi), %%zmm9%{%%k2}         \n"
        "   vmovups    (%%rax, %%rsi, 2), %%zmm10%{%%k2}     \n"
        "   leaq       (%%rax, %%rsi, 2), %%rax                 \n"
        "   vmovups    (%%rax, %%rsi), %%zmm11%{%%k2}        \n"
        "   vmovups    (%%rax, %%rsi, 2), %%zmm12%{%%k2}     \n"
        "   leaq       (%%rax, %%rsi, 2), %%rax                 \n"
        "   vmovups    (%%rax, %%rsi), %%zmm13%{%%k2}        \n"
        "   vmovups    (%%rax, %%rsi, 2), %%zmm14%{%%k2}     \n"
        "   leaq       (%%rax, %%rsi, 2), %%rax                 \n"
        "   vmovups    (%%rax, %%rsi), %%zmm15%{%%k2}        \n"
        "   leaq       (%%rax, %%rsi, 2), %%rax                 \n"

        // input1: first 14 rows of FP32
        "   vmovups    (%%rbx), %%zmm16%{%%k2}               \n"
        "   vmovups    (%%rbx, %%rsi), %%zmm17%{%%k2}        \n"
        "   vmovups    (%%rbx, %%rsi, 2), %%zmm18%{%%k2}     \n"
        "   leaq       (%%rbx, %%rsi, 2), %%rbx                 \n"
        "   vmovups    (%%rbx, %%rsi), %%zmm19%{%%k2}        \n"
        "   vmovups    (%%rbx, %%rsi, 2), %%zmm20%{%%k2}     \n"
        "   leaq       (%%rbx, %%rsi, 2), %%rbx                 \n"
        "   vmovups    (%%rbx, %%rsi), %%zmm21%{%%k2}        \n"
        "   vmovups    (%%rbx, %%rsi, 2), %%zmm22%{%%k2}     \n"
        "   leaq       (%%rbx, %%rsi, 2), %%rbx                 \n"
        "   vmovups    (%%rbx, %%rsi), %%zmm23%{%%k2}        \n"
        "   vmovups    (%%rbx, %%rsi, 2), %%zmm24%{%%k2}     \n"
        "   leaq       (%%rbx, %%rsi, 2), %%rbx                 \n"
        "   vmovups    (%%rbx, %%rsi), %%zmm25%{%%k2}        \n"
        "   vmovups    (%%rbx, %%rsi, 2), %%zmm26%{%%k2}     \n"
        "   leaq       (%%rbx, %%rsi, 2), %%rbx                 \n"
        "   vmovups    (%%rbx, %%rsi), %%zmm27%{%%k2}        \n"
        "   vmovups    (%%rbx, %%rsi, 2), %%zmm28%{%%k2}     \n"
        "   leaq       (%%rbx, %%rsi, 2), %%rbx                 \n"
        "   vmovups    (%%rbx, %%rsi), %%zmm29%{%%k2}        \n"

        // add rows 0-13
        "   vaddps     %%zmm0, %%zmm16, %%zmm0                  \n"
        "   vaddps     %%zmm1, %%zmm17, %%zmm1                  \n"
        "   vaddps     %%zmm2, %%zmm18, %%zmm2                  \n"
        "   vaddps     %%zmm3, %%zmm19, %%zmm3                  \n"
        "   vaddps     %%zmm4, %%zmm20, %%zmm4                  \n"
        "   vaddps     %%zmm5, %%zmm21, %%zmm5                  \n"
        "   vaddps     %%zmm6, %%zmm22, %%zmm6                  \n"
        "   vaddps     %%zmm7, %%zmm23, %%zmm7                  \n"
        "   vaddps     %%zmm8, %%zmm24, %%zmm8                  \n"
        "   vaddps     %%zmm9, %%zmm25, %%zmm9                  \n"
        "   vaddps     %%zmm10, %%zmm26, %%zmm10                \n"
        "   vaddps     %%zmm11, %%zmm27, %%zmm11                \n"
        "   vaddps     %%zmm12, %%zmm28, %%zmm12                \n"
        "   vaddps     %%zmm13, %%zmm29, %%zmm13                \n"

        // input1: last 2 rows (reuse zmm28/zmm29)
        "   vmovups    (%%rbx, %%rsi, 2), %%zmm28%{%%k2}        \n"
        "   leaq       (%%rbx, %%rsi, 2), %%rbx                 \n"
        "   vmovups    (%%rbx, %%rsi), %%zmm29%{%%k2}           \n"
        "   vaddps     %%zmm14, %%zmm28, %%zmm14                \n"
        "   vaddps     %%zmm15, %%zmm29, %%zmm15                \n"
        "   leaq       (%%rbx, %%rsi, 2), %%rbx                 \n"

//---------------------------------------------------------------

        "   vcvtne2ps2bf16  %%zmm0, %%zmm1, %%zmm0            \n"
        "   vpermt2w    %%zmm0, %%zmm31, %%zmm0               \n"

        "   vcvtne2ps2bf16  %%zmm2, %%zmm3, %%zmm1            \n"
        "   vpermt2w    %%zmm1, %%zmm31, %%zmm1               \n"

        "   vcvtne2ps2bf16  %%zmm4, %%zmm5, %%zmm2            \n"
        "   vpermt2w    %%zmm2, %%zmm31, %%zmm2               \n"

        "   vcvtne2ps2bf16  %%zmm6, %%zmm7, %%zmm3            \n"
        "   vpermt2w    %%zmm3, %%zmm31, %%zmm3               \n"

        "   vcvtne2ps2bf16  %%zmm8, %%zmm9, %%zmm4            \n"
        "   vpermt2w    %%zmm4, %%zmm31, %%zmm4               \n"

        "   vcvtne2ps2bf16  %%zmm10, %%zmm11, %%zmm5          \n"
        "   vpermt2w    %%zmm5, %%zmm31, %%zmm5               \n"

        "   vcvtne2ps2bf16  %%zmm12, %%zmm13, %%zmm6          \n"
        "   vpermt2w    %%zmm6, %%zmm31, %%zmm6               \n"

        "   vcvtne2ps2bf16  %%zmm14, %%zmm15, %%zmm7          \n"
        "   vpermt2w    %%zmm7, %%zmm31, %%zmm7               \n"

// Transpose        
//********************************************************************
        "   vunpcklps %%zmm1, %%zmm0, %%zmm24                   \n"
        "   vunpcklps %%zmm3, %%zmm2, %%zmm25                   \n"
        "   vunpcklpd %%zmm25, %%zmm24, %%zmm26                 \n"

        "   vunpcklps %%zmm5, %%zmm4, %%zmm29                   \n"
        "   vunpcklps %%zmm7, %%zmm6, %%zmm30                   \n"
        "   vunpcklpd %%zmm30, %%zmm29, %%zmm27                 \n"

        "   vinsertf32x4    $1, %%xmm27, %%ymm26, %%ymm8        \n"
        "   vmovdqu16    %%ymm8, (%%rdx)                        \n"
        "   cmp     $5, %%r12                                   \n"
        "   jl      1f                                          \n"

        "   vextractf32x4   $1, %%zmm26, %%xmm9                 \n"
        "   vextractf32x4   $1, %%zmm27, %%xmm28                \n"
        "   vinsertf32x4    $1, %%xmm28, %%ymm9, %%ymm9         \n"

        "   vmovdqu16    %%ymm9, (%%rdx, %%rdi, 4)              \n"
        "   cmp     $9, %%r12                                   \n"
        "   jl      1f                                          \n"

        "   vextractf32x4   $2, %%zmm26, %%xmm10                \n"
        "   vextractf32x4   $2, %%zmm27, %%xmm28                \n"
        "   vinsertf32x4    $1, %%xmm28, %%ymm10, %%ymm10       \n"

        "   vmovdqu16    %%ymm10, (%%rdx, %%rdi, 8)             \n"
        "   cmp     $13, %%r12                                   \n"
        "   jl      1f                                          \n"

        "   vextractf32x4   $3, %%zmm26, %%xmm11                \n"
        "   vextractf32x4   $3, %%zmm27, %%xmm28                \n"
        "   vinsertf32x4    $1, %%xmm28, %%ymm11, %%ymm11       \n"

        "   leaq        (%%rdx, %%rdi, 8), %%r14                \n"
        "   vmovdqu16    %%ymm11, (%%r14, %%rdi, 4)             \n"


//--------------------------------------------------------------------

        "1:                                                     \n"
        "   cmp     $2, %%r12                                   \n"
        "   jl      Transpose_END_Store                         \n"

        "   vunpckhpd %%zmm25, %%zmm24, %%zmm26                 \n"
        "   vunpckhpd %%zmm30, %%zmm29, %%zmm27                 \n"

        "   vinsertf32x4    $1, %%xmm27, %%ymm26, %%ymm12       \n"
        "   vmovdqu16    %%ymm12, (%%rdx, %%rdi)                \n"
        "   cmp     $6, %%r12                                   \n"
        "   jl      2f                                          \n"
        "   vextractf32x4   $1, %%zmm26, %%xmm13                \n"
        "   vextractf32x4   $1, %%zmm27, %%xmm28                \n"
        "   vinsertf32x4    $1, %%xmm28, %%ymm13, %%ymm13       \n"

        "   leaq        (%%rdx, %%rdi), %%r14                   \n"
        "   vmovdqu16    %%ymm13, (%%r14, %%rdi, 4)             \n"
        "   cmp     $10, %%r12                                  \n"
        "   jl      2f                                          \n"

        "   vextractf32x4   $2, %%zmm26, %%xmm14                \n"
        "   vextractf32x4   $2, %%zmm27, %%xmm28                \n"
        "   vinsertf32x4    $1, %%xmm28, %%ymm14, %%ymm14       \n"

        "   vmovdqu16    %%ymm14, (%%r14, %%rdi,8)              \n"

        "   cmp     $14, %%r12                                  \n"
        "   jl      2f                                          \n"

        "   vextractf32x4   $3, %%zmm26, %%xmm15                \n"
        "   vextractf32x4   $3, %%zmm27, %%xmm28                \n"
        "   vinsertf32x4    $1, %%xmm28, %%ymm15, %%ymm15       \n"

        "   leaq        (%%r14, %%rdi, 8), %%r14                \n"
        "   vmovdqu16    %%ymm15, (%%r14, %%rdi, 4)             \n"

//--------------------------------------------------------------------

        "2:                                                    \n"
        "   cmp      $3, %%r12                                 \n"
        "   jl       Transpose_END_Store                       \n"

        "   vunpckhps %%zmm1, %%zmm0, %%zmm24                   \n"
        "   vunpckhps %%zmm3, %%zmm2, %%zmm25                   \n"
        "   vunpcklpd %%zmm25, %%zmm24, %%zmm26                 \n"

        "   vunpckhps %%zmm5, %%zmm4, %%zmm29                   \n"
        "   vunpckhps %%zmm7, %%zmm6, %%zmm30                   \n"
        "   vunpcklpd %%zmm30, %%zmm29, %%zmm27                 \n"

        "   vinsertf32x4    $1, %%xmm27, %%ymm26, %%ymm16       \n"
        "   vmovdqu16    %%ymm16, (%%rdx, %%rdi, 2)             \n"
        "   cmp     $7, %%r12                                   \n"
        "   jl      3f                                          \n"
        "   vextractf32x4   $1, %%zmm26, %%xmm17                \n"
        "   vextractf32x4   $1, %%zmm27, %%xmm28                \n"
        "   vinsertf32x4    $1, %%xmm28, %%ymm17, %%ymm17       \n"

        "   leaq        (%%rdx, %%rdi, 2), %%r14                \n"
        "   vmovdqu16    %%ymm17, (%%r14, %%rdi, 4)             \n"
        "   cmp     $11, %%r12                                  \n"
        "   jl      3f                                          \n"

        "   vextractf32x4   $2, %%zmm26, %%xmm18                \n"
        "   vextractf32x4   $2, %%zmm27, %%xmm28                \n"
        "   vinsertf32x4    $1, %%xmm28, %%ymm18, %%ymm18       \n"

        "   vmovdqu16    %%ymm18, (%%r14, %%rdi, 8)             \n"
        "   cmp     $15, %%r12                                  \n"
        "   jl      3f                                          \n"

        "   vextractf32x4   $3, %%zmm26, %%xmm19                \n"
        "   vextractf32x4   $3, %%zmm27, %%xmm28                \n"
        "   vinsertf32x4    $1, %%xmm28, %%ymm19, %%ymm19       \n"

        "   leaq        (%%r14, %%rdi, 8), %%r14                \n"
        "   vmovdqu16    %%ymm19, (%%r14, %%rdi, 4)             \n"


//--------------------------------------------------------------------

        "3:                                                    \n"
        "   cmp      $4, %%r12                                 \n"
        "   jl       Transpose_END_Store                       \n"

        "   vunpckhpd %%zmm25, %%zmm24, %%zmm26                \n"
        "   vunpckhpd %%zmm30, %%zmm29, %%zmm27                \n"

        "   vinsertf32x4    $1, %%xmm27, %%ymm26, %%ymm20       \n"
        "   leaq            (%%rdx, %%rdi, 2), %%r14            \n"
        "   leaq            (%%r14, %%rdi), %%r14               \n"
        "   vmovdqu16    %%ymm20, (%%r14)                       \n"
        "   cmp          $8, %%r12                              \n"
        "   jl           Transpose_END_Store                    \n"

        "   vextractf32x4   $1, %%zmm26, %%xmm21                \n"
        "   vextractf32x4   $1, %%zmm27, %%xmm28                \n"
        "   vinsertf32x4    $1, %%xmm28, %%ymm21, %%ymm21       \n"

        "   vmovdqu16    %%ymm21, (%%r14, %%rdi, 4)             \n"
        "   cmp     $12, %%r12                                  \n"
        "   jl      Transpose_END_Store                         \n"

        "   vextractf32x4   $2, %%zmm26, %%xmm22                \n"
        "   vextractf32x4   $2, %%zmm27, %%xmm28                \n"
        "   vinsertf32x4    $1, %%xmm28, %%ymm22, %%ymm22       \n"

        "   vmovdqu16    %%ymm22, (%%r14, %%rdi, 8)             \n"
        "   cmp     $16, %%r12                                  \n"
        "   jl      Transpose_END_Store                         \n"

        "   vextractf32x4   $3, %%zmm26, %%xmm23                \n"
        "   vextractf32x4   $3, %%zmm27, %%xmm28                \n"
        "   vinsertf32x4    $1, %%xmm28, %%ymm23, %%ymm23       \n"

        "   leaq        (%%r14, %%rdi, 8), %%r14                \n"
        "   vmovdqu16    %%ymm23, (%%r14, %%rdi, 4)             \n"

//--------------------------------------------------------------------


        "Transpose_END_Store:                                   \n"

//************************************************************************** 
        "   addq    $32, %%rdx                                  \n"
        "   subq    $16, %%r13                                  \n"
        "   cmp     $16, %%r13                                  \n"
        "   jge     Add_Transpose_matrix_begin_K                \n"
        
        :
        :
        [dst] "m" (dst),
        [input0] "m" (input0),
        [input1] "m" (input1),
        [M] "m" (M),
        [K]"m" (K),
        [stride_M] "m" (stride_M),
        [stride_N] "m" (stride_N),
        [vpermt2d_control] "m" (vpermt2d_control)
        :"rax", "rbx", "rcx", "rdx", "rsi","rdi","rbp","r8", "r9", "r10", "r11", "r12",
        "r13", "r14", "r15","k2","zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
        "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
        "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
        "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
        "zmm30", "zmm31", "memory"
    );


}

void transpose(uint16_t* Ac, uint16_t* A, int mr, int K, long stride_K)
{

    asm volatile(

    "   mov       %[A], %%rax                       \n"
    "   mov       %[Ac], %%rbx                      \n"
    "   movl      %[mr], %%r8d                      \n"
    "   movl      %[K], %%r12d                      \n"
    "   mov       %[stride_K], %%r11                \n"

    "   mov       %%r8, %%rsi                       \n"
    "   shl       $2, %%rsi                         \n"

    "   mov      $1, %%r13                          \n"
    "   shlx     %%r8, %%r13, %%r13                 \n"
    "   subq     $1, %%r13                          \n"
    "   kmovq    %%r13, %%k2                        \n"

    "   mov       $4, %%rdx                         \n"
    "   shlx      %%rdx, %%rsi, %%rdx               \n"

    "0:                                             \n"

    "   vmovdqu16      (%%rax), %%zmm0              \n"
    "   cmp          $2, %%r8                       \n"
    "   jl           1f                             \n"
    "   vmovdqu16      (%%rax, %%r11), %%zmm1       \n"
    "   cmp          $3, %%r8                       \n"
    "   jl           1f                             \n"
    "   vmovdqu16      (%%rax, %%r11, 2), %%zmm2    \n"

    "   cmp          $4, %%r8                       \n"
    "   jl           1f                             \n"
    "   leaq         (%%rax, %%r11, 2), %%rcx       \n"
    "   vmovdqu16      (%%rcx, %%r11), %%zmm3       \n"
    "   cmp          $5, %%r8                       \n"
    "   jl           1f                             \n"
    "   vmovdqu16      (%%rcx, %%r11, 2), %%zmm4    \n"
    "   cmp          $6, %%r8                       \n"
    "   jl           1f                             \n"
    "   leaq         (%%rcx, %%r11, 2), %%rcx       \n"
    "   vmovdqu16      (%%rcx, %%r11), %%zmm5       \n"
    "   cmp          $7, %%r8                       \n"
    "   jl           1f                             \n"
    "   vmovdqu16      (%%rcx, %%r11, 2), %%zmm6    \n"
    "   cmp          $8, %%r8                       \n"
    "   jl           1f                             \n"
    "   leaq         (%%rcx, %%r11, 2), %%rcx       \n"
    "   vmovdqu16      (%%rcx, %%r11), %%zmm7       \n"
    "   cmp          $9, %%r8                       \n"
    "   jl           1f                             \n"
    "   vmovdqu16      (%%rcx, %%r11, 2), %%zmm8    \n"
    "   cmp          $10, %%r8                      \n"
    "   jl           1f                             \n"
    "   leaq         (%%rcx, %%r11, 2), %%rcx       \n"
    "   vmovdqu16      (%%rcx, %%r11), %%zmm9       \n"

    "   cmp          $11, %%r8                      \n"
    "   jl           1f                             \n"
    "   vmovdqu16      (%%rcx, %%r11, 2), %%zmm10   \n"

    "   cmp          $12, %%r8                      \n"
    "   jl           1f                             \n"
    "   leaq         (%%rcx, %%r11, 2), %%rcx       \n"
    "   vmovdqu16      (%%rcx, %%r11), %%zmm11      \n"
    "   cmp          $13, %%r8                      \n"
    "   jl           1f                             \n"
    "   vmovdqu16      (%%rcx, %%r11, 2), %%zmm12   \n"
    "   cmp          $14, %%r8                      \n"
    "   jl           1f                             \n"
    "   leaq         (%%rcx, %%r11, 2), %%rcx       \n"
    "   vmovdqu16      (%%rcx, %%r11), %%zmm13      \n"
    "   cmp          $15, %%r8                      \n"
    "   jl           1f                             \n"
    "   vmovdqu16      (%%rcx, %%r11, 2), %%zmm14   \n"
    "   cmp          $16, %%r8                      \n"
    "   jl           1f                             \n"
    "   leaq         (%%rcx, %%r11, 2), %%rcx       \n"
    "   vmovdqu16      (%%rcx, %%r11), %%zmm15      \n"

    "1:                                             \n"

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


    "   vmovups         %%zmm24, (%%rbx)%{%%k2}      \n"
    "   vmovups         %%zmm25, (%%rbx, %%rsi, 4)%{%%k2}   \n"
    "   vmovups         %%zmm26, (%%rbx, %%rsi, 8)%{%%k2}   \n"
    "   leaq            (%%rbx, %%rsi, 8), %%r14     \n"
    "   vmovups         %%zmm27, (%%r14, %%rsi, 4)%{%%k2}   \n"

//----------------------------------------------------

    "2:                                              \n"

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

    "   vmovups         %%zmm24, (%%rbx, %%rsi)%{%%k2}      \n"
    "   leaq            (%%rbx, %%rsi), %%r14        \n"
    "   vmovups         %%zmm25, (%%r14, %%rsi, 4)%{%%k2}   \n"
    "   vmovups         %%zmm26, (%%r14, %%rsi, 8)%{%%k2}   \n"
    "   leaq            (%%r14, %%rsi, 8), %%r14     \n"
    "   vmovups         %%zmm27, (%%r14, %%rsi, 4)%{%%k2}   \n"


    "3:                                             \n"
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


    "   vmovups         %%zmm24, (%%rbx, %%rsi, 2)%{%%k2}   \n"
    "   leaq            (%%rbx, %%rsi, 2), %%r14     \n"
    "   vmovups         %%zmm25, (%%r14, %%rsi, 4)%{%%k2}   \n"
    "   vmovups         %%zmm26, (%%r14, %%rsi, 8)%{%%k2}   \n"
    "   leaq            (%%r14, %%rsi, 8), %%r14     \n"
    "   vmovups         %%zmm27, (%%r14, %%rsi, 4)%{%%k2}   \n"


    "4:                                             \n"

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


    "   leaq            (%%rbx, %%rsi, 2), %%r14     \n"
    "   leaq            (%%r14, %%rsi), %%r14        \n"
    "   vmovups         %%zmm24, (%%r14)%{%%k2}             \n"
    "   vmovups         %%zmm25, (%%r14, %%rsi, 4)%{%%k2}   \n"
    "   vmovups         %%zmm26, (%%r14, %%rsi, 8)%{%%k2}   \n"
    "   leaq            (%%r14, %%rsi, 8), %%r14     \n"
    "   vmovups         %%zmm27, (%%r14, %%rsi, 4)%{%%k2}   \n"


    "5:                                             \n"

    "   addq        $64, %%rax                      \n"
    "   addq        %%rdx, %%rbx                    \n"
    "   subq        $32, %%r12                      \n"
    "   cmp         $0, %%r12                       \n"
    "   jg          0b                              \n"


    :
    :
      [A] "m" (A),
      [Ac] "m" (Ac),
      [mr] "m" (mr),
      [K] "m" (K),
      [stride_K] "m" (stride_K)
    :"rax", "rbx", "rcx", "rdx", "rdi", "rsi","rbp","r8", "r9", "r10", "r11", "r12",
     "r13", "r14", "r15","k1","k2","k3","zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
     "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
     "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
     "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
     "zmm30", "zmm31", "memory"
    );

}