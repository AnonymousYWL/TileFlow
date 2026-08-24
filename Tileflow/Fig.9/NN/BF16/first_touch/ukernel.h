void mKernel_32x32x32(uint16_t *C, uint16_t* A, uint16_t* B, uint16_t* Bc,
                    long M, long K, long stride_C,
                    long stride_N,  long stride_K, long kk,
                    uint8_t *vpermt2d_control, long tag)
{

    asm volatile(


    ".macro Interleave_Store_Add                    \n"

//----------------------------------------------------------
    "   tilestored   %%tmm4, (%%rsp, %%rdx, 1)      \n"
    "   tilezero    %%tmm4                          \n"
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
   
//----------------------------------------------------
    "   tilestored   %%tmm5, (%%rsp, %%rdx, 1)      \n"
//---------------------------------------------------

    "   vmovups      (%%rsp), %%zmm16               \n"
    "   vmovups      64(%%rsp), %%zmm17             \n"
    "   vmovups      128(%%rsp), %%zmm18            \n"
    "   vmovups      192(%%rsp), %%zmm19            \n"
    "   vmovups      256(%%rsp), %%zmm20            \n"
    "   vmovups      320(%%rsp), %%zmm21            \n"
    "   vmovups      384(%%rsp), %%zmm22            \n"
    "   vmovups      448(%%rsp), %%zmm23            \n"
    "   vmovups      512(%%rsp), %%zmm24            \n"
    "   vmovups      576(%%rsp), %%zmm25            \n"
    "   vmovups      640(%%rsp), %%zmm26            \n"
    "   vmovups      704(%%rsp), %%zmm27            \n"
    "   vmovups      768(%%rsp), %%zmm28            \n"
    "   vmovups      832(%%rsp), %%zmm29            \n"
    "   vmovups      896(%%rsp), %%zmm30            \n"

    "   tilezero    %%tmm5                          \n"
  
    "   vpmovzxwd     (%%rcx), %%zmm31              \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm0, %%zmm31, %%zmm0         \n"
    "   vpmovzxwd     32(%%rcx), %%zmm31            \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm16, %%zmm31, %%zmm16       \n"
    "   vcvtne2ps2bf16   %%zmm0, %%zmm16, %%zmm0    \n"
    "   vmovdqu16    %%zmm0, (%%rcx)                \n"

    "   vpmovzxwd     (%%rcx, %%rsi, 1), %%zmm31    \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm1, %%zmm31, %%zmm1         \n"
    "   vpmovzxwd     32(%%rcx, %%rsi, 1), %%zmm31  \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm17, %%zmm31, %%zmm17       \n"
    "   vcvtne2ps2bf16   %%zmm1, %%zmm17, %%zmm1    \n"
    "   vmovdqu16    %%zmm1, (%%rcx, %%rsi, 1)      \n"


    "   vpmovzxwd     (%%rcx, %%rsi, 2), %%zmm31    \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm2, %%zmm31, %%zmm2         \n"
    "   vpmovzxwd     32(%%rcx, %%rsi, 2), %%zmm31  \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm18, %%zmm31, %%zmm18       \n"
    "   vcvtne2ps2bf16   %%zmm2, %%zmm18, %%zmm2    \n"
    "   vmovdqu16    %%zmm2, (%%rcx, %%rsi, 2)      \n"


    "   leaq        (%%rcx, %%rsi, 2), %%rcx        \n"
    "   vpmovzxwd     (%%rcx, %%rsi, 1), %%zmm31    \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm3, %%zmm31, %%zmm3         \n"
    "   vpmovzxwd     32(%%rcx, %%rsi, 1), %%zmm31  \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm19, %%zmm31, %%zmm19       \n"    
    "   vcvtne2ps2bf16   %%zmm3, %%zmm19, %%zmm3    \n"
    "   vmovdqu16    %%zmm3, (%%rcx, %%rsi, 1)      \n"


    "   vpmovzxwd     (%%rcx, %%rsi, 2), %%zmm31    \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm4, %%zmm31, %%zmm4         \n"
    "   vpmovzxwd     32(%%rcx, %%rsi, 2), %%zmm31  \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm20, %%zmm31, %%zmm20       \n" 
    "   vcvtne2ps2bf16   %%zmm4, %%zmm20, %%zmm4    \n"
    "   vmovdqu16    %%zmm4, (%%rcx, %%rsi, 2)      \n"


    "   leaq        (%%rcx, %%rsi, 2), %%rcx        \n"
    "   vpmovzxwd     (%%rcx, %%rsi, 1), %%zmm31    \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm5, %%zmm31, %%zmm5         \n"
    "   vpmovzxwd     32(%%rcx, %%rsi, 1), %%zmm31  \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm21, %%zmm31, %%zmm21       \n"   
    "   vcvtne2ps2bf16   %%zmm5, %%zmm21, %%zmm5    \n"
    "   vmovdqu16    %%zmm5, (%%rcx, %%rsi, 1)      \n"


    "   vpmovzxwd     (%%rcx, %%rsi, 2), %%zmm31    \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm6, %%zmm31, %%zmm6         \n"
    "   vpmovzxwd     32(%%rcx, %%rsi, 2), %%zmm31  \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm22, %%zmm31, %%zmm22       \n" 
    "   vcvtne2ps2bf16   %%zmm6, %%zmm22, %%zmm6    \n"
    "   vmovdqu16    %%zmm6, (%%rcx, %%rsi, 2)      \n"


    "   leaq        (%%rcx, %%rsi, 2), %%rcx        \n"
    "   vpmovzxwd     (%%rcx, %%rsi, 1), %%zmm31    \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm7, %%zmm31, %%zmm7         \n"
    "   vpmovzxwd     32(%%rcx, %%rsi, 1), %%zmm31  \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm23, %%zmm31, %%zmm23       \n"   
    "   vcvtne2ps2bf16   %%zmm7, %%zmm23, %%zmm7    \n"
    "   vmovdqu16    %%zmm7, (%%rcx, %%rsi, 1)      \n"


    "   vpmovzxwd     (%%rcx, %%rsi, 2), %%zmm31    \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm8, %%zmm31, %%zmm8         \n"
    "   vpmovzxwd     32(%%rcx, %%rsi, 2), %%zmm31  \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm24, %%zmm31, %%zmm24        \n" 
    "   vcvtne2ps2bf16   %%zmm8, %%zmm24, %%zmm8    \n"
    "   vmovdqu16    %%zmm8, (%%rcx, %%rsi, 2)      \n"


    "   leaq        (%%rcx, %%rsi, 2), %%rcx        \n"
    "   vpmovzxwd     (%%rcx, %%rsi, 1), %%zmm31    \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm9, %%zmm31, %%zmm9         \n"
    "   vpmovzxwd     32(%%rcx, %%rsi, 1), %%zmm31  \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm25, %%zmm31, %%zmm25       \n"  
    "   vcvtne2ps2bf16   %%zmm9, %%zmm25, %%zmm9    \n"
    "   vmovdqu16    %%zmm9, (%%rcx, %%rsi, 1)      \n"

    "   vpmovzxwd     (%%rcx, %%rsi, 2), %%zmm31    \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm10, %%zmm31, %%zmm10       \n"
    "   vpmovzxwd     32(%%rcx, %%rsi, 2), %%zmm31  \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm26, %%zmm31, %%zmm26       \n" 
    "   vcvtne2ps2bf16   %%zmm10, %%zmm26, %%zmm10  \n"
    "   vmovdqu16    %%zmm10, (%%rcx, %%rsi, 2)     \n"


    "   leaq        (%%rcx, %%rsi, 2), %%rcx        \n"
    "   vpmovzxwd     (%%rcx, %%rsi, 1), %%zmm31    \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm11, %%zmm31, %%zmm11       \n"
    "   vpmovzxwd     32(%%rcx, %%rsi, 1), %%zmm31  \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm27, %%zmm31, %%zmm27       \n"  
    "   vcvtne2ps2bf16   %%zmm11, %%zmm27, %%zmm11  \n"
    "   vmovdqu16    %%zmm11, (%%rcx, %%rsi, 1)     \n"


    "   vpmovzxwd     (%%rcx, %%rsi, 2), %%zmm31    \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm12, %%zmm31, %%zmm12       \n"
    "   vpmovzxwd     32(%%rcx, %%rsi, 2), %%zmm31  \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm28, %%zmm31, %%zmm28       \n"
    "   vcvtne2ps2bf16   %%zmm12, %%zmm28, %%zmm12  \n"
    "   vmovdqu16    %%zmm12, (%%rcx, %%rsi, 2)     \n"


    "   leaq        (%%rcx, %%rsi, 2), %%rcx        \n"
    "   vpmovzxwd     (%%rcx, %%rsi, 1), %%zmm31    \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm13, %%zmm31, %%zmm13       \n"
    "   vpmovzxwd     32(%%rcx, %%rsi, 1), %%zmm31  \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm29, %%zmm31, %%zmm29       \n"  
    "   vcvtne2ps2bf16   %%zmm13, %%zmm29, %%zmm13  \n"
    "   vmovdqu16    %%zmm13, (%%rcx, %%rsi, 1)     \n"


    "   vpmovzxwd     (%%rcx, %%rsi, 2), %%zmm31    \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm14, %%zmm31, %%zmm14       \n"
    "   vpmovzxwd     32(%%rcx, %%rsi, 2), %%zmm31  \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm30, %%zmm31, %%zmm30       \n"
    "   vcvtne2ps2bf16   %%zmm14, %%zmm30, %%zmm14  \n"
    "   vmovdqu16    %%zmm14, (%%rcx, %%rsi, 2)     \n"


    "   vmovups      960(%%rsp), %%zmm30            \n"//note
    "   leaq        (%%rcx, %%rsi, 2), %%rcx        \n"
    "   vpmovzxwd     (%%rcx, %%rsi, 1), %%zmm31    \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm15, %%zmm31, %%zmm15       \n"
    "   vpmovzxwd     32(%%rcx, %%rsi, 1), %%zmm31  \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm30, %%zmm31, %%zmm30       \n"  
    "   vcvtne2ps2bf16   %%zmm15, %%zmm30, %%zmm15  \n"
    "   vmovdqu16    %%zmm15, (%%rcx, %%rsi, 1)     \n"

//---------------------------------------------------------------

    "   leaq        (%%rcx, %%rsi, 2), %%rcx        \n"
//----------------------------------------------------------
    "   tilestored   %%tmm6, (%%rsp, %%rdx, 1)      \n"
    "   tilezero    %%tmm6                          \n"   
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

//---------------------------------------------------
    "   tilestored   %%tmm7, (%%rsp, %%rdx, 1)      \n"
//---------------------------------------------------

    "   vmovups      (%%rsp), %%zmm16               \n"
    
    "   vmovups      64(%%rsp), %%zmm17             \n"
    "   vmovups      128(%%rsp), %%zmm18            \n"
    "   vmovups      192(%%rsp), %%zmm19            \n"
    "   vmovups      256(%%rsp), %%zmm20            \n"
    "   vmovups      320(%%rsp), %%zmm21            \n"
    "   vmovups      384(%%rsp), %%zmm22            \n"
    "   vmovups      448(%%rsp), %%zmm23            \n"
    "   vmovups      512(%%rsp), %%zmm24            \n"
    "   vmovups      576(%%rsp), %%zmm25            \n"
    "   vmovups      640(%%rsp), %%zmm26            \n"
    "   vmovups      704(%%rsp), %%zmm27            \n"
    "   vmovups      768(%%rsp), %%zmm28            \n"
    "   vmovups      832(%%rsp), %%zmm29            \n"
    "   vmovups      896(%%rsp), %%zmm30            \n"

    "   tilezero    %%tmm7                          \n"
  
    "   vpmovzxwd     (%%rcx), %%zmm31              \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm0, %%zmm31, %%zmm0         \n"
    "   vpmovzxwd     32(%%rcx), %%zmm31            \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm16, %%zmm31, %%zmm16       \n"
    "   vcvtne2ps2bf16   %%zmm0, %%zmm16, %%zmm0    \n"
    "   vmovdqu16    %%zmm0, (%%rcx)                \n"

    "   vpmovzxwd     (%%rcx, %%rsi, 1), %%zmm31    \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm1, %%zmm31, %%zmm1         \n"
    "   vpmovzxwd     32(%%rcx, %%rsi, 1), %%zmm31  \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm17, %%zmm31, %%zmm17       \n"
    "   vcvtne2ps2bf16   %%zmm1, %%zmm17, %%zmm1    \n"
    "   vmovdqu16    %%zmm1, (%%rcx, %%rsi, 1)      \n"


    "   vpmovzxwd     (%%rcx, %%rsi, 2), %%zmm31    \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm2, %%zmm31, %%zmm2         \n"
    "   vpmovzxwd     32(%%rcx, %%rsi, 2), %%zmm31  \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm18, %%zmm31, %%zmm18       \n"
    "   vcvtne2ps2bf16   %%zmm2, %%zmm18, %%zmm2    \n"
    "   vmovdqu16    %%zmm2, (%%rcx, %%rsi, 2)      \n"


    "   leaq        (%%rcx, %%rsi, 2), %%rcx        \n"
    "   vpmovzxwd     (%%rcx, %%rsi, 1), %%zmm31    \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm3, %%zmm31, %%zmm3         \n"
    "   vpmovzxwd     32(%%rcx, %%rsi, 1), %%zmm31  \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm19, %%zmm31, %%zmm19       \n"    
    "   vcvtne2ps2bf16   %%zmm3, %%zmm19, %%zmm3    \n"
    "   vmovdqu16    %%zmm3, (%%rcx, %%rsi, 1)      \n"


    "   vpmovzxwd     (%%rcx, %%rsi, 2), %%zmm31    \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm4, %%zmm31, %%zmm4         \n"
    "   vpmovzxwd     32(%%rcx, %%rsi, 2), %%zmm31  \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm20, %%zmm31, %%zmm20       \n" 
    "   vcvtne2ps2bf16   %%zmm4, %%zmm20, %%zmm4    \n"
    "   vmovdqu16    %%zmm4, (%%rcx, %%rsi, 2)      \n"


    "   leaq        (%%rcx, %%rsi, 2), %%rcx        \n"
    "   vpmovzxwd     (%%rcx, %%rsi, 1), %%zmm31    \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm5, %%zmm31, %%zmm5         \n"
    "   vpmovzxwd     32(%%rcx, %%rsi, 1), %%zmm31  \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm21, %%zmm31, %%zmm21       \n"   
    "   vcvtne2ps2bf16   %%zmm5, %%zmm21, %%zmm5    \n"
    "   vmovdqu16    %%zmm5, (%%rcx, %%rsi, 1)      \n"


    "   vpmovzxwd     (%%rcx, %%rsi, 2), %%zmm31    \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm6, %%zmm31, %%zmm6         \n"
    "   vpmovzxwd     32(%%rcx, %%rsi, 2), %%zmm31  \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm22, %%zmm31, %%zmm22       \n" 
    "   vcvtne2ps2bf16   %%zmm6, %%zmm22, %%zmm6    \n"
    "   vmovdqu16    %%zmm6, (%%rcx, %%rsi, 2)      \n"


    "   leaq        (%%rcx, %%rsi, 2), %%rcx        \n"
    "   vpmovzxwd     (%%rcx, %%rsi, 1), %%zmm31    \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm7, %%zmm31, %%zmm7         \n"
    "   vpmovzxwd     32(%%rcx, %%rsi, 1), %%zmm31  \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm23, %%zmm31, %%zmm23       \n"   
    "   vcvtne2ps2bf16   %%zmm7, %%zmm23, %%zmm7    \n"
    "   vmovdqu16    %%zmm7, (%%rcx, %%rsi, 1)      \n"


    "   vpmovzxwd     (%%rcx, %%rsi, 2), %%zmm31    \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm8, %%zmm31, %%zmm8         \n"
    "   vpmovzxwd     32(%%rcx, %%rsi, 2), %%zmm31  \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm24, %%zmm31, %%zmm24        \n" 
    "   vcvtne2ps2bf16   %%zmm8, %%zmm24, %%zmm8    \n"
    "   vmovdqu16    %%zmm8, (%%rcx, %%rsi, 2)      \n"


    "   leaq        (%%rcx, %%rsi, 2), %%rcx        \n"
    "   vpmovzxwd     (%%rcx, %%rsi, 1), %%zmm31    \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm9, %%zmm31, %%zmm9         \n"
    "   vpmovzxwd     32(%%rcx, %%rsi, 1), %%zmm31  \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm25, %%zmm31, %%zmm25       \n"  
    "   vcvtne2ps2bf16   %%zmm9, %%zmm25, %%zmm9    \n"
    "   vmovdqu16    %%zmm9, (%%rcx, %%rsi, 1)      \n"

    "   vpmovzxwd     (%%rcx, %%rsi, 2), %%zmm31    \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm10, %%zmm31, %%zmm10       \n"
    "   vpmovzxwd     32(%%rcx, %%rsi, 2), %%zmm31  \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm26, %%zmm31, %%zmm26       \n" 
    "   vcvtne2ps2bf16   %%zmm10, %%zmm26, %%zmm10  \n"
    "   vmovdqu16    %%zmm10, (%%rcx, %%rsi, 2)     \n"


    "   leaq        (%%rcx, %%rsi, 2), %%rcx        \n"
    "   vpmovzxwd     (%%rcx, %%rsi, 1), %%zmm31    \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm11, %%zmm31, %%zmm11       \n"
    "   vpmovzxwd     32(%%rcx, %%rsi, 1), %%zmm31  \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm27, %%zmm31, %%zmm27       \n"  
    "   vcvtne2ps2bf16   %%zmm11, %%zmm27, %%zmm11  \n"
    "   vmovdqu16    %%zmm11, (%%rcx, %%rsi, 1)     \n"


    "   vpmovzxwd     (%%rcx, %%rsi, 2), %%zmm31    \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm12, %%zmm31, %%zmm12       \n"
    "   vpmovzxwd     32(%%rcx, %%rsi, 2), %%zmm31  \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm28, %%zmm31, %%zmm28       \n"
    "   vcvtne2ps2bf16   %%zmm12, %%zmm28, %%zmm12  \n"
    "   vmovdqu16    %%zmm12, (%%rcx, %%rsi, 2)     \n"


    "   leaq        (%%rcx, %%rsi, 2), %%rcx        \n"
    "   vpmovzxwd     (%%rcx, %%rsi, 1), %%zmm31    \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm13, %%zmm31, %%zmm13       \n"
    "   vpmovzxwd     32(%%rcx, %%rsi, 1), %%zmm31  \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm29, %%zmm31, %%zmm29       \n"  
    "   vcvtne2ps2bf16   %%zmm13, %%zmm29, %%zmm13  \n"
    "   vmovdqu16    %%zmm13, (%%rcx, %%rsi, 1)     \n"


    "   vpmovzxwd     (%%rcx, %%rsi, 2), %%zmm31    \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm14, %%zmm31, %%zmm14       \n"
    "   vpmovzxwd     32(%%rcx, %%rsi, 2), %%zmm31  \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm30, %%zmm31, %%zmm30       \n"
    "   vcvtne2ps2bf16   %%zmm14, %%zmm30, %%zmm14  \n"
    "   vmovdqu16    %%zmm14, (%%rcx, %%rsi, 2)     \n"


    "   vmovups      960(%%rsp), %%zmm30            \n"//note
    "   leaq        (%%rcx, %%rsi, 2), %%rcx        \n"
    "   vpmovzxwd     (%%rcx, %%rsi, 1), %%zmm31    \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm15, %%zmm31, %%zmm15       \n"
    "   vpmovzxwd     32(%%rcx, %%rsi, 1), %%zmm31  \n"
    "   vpslld        $16, %%zmm31, %%zmm31         \n"
    "   vaddps      %%zmm30, %%zmm31, %%zmm30       \n"  
    "   vcvtne2ps2bf16   %%zmm15, %%zmm30, %%zmm15  \n"
    "   vmovdqu16    %%zmm15, (%%rcx, %%rsi, 1)     \n"
    "   leaq        (%%rcx, %%rsi, 2), %%rcx        \n"

    ".endm                                          \n"


    ".macro Interleave_Store                        \n"

//----------------------------------------------------------
    "   tilestored   %%tmm4, (%%rsp, %%rdx, 1)      \n"
    "   tilezero    %%tmm4                          \n"   
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
//----------------------------------------------------
    "   tilestored   %%tmm5, (%%rsp, %%rdx, 1)      \n"
    "   tilezero    %%tmm5                          \n"
//---------------------------------------------------

    "   vmovups      (%%rsp), %%zmm16               \n"
    "   vmovups      64(%%rsp), %%zmm17             \n"
    "   vmovups      128(%%rsp), %%zmm18            \n"
    "   vmovups      192(%%rsp), %%zmm19            \n"
    "   vmovups      256(%%rsp), %%zmm20            \n"
    "   vmovups      320(%%rsp), %%zmm21            \n"
    "   vmovups      384(%%rsp), %%zmm22            \n"
    "   vmovups      448(%%rsp), %%zmm23            \n"
    "   vmovups      512(%%rsp), %%zmm24            \n"
    "   vmovups      576(%%rsp), %%zmm25            \n"
    "   vmovups      640(%%rsp), %%zmm26            \n"
    "   vmovups      704(%%rsp), %%zmm27            \n"
    "   vmovups      768(%%rsp), %%zmm28            \n"
    "   vmovups      832(%%rsp), %%zmm29            \n"
    "   vmovups      896(%%rsp), %%zmm30            \n"
  
    "   vcvtne2ps2bf16   %%zmm0, %%zmm16, %%zmm0    \n"
    "   vmovdqu16    %%zmm0, (%%rcx)                \n"
    "   vcvtne2ps2bf16   %%zmm1, %%zmm17, %%zmm1    \n"
    "   vmovdqu16    %%zmm1, (%%rcx, %%rsi, 1)      \n"
    "   vcvtne2ps2bf16   %%zmm2, %%zmm18, %%zmm2    \n"
    "   vmovdqu16    %%zmm2, (%%rcx, %%rsi, 2)      \n"

    "   leaq        (%%rcx, %%rsi, 2), %%rcx        \n"
    "   vcvtne2ps2bf16   %%zmm3, %%zmm19, %%zmm3    \n"
    "   vmovdqu16    %%zmm3, (%%rcx, %%rsi, 1)      \n"
    "   vcvtne2ps2bf16   %%zmm4, %%zmm20, %%zmm4    \n"
    "   vmovdqu16    %%zmm4, (%%rcx, %%rsi, 2)      \n"
    "   leaq        (%%rcx, %%rsi, 2), %%rcx        \n"
    "   vcvtne2ps2bf16   %%zmm5, %%zmm21, %%zmm5    \n"
    "   vmovdqu16    %%zmm5, (%%rcx, %%rsi, 1)      \n"
    "   vcvtne2ps2bf16   %%zmm6, %%zmm22, %%zmm6    \n"
    "   vmovdqu16    %%zmm6, (%%rcx, %%rsi, 2)      \n"


    "   leaq        (%%rcx, %%rsi, 2), %%rcx        \n"
    "   vcvtne2ps2bf16   %%zmm7, %%zmm23, %%zmm7    \n"
    "   vmovdqu16    %%zmm7, (%%rcx, %%rsi, 1)      \n"
    "   vcvtne2ps2bf16   %%zmm8, %%zmm24, %%zmm8    \n"
    "   vmovdqu16    %%zmm8, (%%rcx, %%rsi, 2)      \n"
    "   leaq        (%%rcx, %%rsi, 2), %%rcx        \n"
    "   vcvtne2ps2bf16   %%zmm9, %%zmm25, %%zmm9    \n"
    "   vmovdqu16    %%zmm9, (%%rcx, %%rsi, 1)      \n"
    "   vcvtne2ps2bf16   %%zmm10, %%zmm26, %%zmm10  \n"
    "   vmovdqu16    %%zmm10, (%%rcx, %%rsi, 2)     \n"


    "   leaq        (%%rcx, %%rsi, 2), %%rcx        \n"
    "   vcvtne2ps2bf16   %%zmm11, %%zmm27, %%zmm11  \n"
    "   vmovdqu16    %%zmm11, (%%rcx, %%rsi, 1)     \n"
    "   vcvtne2ps2bf16   %%zmm12, %%zmm28, %%zmm12  \n"
    "   vmovdqu16    %%zmm12, (%%rcx, %%rsi, 2)     \n"
    "   leaq        (%%rcx, %%rsi, 2), %%rcx        \n"
    "   vcvtne2ps2bf16   %%zmm13, %%zmm29, %%zmm13  \n"
    "   vmovdqu16    %%zmm13, (%%rcx, %%rsi, 1)     \n"
    "   vcvtne2ps2bf16   %%zmm14, %%zmm30, %%zmm14  \n"
    "   vmovdqu16    %%zmm14, (%%rcx, %%rsi, 2)     \n"

    "   vmovups      960(%%rsp), %%zmm30            \n"//note
    "   leaq        (%%rcx, %%rsi, 2), %%rcx        \n"
    "   vcvtne2ps2bf16   %%zmm15, %%zmm30, %%zmm15  \n"
    "   vmovdqu16    %%zmm15, (%%rcx, %%rsi, 1)     \n"
    "   leaq        (%%rcx, %%rsi, 2), %%rcx        \n"

//----------------------------------------------------------
    "   tilestored   %%tmm6, (%%rsp, %%rdx, 1)      \n"
    "   tilezero    %%tmm6                          \n"   
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

//---------------------------------------------------
    "   tilestored   %%tmm7, (%%rsp, %%rdx, 1)      \n"
    "   tilezero    %%tmm7                          \n"
//---------------------------------------------------

    "   vmovups      (%%rsp), %%zmm16               \n"
    "   vmovups      64(%%rsp), %%zmm17             \n"
    "   vmovups      128(%%rsp), %%zmm18            \n"
    "   vmovups      192(%%rsp), %%zmm19            \n"
    "   vmovups      256(%%rsp), %%zmm20            \n"
    "   vmovups      320(%%rsp), %%zmm21            \n"
    "   vmovups      384(%%rsp), %%zmm22            \n"
    "   vmovups      448(%%rsp), %%zmm23            \n"
    "   vmovups      512(%%rsp), %%zmm24            \n"
    "   vmovups      576(%%rsp), %%zmm25            \n"
    "   vmovups      640(%%rsp), %%zmm26            \n"
    "   vmovups      704(%%rsp), %%zmm27            \n"
    "   vmovups      768(%%rsp), %%zmm28            \n"
    "   vmovups      832(%%rsp), %%zmm29            \n"
    "   vmovups      896(%%rsp), %%zmm30            \n"
    "   vmovups      960(%%rsp), %%zmm31            \n"
  
    "   vcvtne2ps2bf16   %%zmm0, %%zmm16, %%zmm0    \n"
    "   vmovdqu16    %%zmm0, (%%rcx)                \n"
    "   vcvtne2ps2bf16   %%zmm1, %%zmm17, %%zmm1    \n"
    "   vmovdqu16    %%zmm1, (%%rcx, %%rsi, 1)      \n"
    "   vcvtne2ps2bf16   %%zmm2, %%zmm18, %%zmm2    \n"
    "   vmovdqu16    %%zmm2, (%%rcx, %%rsi, 2)      \n"

    "   leaq        (%%rcx, %%rsi, 2), %%rcx        \n"
    "   vcvtne2ps2bf16   %%zmm3, %%zmm19, %%zmm3    \n"
    "   vmovdqu16    %%zmm3, (%%rcx, %%rsi, 1)      \n"
    "   vcvtne2ps2bf16   %%zmm4, %%zmm20, %%zmm4    \n"
    "   vmovdqu16    %%zmm4, (%%rcx, %%rsi, 2)      \n"
    "   leaq        (%%rcx, %%rsi, 2), %%rcx        \n"
    "   vcvtne2ps2bf16   %%zmm5, %%zmm21, %%zmm5    \n"
    "   vmovdqu16    %%zmm5, (%%rcx, %%rsi, 1)      \n"
    "   vcvtne2ps2bf16   %%zmm6, %%zmm22, %%zmm6    \n"
    "   vmovdqu16    %%zmm6, (%%rcx, %%rsi, 2)      \n"


    "   leaq        (%%rcx, %%rsi, 2), %%rcx        \n"
    "   vcvtne2ps2bf16   %%zmm7, %%zmm23, %%zmm7    \n"
    "   vmovdqu16    %%zmm7, (%%rcx, %%rsi, 1)      \n"
    "   vcvtne2ps2bf16   %%zmm8, %%zmm24, %%zmm8    \n"
    "   vmovdqu16    %%zmm8, (%%rcx, %%rsi, 2)      \n"
    "   leaq        (%%rcx, %%rsi, 2), %%rcx        \n"
    "   vcvtne2ps2bf16   %%zmm9, %%zmm25, %%zmm9    \n"
    "   vmovdqu16    %%zmm9, (%%rcx, %%rsi, 1)      \n"
    "   vcvtne2ps2bf16   %%zmm10, %%zmm26, %%zmm10  \n"
    "   vmovdqu16    %%zmm10, (%%rcx, %%rsi, 2)     \n"


    "   leaq        (%%rcx, %%rsi, 2), %%rcx        \n"
    "   vcvtne2ps2bf16   %%zmm11, %%zmm27, %%zmm11  \n"
    "   vmovdqu16    %%zmm11, (%%rcx, %%rsi, 1)     \n"
    "   vcvtne2ps2bf16   %%zmm12, %%zmm28, %%zmm12  \n"
    "   vmovdqu16    %%zmm12, (%%rcx, %%rsi, 2)     \n"
    "   leaq        (%%rcx, %%rsi, 2), %%rcx        \n"
    "   vcvtne2ps2bf16   %%zmm13, %%zmm29, %%zmm13  \n"
    "   vmovdqu16    %%zmm13, (%%rcx, %%rsi, 1)     \n"
    "   vcvtne2ps2bf16   %%zmm14, %%zmm30, %%zmm14  \n"
    "   vmovdqu16    %%zmm14, (%%rcx, %%rsi, 2)     \n"
    "   leaq        (%%rcx, %%rsi, 2), %%rcx        \n"
    "   vcvtne2ps2bf16   %%zmm15, %%zmm31, %%zmm15  \n"
    "   vmovdqu16    %%zmm15, (%%rcx, %%rsi, 1)     \n"
    "   leaq        (%%rcx, %%rsi, 2), %%rcx        \n"

    ".endm                                          \n"



    ".macro PACK_FMA                                \n"

    "   mov         $16, %%r14                      \n"

    "   tileloadd (%%rax, %%r11, 1), %%tmm0       \n"
    "   addq        $64, %%rax                      \n"
    "   tileloadd   (%%rdi, %%r11, 1), %%tmm1     \n"
    "   addq        $64, %%rdi                      \n"

//-----------------------------------------------------
    "BEGIN_PACK_K16:                                \n"

    "   vmovdqu16   (%%rbx), %%zmm0                 \n"
    "   vmovdqu16   64(%%rbx), %%zmm4               \n"
    "   vmovdqu16   (%%rbx, %%r8, 1), %%zmm1        \n"
    "   vmovdqu16   64(%%rbx, %%r8, 1), %%zmm5      \n"

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
    "   leaq        (%%rbx, %%r8, 2), %%rbx        \n"

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
    "   mov       %[stride_N], %%r13                \n"
    "   mov       %[stride_C], %%rsi                \n"
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

    "   mov      %%rsp, %%r9                        \n"
    "   and      $63, %%r9                          \n"
    "   addq     $1024, %%r9                        \n"
    "   subq     %%r9, %%rsp                        \n"
    "   kmovq    %%r9, %%k2                         \n"

    "   kmovq    %%r13, %%k3                        \n"

    "   tilezero    %%tmm4                          \n"   
    "   tilezero    %%tmm5                          \n"
    "   tilezero    %%tmm6                          \n"
    "   tilezero    %%tmm7                          \n"

    "   cmp     $0, %%r14                           \n"
    "   jg      BEGIN_M                             \n"

    "BEGIN_PACK:                                    \n"

    "   mov         %%r10, %%r13                    \n"
    "   mov         %%rbp, %%rax                    \n"
    "   leaq        (%%rax, %%r11, 8), %%rdi        \n"
    "   leaq        (%%rdi, %%r11, 8), %%rdi        \n"
    
    "   kmovq       %%k1, %%r9                      \n"    //K

    "   kmovq       %%r8, %%k4                      \n" //Mc
    "   kmovq       %%k3, %%r8                      \n"

   "BEGIN_PACK_K:                                   \n"

    "   subq        $32, %%r9                       \n"
    "   PACK_FMA                                    \n"
    "   cmpq        $0, %%r9                        \n"
    "   jg          BEGIN_PACK_K                    \n"

    "   kmovq       %%k4, %%r8                      \n"
    "   jmp         Store32x32                      \n"

    "BEGIN_M:                                       \n"
    
    "   mov         %%r10, %%rbx                    \n"

    "   mov         %%rbp, %%rax                    \n"
    "   leaq        (%%rax, %%r11, 8), %%rdi        \n"
    "   leaq        (%%rdi, %%r11, 8), %%rdi        \n"

    "   kmovq       %%k1, %%r9                      \n"    //K
//-----------------------------------------------------------
    "BEGIN_K:                                       \n"

    "   subq        $32, %%r9                       \n"
    "   tileloadd   (%%rax, %%r11, 1), %%tmm0       \n"
    "   addq        $64, %%rax                      \n"
    "   tileloadd   (%%rbx, %%rdx, 1), %%tmm2       \n"
    "   tdpbf16ps   %%tmm2, %%tmm0, %%tmm4          \n"
    "   tileloadd   1024(%%rbx, %%rdx, 1), %%tmm3   \n"
    "   addq        $2048, %%rbx                    \n"
    "   tdpbf16ps   %%tmm3, %%tmm0, %%tmm5          \n"
    "   tileloadd   (%%rdi, %%r11, 1), %%tmm1       \n"
    "   addq        $64, %%rdi                      \n"
    "   tdpbf16ps   %%tmm2, %%tmm1, %%tmm6          \n"
    "   tdpbf16ps   %%tmm3, %%tmm1, %%tmm7          \n"

    "   cmpq        $0, %%r9                        \n"
    "   jg          BEGIN_K                         \n"

    "Store32x32:                                    \n"


    "   cmp        $0, %%r15                        \n"
    "   je         Store                            \n"
    "   Interleave_Store_Add                        \n"
    "   jmp         END_Store                       \n"
    "Store:                                         \n"
    "   Interleave_Store                            \n"
    "END_Store:                                     \n"
//------------------------------------------------------
    "   leaq        (%%rbp, %%r11, 8), %%rbp        \n"
    "   leaq        (%%rbp, %%r11, 8), %%rbp        \n"
    "   leaq        (%%rbp, %%r11, 8), %%rbp        \n"
    "   leaq        (%%rbp, %%r11, 8), %%rbp        \n"
    
    "   subq    $32, %%r8                           \n"
    "   cmp     $0, %%r8                            \n"
    "   jg      BEGIN_M                             \n"


    "   kmovq       %%k2, %%r12                     \n"
    "   addq        %%r12, %%rsp                    \n"

    
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
      [stride_C] "m" (stride_C),
      [kk] "m" (kk),
      [vpermt2d_control] "m" (vpermt2d_control),
      [tag] "m" (tag)
    :"rax", "rbx", "rcx", "rdx", "rdi", "rsi","rbp","r8", "r9", "r10", "r11", "r12",
     "r13", "r14", "r15","k1","k2","k3","k4","zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
     "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
     "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
     "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
     "zmm30", "zmm31", "memory"
    );

}
