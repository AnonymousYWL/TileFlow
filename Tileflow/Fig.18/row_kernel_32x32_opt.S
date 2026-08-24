    .text 
    .global _row_kernel_32x32_opt
    .align  4

.macro PACKB_KERNEL_M32N32_K1
    ld1w    {z0.s}, p0/z, [x24]       //packa
    ld1w    {z1.s}, p1/z, [x24, x29, lsl #2]      //packa + SVLs * KC
    ld1w    {z4.s-z7.s}, pn15/z, [x19]   //b_ptr

    fmopa   za0.s, p0/m, p2/m, z0.s, z4.s
    fmopa   za1.s, p0/m, p3/m, z0.s, z5.s
    fmopa   za2.s, p1/m, p2/m, z1.s, z4.s
    fmopa   za3.s, p1/m, p3/m, z1.s, z5.s

    st1w    {z4.s-z5.s}, pn9, [x10]        //packb
    st1w    {z6.s-z7.s}, pn14, [x10, x22, lsl #2]  //packb + 2 * SVLs * KC

    add     x19, x19, x8, lsl #2    // b_ptr+= ldb
    addvl   x24, x24, #1            // packa += SVLb
    addvl   x10, x10, #2            // packb += 2*SVLb (bytes)
.endm

.macro  PACKB_KERNEL_M32N32_K4
    ld1w    {z0.s-z3.s}, pn15/z, [x24]  //packa
    ld1w    {z4.s-z7.s}, pn15/z, [x19]  //b_ptr
    ld1w    {z8.s-z11.s}, pn15/z, [x24, x29, lsl #2]     //packa + SVLs * KC

    fmopa   za0.s, p0/m, p2/m, z0.s, z4.s
    fmopa   za1.s, p0/m, p3/m, z0.s, z5.s
    fmopa   za2.s, p1/m, p2/m, z8.s, z4.s
    fmopa   za3.s, p1/m, p3/m, z8.s, z5.s

    st1w    {z4.s-z5.s}, pn9, [x10]        //packb
    st1w    {z6.s-z7.s}, pn14, [x10, x22, lsl #2]  //packb + 2 * SVLs * KC

    add     x19, x19, x8, lsl #2    // b_ptr+= ldb
    addvl   x24, x24, #4            // packa += 4*SVLb
    addvl   x10, x10, #2            // packb += 2*SVLb (bytes)

    ld1w    {z12.s-z15.s}, pn15/z, [x19]
    fmopa   za0.s, p0/m, p2/m, z1.s, z12.s
    fmopa   za1.s, p0/m, p3/m, z1.s, z13.s
    fmopa   za2.s, p1/m, p2/m, z9.s, z12.s
    fmopa   za3.s, p1/m, p3/m, z9.s, z13.s
    
    st1w    {z12.s-z13.s}, pn9, [x10]        //packb
    st1w    {z14.s-z15.s}, pn14, [x10, x22, lsl #2]  //packb + 2 * SVLs * KC

    add     x19, x19, x8, lsl #2    // b_ptr+= ldb
    addvl   x10, x10, #2            // packb += 2*SVLb (bytes)

    ld1w    {z16.s-z19.s}, pn15/z, [x19]
    fmopa   za0.s, p0/m, p2/m, z2.s, z16.s
    fmopa   za1.s, p0/m, p3/m, z2.s, z17.s
    fmopa   za2.s, p1/m, p2/m, z10.s, z16.s
    fmopa   za3.s, p1/m, p3/m, z10.s, z17.s

    st1w    {z16.s-z17.s}, pn9, [x10]        //packb
    st1w    {z18.s-z19.s}, pn14, [x10, x22, lsl #2]  //packb + 2 * SVLs * KC

    add     x19, x19, x8, lsl #2    // b_ptr+= ldb
    addvl   x10, x10, #2            // packb += 2*SVLb (bytes)

    ld1w    {z20.s-z23.s}, pn15/z, [x19]
    fmopa   za0.s, p0/m, p2/m, z3.s, z20.s
    fmopa   za1.s, p0/m, p3/m, z3.s, z21.s
    fmopa   za2.s, p1/m, p2/m, z11.s, z20.s
    fmopa   za3.s, p1/m, p3/m, z11.s, z21.s

    st1w    {z20.s-z21.s}, pn9, [x10]        //packb
    st1w    {z22.s-z23.s}, pn14, [x10, x22, lsl #2]  //packb + 2 * SVLs * KC

    add     x19, x19, x8, lsl #2    // b_ptr+= ldb
    addvl   x10, x10, #2            // packb += 2*SVLb (bytes)
.endm

.macro KERNEL_M32N32_K1
    ld1w    {z0.s}, p0/z, [x24]    //packa 
    ld1w    {z1.s}, p1/z, [x24, x29, lsl #2]      //packa + SVLs * KC
    ld1w    {z4.s-z5.s}, pn9/z, [x10]    //packb 
    
    fmopa   za0.s, p0/m, p2/m, z0.s, z4.s
    fmopa   za1.s, p0/m, p3/m, z0.s, z5.s
    fmopa   za2.s, p1/m, p2/m, z1.s, z4.s
    fmopa   za3.s, p1/m, p3/m, z1.s, z5.s

    addvl x10, x10, #2                     // packb += 2*svls
    addvl x24, x24, #1                     // packa += svls 
.endm 

.macro KERNEL_M32N32_K4
    ld1w    {z0.s-z3.s}, pn15/z, [x24]  //packa
    ld1w    {z4.s-z7.s}, pn15/z, [x24, x29, lsl #2] //packa + SVLs * KC
    addvl   x24, x24, #4                //packa += 4*svls

    ld1w    {z8.s-z9.s}, pn9/z, [x10]   //packb 
    addvl   x10, x10, #2               // packb += 2*svls  
    fmopa   za0.s, p0/m, p2/m, z0.s, z8.s
    fmopa   za1.s, p0/m, p3/m, z0.s, z9.s
    fmopa   za2.s, p1/m, p2/m, z4.s, z8.s
    fmopa   za3.s, p1/m, p3/m, z4.s, z9.s

    ld1w    {z10.s-z11.s}, pn9/z, [x10]
    addvl   x10, x10, #2               // packb += 2*svls
    fmopa   za0.s, p0/m, p2/m, z1.s, z10.s
    fmopa   za1.s, p0/m, p3/m, z1.s, z11.s
    fmopa   za2.s, p1/m, p2/m, z5.s, z10.s
    fmopa   za3.s, p1/m, p3/m, z5.s, z11.s

    ld1w    {z12.s-z13.s}, pn9/z, [x10]
    addvl   x10, x10, #2               // packb += 2*svls
    fmopa   za0.s, p0/m, p2/m, z2.s, z12.s
    fmopa   za1.s, p0/m, p3/m, z2.s, z13.s
    fmopa   za2.s, p1/m, p2/m, z6.s, z12.s
    fmopa   za3.s, p1/m, p3/m, z6.s, z13.s

    ld1w    {z14.s-z15.s}, pn9/z, [x10]
    addvl   x10, x10, #2               // packb += 2*svls
    fmopa   za0.s, p0/m, p2/m, z3.s, z14.s
    fmopa   za1.s, p0/m, p3/m, z3.s, z15.s
    fmopa   za2.s, p1/m, p2/m, z7.s, z14.s
    fmopa   za3.s, p1/m, p3/m, z7.s, z15.s
.endm

.macro KERNEL_NEXT_M32N32_K1
    ld1w    {z0.s}, p0/z,  [x24]    //packa 
    ld1w    {z1.s}, p1/z,  [x24, x29, lsl #2] //packa + SVLs * KC
    ld1w    {z4.s-z5.s}, pn14/z,  [x30]    //packb 
    
    fmopa   za0.s, p0/m, p4/m, z0.s, z4.s
    fmopa   za1.s, p0/m, p5/m, z0.s, z5.s
    fmopa   za2.s, p1/m, p4/m, z1.s, z4.s
    fmopa   za3.s, p1/m, p5/m, z1.s, z5.s

    addvl x30, x30, #2                     // packb += 2*svls
    addvl x24, x24, #1                     // packa += svls 
.endm 

.macro  KERNEL_NEXT_M32N32_K4
    ld1w    {z0.s-z3.s}, pn15/z, [x24]  //packa
    ld1w    {z4.s-z7.s}, pn15/z, [x24, x29, lsl #2] //packa + SVLs * KC
    addvl   x24, x24, #4                //packa += 4*svls

    ld1w    {z8.s-z9.s}, pn14/z, [x30]   //packb 
    addvl   x30, x30, #2               // packb += 2*svls  
    fmopa   za0.s, p0/m, p4/m, z0.s, z8.s
    fmopa   za1.s, p0/m, p5/m, z0.s, z9.s
    fmopa   za2.s, p1/m, p4/m, z4.s, z8.s
    fmopa   za3.s, p1/m, p5/m, z4.s, z9.s

    ld1w    {z10.s-z11.s}, pn14/z, [x30]
    addvl   x30, x30, #2               // packb += 2*svls
    fmopa   za0.s, p0/m, p4/m, z1.s, z10.s
    fmopa   za1.s, p0/m, p5/m, z1.s, z11.s
    fmopa   za2.s, p1/m, p4/m, z5.s, z10.s
    fmopa   za3.s, p1/m, p5/m, z5.s, z11.s

    ld1w    {z12.s-z13.s}, pn14/z, [x30]
    addvl   x30, x30, #2               // packb += 2*svls
    fmopa   za0.s, p0/m, p4/m, z2.s, z12.s
    fmopa   za1.s, p0/m, p5/m, z2.s, z13.s
    fmopa   za2.s, p1/m, p4/m, z6.s, z12.s
    fmopa   za3.s, p1/m, p5/m, z6.s, z13.s

    ld1w    {z14.s-z15.s}, pn14/z, [x30]
    addvl   x30, x30, #2               // packb += 2*svls
    fmopa   za0.s, p0/m, p4/m, z3.s, z14.s
    fmopa   za1.s, p0/m, p5/m, z3.s, z15.s
    fmopa   za2.s, p1/m, p4/m, z7.s, z14.s
    fmopa   za3.s, p1/m, p5/m, z7.s, z15.s

.endm

.macro MACRO_LOAD_C_START
    mov     w13, #0
    mov     x11, x20                // XC_ptr
    add     x23, x20, x17, lsl #2   // XC_ptr + SVLs * ldc

LOAD_C_START:
    psel    pn10, pn9, p0.s[w13, 0]
    psel    pn11, pn9, p0.s[w13, 1]
    psel    pn12, pn9, p0.s[w13, 2]
    psel    pn13, pn9, p0.s[w13, 3]
    ld1w    {z0.s, z8.s}, pn10/z, [x11]               // c_ptr0
    ld1w    {z1.s, z9.s}, pn11/z, [x11, x8, lsl #2]   // c_ptr0 + ldc
    ld1w    {z2.s, z10.s}, pn12/z, [x11, x25, lsl #2]  // c_ptr0 + ldc*2
    ld1w    {z3.s, z11.s}, pn13/z, [x11, x15, lsl #2]  // c_ptr0 + ldc*3
    mova    za0h.s[w13,0:3], {z0.s-z3.s}
    mova    za1h.s[w13,0:3], {z8.s-z11.s}

    psel    pn10, pn9, p1.s[w13, 0]
    psel    pn11, pn9, p1.s[w13, 1]
    psel    pn12, pn9, p1.s[w13, 2]
    psel    pn13, pn9, p1.s[w13, 3]
    ld1w    {z4.s, z12.s}, pn10/z, [x23]               // c_ptr1 = c_ptr0 + SVLs * ldc
    ld1w    {z5.s, z13.s}, pn11/z, [x23, x8, lsl #2]   // c_ptr1 + ldc
    ld1w    {z6.s, z14.s}, pn12/z, [x23, x25, lsl #2]  // c_ptr1 + ldc*2
    ld1w    {z7.s, z15.s}, pn13/z, [x23, x15, lsl #2]  // c_ptr1 + ldc*3
    mova    za2h.s[w13,0:3], {z4.s-z7.s}
    mova    za3h.s[w13,0:3], {z12.s-z15.s}

    add     x11, x11, x8, lsl #4         //c_ptr0 += 4*ldc FP32 bytes
    add     x23, x23, x8, lsl #4        //c_ptr1 += 4*ldc FP32 bytes
    add     w13, w13, #4

    cmp     w13, w12
    b.mi    LOAD_C_START
.endm

.macro  MACRO_STORE_C_START
    mov     w13, #0
    mov     w14, #0
    mov     x11, x20      // XC_ptr
    add     x23, x20, x17, lsl #2   // Xc_ptr + SVLs * ldc

STORE_C_START:
    psel    pn10, pn9, p0.s[w13, #0]
    psel    pn11, pn9, p1.s[w13, #0]
    psel    pn12, pn9, p0.s[w13, #1]
    psel    pn13, pn9, p1.s[w13, #1]

    mova    {z0.b-z3.b}, za0h.b[w14, 0:3]       //za0h.s[0], za1h.s[0], za2h.s[0], za3h.s[0] 
    mova    {z4.b-z7.b}, za0h.b[w14, 4:7]

    st1w    {z0.s-z1.s},  pn10, [x11]               // c_ptr0 
    st1w    {z2.s-z3.s},  pn11, [x23]                  // c_ptr1
  
    add     x11, x11, x8, lsl #2         //c_ptr0 += ldc FP32 bytes
    add     x23, x23, x8, lsl #2         //c_ptr1 += ldc FP32 bytes

    st1w    {z4.s-z5.s},  pn12, [x11]               // c_ptr0 
    st1w    {z6.s-z7.s},  pn13, [x23]                  // c_ptr1

    add     x11, x11, x8, lsl #2         //c_ptr0 += ldc FP32 bytes
    add     x23, x23, x8, lsl #2         //c_ptr1 += ldc FP32 bytes

    add     w13, w13, #2
    add     w14, w14, #8

    cmp     w13, w12
    b.mi    STORE_C_START
.endm



.macro MACRO_LOAD_C_START_1
    mov     w13, #0
    mov     x11, x20                // XC_ptr
    add     x23, x20, x17, lsl #2   // XC_ptr + SVLs * ldc

LOAD_C_START_1:
    psel    pn10, pn9, p0.s[w13, 0]
    psel    pn11, pn9, p0.s[w13, 1]
    psel    pn12, pn9, p0.s[w13, 2]
    psel    pn13, pn9, p0.s[w13, 3]
    ld1w    {z0.s, z8.s}, pn10/z, [x11]               // c_ptr0
    ld1w    {z1.s, z9.s}, pn11/z, [x11, x8, lsl #2]   // c_ptr0 + ldc
    ld1w    {z2.s, z10.s}, pn12/z, [x11, x25, lsl #2]  // c_ptr0 + ldc*2
    ld1w    {z3.s, z11.s}, pn13/z, [x11, x15, lsl #2]  // c_ptr0 + ldc*3
    mova    za0h.s[w13,0:3], {z0.s-z3.s}
    mova    za1h.s[w13,0:3], {z8.s-z11.s}

    psel    pn10, pn9, p1.s[w13, 0]
    psel    pn11, pn9, p1.s[w13, 1]
    psel    pn12, pn9, p1.s[w13, 2]
    psel    pn13, pn9, p1.s[w13, 3]
    ld1w    {z4.s, z12.s}, pn10/z, [x23]               // c_ptr1 = c_ptr0 + SVLs * ldc
    ld1w    {z5.s, z13.s}, pn11/z, [x23, x8, lsl #2]   // c_ptr1 + ldc
    ld1w    {z6.s, z14.s}, pn12/z, [x23, x25, lsl #2]  // c_ptr1 + ldc*2
    ld1w    {z7.s, z15.s}, pn13/z, [x23, x15, lsl #2]  // c_ptr1 + ldc*3
    mova    za2h.s[w13,0:3], {z4.s-z7.s}
    mova    za3h.s[w13,0:3], {z12.s-z15.s}

    add     x11, x11, x8, lsl #4         //c_ptr0 += 4*ldc FP32 bytes
    add     x23, x23, x8, lsl #4        //c_ptr1 += 4*ldc FP32 bytes
    add     w13, w13, #4

    cmp     w13, w12
    b.mi    LOAD_C_START_1
.endm

.macro  MACRO_STORE_C_START_1
    mov     w13, #0
    mov     w14, #0
    mov     x11, x20      // XC_ptr
    add     x23, x20, x17, lsl #2   // Xc_ptr + SVLs * ldc

STORE_C_START_1:
    psel    pn10, pn9, p0.s[w13, #0]
    psel    pn11, pn9, p1.s[w13, #0]
    psel    pn12, pn9, p0.s[w13, #1]
    psel    pn13, pn9, p1.s[w13, #1]

    mova    {z0.b-z3.b}, za0h.b[w14, 0:3]       //za0h.s[0], za1h.s[0], za2h.s[0], za3h.s[0] 
    mova    {z4.b-z7.b}, za0h.b[w14, 4:7]

    st1w    {z0.s-z1.s},  pn10, [x11]               // c_ptr0 
    st1w    {z2.s-z3.s},  pn11, [x23]                  // c_ptr1
  
    add     x11, x11, x8, lsl #2         //c_ptr0 += ldc FP32 bytes
    add     x23, x23, x8, lsl #2         //c_ptr1 += ldc FP32 bytes

    st1w    {z4.s-z5.s},  pn12, [x11]               // c_ptr0 
    st1w    {z6.s-z7.s},  pn13, [x23]                  // c_ptr1

    add     x11, x11, x8, lsl #2         //c_ptr0 += ldc FP32 bytes
    add     x23, x23, x8, lsl #2         //c_ptr1 += ldc FP32 bytes

    add     w13, w13, #2
    add     w14, w14, #8

    cmp     w13, w12
    b.mi    STORE_C_START_1
.endm


.macro MACRO_LOAD_NEXT_C_START
    mov     w13, #0
    mov     x11, x20                // XC_ptr
    add     x23, x20, x17, lsl #2   // XC_ptr + SVLs * ldc

LOAD_NEXT_C_START:
    psel    pn10, pn14, p0.s[w13, 0]
    psel    pn11, pn14, p0.s[w13, 1]
    psel    pn12, pn14, p0.s[w13, 2]
    psel    pn13, pn14, p0.s[w13, 3]
    ld1w    {z0.s, z8.s}, pn10/z, [x11]               // c_ptr0
    ld1w    {z1.s, z9.s}, pn11/z, [x11, x8, lsl #2]   // c_ptr0 + ldc
    ld1w    {z2.s, z10.s}, pn12/z, [x11, x25, lsl #2]  // c_ptr0 + ldc*2
    ld1w    {z3.s, z11.s}, pn13/z, [x11, x15, lsl #2]  // c_ptr0 + ldc*3
    mova    za0h.s[w13,0:3], {z0.s-z3.s}
    mova    za1h.s[w13,0:3], {z8.s-z11.s}

    psel    pn10, pn14, p1.s[w13, 0]
    psel    pn11, pn14, p1.s[w13, 1]
    psel    pn12, pn14, p1.s[w13, 2]
    psel    pn13, pn14, p1.s[w13, 3]
    ld1w    {z4.s, z12.s}, pn10/z, [x23]               // c_ptr1 = c_ptr0 + SVLs * ldc
    ld1w    {z5.s, z13.s}, pn11/z, [x23, x8, lsl #2]   // c_ptr1 + ldc
    ld1w    {z6.s, z14.s}, pn12/z, [x23, x25, lsl #2]  // c_ptr1 + ldc*2
    ld1w    {z7.s, z15.s}, pn13/z, [x23, x15, lsl #2]  // c_ptr1 + ldc*3
    mova    za2h.s[w13,0:3], {z4.s-z7.s}
    mova    za3h.s[w13,0:3], {z12.s-z15.s}

    add     x11, x11, x8, lsl #4         //c_ptr0 += 4*ldc FP32 bytes
    add     x23, x23, x8, lsl #4        //c_ptr1 += 4*ldc FP32 bytes
    add     w13, w13, #4

    cmp     w13, w12
    b.mi    LOAD_NEXT_C_START
.endm

.macro  MACRO_STORE_NEXT_C_START
    mov     w13, #0
    mov     w14, #0
    mov     x11, x20      // XC_ptr
    add     x23, x20, x17, lsl #2   // Xc_ptr + SVLs * ldc

STORE_NEXT_C_START:
    psel    pn10, pn14, p0.s[w13, #0]
    psel    pn11, pn14, p1.s[w13, #0]
    psel    pn12, pn14, p0.s[w13, #1]
    psel    pn13, pn14, p1.s[w13, #1]

    mova    {z0.b-z3.b}, za0h.b[w14, 0:3]       //za0h.s[0], za1h.s[0], za2h.s[0], za3h.s[0] 
    mova    {z4.b-z7.b}, za0h.b[w14, 4:7]

    st1w    {z0.s-z1.s},  pn10, [x11]               // c_ptr0 
    st1w    {z2.s-z3.s},  pn11, [x23]                  // c_ptr1
  
    add     x11, x11, x8, lsl #2         //c_ptr0 += ldc FP32 bytes
    add     x23, x23, x8, lsl #2         //c_ptr1 += ldc FP32 bytes

    st1w    {z4.s-z5.s},  pn12, [x11]               // c_ptr0 
    st1w    {z6.s-z7.s},  pn13, [x23]                  // c_ptr1

    add     x11, x11, x8, lsl #2         //c_ptr0 += ldc FP32 bytes
    add     x23, x23, x8, lsl #2         //c_ptr1 += ldc FP32 bytes

    add     w13, w13, #2
    add     w14, w14, #8

    cmp     w13, w12
    b.mi    STORE_NEXT_C_START
.endm

_row_kernel_32x32_opt:
//x0:mc, x1:kc, x2:nc, x3: XB, x4: packa, x5: packb, x6:c, x7:KC, x8:ldc, x9:pc

    ldr     w8,  [sp]        // ldc = ldb = n
    ldr     w9,  [sp, #4]    // pc

    stp     x19, x20, [sp, #-96]!
    stp     x21, x22, [sp, #16]
    stp     x23, x24, [sp, #32]
    stp     x25, x26, [sp, #48]
    stp     x27, x28, [sp, #64]
    stp     x29, x30, [sp, #80]

    smstart

    cntw    x12                     //   SVLs -> 16

    lsl     x25, x8, #1   // ldc * 2
    add     x15, x25, x8  // ldc * 3

    mul     x17, x12, x8   // SVLs * ldc

    mul     x29, x12, x1    // SVLs * kc
    lsl     x22, x29, #1    // 2 * SVLs * kc

    add    x21, x3, x2, lsl #2     // Exit condition for N loop
    mov    x16, x3                  // b_base
    whilelt pn9.b, x16, x21, vlx2   // tiles predicate (N dimension)

    ptrue pn15.b

LOOP_NR:
    addvl   x27, x16, #2            // b_base + 2 * SVLs
    whilelt pn14.b, x27, x21, vlx2   // tiles predicate (N dimension)
    
    mov    x10, x5                  // packb
    mov    x20, x6                  // c_base

    pext   {p2.b, p3.b}, pn9[0]      // tiles predicate (N dimension)
    pext   {p4.b, p5.b}, pn14[0]    

    mov x26, #0
    whilelt pn8.s, x26, x0, vlx2           // tiles predicate (M dimension)
    pext   {p0.s, p1.s}, pn8[0]      // tiles predicate (M dimension)
//------------ first nr
    zero    {za}
    cmp     x9, x7
    b.lt    PACKB_KERNEL

//----------------LOAD  C---------------------
    MACRO_LOAD_C_START

PACKB_KERNEL:
    mov x19, x16                     // b_ptr = b_base
    mov x24, x4                     // packa
//---------------------KC LOOP PACKB------------------------------------------------
    lsr x23, x1, #2                 // KC/4
    and x13, x1, #0x3               // KC%4
    
    cbz   x23, 1f
PACKB_M32xN32_LOOP_K4:
    PACKB_KERNEL_M32N32_K4
    subs  x23, x23, #1
    b.ne PACKB_M32xN32_LOOP_K4

1:
    cbz   x13, 2f
PACKB_M32xN32_LOOP_K1:
    PACKB_KERNEL_M32N32_K1
    subs  x13, x13, #1
    b.ne PACKB_M32xN32_LOOP_K1     
    
2:
    //nop

//----------------STORE C
    MACRO_STORE_C_START

//next mr
    incw    x26, all, mul #2
    whilelt pn8.s, x26, x0, vlx2           // tiles predicate (M dimension)

LOOP_MR:
    add     x24, x24, x29, lsl #2          // packa += SVLs * KC  
    mov     x10, x5                       // packb 
    add     x20, x20, x17, lsl #3         // XC_ptr += 2 * SVLs * ldc

    pext   {p0.s, p1.s}, pn8[0]      // tiles predicate (M dimension)

    zero    {za}
    cmp     x9, x7
    b.lt    KERNEL_START

//----------------LOAD  C---------------------
    MACRO_LOAD_C_START_1

//---------------  
KERNEL_START:
//---------------------KC LOOP KERNEL
    lsr x23, x1, #2                 // KC/4
    and x13, x1, #0x3               // KC%4
    
    cbz   x23, 3f
KERNEL_M32xN32_LOOP_K4:
    KERNEL_M32N32_K4
    subs  x23, x23, #1
    b.ne KERNEL_M32xN32_LOOP_K4

3:
    cbz   x13, 4f
KERNEL_M32xN32_LOOP_K1:
    KERNEL_M32N32_K1
    subs  x13, x13, #1
    b.ne KERNEL_M32xN32_LOOP_K1    
    
4:
    //nop
//----------------STORE C
    MACRO_STORE_C_START_1

//------------loop mr
    incw    x26, all, mul #2
    whilelt pn8.s, x26, x0, vlx2           // tiles predicate (M dimension)
    b.first LOOP_MR

//------------ second nr
    mov x26, #0
    whilelt pn8.s, x26, x0, vlx2           // tiles predicate (M dimension)
    addvl   x20, x6, #2                   // XC_ptr += 2 * SVLs
    mov     x24, x4                       // packa

LOOP_NEXT_MR:

    pext   {p0.s, p1.s}, pn8[0]      // tiles predicate (M dimension)
    mov    x30, x10                   // packb += 2 * SVLs * KC
    
    zero    {za}
    cmp     x9, x7
    b.lt    KERNEL_NEXT_START

    //load c    
    MACRO_LOAD_NEXT_C_START

KERNEL_NEXT_START:
//---------------------KC LOOP KERNEL
    lsr x23, x1, #2                 // KC/4
    and x13, x1, #0x3               // KC%4
    
    cbz   x23, 5f
KERNEL_NEXT_M32xN32_LOOP_K4:
    KERNEL_NEXT_M32N32_K4
    subs  x23, x23, #1
    b.ne KERNEL_NEXT_M32xN32_LOOP_K4

5:
    cbz   x13, 6f
KERNEL_NEXT_M32xN32_LOOP_K1:
    KERNEL_NEXT_M32N32_K1
    subs  x13, x13, #1
    b.ne KERNEL_NEXT_M32xN32_LOOP_K1    
    
6:
    //nop

    //store c
    MACRO_STORE_NEXT_C_START

    add     x24, x24, x29, lsl #2         // packa += SVLs * KC  
    add     x20, x20, x17, lsl #3         // XC_ptr += 2 * SVLs * ldc
    
    incw    x26, all, mul #2
    whilelt pn8.s, x26, x0, vlx2           // tiles predicate (M dimension)
    b.first LOOP_NEXT_MR


//------------loop nr
    addvl   x6, x6, #4                     // c_base += 4 * SVLs

    addvl   x16, x16, #4                  // b_base + 4 * SVLs
    whilelt pn9.b, x16, x21, vlx2       // tiles predicate (N dimension)
    b.first LOOP_NR

    smstop

    ldp     x29, x30, [sp, #80]
    ldp     x27, x28, [sp, #64]
    ldp     x25, x26, [sp, #48]
    ldp     x23, x24, [sp, #32]
    ldp     x21, x22, [sp, #16]
    ldp     x19, x20, [sp], #96
    

    ret