    .text
    .global _row_packa1

_row_packa1:  
// x0:mc, x1:kc, x2:matleft, x3:lda(k), x4:packa
    stp     x19, x20, [sp, #-48]!
    stp     x21, x22, [sp, #16]
    stp     x23, x24, [sp, #32]

    smstart

    zero    {za}

    cntw    x5                          // SVLs
    
    mul     x11, x5, x3                 // SVLs*lda

    mul     x21, x5, x1                 // SVLs*kc
    
    lsl     x16, x3, #1                 // 2*lda
    add     x17, x16, x3                // 3*lda

    mul     x14, x5, x5               // SVLs*SVLs
    lsl     x20, x14, #1              // 2*SVLs*SVLs
    
    mov     x8, #0                          // Loop_M counter
    whilelt pn9.s, x8, x0, vlx2             // M dimension predicate

Loop_M:
    mov     x7, x4                   // packa
    mov     x10, x2                  // XA

    pext   {p0.s, p1.s}, pn9[0]


    mov     x9, #0                   // Loop_K counter
    whilelt pn8.s, x9, x1, vlx2     // K dimension predicate-as-counter

Loop_K:
    pext   {p2.s, p3.s}, pn8[0]

    mov     x6, x10                  // XA_ptr -> load
    add     x19, x10, x11, lsl #2    // XA_ptr + SVLs*lda
    mov     w12, #0                  // Loop_load counter

Loop_load:
    psel    pn10, pn8, p0.s[w12, #0]
    psel    pn11, pn8, p0.s[w12, #1]
    psel    pn12, pn8, p0.s[w12, #2]
    psel    pn13, pn8, p0.s[w12, #3]
    ld1w    {z0.s,z8.s}, pn10/z, [x6]      // a_ptr
    ld1w    {z1.s,z9.s}, pn11/z, [x6, x3, lsl #2]  // a_ptr + lda
    ld1w    {z2.s,z10.s}, pn12/z, [x6, x16, lsl #2] // a_ptr + lda*2
    ld1w    {z3.s,z11.s}, pn13/z, [x6, x17, lsl #2] // a_ptr + lda*3
    mova    za0h.s[w12, 0:3], {z0.s-z3.s} 
    mova    za2h.s[w12, 0:3], {z8.s-z11.s}

    psel    pn10, pn8, p1.s[w12, #0]
    psel    pn11, pn8, p1.s[w12, #1]
    psel    pn12, pn8, p1.s[w12, #2]
    psel    pn13, pn8, p1.s[w12, #3]
    ld1w    {z4.s,z12.s}, pn10/z, [x19]      // a_ptr + SVLs * lda
    ld1w    {z5.s,z13.s}, pn11/z, [x19, x3, lsl #2]  // a_ptr + SVLs * lda + lda
    ld1w    {z6.s,z14.s}, pn12/z, [x19, x16, lsl #2] // a_ptr + SVLs * lda + lda*2
    ld1w    {z7.s,z15.s}, pn13/z, [x19, x17, lsl #2] // a_ptr + SVLs * lda + lda*3
    mova    za1h.s[w12, 0:3], {z4.s-z7.s} 
    mova    za3h.s[w12, 0:3], {z12.s-z15.s}

    add     w12, w12, #4                  // Loop_load counter increment
    add     x6, x6, x3, lsl #4            // a_ptr += 4*lda fp32 elems 
    add     x19, x19, x3, lsl #4          // a_ptr + SVLs * lda += 4*lda fp32 elems 

    cmp     w12, w5
    b.mi    Loop_load


    mov     w12, #0                         // Loop_store counter
Loop_store:
    psel    p4, p0, p2.s[w12, #0]
    psel    p5, p0, p3.s[w12, #0]
    psel    p6, p1, p2.s[w12, #0]
    psel    p7, p1, p3.s[w12, #0]
    
    mova    z0.s, p4/m, za0v.s[w12, #0]
    mova    z1.s, p6/m, za1v.s[w12, #0]
    mova    z2.s, p5/m, za2v.s[w12, #0]
    mova    z3.s, p7/m, za3v.s[w12, #0]

    st1w    {z0.s}, p4, [x7]                 // packa
    st1w    {z2.s}, p5, [x7, x20, lsl #2]    // packa + 2*SVLs*SVLs
    addvl   x7, x7, #1

    st1w    {z1.s}, p6, [x7]    // packa + SVLs
    st1w    {z3.s}, p7, [x7, x20, lsl #2]    // packa + SVLs + 2*SVLs*SVLs   
    addvl   x7, x7, #1           // packa +=  SVLs fp32 elems

    add     w12, w12, #1
    cmp     w12, w5
    b.mi    Loop_store


    addvl   x10, x10, #2          // a_base += 2*SVLs fp32 elems
    add     x7, x7, x14, lsl #3   // packa += 2*SVLs*SVLs fp32 elems

    incw    x9, all, mul #2
    whilelt pn8.s, x9, x1, vlx2  // K dimension predicate-as-counter
    b.first Loop_K

    add     x2, x2, x11, lsl #3           // &a += 2*SVLs*lda fp32 elems
    add     x4, x4, x21, lsl #3           // &packa += 2*SVLs*kc fp32 elems

    incw    x8, all, mul #2
    whilelt pn9.s, x8, x0, vlx2
    b.first Loop_M

    smstop        
    
    ldp     x23, x24, [sp, #32]
    ldp     x21, x22, [sp, #16]
    ldp     x19, x20, [sp], #48            

    ret