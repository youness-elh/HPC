
Fill the current report and commit it as any other file (be compact, a few lines per Section should be enough).

# At the end of the practical work
* Section 5: 
    * lscpu: gives information about the architecture overview of the processor.
    * cat /proc/cpuinfo: gives detailed information on every processor.

* Section 6:
    * In fact the vectorization is only enabled when -O3 is added to compiler options. It is seen on the godbolt compiler from the the floating point registers which changed from %xmm0-7 to %ymm0-7.
    * Advanced Vector Extensions (AVX, also known as Sandy Bridge New Extensions) are extensions to the x86 instruction set architecture for microprocessors from Intel and AMD.
    * AVX uses sixteen YMM registers to perform a Single Instruction on Multiple pieces of Data (see SIMD).

* Section 7:
    * AVX2 widden most of SSE and AVX 128 bits to 256 bits.
    * Each YMM register can hold and do simultaneous operations (math) on:
        * eight 32-bit single-precision floating point numbers or
        * four 64-bit double-precision floating point numbers. (this our case in dot_avx2)
    * AVX_dot_aligned is obtained by the same AVX2_dot and changing loadu by load as mentionned in the excercise sheet.

* Section 8:
    * We first multiply vĺine vectors for both A and B
    * Then, we load the output line 
    * we sum on i axis the calculated vector multiplication 
    * then we have the ith line of the output
    * we store it at last

## What worked

## What did not work



## Final status

## What works

## What does not work

