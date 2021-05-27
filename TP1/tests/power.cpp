
/*
We want to do:
long int power_c(const long int val){
   return val*val;
}
(but currently it is "return val+1;")
*/
/*
extern "C" long int power_long_int_asm(const long int val);
__asm__(
".global power_long_int_asm \n"
"power_long_int_asm: \n"
"add  $1, %rdi;\n"
"mov  %rdi, %rax;\n"
"ret;\n"
);
*/
//my power function for long int
extern "C" long int power_long_int_asm(const long int val);
__asm__(
".global power_long_int_asm \n"
"power_long_int_asm: \n"
"pushq   %rbp;\n"
"movq    %rsp, %rbp;\n"
"movq    %rdi, -8(%rbp);\n"
"movq    -8(%rbp), %rax;\n"
"imulq   %rax, %rax;\n"
"popq    %rbp;\n"
"ret;\n"
);

//my power function for double
extern "C" double power_double_asm(const double val);
__asm__(
".global power_double_asm \n"
"power_double_asm: \n"
"pushq   %rbp;\n"
"movq    %rsp, %rbp;\n"
"movsd   %xmm0, -8(%rbp);\n"
"movsd   -8(%rbp), %xmm0;\n"
"mulsd   %xmm0, %xmm0;\n"
"movq    %xmm0, %rax;\n"
"movq    %rax, %xmm0;\n"
"popq    %rbp;\n"
"ret;\n"
);

/*
We want to do:
double power_c(const double val){
   return val*val;
}
(but currently it is "return val+1;")
*/
/*
extern "C" double power_double_asm(const double val);
__asm__(
".global power_double_asm \n"
"power_double_asm: \n"
"vaddsd  .LC0_bis(%rip), %xmm0, %xmm0;\n"
"ret;\n"
".LC0_bis:\n"
".long   0\n"
".long   1072693248\n"
);
*/
#include "utils.hpp"
#include <iostream>

void test(){
    const long int TestSize = 1000;

    std::cout << "Check power_long_int_asm" << std::endl;

    for(long int idx = 0 ; idx < TestSize ; ++idx){
        CheckEqual(idx*idx,power_long_int_asm(idx));
    }

    std::cout << "Check power_double_asm" << std::endl;

    for(long int idx = 0 ; idx < TestSize ; ++idx){
        CheckEqual(double(idx)*double(idx),power_double_asm(double(idx)));
    }
}

int main(){
    test();

    return 0;
}
