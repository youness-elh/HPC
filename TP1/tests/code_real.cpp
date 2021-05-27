/*
g++ -S -fverbose-asm -g -O2 code_real1.cpp -o code_real1.s
as -alhnd code_real1.s > code_real1.lst

OR

g++ -g -O -Wa,-aslh code_real1.cpp > code_real1.txt

*/

double just_add(double val){
   val += 1;
   return val;
}

int main(){}