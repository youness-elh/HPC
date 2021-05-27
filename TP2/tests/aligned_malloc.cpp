#include <memory>
#include <cstdlib>
#include <iostream>
#include <cassert>

#include "utils.hpp"

template <class ObjectType>
std::size_t alignementOfPtr(const ObjectType* ptr){
    if(ptr == nullptr){
        return 0;
    }
    //added lines
    size_t ptr2 = (size_t) ptr;
    int i=1;
    while ((ptr2%i)==0){	
    i*=2;
    }
    //std::cout  << i/2 << std::endl; 
    return i/2; 
}


void* custom_aligned_alloc(const std::size_t Alignement, const std::size_t inSize){
    if (inSize == 0) {
        return nullptr;
    }
    assert(Alignement != 0 && ((Alignement - 1) & Alignement) == 0);
    
    // added lines
    
    //allocate more memory than asked
    std::size_t align = Alignement;
    void * p = malloc(inSize*10); //making sure to have enough memory to find an alligned adress
    //std::cout << " --------p address---------" << p << std::endl;
    size_t ptr_to_align = size_t(p);
    //std::cout << " --------ptr to align address---------" << ptr_to_align << std::endl;

    //method 1
    // while (alignementOfPtr(ptr_to_align)!=align)
    // {
    //     ptr_to_align = ptr_to_align + 1;
    // }
    //method 2
    int mask =~(align-1);
    ptr_to_align = (size_t(ptr_to_align)+align) & mask;

    //std::cout << " --------ptr aligned address---------" << ptr_to_align << std::endl;

    // store the pointer to free 
    void * aligned_ptr = (void*) ptr_to_align ;
    void **adresssave =  (void**)((char*) aligned_ptr + sizeof(void*));//on decale de sizeof(void*) pour mettre l'adress de p
    *adresssave = p;

    return aligned_ptr;
}

void custom_free(void* ptr){

    if (ptr)
    {
        void ** ptr_to_free =  (void**)((char*)ptr + sizeof(void*));
        //std::cout << " -------------------------ptr aligned---------" << ptr << std::endl;
        void * unaligned_ptr = *ptr_to_free;
        //std::cout << " -------------------------ptr freeed---------" << unaligned_ptr << std::endl;

        free(unaligned_ptr);
    }
}

void test(){
    std::cout << "Test with hard numbers" << std::endl;
    {
        std::cout << "Address " << (1) << std::endl;
        std::cout << ">> Alignement " << alignementOfPtr(reinterpret_cast<unsigned char*>(1)) << std::endl;
        CheckEqual(1UL, alignementOfPtr(reinterpret_cast<unsigned char*>(1)));

        std::cout << "Address " << (2) << std::endl;
        std::cout << ">> Alignement " << alignementOfPtr(reinterpret_cast<unsigned char*>(2)) << std::endl;
        CheckEqual(2UL, alignementOfPtr(reinterpret_cast<unsigned char*>(2)));

        std::cout << "Address " << (4) << std::endl;
        std::cout << ">> Alignement " << alignementOfPtr(reinterpret_cast<unsigned char*>(4)) << std::endl;
        CheckEqual(4UL, alignementOfPtr(reinterpret_cast<unsigned char*>(4)));

        std::cout << "Address " << (8) << std::endl;
        std::cout << ">> Alignement " << alignementOfPtr(reinterpret_cast<unsigned char*>(8)) << std::endl;
        CheckEqual(8UL, alignementOfPtr(reinterpret_cast<unsigned char*>(8)));

        std::cout << "Address " << (6) << std::endl;
        std::cout << ">> Alignement " << alignementOfPtr(reinterpret_cast<unsigned char*>(6)) << std::endl;
        CheckEqual(2UL, alignementOfPtr(reinterpret_cast<unsigned char*>(6)));

        std::cout << "Address " << (7) << std::endl;
        std::cout << ">> Alignement " << alignementOfPtr(reinterpret_cast<unsigned char*>(7)) << std::endl;
        CheckEqual(1UL, alignementOfPtr(reinterpret_cast<unsigned char*>(7)));
    }
    std::cout << "Perform some allocations" << std::endl;
    {
        const int nbAllocs = 10;
        for(int idx = 0 ; idx < nbAllocs ; ++idx){
            std::unique_ptr<int[]> ptr(new int[141234]);
            std::cout << "Address " << ptr.get() << std::endl;
            std::cout << ">> Alignement " << alignementOfPtr(ptr.get()) << std::endl;

            std::size_t alignement = alignementOfPtr(ptr.get());
            CheckEqual(true, (std::size_t(ptr.get()) & (alignement)) != 0);
            CheckEqual(true, (std::size_t(ptr.get()) & (alignement-1)) == 0);
        }
    }
    std::cout << "Test with C11" << std::endl;
    {
        const int nbAllocs = 10;
        for(std::size_t alignment = 1 ; alignment <= 16 ; alignment *= 2){
            std::cout << "alignment = " << alignment << std::endl;
            for(int idx = 0 ; idx < nbAllocs ; ++idx){
                int* ptr = reinterpret_cast<int*>(aligned_alloc( alignment, sizeof(int)*141234));
                std::cout << "Address " << ptr << std::endl;
                std::cout << ">> Alignement " << alignementOfPtr(ptr) << std::endl;

                std::size_t alignement = alignementOfPtr(ptr);
                CheckEqual(true, (std::size_t(ptr) & (alignement)) != 0);
                CheckEqual(true, (std::size_t(ptr) & (alignement-1)) == 0);
                free(ptr);
            }
        }
    }
    std::cout << "Test with custom kernel" << std::endl;
    {
        const int nbAllocs = 10;
        for(std::size_t alignment = 1 ; alignment <= 16 ; alignment *= 2){
            std::cout << "alignment = " << alignment << std::endl;
            for(int idx = 0 ; idx < nbAllocs ; ++idx){
                int* ptr = reinterpret_cast<int*>(custom_aligned_alloc( alignment, sizeof(int)*141234));
                std::cout << "Address " << ptr << std::endl;
                std::cout << ">> Alignement " << alignementOfPtr(ptr) << std::endl;

                std::size_t alignement = alignementOfPtr(ptr);
                CheckEqual(true, (std::size_t(ptr) & (alignement)) != 0);
                CheckEqual(true, (std::size_t(ptr) & (alignement-1)) == 0);
                custom_free(ptr);
            }
        }
    }
}


int main(){
    test();

    return 0;
}
