#include "utils.hpp"
#include "timer.hpp"
#include <iostream>
#include <random>
#include <vector>
#include <cassert>

bool has_unique_chars(const char inSentence[], const size_t length){
    // TODO return true if all chars in inSentence are unique
}

int main(int /*argc*/, char** /*argv*/){
    {
        CheckEqual(true, has_unique_chars("abc", 3));
    }
    {
        CheckEqual(false, has_unique_chars("abca", 4));
    }
    {
        CheckEqual(true, has_unique_chars("abcA", 4));
    }
    {
        CheckEqual(false, has_unique_chars("abc--", 5));
    }
    
    return 0;
}