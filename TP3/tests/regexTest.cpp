#include <iostream>
#include <fstream>
#include <string>
#include <regex>
#include <vector>
#include <utility>

int main(int argc, char** argv){
    if(argc != 2){
        std::cout << "You must provide the source file." << std::endl;
        return -1;
    }

    
    std::string allLines;
    {
        const std::string text_file = argv[1];
        std::cout << "Will read " << text_file << std::endl;
        
        std::ifstream file (text_file);
        if (!file.is_open()){
            std::cout << "Cannot open file" << std::endl;
            return -1;
        }
    
        std::string line;
        while ( getline (file,line) ){
            allLines.append(line);
            allLines.append("\n");
        }
        file.close();  
    }
    
    {
        std::vector<std::pair<std::string, std::regex>> regexs;
        regexs.emplace_back(std::pair<std::string, std::regex>("var_decl", std::regex("(int | double |float|string).*[;]")));
        
        regexs.emplace_back(std::pair<std::string, std::regex>("decimal_number", std::regex("[0-9][0-9]*[.]?[0-9]*[E]?[+-]?[0-9]*")));
        
        regexs.emplace_back(std::pair<std::string, std::regex>("comments", std::regex("\\/\\*([\n]|[^*]|\\*[^/])*\\*\\/")));//"\\/\\*([a-zA-Z0-9\\n_]+)\\*\\/"))); 
            
        for(auto& iter : regexs){
            std::sregex_iterator next(allLines.begin(), allLines.end(), iter.second);
            std::sregex_iterator end;
            while (next != end) {
                std::smatch match = *next;
                std::cout << "Match " << iter.first << ": " <<  match.str() << "\n";
                next++;
            }
        }      
    }
    
    return 0;
}

    