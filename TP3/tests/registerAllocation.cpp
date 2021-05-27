#include <string>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <set>
#include <cassert>


enum InstructionType{
    ASSIGNEMENT,
    OPERATION,
    RET
};

struct Instruction{
    InstructionType type;
    
    std::string dest;
    std::string src1;
    std::string op;
    std::string src2;
    
    void print() const {
        std::cout << " >>";
        switch(type){
        case ASSIGNEMENT:
            std::cout << dest << " = " << src1 << std::endl;
            break;
        case OPERATION:
            std::cout << dest << " = " << src1 << op << src2 << std::endl;
            break;
        case RET:
            std::cout << "return " << dest << std::endl;
            break;
        default:
            std::cout << "Unknown type " << type << " : " << dest << "," << src1 << "," << op << "," << src2 << std::endl;
        }
    }
};

std::vector<Instruction> generateInstructions(){
    std::vector<Instruction> instructions;

    instructions.emplace_back(Instruction{ASSIGNEMENT, "t0", "$0", "", ""});
    instructions.emplace_back(Instruction{ASSIGNEMENT, "t1", "$10", "", ""});
    instructions.emplace_back(Instruction{OPERATION, "t3", "t0", "+", "x"});
    instructions.emplace_back(Instruction{OPERATION, "t4", "t0", "*", "t1"});
    instructions.emplace_back(Instruction{OPERATION, "t5", "t4", "+", "t0"});
    instructions.emplace_back(Instruction{OPERATION, "t6", "y", "*", "t1"});
    instructions.emplace_back(Instruction{OPERATION, "t7", "t6", "*", "t1"});
    instructions.emplace_back(Instruction{OPERATION, "t8", "t6", "*", "t1"});
    instructions.emplace_back(Instruction{OPERATION, "t9", "t0", "*", "t8"});
    instructions.emplace_back(Instruction{OPERATION, "t10", "t4", "*", "t8"});
    instructions.emplace_back(Instruction{OPERATION, "t11", "t4", "*", "t8"});
    instructions.emplace_back(Instruction{OPERATION, "t12", "t3", "*", "t7"});
    instructions.emplace_back(Instruction{OPERATION, "t13", "t12", "*", "t11"});
    instructions.emplace_back(Instruction{OPERATION, "t14", "t12", "*", "t11"});
    instructions.emplace_back(Instruction{RET, "t13", "", "", ""});

    return instructions;
}

void printAllInstructions(const std::vector<Instruction>& instructions){
    for(const auto ins : instructions){
        ins.print();
    }
}

int main(){
    std::vector<Instruction> instructions = generateInstructions();
    
    std::cout << "Original instructions:----------------" << std::endl;
    printAllInstructions(instructions);
    
    // Constant propagation
    {
        // Use it to know if a variable can be replaced by a formula
        std::unordered_map<std::string, std::string> constValues;

        //added lines

        for(const auto ins : instructions){
            if (ins.type == ASSIGNEMENT){
                constValues[ins.dest] = ins.src1;
                //std::cout << "----------: " <<ins.src1 << std::endl;
            }
        }

        for (std::vector<Instruction>::iterator it = instructions.begin(); it != instructions.end(); ++it)
        {
            //std::cout << "-------------------repalce each variable in every instruction-------------------" << (*it)<<std::endl;

            if (constValues[it->src1] !="" && constValues[it->src2] !="" ){

                it->type = ASSIGNEMENT;
                it->src1 = "$("+constValues[it->src1]+ it->op+constValues[it->src2]+")";
                it->op = "";
                it->src2 = "";
                constValues[it->dest] = it->src1;
            }

            else if (constValues[it->src1] =="" && constValues[it->src2] !="" ){

                it->src2 = constValues[it->src2];
            }

             else if (constValues[it->src1] !="" && constValues[it->src2] =="" ){
                it->src1 = constValues[it->src1];
            }

        }

        
        std::cout << "After constant propagation:" << std::endl;
        printAllInstructions(instructions);
    }
    
    // Remove unused
    {  
        std::cout << "Proceed to remove unused variables:" << std::endl;
        // Consider that "used" will store all the variables that are really used
        std::set<std::string> used;
        
        assert(instructions.size() && instructions.back().type == RET);
        
        // added lines to fill "used"
        used.insert(instructions.back().dest);
        //std::cout << "------------"<<used.size() << std::endl;
        
        //loop on the instructions
        for ( std::vector<Instruction>::iterator it = instructions.end()-1; it != instructions.begin(); --it){
            //std::cout << "------------"<<it->dest << std::endl;
            //only if the instruction contains used variables is set used
            if (used.find(it->dest) != used.end()){
               
                if ((*it).src1.rfind("$",0) != 0){
                    used.insert(it->src1); 
                }

                if ((*it).src1.rfind("$",0) != 0){ //or it->src2[0] !='$'
                    used.insert(it->src2); 
                }
            }    

        }
        {
            // We remove all variables not in "used"
            auto iterEnd = instructions.end();
            for(auto iter = instructions.begin() ; iter != iterEnd ;){
                if(used.find((*iter).dest) == used.end()){
                    std::cout << "Erase : " << (*iter).dest << std::endl;
                    instructions.erase(iter);
                    iterEnd = instructions.end();
                }
                else{
                     ++iter;
                }
            }
        }
        
        std::cout << "After removing unused variables:" << std::endl;
        printAllInstructions(instructions);
    }
        
    // // Dumb register allocations
    // {
    //     const long int NbRegistersInCPU = 3;
        
    //     auto iterEnd = instructions.end();
    //     for(auto iter = instructions.begin() ; iter != iterEnd ;){
    //         // If this is an operation
    //         if((*iter).type == OPERATION){
    //             long int cptRegUse = 0;
    //             if((*iter).src1.rfind("$",0) != 0){
    //                 iter = instructions.insert(iter, Instruction{ASSIGNEMENT, "%r" + std::to_string(cptRegUse), (*iter).src1, "", ""});                
    //                 ++iter;
    //                 (*iter).src1 = "%r" + std::to_string(cptRegUse);
    //                 cptRegUse += 1;
    //             }
                
    //             iter = instructions.insert(iter+1, Instruction{ASSIGNEMENT, (*iter).dest, "%r" + std::to_string(cptRegUse), "", ""});
    //             --iter;
    //             (*iter).dest = "%r" + std::to_string(cptRegUse);
    //             cptRegUse += 1;
    //             iterEnd = instructions.end();
    //             iter += 2;
    //             assert(cptRegUse <= NbRegistersInCPU);
    //         }
    //         else{
    //              ++iter;
    //         }
    //     }
        
    //     // TODO try to do better than this dumb algorithm
        
    //     std::cout << "After register allocation:" << std::endl;
    //     printAllInstructions(instructions);
    // }

    
    // enhanced register allocations v0: take into account the src2
    // {
    //     const long int NbRegistersInCPU = 3;
        
    //     auto iterEnd = instructions.end();
    //     for(auto iter = instructions.begin() ; iter != iterEnd ;){
    //         // If this is an operation
    //         if((*iter).type == OPERATION){
    //             long int cptRegUse = 0;
    //             if((*iter).src1.rfind("$",0) != 0){
    //                 if((*iter).src2.rfind("$",0) != 0){
    //                     iter = instructions.insert(iter, Instruction{ASSIGNEMENT, "%r" + std::to_string(cptRegUse), (*iter).src1, "", ""});                
    //                     ++iter;
    //                     (*iter).src1 = "%r" + std::to_string(cptRegUse);
    //                     cptRegUse += 1;
    //                 }
    //             }
                
    //             iter = instructions.insert(iter+1, Instruction{ASSIGNEMENT, (*iter).dest, "%r" + std::to_string(cptRegUse), "", ""});
    //             --iter;
    //             (*iter).dest = "%r" + std::to_string(cptRegUse);
    //             cptRegUse += 1;
    //             iterEnd = instructions.end();
    //             iter += 2;
    //             assert(cptRegUse <= NbRegistersInCPU);
    //         }
    //         else{
    //              ++iter;
    //         }
    //     }
        
    //     // TODO try to do better than this dumb algorithm
        
    //     std::cout << "After register allocation:" << std::endl;
    //     printAllInstructions(instructions);
    // }

    // enhanced register allocations v1: take into account variables x and y as already put in registers in compilation (cause function variables <7 are registers)
        {
        std::set<std::string> instruction_var;       
        //put destination variables in a set
        for ( std::vector<Instruction>::iterator it = instructions.end(); it != instructions.begin(); --it){
               
            instruction_var.insert(it->dest); 
        }

        const long int NbRegistersInCPU = 3;
        int count_var_func = 0;

        auto iterEnd = instructions.end();
        for(auto iter = instructions.begin() ; iter != iterEnd ;){
            // If this is an operation
            if((*iter).type == OPERATION){
                long int cptRegUse = 0;

                if((instruction_var.find((*iter).src1) == instruction_var.end()) && (instruction_var.find((*iter).src2) == instruction_var.end()) && (count_var_func<7)){ 
                    //do nothing cause the dest is aleady assigned registerOPregister;
                    iterEnd = instructions.end();
                    ++iter;
                    if((*iter).src1.rfind("$",0) == 0){
                        count_var_func++;
                    }
                        
                    if((*iter).src2.rfind("$",0) == 0){
                        count_var_func++;
                    }

                }

                else{
                    if((*iter).src1.rfind("$",0) != 0){
                        if((*iter).src2.rfind("$",0) != 0){
                            iter = instructions.insert(iter, Instruction{ASSIGNEMENT, "%r" + std::to_string(cptRegUse), (*iter).src1, "", ""});                
                            ++iter;
                            (*iter).src1 = "%r" + std::to_string(cptRegUse);
                            cptRegUse += 1;
                        }
                    }

                    iter = instructions.insert(iter+1, Instruction{ASSIGNEMENT, (*iter).dest, "%r" + std::to_string(cptRegUse), "", ""});
                    --iter;
                    (*iter).dest = "%r" + std::to_string(cptRegUse);
                    cptRegUse += 1;
                    iterEnd = instructions.end();
                    iter += 2;
                    assert(cptRegUse <= NbRegistersInCPU);
                }
            }
            else{
                 ++iter;
            }
        }
        
        // TODO try to do better than this dumb algorithm
        
        std::cout << "After register allocation:" << std::endl;
        printAllInstructions(instructions);
    }


    return 0;
}