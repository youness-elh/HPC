#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sys/types.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sched.h>

#include <iostream>
#include <vector>
#include <cassert>

//added lines
#include <map>


int main(){
//added lines
	std::map<int, int> count_map; 
	std::map<int, int>::iterator itr; 

	int cpu = 0;
	int prev = 100;
 
	while(true){
		cpu = sched_getcpu();
		if (prev != cpu){
			std::cout << "move to core " << cpu << std::endl;
			count_map[cpu] =  count_map[cpu]+1; 

			//counting number of moves in cores
			std::cout << "Core count : ";
			for (itr = count_map.begin(); itr != count_map.end(); ++itr) 
			{ 
				std::cout <<"["<< itr->first << "]  " << itr->second << "     "; 
				
			} 
			std::cout << std::endl;
			prev = cpu;
			
		}
	}
	return 0;
}
