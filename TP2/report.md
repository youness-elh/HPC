Fill the current report and commit it as any other file (be compact, a few lines per Section should be enough).

# At the end of the practical work

## What worked
* **section 5:**
* 5.1/ ok
	* I have 8 cores so the available cores were: 0 1 2 3 4 5 6 7
	* taskset 0x00000001 (or 1 in hexadecimal) ./print_av_cores pin a single process on core 0 using the affinity mask of bits 0x00000001 = 1\*16⁰ -> 1\*2⁰. With the affinity mask  0x00000002 = 2 \* 16⁰ -> 2\*2⁰ we pin a single process on core 1. 
	* To pin two process on cores 0 and 1 we can use affinity mask = 0x00000003 = 3\*16⁰ -> 1\*2⁰+1\*2¹. 

	* To pin on 7 and 0 the chosen affinity mask is 0x00000081 = 8\*16¹ + 1\*16⁰ -> 2³\*2⁴ + 1* 2⁰.

* 5.2/ ok
	* Each process might run on a different core according to some criteria (priority etc). The loop while simulate many processes and th OS is in charge to put the process on a chosen core. The code has an account of the moves on every core. To force the OS to change the assigned core one can run other processes in parallel (run script on different terminals, open softwares, pages...).

* 5.3/ ok
	* "sched_getaffinity" gets a process's CPU affinity mask which determines the set of CPUs on which it is eligible to run. This is done in GetBinding() function followed by  "sched_setaffinity" which sets a process's CPU affinity mask in order to specify the single core within BindToCore().

* **section 6:**
* 6.1/ ok
	* Memory address p has an allignement x if this latter fulfill p%x = 0.
* 6.2/ok
	* in order to make sure that we allocate an alligned memory adress first we allocate enough memory and we look for the first alligned adress based on the givn allignement. We store this adress using a pointer of a pointer next to the found alligned pointer so we get easily and we free it later. To find the next alligned pointer to allignement two methods are possible:
		* Advancing through adjacent adresses till we find the alligned adress.
		* Or, using the fact that the alligned adress should have y-1 zeroes in binary (e.g assuming that 2**y is the allignement) and deducing a mask 11...1000..0 containing y-1 zeroes. After applying this mask to the not alligned adress, and making sure that it has at least one allignement by adding one allignemnt to the not alligned adress, we have the first alligned address in block.

## What did not work

* Something wrong with the pipeline yml file.

## Final status

* All done

## What works



## What does not work


