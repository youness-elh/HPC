Fill the current report and commit it as any other file (be compact, a few lines per Section should be enough).

# At the end of the practical work

* Section 5: 
    * OMP_NUM_THREADS=4 ./simplyparallel: 
Gives expected results.The threads were assigned to cores freely based on availability.
    * OMP_PROC_BIND=true OMP_NUM_THREADS=4 ./simplyparallel: 
Gives expected results.The threads were assigned to cores according to a specific choice, a binded core.

* Section 6:
The used clause for such operation was "reduction" with "+" as identifier and "sum" as list.
* Section 7:
After creating an intuitive version, the following steps were included:

    * The second task was assigned to the current thread and not the first otherwise it will be sequential (e.g the other thread has to wait for the master).
  
    * Avoid creating tasks for n < 15.

    * Use higher priority for tasks at the top. One have to note that the argument of priority(arg) is the index of priority. the higher it is the important the task is. in order to give priority to n when it is small we used a decreasing function weighting small values of n (i.e x -> exp(-x)).

* Section 8:
    * The kernel is carrying out the following operation: w0 += r0 * r1 so the appropriate dependencies are: depend(in,r0) ,  depend(in,r1) --> depend(out,w0)
			     

## What worked

## What did not work



## Final status

## What works

## What does not work
