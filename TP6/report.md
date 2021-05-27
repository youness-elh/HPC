Fill the current report and commit it as any other file (be compact, a few lines per Section should be enough).

# At the end of the practical work

## What worked

* **Question 5:**
Another kernel is made called distance. The matrix is then calculated using the examples of blockmat and threadmat. every sub-matrix of (2,2) is calculated by a thread. There are (2,2) threads in our case.

* **Question 6:**
The kernel of the matrix multiplication is made so that each thread computes one element of the block sub-matrix (1,1). We have (N,N) threads per block. the assigned thread (previous thread +1 of the current block) is doing the calculation of the sub-matrix (1,1). I had to be careful with the following issues:
    * CudaCpu: takes as arguments: threadsPerBlock(N, N), blocksPerGrid(1, 1)
    * cudaMemcpy: the copy direction: cudaMemcpyHostToDevice or cudaMemcpyDeviceToHost
    * get() method should be with std::unique_ptr object. 

* **Question 7:**
In this excercice the kernel is a 1D incrementing over grids the values in each thread of the sub groups of the grid. The sum is in parallele of N, then N/2 as explained in the excercise sheet and stored in the buffer which is exchanged with reinserted new table for each iteration. I had to imagine that there are many grids to be able to solve this excercice.


## What worked

## What did not work



## Final status

## What works

## What does not work

