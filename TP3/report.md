Fill the current report and commit it as any other file (be compact, a few lines per Section should be enough).

# At the end of the practical work
* At the lecture: Only the first question was answered. the other question were treated later.

## What worked
* **Question 5:**
It is done based on the hints given in https://regex101.com/.
* **Question 6:**
The constante values were stored in a map with the destination variable as key. There are 3 cases when checking the instructions:
    * No constante in both sources.
    * left source is constante.
    * right source is constante.
After replacing sources (variables) with constantes according to the above cases we proceed to removing the unused variables. First, The used non constante variables are stored in a set and then deleted.
* **Question 7:**
The enhancement is done on three levels:
    * When the constante value is on the source 2 
    * When both sources are constante values, the variable is considered a register
    * The arguments x, y are considered registers as long as there are no more than 6 function arguments.

## What did not work
The pipline but fixed by including the path to the .cpp file


## Final status
* All tasks are done

## What works
* All

## What does not work
* I included the scond argument of the rgextext exc file in the cmakefile so that the pipline is running properly.

