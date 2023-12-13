Welcome to Induced Block Funtions and Matchings (IBloFunMatch)!!!

Installation instructions:
==========================
Basic Requirements
------------------
- Have a C++ compiler installed (e.g. GNU compiler)
- Have Python installed with numpy, matplotlib,...
- Make sure you have already installed:
  CGAL, mpfr, gmp and boost

Add local copies of header files from PerMoVec and GUDHI
--------------------------------------------------------
- Download the "PerMoVEC" bitbucket repository inside the "ext" folder of "IBloFunMatch"
- Clone the GUDHI repository "https://github.com/GUDHI/gudhi-devel" somewhere in your computer. (*)
- Take the "Flag_complex_edge_collapser.h" file from the "PerMoVec" folder and substitute the original file 	
  from (*) located at "include/gudhi/Flag_complex_edge_collapser.h"
- Take the "include/gudhi" folder from (*) and copy it inside the "ext" folder with the name "gudhi"
	

Add a copy of PHAT with preimage support 
----------------------------------------
- Clone the PHAT repository branch "pm_matrix" (BRANCH NAME IMPORTANT) somewhere in your computer
- take the "include/phat" folder and copy it inside the "ext" folder from IBloFunMatch with the name "phat"


Build IBloFunMatch
------------------

- Compile "IBloFunMatch.cc" using CMake:
```sh
	mkdir build
	cd build 
	cmake ..
	cmake --build .
```
- Inside the build directory there is an executable "IBloFunMatch" which now can be run from the Notebooks and the terminal.
- For example (inside the build directory), executing 
```sh
	./IBloFunMatch -h 
```
  will print the help menu of the program.

- You can now run the jupyter notebooks inside the "Notebooks" folder. 

- (! Not working ATM) You can also execute the python script "IBloFunMatch.py" which reads a data file and plots the matching as follows:
	python IBloFunMatch.py {path to executable file} {sample percentage (from 0 to 1)} {path to data points} 
	(e.g. python IBloFunMatch.py build/IBloFunMatch.exe 0.4 samples/dino_data.txt)