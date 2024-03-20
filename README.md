Welcome to Induced Block Funtions and Matchings (IBloFunMatch)!!!

Installation instructions:
==========================
Basic Requirements
------------------
- Have a C++ compiler installed (e.g. GNU compiler)
- Have Python installed with numpy, matplotlib,...
- Make sure you have already installed:
  CGAL, mpfr, gmp and boost

Build IBloFunMatch
------------------

- Compile "IBloFunMatch.cc" using CMake:
```sh
	mkdir build
	cd build 
	cmake -DCMAKE_BUILD_TYPE=Debug ..
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