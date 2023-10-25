Welcome to Induced Block Funtions and Matchings (IBloFunMatch)!!!

Installation instructions:
--------------------------
- Have a C++ compiler installed (e.g. GNU compiler)
- Have Python installed with numpy, matplotlib,...
- Make sure you have already installed:
CGAL, GUDHI (and their dependencies:mpfr, gmp and boost)
- Maker sure that "gudhi" and "cgal" are in your compiler include path
- Download the PHAT copy from the branch "pm_matrix"
- Make sure that the "phat" folder within "{PHAT copy path}/include" is in your C++ compiler include path
- Download the "PerMoVEC" bitbucket repository
- Take the "Flag_complex_edge_collapser.h" file and substitute the original file on the gudhi repository 
- (on the previous step you might want to call the older file "Flag_complex_edge_collapser_old.h" to keep it)
- Make sure the "permovec.h" file is in your compiler include path

Now we are ready! 

- Compile "IBloFunMatch.cc" and produce the ".exe" file (e.g. IBloFunMatch.exe)
- If you execute the ".exe" file with the -h flag you will get a menu detailing the program.
- You can also execute the python script "IBloFunMatch.py" which reads a data file and plots the matching as follows:
	python IBloFunMatch.py {path to .exe file} {sample percentage (from 0 to 1)} {path to data points} 
	(e.g. python IBloFunMatch.py IBloFunMatch.exe 0.4 samples/dino_data.txt)