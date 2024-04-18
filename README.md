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

Python Installation
-------------------
To run the notebooks, it is required to install iblofunmatch. For this, we recommend creating a virtual environment. Once created and activated, inside the main IBloFunMatch folder, run the following command
```sh
	python3 -m pip install .
```
After the installation, one should be able to load the module `iblofunmatch', to check it, one can run the following command:
```sh
	python3 -c "import iblofunmatch"
```
If there are no error messages, the module is correctly installed.
