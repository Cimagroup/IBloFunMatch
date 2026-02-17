Welcome to Induced Block Funtions and Matchings (IBloFunMatch)!!!

Installation instructions:
==========================
Basic Requirements
------------------
- Have a C++ compiler installed (e.g. GNU compiler)
- Have Python installed (version >=3.10)
- Make sure you have already installed:
  CGAL, mpfr, gmp and boost
- Required python packages:
    - scipy (>=1.15),
    - gudhi (>=3.11),
    - tqdm (>=4.67),
    - numpy (>=2.2),
    - matplotlib (>=3.10)

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
	python3 -m pip install --editable .
```
We need to add the "editable" option since we will use the code built in this folder.
After the installation, one should be able to load the module `iblofunmatch', to check it, one can run the following command:
```sh
	python3 -c "import iblofunmatch.inter as ibfm"
```
The path to the executable file should appear printed. If there are no error messages, the module is correctly installed.
