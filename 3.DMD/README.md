# 3. `DMD`

This directory contains the necessary files for the actual step of performing DMD. It contains several files.

- The directory `input` contains sampling results from the last step: PySCF checkfiles and Density Matrix (DM) files. They are required to run the DMD program.

- `DMD.py` is the program itself. It requires an input file to specify all the parameters of the DMD process. There are four examples of DMD input files (`*.dmd`) included in the directory, each one describing a different model Hamiltonian for the FeSe molecule.

- `params_lib.py` is the parameter library file, a Python library for the model Hamiltonian. It contains the necessary information to translate the model parameter names into the correct combination of density matrix elements. Note that the indexing of the DM elements has to be consistent with the orbital ordering chosen in the previous steps of the calculation. Depending on the system and on the chosen model Hamiltonian, different parameters can be formulated and defined in this file.

- `MH FeSe.pdf` contains a graphic description of the valence orbitals of FeSe and a list of the different parameters that can be used to describe a model Hamiltonian for this system. The same name convention of the parameters is used in the DMD input file and in the parameter library file.  

## The DMD Input File

It is a plain text file with the input parameters of the program. Each line contains the definition of one of the program variables. Lines starting with `#` are ignored, and `#`  also can be inserted after the definition of a variable to insert a comment. The variable declaration follows the syntax `variable_name = variable_value`, and blankspace characters are ignored.

#### List of variables

##### Fundamental variables

- **nspin** - list of scanned spin multiplicities (integer values separated by commas)
- **norb**    - number of orbitals in active space
- **nelec**   - number of electrons in active space (by default it is equal to norb)
- **nroots**  - number of roots (eigenstates) calculated in CASCI/DMRG
- **params**  - List of the names of the parameters considered for the effective model. In the given example file, the supported parameters are `E0, J, tσ, tπ, Ud, Up, Vpd, εpσ, εdσ, εpπ, εdπ, and εdδ`. Those parameters are defined in the Parameter library file.
- **nsamples** - number of non-eigenstates generated from each eigenstate. The default is 0 (pure eigenstate method). 
- **var**      - "variance" (actually it is the standard deviation) allowed in the CI coefficients of non-eigenstates. The setting of this variable is very complex because a number of different strategies were tried. Here are its possible settings:

> * If only one number is used, this will be the standard deviation for all non-eigenstates. (Ex.: `var = 0.01`)
> * If a small number of values, the non-eigenstates will be evenly distributed through the designed deviation values (Ex.: `nsamples = 7`, `var = 0.001, 0.002, 0.003`; the first two samples will have SD = 0.001, the next two will have SD = 0.002 and the three remaining ones will have SD = 0.003).
> * If it is a list of three values and the first value is zero: this triggers the random selector, for each non-eigenstate a random value of standard deviation is chosen, within the range defined by the second and third value of the list.
> * If it is a list of `nspin + 1` values, and the first value is 10: the standard deviation is chosen according to the spin value of the state.

##### File related variables

- **inputdir**  - Path of the directory where checkpoint files and DM files are located. You can use relative paths (e.g. ".." for the above directory), but paths cannot contain blankspaces. The default value is the current directory.
- **inputname** - allows using a suffix for the inputfile names. The program will read the files `mf_S{spin}{inputname}.hdf5`, `mc_S{spin}{inputname}.hdf5`, `dm1_S{spin}{inputname}.dat` , and `dm2_S{spin}{inputname}.dat`.
- **outputdir** - Path of the directory to write output files.

##### Other variables

- **le_window**     - low-energy window in electron-volts (states with higher relative energy to the ground state are removed from sampling)
- **fixed_pnames**  - List of the names of the parameters which will have a fixed value during the optimization
- **fixed_pvalues** - List of fixed parameter values (in Hartree)
- **tr_tol**        - Maximum tolerance for the relative error of 1-DM trace in each state (default = 0.02). This filtering step is probably redundant using CASCI / DMRG but it was necessary in early versions of the program when CISD sampling was used.
- **cross**         - whether to generate non-diagonal terms of the non-eigenstates (also probably redundant using CASCI, because the crossed states are often eliminated by the tolerance criterion)

- **dmrg**    - whether the input data is from DMRG calculation (you can set to `yes` | `true` |`no` | `false`)
- **dmread**  - whether the density matrices are read from a file (else they are calculated during the run) 
- **save**    - whether the descriptors are written to files
- **verbose** - whether the program outputs in the terminal console besides of textfile output
- **debug**   - whether the non-eigenstate determinant expansions iare written to files
- **textbox** - whether the optimized parameters are printed in the plot
- **rel**     - whether the plot shows relative energies
- **unit**    - unit to display data in the plot (au,ha= hartree; ev,eV=electron-volt; cm=cm^-1)

## Running the DMD program

You need Python (obviously) with the Numpy, MatPlotLib, and PySCF libraries installed. Just run in the terminal:

```$ python3 DMD.py inputfile_name.dmd```

Please note that the program needs the PySCF checkpoint files to be located in the directory specified by `inputdir` in the input file. If `dmread` is set true in the input file, the program will also require the density matrix files to be in `inputdir`.

Also note that both SCF and multiconfigurational checkpoint files are required. In the provided example, the sampling was performed for the FeSe molecule, with spin multiplicities S = 0, S = 1, and S = 2. The multiconfigurational calculation results are saved in the files `mc_S0.hdf5`, `mc_S1.hdf5` and `mc_S2.hd5`. Ideally, the program would expect to have the SCF checkpoint files `mf_S0.hdf5`, `mf_S1.hdf5`, and `mf_S2.hdf5`, but it also works with one checkpoint file for the SCF result named `mf.chk`. Both file extensions `.chk` or `.hdf5` can be used for the checkpoint files.

By default, the program will generate 3 output files: a text file `*.out` with the results of the calculation, a `*.png` image with a simple plot of the result, and a text file `*_PLOT.dat` containing the sampled energy values and the corresponding model energy values.   

If the keyword `save` is set true in the input file, an additional text files `*_DATA.plot` is created, containing a table of all descriptors of the model parameters. Note that this file is required for the next step (matching pursuit algorithm).

If the keyword `verbose` is set true in the input file, the program will also show results in the terminal screen.

