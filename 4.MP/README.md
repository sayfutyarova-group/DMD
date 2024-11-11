# 4. Matching pursuit algorithm

This directory contain two Python files:

## `matching_pursuit.py`

It uses a previously done DMD calculation to optimize the set of parameters in the model Hamiltonian. Note that you need to perform a DMD fitting using the whole set of parameters, in order to have the full set of descriptors needed to perform this step. In the provided example, this was done in the `FeSe_model4` case.

Before running this script, please edit the input variables in it:

- `params` should contain the same exact list of model parameters used in the previous DMD calculation. 
- `e_file` should contain the path to the corresponding `*_PLOT.dat` file.
- `d_file` should contain the path to the corresponding `*_DATA.dat` file.

To run the script, type in the terminal:

```$ python3 matching_pursuit.py > matching_pursuit.out```

The output file `matching_pursuit.out` is necessary to run the next script.

## `mp_plot.py`

It extracts the results from the matching pursuit output file and creates plots of them. 

