# 2. `sampling` section

This step is the most important and time-consuming of the DMD process: sampling. 

Here DMRG is used as the CASCI solver, although it would not be necessary for such a small system. The Block2 program is used along with PySCF.

The sampling spans the 30 lowest-energy CASCI roots of the system for each spin multiplicity, S = 0, 1, 2.

The `chkfile` and the localized orbital file from the last step are necessary for this step.
