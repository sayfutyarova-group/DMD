# 1. `prep` section

This is the first step, obtaining localized orbitals for the system in order to proceed with the sampling. The FeSe molecule is used as an example.

The restricted open-shell Kohn-Sham DFT with the PBE0 functional is used.

There are two files in this directory, `1.1_FeSe_prep.py` and `1.2_FeSe_manualreorder.py`.

## 1.1 `FeSe_prep`

First, we calculate the high-spin state of the system (S = 2), because we will have the most number of ocuppied orbitals in the SCF calculation. This is useful for obtaining a good set of orbitals, which will be used for all spin states (S = 0,1,2) in the next step (CASCI calculation).

Then, orbital localization is performed to obtain the set of orbitals for the next step. There are various ways of localizing orbitals. The script provides two examples: using IAOs (Intrinsic Atomic Orbitals) or performing AVAS (Atomic Valence Active Spaces) and then performing a localization in the active space subset. These methods will yield different CASCI energies, so one should test both of them and decide which is best. Ideally, one would work with the CASSCF optimized orbitals, but this is not feasible for larger systems.

## 1.1 `FeSe_manualreorder`

An additional step of manual orbital reordering is necessary. Open the `molden` file with the localized orbitals in some visualization program (e.g. Jmol) and inspect the orbitals. Determine the preferred order for the active orbitals, type it in the file (in the variable `caslist`) and then run the script.
