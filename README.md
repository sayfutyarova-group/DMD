# DMD

Density matrix downfolding code from quantum chemical calculations using [PySCF](https://github.com/pyscf/pyscf) and DMRG from [Block2](https://github.com/block-hczhai/block2-preview).

*Author:* David W. O. de Sousa ([davidsousarj](https://github.com/davidsousarj)), The Pennsylvanis State University, November 2024.

Besides the actual DMD code, this repo contains a complete workflow sample with all steps, from preliminary quantum chemical calculations to high-quality plot generation. The FeSe molecule is used as example. The workflow consists of five steps:

1. `prep`: preliminary quantum chemical calculations using PySCF, using Kohn-Sham DFT and orbital localization to obtain a set of initial orbitals.
2. `sampling`: multiconfigurational calculation (CASCI using DMRG solver) to obtain a number of low-energy states of the molecule.
3. `DMD`: the fitting of a model Hamiltonian to the sampled low-energy states.
4. `MP`: the use the matching pursuit algorithm to optimize the size of the model Hamiltonian. 
5. `figures`: the generation of high-quality figures to be published on scientific journals.

Please check out inside the directories for specific documentation.
