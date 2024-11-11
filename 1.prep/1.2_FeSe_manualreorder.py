##
## Example files for Density Matrix Downfolding
##
## Author: David W.O. de Sousa
##
## An additional step of manual orbital reordering is necessary. Open the molden file with the
## localized orbitals in some visualization program (e.g. Jmol) and inspect the orbitals.
## Determine the preferred order for the active orbitals, fill the entries below and run the script.
##
import numpy as np
from pyscf import scf
from pyscf.tools import molden

# Inputs
chkfile = 'mf.chk'
caslist = [31,33,32,26,25,27,28,29,30] # The 1-based index of the orbitals, as shown in Jmol
ncore = 24
filename = 'FeSe_AVAS.Boys'

mol = scf.chkfile.load_mol(chkfile)
mf_dict = scf.chkfile.load(chkfile, 'scf')
norb = mf_dict['mo_coeff'].shape[1]
ncas = len(caslist)

caslist = np.array(caslist)
caslist -= 1 # 0-based index

MOs = np.fromfile(filename + ".dat").reshape((norb, norb))
MOlist = np.hstack((np.arange(ncore), caslist, np.arange(ncore + ncas, norb)))
FinalMOs = np.array([ [o[j] for j in MOlist ] for o in MOs ])
FinalMOs.tofile(filename + "_ordered.dat")
molden.from_mo(mol, filename + "_ordered.molden", FinalMOs)
