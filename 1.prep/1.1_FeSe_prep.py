##
## Example files for Density Matrix Downfolding
##
## Author: David W.O. de Sousa
##
## This is the first step, obtaining localized orbitals for the system in order to proceed with the
## sampling. The FeSe molecule is used as an example.
##
import numpy as np
from pyscf import gto, scf, lo
from pyscf.tools import molden, mo_mapping
from pyscf.mcscf import avas

##
## Consider the high-spin state of the system (S = 2), because we will have the most number of 
## ocuppied orbitals in the SCF calculation.
## This is useful for obtaining a good set of orbitals, which will be used for all spin states
## (S = 0,1,2) in the CASCI calculation.
## Note that PySCF defines `mol.spin` as 2*S, not S.
## 
SPIN = 4
ActiveOrbSets = ['Fe 3d', 'Fe 4s', 'Se 4p']
ActiveOrbPops = [5, 1, 3]
chkfile = 'mf.chk'


## Defining the system
mol = gto.Mole()
mol.atom = 'Fe 0 0 0; Se 0 0 2.4'
mol.basis = 'def2-tzvp'
mol.verbose= 4
mol.spin = SPIN
mol.build()

## Restricted open-shell Kohn-Sham calculation
mf = scf.ROKS(mol)
mf.xc = 'PBE0'
mf.max_cycle = 500
mf.conv_tol = 1.e-8
mf.chkfile = chkfile
mf = scf.newton(mf)
mf.kernel()

## Save canonical MOs for further use or inspection
#mf.mo_coeff.tofile('FeSe_MO.dat')
#molden.from_mo(mol, 'FeSe_MO.molden', mf.mo_coeff)


##
## If you are re-running this script and the above calculations are already done,
## comment the above lines from the definition of the system and un-comment the
## lines below
##
#mol = scf.chkfile.load_mol(chkfile)
#mf = scf.ROKS(mol)
#mf.__dict__.update(scf.chkfile.load(chkfile, 'scf'))

##
## There are two possible ways of doing orbital localization - using IAOs (Intrinsic Atomic
## Orbitals) or performing AVAS (Atomic Valence Active Spaces) and then performing a
## localization in the active space subset. These methods will yield different CASCI energies,
## so one should test both of them and decide which is best. Ideally, one would work with the
## CASSCF optimized orbitals, but this is not feasible for larger systems. 
##

##
## AVAS procedure
##
ncas, nelecas, AVAS_MOs = avas.avas(mf, ActiveOrbSets)
ncore = (mol.nelectron - nelecas)//2
#AVAS_MOs.tofile('FeSe_AVAS.dat')
#molden.from_mo(mol, 'FeSe_AVAS.molden', AVAS_MOs)

CoreMOs = AVAS_MOs[: , : ncore]
ActMOs  = AVAS_MOs[: , ncore : ncore + ncas]
VirtMOs = AVAS_MOs[: , ncore + ncas : ]

# Boys localization in the active subset
localizer = lo.boys.Boys(mol, ActMOs)
## Sometimes the line below helps with the localization
#localizer.init_guess='random'
localizer.verbose = 4
LocAct = localizer.kernel()

# Join localized active orbitals with core and virtual
FinalMOs = np.column_stack((CoreMOs, LocAct))
FinalMOs = np.column_stack((FinalMOs, VirtMOs))

FinalMOs.tofile('FeSe_AVAS.Boys.dat')
molden.from_mo(mol, 'FeSe_AVAS.Boys.molden', FinalMOs)

##
## IAO orbital localization
##
IAO = lo.iao.iao(mol, mf.mo_coeff[:, mf.mo_occ>0])
IAO_orth = lo.vec_lowdin(IAO, mf.get_ovlp())
ActiveList = np.array([], dtype=int)

## Get active IAOs
for i in range(len(ActiveOrbSets)):
    OrbPop = mo_mapping.mo_comps(ActiveOrbSets[i], mol, IAO_orth)
    PopList = OrbPop.argsort()[-ActiveOrbPops[i]:]
    ActiveList = np.hstack((ActiveList, PopList))

ActiveIAOs = np.array([ [a[j] for j in ActiveList ] for a in IAO_orth ])

## Join IAOs with core and virtual orbitals
MOs = mf.mo_coeff
CoreMOs = MOs[: , : ncore]
VirtMOs = MOs[: , ncore + ncas : ]

FinalMOs = np.column_stack((CoreMOs, ActiveIAOs))
FinalMOs = np.column_stack((FinalMOs, VirtMOs))

FinalMOs.tofile('FeSe_IAO.dat')
molden.from_mo(mol, 'FeSe_IAO.molden', FinalMOs)
