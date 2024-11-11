##
## Example files for Density Matrix Downfolding
##
## Author: David W.O. de Sousa
##
## This step is the most important and time-consuming of the DMD process: sampling.
## Here DMRG is used as the CASCI solver, although it would not be necessary for
## such a small system. The Block2 program is used along with PySCF.
##
## The sampling spans the 30 lowest-energy CASCI roots of the system for each
## spin state, S = 0, 1, 2.
##
## The chkfile and the localized orbital file from the last step are necessary
## for this step.
##
from pyscf import scf, mcscf
from pyscf import dmrgscf
from pyscf.dmrgscf import *
from pyscf.dmrgscf.dmrgci import *
import h5py
import numpy as np
import glob, os

spins = [2,1,0]
nroots = 30
ncore = 24
ncas = 9
nelecas = 12
nbasis = 93
chkfile = 'mf.chk'
filename = 'FeSe_IAO_ordered'
#filename = 'FeSe_AVAS.Boys_ordered'
M = 2500 # bond dimension of the DMRG calculation

dmrgscf.settings.BLOCKEXE = os.popen("which block2main").read().strip()
dmrgscf.settings.MPIPREFIX = '' # use 'mpirun -n 1' if running on the cluster

## Generate 1- and 2-electron density matrices from the DMRG results
def make_rdm12s(DMRGCI, state, norb, nelec, **kwargs):
    if isinstance(nelec, (int, np.integer)):
        na = nelec // 2 + nelec % 2
        nb = nelec - na
    else:
        na, nb = nelec

    file2pdm = f"2pdm-{state}-{state}.npy" if DMRGCI.nroots > 1 else "2pdm.npy"
    dm2 = np.load(os.path.join(DMRGCI.scratchDirectory, "node0", file2pdm))
    dm2 = dm2.transpose(0, 1, 4, 2, 3)
    ## The 1-electron density matrix can be obtained from the 2-electron
    ## using the lines below. I prefer asking Block to generate a separate
    ## file for the 1-DM and read it directly.
    #dm1a = np.einsum('ikjj->ki', dm2[0]) / (na - 1)
    #dm1b = np.einsum('ikjj->ki', dm2[2]) / (nb - 1)
    #return (dm1a, dm1b), dm2
    file1pdm = f"1pdm-{state}-{state}.npy" if DMRGCI.nroots > 1 else "1pdm.npy"
    dm1 = np.load(os.path.join(DMRGCI.scratchDirectory, "node0", file1pdm))
    return dm1, dm2

## Load entry data
mol = scf.chkfile.load_mol(chkfile)
mo = np.fromfile(filename + '.dat').reshape((nbasis,nbasis))

for spin in spins:
    mol.spin = 2*spin
    mol.build()
    mf =scf.ROHF(mol)
    nelec = ((nelecas+2*spin)//2, (nelecas-2*spin)//2)

    # configure DMRGSCF
    SCRATCH = f'./scratch_{spin}'
    dmrgscf.settings.BLOCKSCRATCHDIR = SCRATCH
    dmrgscf.settings.BLOCKRUNTIMEDIR = SCRATCH

    # DMRG
    mycas = mcscf.CASCI(mf, ncas, nelec)
    mycas.fcisolver = DMRGCI(mol, maxM=M, tol=1e-6)
    mycas.fcisolver.spin = 2*spin
    mycas.fcisolver.nroots = nroots
    #mycas.fcisolver.block_extra_keyword.append('num_thrds 20')
    #mycas.fcisolver.block_extra_keyword.append('mem 40 g')
    mycas.fcisolver.block_extra_keyword.append('onepdm')
    mycas.natorb = False
    mycas.kernel(mo)

    with h5py.File(f"mc_S{spin}.hdf5", "w") as f:
        f.create_group("ci")
        f["ci/ncas"]      = mycas.ncas
        f["ci/nelecas"]   = list(mycas.nelecas)
        f["ci/ncore"]     = mycas.ncore
        f["ci/e_tot"]     = mycas.e_tot
        f["ci/e_cas"]     = mycas.e_cas
        f["ci/ci"]        = mycas.ci
        f["ci/mo_coeff"]  = mycas.mo_coeff

    # Save density matrices
    DM1 = np.zeros((nroots,2,ncas,ncas))
    DM2 = np.zeros((nroots,3,ncas,ncas,ncas,ncas))
    for i in range(nroots):
        dm1, dm2 = make_rdm12s(mycas.fcisolver, mycas.ci[i], norb=mycas.ncas, nelec=mycas.nelecas)
        DM1[i] = dm1
        DM2[i] = dm2
    DM1.tofile(f"dm1_S{spin}.dat")
    DM2.tofile(f"dm2_S{spin}.dat")

    # Delete scratch to save space
    for f in glob.glob(f"{SCRATCH}/F*.*"):
        os.remove(f)
