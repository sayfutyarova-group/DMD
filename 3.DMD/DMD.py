#!/usr/bin/env python
# coding: utf-8

__version__='1.0'

import os
import pyscf
import pyscf.mcscf
import pyscf.ao2mo
import numpy as np
from matplotlib import pyplot as plt
from params_lib import *

ha2ev = 27.211399 # conversion from Hartree to electron-volt
ha2cm = 219474.63 # conversion from Hartree to cm^-1

def load_state(state, norb, nelec, nroots, inputdir, inputname):
    '''Loads checkfiles and build mol, mf and mc objects.'''
    
    # Locate files
    SCFfile = f'{inputdir}/mf_{state}{inputname}.hdf5'
    found = os.path.isfile(SCFfile)
    if not found:
        SCFfile = f'{inputdir}/mf_{state}{inputname}.chk'
        found = os.path.isfile(SCFfile)
    if not found:
        SCFfile = f'{inputdir}/mf{inputname}.hdf5'
        found = os.path.isfile(SCFfile)
    if not found:
        SCFfile = f'{inputdir}/mf{inputname}.chk'
        found = os.path.isfile(SCFfile)
    if not found:
        raise FileNotFoundError(SCFfile)
    CASfile = f'{inputdir}/mc_{state}{inputname}.hdf5'
    found = os.path.isfile(CASfile)
    if not found:
        raise FileNotFoundError(CASfile)
        
    # Load SCF result
    mol, mf_dict = pyscf.scf.chkfile.load_scf(SCFfile)
    mf = pyscf.scf.RHF(mol) 
    mf.__dict__.update(mf_dict)
    
    # Load CASCI result
    mc = pyscf.mcscf.CASCI(mf, norb, nelec)
    mc.natorb=False
    mc.canonicalization=False
    mc.sorting_mo_energy=False
    mc.fcisolver.nroots=nroots
    mc_dict = pyscf.lib.chkfile.load(CASfile, 'ci')
    mc.__dict__.update(mc_dict)
    
    return mol, mf, mc

def gen_noneigs(ci0, nsamples=10, var=[1.0], 
                debug=False, outputdir='.', ispin=0):
    '''
    Manages the non-eigenstate generation method depending on the
    format of `var`.
    
    Method 1: `var` is an array of different values. The non-eigenstates
    will be evenly distributed through the designed deviation values.
    Ex.: nsamples = 7, var = 0.001, 0.002, 0.003; the first two samples
    will have SD = 0.001, the next two will have SD = 0.002 and the
    three remaining ones will have SD = 0.003).
    
    Method 2: var[0] = 0 and var[1:2] = lower and upper range limits.
    For each non-eigenstate a random value of standard deviation is
    chosen, within the range defined by the var[1] and var[2].
    
    Method 3: var[0] = 10 and nspin remaining values. The following
    values are the std devs for each spin state.
    '''
    size    = ci0.shape
    noneigs = np.zeros((nsamples,*size))
    # Method 1
    if var[0] != 0 and var[0] != 10:
        nvar = len(var)
        npts = nsamples//nvar
        extra = 0
        for v in range(nvar):
            if v == nvar - 1:  extra = nsamples%nvar
            for n in range(npts + extra):
                noneigs[v*npts + n] = noneig(var[v], size, ci0)

    # Method 3
    elif var[0] == 10:
        # Check errors
        try:
            vi = var[ispin+1]
        except IndexError:
            print(""" ERROR. When var[0] = 10 there must be `nspin` var
 elements afterwards.""")
            exit(1)
        for i in range(nsamples):
            noneigs[i] = noneig(vi, size, ci0)

    # Method 2
    else:
        # Check for input errors
        if len(var) < 3:
            print(""" ERROR. When var[0] = 0 the next two elements 
 (lowerlimit, upperlimit) are required.""")
            exit(1)
        if var[1] > var[2]:
            print(""" ERROR. var[1] (lowerlimit) should be less than 
 var[2] (upperlimit).""")
            exit(1)
        vi = np.random.uniform(var[1], var[2], nsamples)
        for i in range(nsamples):
            noneigs[i] = noneig(vi[i], size, ci0)

    if debug:
        for i in range(nsamples):
            np.savetxt(f'{outputdir}/{outname}_neig_{ispin}{i}.dat', 
                       noneigs[i], delimiter=" ", fmt="%16.10f")

    return noneigs

def noneig(v, size, ci0):
    '''Non-eigenstate generator.'''
    R = np.random.normal(0, v, size)
    cimod = ci0 + R                                   # additive var
    cimod = cimod / np.sqrt( np.square(cimod).sum() ) # renormalize
    return cimod

def old_noneig_energy(mol, mf, mc, ci0):
    '''Non-eigenstate CI energy calculator (inefficient).'''
    dm1, dm2 = pyscf.mcscf.addons.make_rdm12(mc, mo_coeff=mc.mo_coeff, 
    ci=ci0)
    h1e = mf.get_hcore()
    eri = mol.intor('int2e', aosym='s1')
    E = np.einsum('ij,ji', h1e, dm1) \
        + 0.5*np.einsum('ijkl,ijkl', eri, dm2) + mol.energy_nuc()
    return E

def noneig_energy(mc, ci0, debug):
    '''Returns the energy of a CI state.'''
    global outfile, verbose
    # DM in MO basis
    if debug: pprint("   Entered function noneig_energy. Calculating active space DMs...", outfile, verbose)
    dm1, dm2 = mc.fcisolver.make_rdm12(ci0, mc.ncas, mc.nelecas)
    if debug: pprint("   DMs calculated. Getting 1-electron effective H...", outfile, verbose)
    h1_eff, h0 = mc.get_h1eff()
    if debug: pprint("   H1_eff calculated. Getting 2-electron effective H...", outfile, verbose)
    h2_eff = mc.get_h2eff()
    if debug: pprint("   H2_eff calculated. Transforming H2eff matrix...", outfile, verbose)
    h2_eff = pyscf.ao2mo.restore(1, h2_eff, mc.ncas) # h2_eff is now 4-D
    if debug: pprint("   H2_eff transformed. Calculating energy...", outfile, verbose)
    E = h0 + h1_eff.ravel().dot(dm1.ravel()) \
        + 0.5*h2_eff.ravel().dot(dm2.ravel())
    return E

def cross_energy(mc, ci_i, ci_j):
    '''Returns the hamiltonian matrix element <Phi_i|\hat{H}|Phi_j>.'''
    tdm1, tdm2 = mc.fcisolver.trans_rdm12(ci_i, ci_j, mc.ncas, mc.nelecas) 
    ovlp = ci_i.ravel().dot(ci_j.ravel())
    h1_eff, h0 = mc.get_h1eff()
    h2_eff = mc.get_h2eff()
    h2_eff = pyscf.ao2mo.restore(1, h2_eff, mc.ncas) # h2_eff is now 4-D
    return ((h0 * ovlp)
            +     h1_eff.ravel().dot(tdm1.ravel())
            + (0.5*h2_eff.ravel().dot(tdm2.ravel()) ))

def get_descriptors_eig(mc, nroots, params_list, nparams, spin, dmread,
                        inputdir, inputname):
    '''
    Gets the DMs for all eigenstates of a certain spin state and calls
    the descriptor generator (`get_parameter`). 
    '''
    A=np.zeros((nroots, nparams))
    trace=np.zeros(nroots)
    ncas=mc.ncas
    nelec = mc.nelecas #((nelec+2*spin)//2, (nelec-2*spin)//2)

    if dmread:
        # DM index format
        # DM1 = [ nroots, alpha/beta, p, q]
        # DM2 = [ nroots, aa/ab/bb,   p, q, r, s]
        DM1 = np.fromfile(f"{inputdir}/dm1_S{spin}{inputname}.dat")
        DM2 = np.fromfile(f"{inputdir}/dm2_S{spin}{inputname}.dat")
        total_nroots = len(DM1)//ncas//ncas//2
        DM1 =  DM1.reshape((total_nroots,2,ncas,ncas))
        DM2 =  DM2.reshape((total_nroots,3,ncas,ncas,ncas,ncas))

    for i in range(nroots):
        if dmread:
            d1 = DM1[i]
            d2 = DM2[i]
        else:
            d1, d2 = mc.fcisolver.make_rdm12s(mc.ci[i], norb=ncas,
                                              nelec=nelec)

        for p in range(nparams):
            A[i,p] = get_parameter(d1, d2, ncas, params_list[p], eig=True)

        trace[i] = np.trace(d1[0]) + np.trace(d1[1])
    
    return mc.e_tot[:nroots], A, trace

def get_descriptors_noneig(mc, noneigs, params_list, nparams, energies,
                           spin, cross):
    '''Gets the DMs for all non-eigenstates of a certain spin state 
    and calls the descriptor generator (`get_parameter`). '''
    nsamples  = len(energies)
    nsamples2 = nsamples*nsamples if cross else nsamples
    Hij       = np.zeros(nsamples2)
    A         = np.zeros((nsamples2, nparams))
    trace     = np.zeros(nsamples2)
    x = 0
    for i in range(nsamples):
        for j in range(nsamples):
            if i==j:
                Hij[x] = energies[i]
            else:
                if not cross: continue
                Hij[x] = cross_energy(mc, noneigs[i], noneigs[j])

            #nelec = ((mc.nelecas+2*spin)//2, (mc.nelecas-2*spin)//2)
            d1, d2 = mc.fcisolver.trans_rdm12s(noneigs[i], noneigs[j],
                                               norb=mc.ncas, 
                                               nelec=mc.nelecas)

            for p in range(nparams):
                A[x,p] = get_parameter(d1, d2, mc.ncas, params_list[p],
                                       eig=False)

            trace[x] = np.trace(d1[0]) + np.trace(d1[1])

            x+=1

    return Hij, A, trace

def filter_points(Hij, A, trace, nelec, tr_tol, le_window):
    '''
    Eliminates sample points that do not obey the trace and
    low energy window criteria.'''
    npts = len(Hij)
    Emin = min(Hij)
    exclude=[]
    for i in range(npts):
        if abs(trace[i] - nelec)/nelec > tr_tol or \
           (Hij[i] - Emin)*ha2ev > le_window:
            exclude.append(i)

    HIJ   = np.array([Hij[i] for i in range(npts) if i not in exclude])
    DATA  = np.array([A[i] for i in range(npts) if i not in exclude])
    TRACE = np.array([trace[i] for i in range(npts) if i not in exclude])

    return HIJ, DATA, TRACE

def r2(x, y):
    import warnings
    warnings.filterwarnings("ignore") # ignore division by zero
    zx = (x-np.mean(x))/np.std(x, ddof=1)
    zy = (y-np.mean(y))/np.std(y, ddof=1)
    r = np.sum(zx*zy)/(len(x)-1)
    return r**2

def test_correlations(DATA, params_list, nparams):
    global outfile, verbose
    pprint(f"Correlation between parameters:", outfile, verbose, skip=True)
    for x in range(nparams - 1):
        for y in range(x + 1, nparams):
            r2coeff = r2(DATA[:,x], DATA[:,y])
            pprint(f"  {params_list[x]:3s} vs. {params_list[y]:3s}\
 : R^2 = {r2coeff:.6f}", outfile, verbose)


def fix_model(HIJ, DATA, params_list, fixed_pnames, fixed_p):
    """Removes fixed parameters from the DATA table."""
    fixed_pindex = [ params_list.index(p) for p in fixed_pnames ]
    fixed_cols   = np.transpose( [ DATA[:,n] for n in fixed_pindex ] )
    # delete fixed parameter columns
    DATAfix    = np.delete(DATA, fixed_pindex, axis=1) 
    HIJfix     = HIJ - np.dot(fixed_cols, fixed_p)
    return HIJfix, DATAfix, fixed_pindex

def DMD_fit_fixp(Hij, A, fixed_p, fixed_pindex):
    P = np.linalg.lstsq(A, Hij, rcond=None)
    sol = P[0]
    fixed = sorted(list(zip(fixed_pindex, fixed_p)))
    for i in range(len(fixed)):
        sol = np.insert(sol, fixed[i][0], fixed[i][1])
    return (sol, P[1], P[2], P[3])

def DMD_fit(Hij, A):
    P = np.linalg.lstsq(A, Hij, rcond=None)
    return P

def DMD_results(Hij, A, sol, params_list, nparams, outfile, outname,
                textbox, unit, rel, dmrg, fixed_pnames, outputdir):
    global verbose
    # Units treatment
    if unit.lower() in ['au','ha','hartree']:
        unitname = 'Ha'
        factor  = 1.
        d = 4
    elif unit.lower() == 'ev':
        unitname = 'eV'
        factor  = ha2ev
        d = 2
    elif unit.lower() == 'cm':
        unitname = 'cm^-1'
        factor  = ha2cm
        d = 1
    else:
        unitname = 'Ha'
        factor  = 1.
        pprint(f"WARNING: energy unit {unit} not recognized. Using\
 atomic units instead", outfile, verbose)
        d = 4

    # Statistical treatment
    dE = Hij - A@sol[0]
    dE_max = max(dE)
    dE_rms = np.sqrt( np.square(dE).sum()/ len(dE) )
    # Calculate R² only if correct data rank
    fit_ok = sol[2] >= nparams and fixed_pnames == "" or \
             sol[2] >= nparams - len(fixed_pnames)
    if  fit_ok: [R2] = 1 - sol[1]/(Hij.size*Hij.var())

    # Print results
    pprint("DMD Results", outfile, verbose, uline=True, skip=True)
    pprint(f"{params_list}  = {sol[0]*factor} {unitname}", outfile, verbose)
    pprint(f"residue = {sol[1]*factor} {unitname}", outfile, verbose, skip=True)
    pprint(f"dE_max  = {dE_max*factor} {unitname}", outfile, verbose)
    pprint(f"dE_rms  = {dE_rms*factor} {unitname}", outfile, verbose)
    if fit_ok:
        pprint(f"R^2     = {R2}", outfile, verbose)
    else:
        pprint(f"WARNING: possible redundancy in fit data. rank\
 ={sol[2]} but fit has {nparams} parameters.", outfile, verbose)

    # Fixed parameters treatment
    f = ""
    for p in params_list:
        if p in fixed_pnames:
            f += "*"
        else:
            f += " "

    # Plot fit
    plt.figure()
    ecasci, emodel = Hij*factor, A@sol[0]*factor
    plt.plot(ecasci, emodel, 'o')
    line = np.linspace(np.amin(ecasci), np.amax(ecasci), 100)
    plt.plot(line, line)
    if dmrg:
        plt.xlabel(f"DMRG Energy / {unitname}")
    else:
        plt.xlabel(f"CASCI Energy / {unitname}")
    if rel:
        plt.ylabel(f"Model Relative Energy / {unitname}")
    else:
        plt.ylabel(f"Model Energy / {unitname}")
    if textbox:
        x, y, dx, dy = 0.03, 0.95, 0.28, 0.05 
        for i in range(len(params_list)):
            if i>0 and i%4 == 0: x += dx
            plt.text(x, y - dy*(i%4),
        f"{params_list[i]} = {sol[0][i]*factor:.{d}f} {unitname}{f[i]}",
        ha='left', va='center', transform = plt.gca().transAxes)
            
        x -= dx*((len(params_list)-1)//4)
        plt.text(x, y -dy*5, 
        r"$\Delta E_{max} =$" + f"{dE_max*factor:.{d}f} {unitname}", 
        ha='left', va='center', transform = plt.gca().transAxes)
        plt.text(x, y -dy*6, 
        r"$\Delta E_{rms} =$" + f"{dE_rms*factor:.{d}f} {unitname}", 
        ha='left', va='center', transform = plt.gca().transAxes)
        if fit_ok:
            plt.text(x, y -dy*7, 
            r"$R^2 =$" + f"{R2:.5f}", ha='left', va='center', 
            transform = plt.gca().transAxes)
        else:
            plt.text(x, y -dy*7, 
            r"$FIT\  ERROR.\  No\  R^2$", ha='left', va='center', 
            transform = plt.gca().transAxes)
        if fixed_pnames != "":
            plt.text(x, y -dy*8, 
            "* Fixed parameters.", ha='left', va='center', 
            transform = plt.gca().transAxes)

    plt.tight_layout()
    plt.savefig(outputdir + "/" + outname + ".png", dpi=300)
    if verbose:
        plt.show()

def print_header(INP, params_list, SIZE, outfile, verbose):
    pprint( "Density Matrix Downfolding from CASCI/DMRG", outfile, verbose)
    pprint(f"              Version {__version__}", outfile, verbose)
    pprint( "     Written by David W.O. de Sousa", outfile, verbose)
    pprint( "The Pennsylvania State University, 2023", outfile, verbose)
    pprint("Program Parameters", outfile, verbose, uline=True, skip=True)
    pprint(f"Number of orbitals in active space:          {INP['norb']}", outfile, verbose)
    pprint(f"Scanned spin states:                         {INP['nspin']}", outfile, verbose)
    pprint(f"Number of eigenstates per spin state:        {INP['nroots']}", outfile, verbose)
    pprint(f"Number of non-eigenstates per eigenstate:    {INP['nsamples']}", outfile, verbose)
    if nsamples > 0:
        pprint(f"Standard deviation of CI coeffs:             {INP['var']}", outfile, verbose)
    pprint(f"Total number of points:                      {SIZE}", outfile, verbose)
    pprint(f"Tolerance factor of descriptor assessment:   {INP['tr_tol']*100:.1f}%", outfile, verbose)
    pprint(f"Low-energy space energy window:              {INP['le_window']} eV", outfile, verbose, skip=True)
    pprint(f"Parameters in the effective model:           {params_list}", outfile, verbose)
    if INP['inputdir'] != '.' or INP['inputname'] != '' or INP['outputdir'] != '.':
        pprint("I/O Parameters", outfile, verbose, uline=True, skip=True)
        if INP['inputdir'] != '.':
            pprint(f"Read inputs from directory:                  {INP['inputdir']}", outfile, verbose)
        if INP['inputname'] != '':
            pprint(f"Suffix for input filenames:                  {INP['inputname']}", outfile, verbose)
        if INP['outputdir'] != '.':
            pprint(f"Write outputs to directory:                  {INP['outputdir']}", outfile, verbose)

def pprint(text, outfile, verbose, margin=1, uline=False, skip=False):
    s=" "
    l="="
    if skip:
        _pprint('', outfile, verbose)
    _pprint(s*margin + text, outfile, verbose)
    if uline:
        _pprint(s*margin + l*len(text), outfile, verbose)

def _pprint(text, outfile, verbose=True):
    if verbose: print(text)
    outfile.write(text+"\n")

def getLine(source, string, pos=0, none=True, comments=True):
    '''
    Gets the line index of the (`pos`+1)-th occurrence of `string` in `source`.
    '''
    count=0
    for index in range(len(source)):
        if string in source[index]:
            if source[index][0] == "#":
                continue # Ignore comments
            if pos == count:
                return index
            count += 1
    if none:
        return None
    else:
        raise ValueError(f"string {string} not found.")

def truefalse(s):
  return s.lower() in ("yes", "true", "t", "y", "1")

def parse(lines, string, default="", vtype='str', req=False):
    n = getLine(lines, string)
    if n is not None:
        line = lines[n].replace(" ","") # remove spaces
        line = line.split("#", 1)[0] # remove comments
        result = line.split("=")[1]
    else:
        if not req:
            result = default
        else:
            print(f" ERROR. Variable {string} is required.")
            exit(1)            

    if vtype == 'str':
        return result.replace("'","").replace('"','')
    elif vtype == 'int':
        return int(result)
    elif vtype == 'bool':
        return truefalse(result)
    elif vtype == 'float':
        return float(result)
    elif vtype == 'afloat':
        return list(map(float, result.split(",")))
    elif vtype == 'aint':
        return list(map(int, result.split(",")))
    else:
        raise TypeError(f"Wrong type: '{vtype}'.")

def parse_input(inputfile):
    global supported_params
    f = open(inputfile, 'r').read().split("\n")
    INPUTVARS = {}
    # Required variables
    INPUTVARS['params'] = parse(f, "params", req=True)
    INPUTVARS['norb'] = parse(f, "norb", req=True, vtype='int')
    INPUTVARS['nroots'] = parse(f, "nroots", req=True, vtype='int')
    
    INPUTVARS['nspin'] = parse(f, "nspin", default="0", vtype='aint')
    INPUTVARS['nelec'] = parse(f, "nelec", default=INPUTVARS['norb'], vtype='int')    
    INPUTVARS['nsamples'] = parse(f, "nsamples", default=0, vtype='int')
    INPUTVARS['var'] = parse(f, "var", default="1.0", vtype='afloat')
    INPUTVARS['tr_tol'] = parse(f, "tr_tol", default=0.02, vtype='float')
    INPUTVARS['memory'] = parse(f, "memory", default=0, vtype='int')
    
    INPUTVARS['save'] = parse(f, "save", default="false", vtype='bool')
    INPUTVARS['cross'] = parse(f, "cross", default="false", vtype='bool')
    INPUTVARS['verbose'] = parse(f, "verbose", default="false", vtype='bool')
    INPUTVARS['debug'] = parse(f, "debug", default="false", vtype='bool')
    INPUTVARS['textbox'] = parse(f, "textbox", default="false", vtype='bool')
    INPUTVARS['dmrg'] = parse(f, "dmrg", default="false", vtype='bool')
    INPUTVARS['dmread'] = parse(f, "dmread", default="false", vtype='bool')
    
    INPUTVARS['fixed_pnames'] = parse(f, "fixed_pnames")
    INPUTVARS['fixed_p'] = parse(f, "fixed_pvalues", default='nan', vtype='afloat')
    INPUTVARS['le_window'] = parse(f, "le_window", default='inf', vtype='float')
    INPUTVARS['rel'] = parse(f, "rel", default="false", vtype='bool')
    INPUTVARS['unit'] = parse(f, "unit", default="au")
    
    INPUTVARS['inputdir'] = parse(f, "inputdir", default=".")
    INPUTVARS['inputname'] = parse(f, "inputname", default="")
    INPUTVARS['outputdir'] = parse(f, "outputdir", default=".")

    params_list = INPUTVARS['params'].split(",")
    for p in params_list:
        if p not in supported_params:
            print(f"ERROR. Parameter {p} not supported.")
            exit(1)

    return INPUTVARS, params_list

########################################################################
# Main program
if __name__=='__main__':
    # Load program input
    import sys
    import os
    import time
    t = time.time()

    if len(sys.argv) < 2:
        print(" ERROR. Input file not specified.")
        exit(1)

    inputfile = sys.argv[1]
    if not os.path.isfile(inputfile):
        print(" ERROR. Input file does not exist.")
        exit(1)

    INP, params_list = parse_input(inputfile)
    norb = INP['norb']
    nelec = INP['nelec']
    nroots = INP['nroots'] 
    nspin = INP['nspin']
    nsamples = INP['nsamples']
    nparams = len(params_list)
    nsamples2 = nsamples*nsamples if INP['cross'] else nsamples 
    SIZE = len(nspin)*(nroots*(1 + nsamples2))

    # Create output
    if not os.path.exists(INP['outputdir']):
        os.makedirs(INP['outputdir'])
    outname = inputfile.replace(".dmd","")
    outfile = open(INP['outputdir'] + "/" + outname + ".out", 'w')
    verbose = INP['verbose']

    print_header(INP, params_list, SIZE, outfile, verbose)

    pprint("Load Data", outfile, verbose, uline=True, skip=True)
    HIJ   = np.zeros(SIZE)
    DATA  = np.zeros((SIZE, nparams))
    TRACE = np.zeros(SIZE)

    idx1 = 0
    ispin = 0
    for spin in nspin:
        pprint(f"Loading calculation files for spin = {spin}...", 
               outfile, verbose, skip=spin!=0)
        mol, mf, mc = load_state(f"S{spin}", norb, nelec, nroots,
                                 INP['inputdir'], INP['inputname'])

        if INP['memory'] > 0: # ideally if memory > 4000
                mem = INP['memory']
                mol.max_memory = mem
                mf.max_memory = mem
                mc.max_memory = mem            
                pprint(f"Memory set to {mem} MB.", outfile, verbose)

        if not INP['dmread']:
            pprint("Generating descriptors...", outfile, verbose)
        else:
            pprint("Reading descriptors from file...", outfile, verbose)

        Hij,A,trace = get_descriptors_eig(mc, nroots, params_list,
                                          nparams, spin, INP['dmread'],
                                          INP['inputdir'],
                                          INP['inputname'])
        idx2 = idx1 + nroots
        
        HIJ[idx1:idx2] = Hij
        DATA[idx1:idx2] = A
        TRACE[idx1:idx2] = trace
        
        idx1 = idx2

        if nsamples > 0:
            if INP['dmread']:
                # dmread = True implies that the mc object does not have
                # the determinant coefficients of each CI state,
                # therefore the density matrices cannot be evaluated in
                # runtime. Without the determinant coefficients it is
                # impossible to perform the non-eiigenstate generation.
                #
                # TO BE IMPLEMENTED: generate an approximate determinant
                # expansion of DMRG calculation during the previous step
                print(" ERROR. Non-eigenstate sampling not supported for DM read mode.")
                exit(1)
            for iroot in range(nroots):
                if INP['debug']: pprint(f" noneig {iroot}:", outfile, verbose)
                noneigs = gen_noneigs(mc.ci[iroot], nsamples, INP['var'],
                                      INP['debug'], INP['outputdir'], ispin)

                energies = np.zeros(nsamples)
                for sample in range(nsamples):
                    if INP['debug']: pprint(f"  energy {sample}:", outfile, verbose)
                    energies[sample] = noneig_energy(mc, noneigs[sample], INP['debug'])
                    if INP['debug']: pprint(f"  energy {sample} done.", outfile, verbose)
                    
                Hij,A,trace = get_descriptors_noneig(mc, noneigs, params_list,
                                                     nparams, energies, spin,
                                                     INP['cross'])
                idx2 = idx1 + nsamples2
                
                HIJ[idx1:idx2] = Hij
                DATA[idx1:idx2] = A
                TRACE[idx1:idx2] = trace
                idx1 = idx2
        ispin += 1

    if INP['rel']:
        HIJ = HIJ - min(HIJ)

    if INP['save']:
        np.savetxt(f'{INP["outputdir"]}/{outname}_DATA.dat',
                   DATA, delimiter=" ", fmt="%16.10f")
        
    # Assess descriptors
    pprint("Assess Data", outfile, verbose, uline=True, skip=True)
    pprint("Assessing descriptors...", outfile, verbose)
    HIJ, DATA, TRACE = filter_points(HIJ, DATA, TRACE, nelec,
                                     INP['tr_tol'], INP['le_window'])
    pprint(f"Points rettained after assessment : {len(HIJ)}", outfile, verbose)
    test_correlations(DATA, params_list, nparams)

    # Correct model to fixed parameters
    fixed_pnames = INP['fixed_pnames']
    fixed_p = INP['fixed_p']
    if fixed_pnames != "":
        if len(fixed_p) == 1:
            fixed_pnames = [fixed_pnames]
        else:
            fixed_pnames = fixed_pnames.split(',')
        pprint(f"Fixed parameters       : {fixed_pnames}", outfile, verbose, skip=True)
        pprint(f"Fixed parameter values : {fixed_p}", outfile, verbose)
        HIJfix, DATAfix, fixed_pindex = fix_model(HIJ, DATA, params_list,
                                                 fixed_pnames, fixed_p)
        # Numpy Least Squares
        P = DMD_fit_fixp(HIJfix, DATAfix, fixed_p, fixed_pindex)
    else:
        P = DMD_fit(HIJ, DATA)

    t= time.time() - t

    DMD_results(HIJ, DATA, P, params_list, nparams, outfile, outname,
                INP['textbox'], INP['unit'], INP['rel'], INP['dmrg'],
                fixed_pnames, INP['outputdir'])

    EMODEL = DATA@P[0]
    with open(f'{INP["outputdir"]}/{outname}_PLOT.dat', 'w') as f:
        f.write(" Calc. Energy    Model Energy     DM1 Trace\n")
        for z in range(SIZE):
            f.write(f'{HIJ[z]:16.10f} {EMODEL[z]:16.10f} {TRACE[z]:16.10f}\n')

    pprint(f"Total calculation time: {t:.1f} seconds ",outfile, verbose, skip=True)

    outfile.close()
