import numpy as np

# User input
params = "E0,J,tσ,tσps,tπ,Ud,Up,Vpd,Vps,εs,εdσ,εpσ,εpπ,εdπ,εdδ".split(',')
e_file = "input/FeSe_model4_PLOT.dat"
d_file = "input/FeSe_model4_DATA.dat"

def r2(x, y):
    import warnings
    warnings.filterwarnings("ignore") # ignore division by zero
    zx = (x-np.mean(x))/np.std(x, ddof=1)
    zy = (y-np.mean(y))/np.std(y, ddof=1)
    r = np.sum(zx*zy)/(len(x)-1)
    return r**2

def LinearFit(x, A):
    P = np.linalg.lstsq(A, x, rcond=None)
    return P

ORDER = []
SOLUTIONS = []
DE_RMS = []
rem_params = params[:]
FIT_R2=[]

energies, dummy, dummy = np.loadtxt(e_file, skiprows=1, unpack=True)
descriptors = np.loadtxt(d_file).transpose()

# 0th iteration
maxcorr = "E0"
ORDER.append(maxcorr)
rem_params.remove(maxcorr)
sol = np.array([energies.mean()])
SOLUTIONS.append(sol)
A = np.transpose([ descriptors[params.index(x)] for x in ORDER ])
de = energies - A@sol
de_rms = np.sqrt( np.square(de).sum()/ len(de) )
DE_RMS.append(de_rms)
R2 = r2(energies, A@sol)
FIT_R2.append(R2)
print("Iteration 0: E0 = ",sol)
print("solution: ",sol," R2 = ",R2," DeltaE_rms = ",de_rms)
print()

# other iterations
for i in range(len(params)-1):
    residue = energies - A@sol

    CORRELS = []

    for j in rem_params:
        # This is the Pearson Correlation Coefficient and NOT the R²!
        # np.corrcoef returns 2x2 matrix, off-diagonal element is corr(x,y)
        corrcoef = np.corrcoef(descriptors[params.index(j)], residue)[0,1]
        print(j,corrcoef)
        CORRELS.append( abs(corrcoef) ) # testar

    print("Iteration ",i+1,": correlations")
    print(CORRELS)

    maxcorr = rem_params[ CORRELS.index(max(CORRELS)) ]

    print("Max correlation: ",maxcorr)
    ORDER.append(maxcorr)
    rem_params.remove(maxcorr)
    A = np.transpose([ descriptors[params.index(x)] for x in ORDER ])
    P = LinearFit(energies, A)
    sol = P[0]
    R2 = r2(energies, A@sol)
    de = energies - A@sol
    de_rms = np.sqrt( np.square(de).sum()/ len(de) )
    FIT_R2.append(R2)
    SOLUTIONS.append(sol)
    DE_RMS.append(de_rms)
    print("solution: ",sol," R2 = ",R2," DeltaE_rms = ",de_rms)
    print()

# make solutions square
for i in range(len(params)):
	SOLUTIONS[i] = np.append(SOLUTIONS[i], [0]*(len(params)-1-i) )

SOLUTIONS = np.array(SOLUTIONS)

# Print out results
print()
print("MATCHING PURSUIT RESULTS")
print(" Iter ", end="")
for i in range(len(params)):
	print(f"{ORDER[i]:18s}", end="")
print()
for i in range(len(params)):
	print(f"{i:5d} ", end="")
	for j in range(len(params)):
		print(f"{SOLUTIONS[i][j]:18.6f}", end="")
	print()

print()
print("ERRORS AND CORRELATIONS")
print(" Iter   DE_rms     R2      ")
for i in range(len(params)):
	print(f"{i:5d} ", end="")
	print(f"{DE_RMS[i]:18.6f}", end="")
	print(f"{FIT_R2[i]:18.6f}")

