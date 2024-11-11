import numpy as np
import os
from matplotlib import pyplot as plt

def getLine(source, string, pos=0, none=True, comments=True):
	"""Gets the line index of the (`pos`+1)-th
       occurrence of `string` in `source`."""
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
		raise ValueError('string not found')

if not os.path.isfile("result1.dat") \
   or not os.path.isfile("result2.dat"):
	f = open("matching_pursuit.out",'r').readlines()
	n1 = getLine(f, "MATCHING PURSUIT RESULT") + 1
	n2 = getLine(f, "ERRORS AND CORRELATIONS") + 1
	n3 = len(f)

	a = open("result1.dat",'w')
	for i in range(n1,n2-1):
		a.write(f[i])
	a.close()

	a = open("result2.dat",'w')
	for i in range(n2,n3):
		a.write(f[i])
	a.close()
	
params = np.loadtxt("result1.dat", skiprows=1).transpose()
labels = open("result1.dat").readline().split()

i,DE,R2 = np.loadtxt("result2.dat", skiprows=1, unpack=True)

ha2ev=27.211399

params *= ha2ev
DE *= ha2ev

colors=["rosybrown","maroon","seagreen","gold","orange",
        "goldenrod","plum","olive","magenta","red",
        "blue","black","cyan","purple","grey","pink"]

print(len(colors), len(params))

plt.figure()
ax = plt.subplot(111)
for j in range(2,len(params)):
	ax.plot(i[:-1],params[j][:-1], '-o', color=colors[j], label=labels[j])
ax.set_xlabel("MP step")
ax.set_ylabel("Parameter (eV)")
ax.set_ylim((-10.5,10.5))

box = ax.get_position() # Shrink current axis by 20%
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5)) # Put a legend to the right of the current axis

plt.savefig("MP1.png")
plt.show()

plt.figure()
plt.bar(list(range(len(DE))), DE)
plt.xlabel("MP step")
plt.ylabel("RMS error (eV)")
plt.savefig("MP2.png")
plt.show()
