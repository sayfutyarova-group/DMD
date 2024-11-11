import numpy as np

supported_params = ["E0","tσ","tσps","tσall","tπ",
                    "Up","Udσ","Udπ","Udδ","Ud","Vpd","Vps","J",
                    "εs","εdσ","εpπ","εdπ","εpσ","εdδ","εd","εp","t","U"]

def get_parameter(d1, d2, norb, param, eig):
    if param == "E0": return 1.

    if eig:
        (dm1a, dm1b) = d1
        (dm2aa, dm2ab, dm2bb) = d2
        dm2ba = dm2ab.transpose((2,3,0,1)) # (iijj) --> (jjii)
    else:
        (dm1a, dm1b) = d1
        (dm2aa, dm2ab, dm2ba, dm2bb) = d2

    # 0-body
    if param == "E0":
        return 1.
    # 1-body
    elif param == "tσ":
        return -dm1a[0,3] - dm1a[3,0] - dm1b[0,3] - dm1b[3,0]
    elif param == "tσps":
        return -dm1a[0,8] - dm1a[8,0] - dm1b[0,8] - dm1b[8,0]
    elif param == "tσall":
        return -(dm1a[0,3] + dm1a[3,0] + dm1b[0,3] + dm1b[3,0] + dm1a[0,8] + dm1a[8,0] + dm1b[0,8] + dm1b[8,0])
    elif param == "tπ":
        return -(dm1a[1,4] + dm1a[4,1] + dm1a[2,5] + dm1a[5,2] + dm1b[1,4] + dm1b[4,1] + dm1b[2,5] + dm1b[5,2])
    elif param == "t":
        return -(dm1a[1,4] + dm1a[4,1] + dm1a[2,5] + dm1a[5,2] + dm1b[1,4] + dm1b[4,1] + dm1b[2,5] + dm1b[5,2]) \
               -(dm1a[0,3] + dm1a[3,0] + dm1b[0,3] + dm1b[3,0] + dm1a[0,8] + dm1a[8,0] + dm1b[0,8] + dm1b[8,0])
    elif param == "εs":
        return dm1a[8,8] + dm1b[8,8]
    elif param == "εdσ":
        return dm1a[3,3] + dm1b[3,3]
    elif param == "εpπ":
        return dm1a[1,1] + dm1a[2,2] + dm1b[1,1] + dm1b[2,2]
    elif param == "εdπ":
        return dm1a[4,4] + dm1a[5,5] + dm1b[4,4] + dm1b[5,5]
    elif param == "εpσ":
        return dm1a[0,0] + dm1b[0,0]
    elif param == "εdδ":
        return dm1a[6,6] + dm1a[7,7] + dm1b[6,6] + dm1b[7,7]
    elif param == "εp":
        return dm1a[0,0] + dm1b[0,0] + dm1a[1,1] + dm1a[2,2] + dm1b[1,1] + dm1b[2,2]
    elif param == "εd":
        return dm1a[3,3] + dm1b[3,3] + dm1a[4,4] + dm1a[5,5] + dm1b[4,4] + dm1b[5,5] + dm1a[6,6] + dm1a[7,7] + dm1b[6,6] + dm1b[7,7]

    # 2-body
    elif param == "Up":
        s = 0.0
        for i in range(3):
            s += dm2aa[i,i,i,i] + dm2ab[i,i,i,i] + dm2ba[i,i,i,i] + dm2bb[i,i,i,i]
        return s
    elif param == "Udσ":
        return dm2aa[3,3,3,3] + dm2ab[3,3,3,3] + dm2ba[3,3,3,3] + dm2bb[3,3,3,3]
    elif param == "Udπ":
        s = 0.0
        for i in range(4,4+2):
            s += dm2aa[i,i,i,i] + dm2ab[i,i,i,i] + dm2ba[i,i,i,i] + dm2bb[i,i,i,i]
        return s
    elif param == "Udδ":
        s = 0.0
        for i in range(6,6+2):
            s += dm2aa[i,i,i,i] + dm2ab[i,i,i,i] + dm2ba[i,i,i,i] + dm2bb[i,i,i,i]
        return s
    elif param == "Ud":
        s = 0.0
        for i in range(3,3+5):
            s += dm2aa[i,i,i,i] + dm2ab[i,i,i,i] + dm2ba[i,i,i,i] + dm2bb[i,i,i,i]
        return s
    elif param == "U":
        s = 0.0
        for i in range(norb):
            s += dm2aa[i,i,i,i] + dm2ab[i,i,i,i] + dm2ba[i,i,i,i] + dm2bb[i,i,i,i]
        return s
    elif param == "Vpd":
        s = 0.0
        for i in range(3):
            for j in range(3,3+5):
                s += dm2aa[i,i,j,j] + dm2ba[i,i,j,j] + dm2ab[i,i,j,j] + dm2bb[i,i,j,j] #+ etc. d2[j,j,i,i]
        return s
    elif param == "Vps":
        s = 0.0
        for i in range(3):
            s += dm2aa[i,i,8,8] + dm2ba[i,i,8,8] + dm2ab[i,i,8,8] + dm2bb[i,i,8,8] #+ etc. d2[8,8,i,i]
        return s
    # mixed
    elif param == "J":
        s = 0.0
        for i in range(3,8):
            for j in range(i+1,8):
                s += -0.5*(dm2ab[i,j,j,i] + dm2ba[i,j,j,i])
                s += 0.25*(dm1a[i,i] - dm1b[i,i])*(dm1a[j,j] - dm1b[j,j])
        return s
    else:
        return 0.
