import numpy as np
import matplotlib.pyplot as plt

# variable parts of payoffs

(A,B,C,D) = (0,0,0,0)
(Ap,Bp,Cp,Dp) = (0,0,0,0)

# demography
s = 0.66
pi = 0.75
F = (1+s**2 - 4*pi*(1-pi)*s**2) / (2 - 4*pi*(1-pi)*s**2)

# set up recursions

def palpha(phi):
    denom = phi[1,1] * (1+A) + phi[1,0] * (1+B) + phi[0,1] * (1+C) + phi[0,0] * (1+D)
    numer = phi[1,1] * (1+A) + phi[1,0] * (1+B)
    return numer / denom

def pomega(phi):
    denom = phi[1,1] * (1+Ap) + phi[1,0] * (1+Bp) + phi[0,1] * (1+Cp) + phi[0,0] * (1+Dp)
    numer = phi[1,1] * (1+Ap) + phi[0,1] * (1+Cp)
    return numer / denom

def heterog(phi, s, pi, pa, pw):
    Phat = pi * pa + (1-pi) * pw
    term01 = (s**2) * pi * (1-pi) + s * (1-s) * (pi * Phat + (1-pi) * (1-Phat))
    term10 = (s**2) * pi * (1-pi) + s * (1-s) * ((1-pi) * Phat + pi * (1-Phat))
    term11 = s * (1-s) * (1-Phat)
    return term01 * phi[0,1] + term10 * phi[1,0] + term11 * (phi[0,0] + phi[1,1]) + Phat * (1-Phat) * (1-s)**2

def homog(phi, s, pi, pa, pw):
    Phat = pi * pa + (1-pi) * pw
    term01 = (s * (1-pi))**2 + 2 * s * (1-s) * (1-pi) * Phat
    term10 = (s * pi)**2 + 2 * s * (1-s) * pi * Phat
    term11 = s**2 + 2 * s * (1-s) * Phat
    return term01 * phi[0,1] + term10 * phi[1,0] + term11 * phi[1,1] + (Phat * (1-s))**2

# data storage
Tf = 100
data = np.empty((Tf, 2, 2), dtype=float)

# initialize population

# randomly
#phi = np.random.uniform(0,1,size=(2,2))
#ttl = np.sum(np.sum(phi))
#phi[:,:] = phi[:,:] / ttl
#phi[0,0] = 1 - phi[0,1] - phi[1,0] - phi[1,1]

phi = np.array([[0.2,0.05],[0.35,0.4]])

tmp = np.empty((2,2),dtype=float)

for t in range(Tf):
    data[t,:,:] = np.copy(phi)
    tmp[0,1] = heterog(phi, s, pi, palpha(phi), pomega(phi))
    tmp[1,0] = heterog(phi, s, pi, palpha(phi), pomega(phi))
    tmp[1,1] = homog(phi, s, pi, palpha(phi), pomega(phi))
    tmp[0,0] = 1-tmp[0,1]-tmp[1,0]-tmp[1,1]
    phi = np.copy(tmp)


# 89mm single col (3.504in), 183mm double column (7.205in)
# can be 1.5 col 120mm (4.724in) or 136mm (5.354in) if necessary
# labels 7pt font, with lowercase type but first letter capitalized
label = np.array([[r"$\phi_{0,0}$", r"$\phi_{0,1}$"],[r"$\phi_{1,0}$", r"$\phi_{1,1}$"] ])
color = np.array([["cyan", "orange"],["magenta", "black"]])
ls = np.array([['-','-'],['--', '--']])
fs = 12
plt.figure(figsize = (4.724, 3.75))
plt.plot([0,1.25 * Tf],[0.5 * F, 0.5 * F], ':', color="gray" )
plt.text(1.125 * Tf, 0.5*F -0.025, r"$F/2$", ha="center", va="top", fontsize=10)
plt.text(1.125 * Tf, 0.5*(1-F)-0.025, r"$(1-F)/2$", ha="center", va="top", fontsize=10)
plt.plot([0,1.25 * Tf],[0.5 * (1-F), 0.5 * (1-F)], ':', color="gray" )
for i in range(2):
    for j in range(2):
        plt.plot( np.arange(Tf), data[:,i,j], "-", color=color[i,j], label=label[i,j], ls=ls[i,j])
plt.ylim([0,1])
plt.ylabel(r"frequency, $\phi_{i,j}$", fontsize=12)
plt.xlabel("time in generations", fontsize=fs)
plt.legend(frameon=False, fontsize=12)
plt.show()
#plt.savefig("FigureA1.pdf", dpi=600)