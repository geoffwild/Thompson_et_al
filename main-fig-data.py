import numpy as np

datafile = open("main-fig-data.csv", "w")

# fixed payoffs
sel = 1e-01
(A,Ap) = (sel, sel)
(D,Dp) = (0,0)

# list of variable payoffs
Tvals = np.linspace(0,2,40)
Svals = np.linspace(-1,1,40)

# write header for data file
str2write = "pi,s,R,A,D,C,B,Phat,WI\n"
datafile.write(str2write)

for RV in [0.6,0.9]:
    for philopatry in [0.25,0.5,0.75]:
        for Temptn in Tvals:
            for Sucker in Svals:
                [C,Bp] = [sel*Temptn,sel*Temptn]
                [B,Cp] = [sel*Sucker,sel*Sucker]
                
                # population-genetic simulation
                ###############################
                  
                # set up pop-gen recursions    
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
                    
                # reproductive value of alphas
                pi = RV
                s=philopatry
                     
                # initialize population randomly
                phi = np.random.uniform(0,1,size=(2,2))
                ttl = np.sum(np.sum(phi))
                phi[:,:] = phi[:,:] / ttl
                phi[0,0] = 1 - phi[0,1] - phi[1,0] - phi[1,1]
                tmp = np.empty((2,2),dtype=float)
                
                # iterate until weighted allele frequency is near zero
                tol=1e-12
                maxiter = 1000
                for t in range(maxiter):
                    tmp[0,1] = heterog(phi, s, pi, palpha(phi), pomega(phi))
                    tmp[1,0] = heterog(phi, s, pi, palpha(phi), pomega(phi))
                    tmp[1,1] = homog(phi, s, pi, palpha(phi), pomega(phi))
                    tmp[0,0] = 1-tmp[0,1]-tmp[1,0]-tmp[1,1]
                    # weighted allele frequency
                    wtp_new = pi*(tmp[1,0]+tmp[1,1]) + (1-pi)*(tmp[0,1]+tmp[1,1])
                    wtp_old = pi*(phi[1,0]+phi[1,1]) + (1-pi)*(phi[0,1]+phi[1,1])
                    if np.abs(wtp_new - wtp_old) < tol:
                        break
                    phi = np.copy(tmp)
                    
                # incl fitness data
                ###################
                F = (1+s**2 - 4*pi*(1-pi)*s**2) / (2 - 4*pi*(1-pi)*s**2)
                
                negcalC = (pi*(B-D)+(1-pi)*(Cp-Dp))
                calB = pi*(C-D)+(1-pi)*(Bp-Dp)
                calD = pi*(A+D-B-C) + (1-pi)*(Ap+Dp-Bp-Cp)
                
                WI = negcalC + 0.5*calD + (calB + 0.5*calD) * (2 * F - 1)
                
                # write data to file
                ####################
                str2write = "{:4.3f},{:4.3f},{:4.3f},".format(pi,s,2*F-1)
                str2write +="{:4.3f},{:4.3f},{:4.3f},{:4.3f},".format(A,D,C,B)
                str2write +="{:6.5f},{:6.5f}\n".format(wtp_new,WI)
                datafile.write(str2write)
        
datafile.close()