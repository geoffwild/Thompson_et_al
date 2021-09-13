import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv( "main-fig-data.csv", header = 0 )

def alphas(Phat):
    if Phat < 0.5:
        return 0.125
    else:
        return 0.75

def colors(WI):
    if WI > 0:
        return "blue"
    else:
        return "red"

def panel(pi, s):
    if pi > 0.625:
        row = 0
    else:
        row = 1
    if s < 0.375:
        col = 0
    elif (s > 0.375 and s < 0.625):
        col = 1
    else:
        col = 2
    return [row, col]
    
fig, ax = plt.subplots( nrows=2, ncols=3, figsize=(9,7), sharex='col',sharey='row' )

Rvals=-np.ones( (2,3), dtype=float )

for index in df.index:
    # determine panel in which to plot data
    pi = df.loc[index,'pi']
    s = df.loc[index,'s']
    i,j = panel(pi,s)
    
    # grab the relatedness value
    R = df.loc[index,'R']
    if Rvals[i,j] < 0:
        # we haven't yet recorded the R for this panel
        Rvals[i,j] = R
    
    # find x and y coords to plot
    xcoord = df.loc[index,'C']
    ycoord = df.loc[index,'B']
    
    # determine style of point to plot
    Phat=df.loc[index,'Phat']
    alpha=alphas(Phat)
    WI = df.loc[index,'WI']
    color=colors(WI)

    # plot the data
    ax[i,j].plot(xcoord, ycoord, 'o', color=color, alpha=alpha)
    
# tidy up the plot
for i in range(2):
    for j in range(3):
        ax[i,j].vlines(0.1,-0.1,0.1,lw=4,color='k')
        ax[i,j].hlines(0.0,0.0,0.2,lw=4,color='k')
        ax[i,j].set_xlim(0.0,0.2)
        ax[i,j].set_ylim(-0.1,0.1)
        Rstr = r"$R =$ {:3.2f}".format(Rvals[i,j])
        ax[i,j].text(0.01, 0.08, Rstr, color="white", fontsize=14)
        ax[i,j].text(0.01, -0.075, "SH", color="white")
        ax[i,j].text(0.17, 0.08, "SG", color="white")
        if (i==0 and j==2):
            ax[i,j].text(0.17, -0.075, "PD", color="white")
        else:
            ax[i,j].text(0.17, -0.075, "PD", color="black")

for i in range(2):
    ax[i,0].set_yticks(ticks=[-0.10,-0.05,0,0.05,0.10])
    ax[i,0].set_ylabel(r"$B = C'$", fontsize=12)

for j in range(3):
    ax[1,j].set_xticks(ticks=[0,0.05,0.10,0.15,0.20])
    ax[1,j].set_xlabel(r"$C = B'$", fontsize=12)

plt.subplots_adjust( left= 0.1, right=0.975, wspace=0.15, hspace=0.1, bottom=0.1, top=0.95 )

plt.savefig("main-fig.pdf", dpi=600)