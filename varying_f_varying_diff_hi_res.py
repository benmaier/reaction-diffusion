# import necessary libraries
import numpy as np
import matplotlib.pyplot as pl

# for the animation
import matplotlib.animation as animation
from matplotlib.colors import Normalize

from scipy.sparse import spdiags

import matplotlib as mpl

def get_laplacian(N):
    """Construct a sparse matrix that applies the 5-point discretization"""
    N = N
    e = np.ones(N**2)
    e2 = ([1]*(N-1)+[0])*N
    e3 = ([0]+[1]*(N-1))*N
    L = spdiags([-4*e,e2,e3,e,e],[0,-1,1,-N,N],N**2,N**2)
    return L

# ============ define relevant functions =============

# an efficient function to compute a mean over neighboring cells
def apply_laplacian(mat):
    """This function applies a discretized Laplacian
    in periodic boundary conditions to a matrix
    For more information see 
    https://en.wikipedia.org/wiki/Discrete_Laplace_operator#Implementation_via_operator_discretization
    """

    # the cell appears 4 times in the formula to compute
    # the total difference
    neigh_mat = -4*mat.copy()

    # Each direct neighbor on the lattice is counted in
    # the discrete difference formula
    neighbors = [ 
                    ( 1.0,  (-1, 0) ),
                    ( 1.0,  ( 0,-1) ),
                    ( 1.0,  ( 0, 1) ),
                    ( 1.0,  ( 1, 0) ),
                ]

    # shift matrix according to demanded neighbors
    # and add to this cell with corresponding weight
    for weight, neigh in neighbors:
        neigh_mat += weight * np.roll(mat, neigh, (0,1))

    return neigh_mat

# Define the update formula for chemicals A and B
def update(A, B, DA, DB, f, k, delta_t, L=None):
    """Apply the Gray-Scott update formula"""

    # compute the diffusion part of the update
    if L is None:
        diff_A = DA * apply_laplacian(A)
        diff_B = DB * apply_laplacian(B)
    else:
        diff_A = DA * L.dot(A)
        diff_B = DB * L.dot(B)
    
    # Apply chemical reaction
    reaction = A*B**2
    diff_A -= reaction
    diff_B += reaction

    # Apply birth/death
    diff_A += f * (1-A)
    diff_B -= (k+f) * B

    A += diff_A * delta_t
    B += diff_B * delta_t

    return A, B

def get_initial_A_and_B(N, random_influence = 0.2):
    """get the initial chemical concentrations"""

    # get initial homogeneous concentrations
    A = (1-random_influence) * np.ones((N,N))
    B = np.zeros((N,N))

    # put some noise on there
    A += random_influence * np.random.random((N,N))
    B += random_influence * np.random.random((N,N))

    # get center and radius for initial disturbance
    N2, r = N//2, 50

    # apply initial disturbance
    A[N2-r:N2+r, N2-r:N2+r] = 0.50
    B[N2-r:N2+r, N2-r:N2+r] = 0.25

    return A.flatten(), B.flatten()

def get_initial_artists(A, B):
    """return the matplotlib artists for animation"""
    dpi = mpl.rcParams['figure.dpi']
    figsize = N / float(dpi), N / float(dpi) 
    fig = pl.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    imA = ax.imshow(A.reshape(N,N), animated=True,vmin=0,vmax=1,cmap='Greys')

    return fig, imA

def updatefig(frame_id,updates_per_frame,*args):
    """Takes care of the matplotlib-artist update in the animation"""

    # update x times before updating the frame
    for u in range(updates_per_frame):
        A, B = update(*args)

    # update the frame
    imA.set_array(A.reshape(N,N))

    # renormalize the colors
    #imA.set_norm(Normalize(vmin=np.amin(A),vmax=np.amax(A)))
    #imB.set_norm(Normalize(vmin=np.amin(B),vmax=np.amax(B)))


    # return the updated matplotlib objects
    return imA,

# =========== define model parameters ==========
# grid size
N = 1000
L = get_laplacian(N)


# update in time
delta_t = 1.1

# Diffusion coefficients
DA = 0.16*1.3
DB = 0.08*1.3

D = (np.ones((N,N)) * np.linspace(0.4,1.3,N)[:,None])

DA = 0.16*D
DB = 0.08*D

DA = DA.flatten()
DB = DB.flatten()

# define birth/death rates
f = (np.ones((N,N)) * np.linspace(0.016,0.040,N)[None,:]).flatten()
f = np.ones((N,N))
for i in range(N):
    for j in range(N):
        r = np.sqrt((i-N/2)**2+(j-N/2)**2) / (N/2)
        f[i,j] = r * (0.045-0.017)+0.016

f = f.flatten()
#f = 0.060
k = 0.062

# intialize the figures
A, B = get_initial_A_and_B(N)


# how many updates should be computed before a new frame is drawn
updates_per_frame = 30

# these are the arguments which have to passed to the update function
animation_arguments = (updates_per_frame, A, B, DA, DB, f, k, delta_t, L)

from progressbar import ProgressBar as PB

bar = PB()

for step in bar(range(30000)):

    update(*animation_arguments[1:])

    if step in [ 400, 1000, 4000, 8000, 15000, 29999 ]:
        fig, imA = get_initial_artists(A, B)
        fig.savefig('img/n_1000_hires_{:d}.png'.format(step))


# show the animation
#ani.save('img/gray_scott_varying_feed_rate.mp4', writer=writer)
# show the animation
