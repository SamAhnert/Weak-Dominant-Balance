'''
This code contains all necessary implementation of the pointwise method, appropriate pipeline for loading the wavy channel data, and appropriate 
hyperparameters to obtain baseline GMM clusters under the RANS equation. application of sPCA can be done in the 
"PIV_Pointwise_WavyChannel_Baseline.ipynb" file under the "/plotting/" folder to generate all components of our results figure.

Warning: Running this MAY OVERWRITE the current stored results from the GMM used in the paper! 
(although likely you will encounter "folder already exists" errors before it is able to override current results).

Important: It is not necessary to run this code to recreate plots for results, "PIV_Pointwise_WavyChannel_Baseline.ipynb" may be run 
independent of/prior to running this.
'''

import h5py
import jax
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp
from jax import jit ,vmap
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.signal import convolve2d, fftconvolve # check jax
# from jax.scipy.signal import fftconvolve
# from jax.scipy.interpolate import RegularGridInterpolator
from jax.scipy.ndimage import map_coordinates
import time
from scipy import sparse

import os
import cv2
import numpy as np
import natsort

from numpy.random import randint
import sklearn as sk
from sklearn.mixture import GaussianMixture # jaxxx??? # Hyakkk
from sklearn.decomposition import SparsePCA
from scipy.io import loadmat
import matplotlib as mpl
from matplotlib.colors import ListedColormap
# Seaborn colormap
import seaborn as sns
import colorcet as cc
sns_list = sns.color_palette(cc.glasbey,n_colors=20).as_hex()
sns_list.pop(4)

# print(sns_list)
# raise Exception('stop')

sns_list.insert(0, '#ffffff')  # Insert white at zero position
sns_list.pop(1)
sns_cmap = ListedColormap(sns_list)
cm = sns_cmap


mpl_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
            '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
            '#bcbd22', '#17becf']

def train_gmm_model(nc, features, seed=-1, sample_pct=0.95, mode='kmeans'):
    # use random seed if not specified
    if seed == -1:
        seed = randint(2**32)
    else:
        pass
    
    print(seed)
    model = GaussianMixture(n_components=nc, random_state=seed, n_init = 5, init_params=mode)

    # PERMUTATION
    mask = np.random.permutation(features.shape[0])[:int(sample_pct*features.shape[0])]
    model.fit(features[mask, :])

    return model

labels = [r'$\bar{u} \bar{u}_x$', r'$\bar{v}\bar{u}_y$',
        r'$\nu \nabla \bar{\bf{u}}$',
        r'$\overline{({u^\prime} ^2)}_x$', r'$\overline{(u^\prime v^\prime)}_y$']

# labels = [r'$\bar{u} \bar{u}_x$', r'$\bar{v}\bar{u}_y$',
#         r'$\nu \nabla \bar{\bf{u}}$', 
#         r'$\overline{({u^\prime} ^2)}_x$', r'$\overline{(u^\prime v^\prime)}_y$']

'''
Import Data
'''
uv_file = 'data/uv_RAFT.npy'
uu_file = 'data/uu_RAFT.npy'


input_dir = 'data/mean_velocity.npz'
mask_dir = 'data/mask_2D2C_2014.12_W100_10_V1ms_PD350_B00001_A.png'

wavelength_real = 0.1 # wavelength in [m]
amplitude_real = 5e-3 # wave amplitude in [m]
del_ts = 350e-6 # temporal distance between two consecutive samples [s]
fsize = 20 # plots, fontsize
u_tau_theory = 0.0629 # for a flat (classical) turbulent channel flow at the same Re_tau = 201
NU = 1.562e-5 # at 25 degree  
ETA = 1.849e-5
RHO = ETA/NU
HalfHeight = 0.05 # channel half height in [m]

#Exp_no = 13 # cd NEW_DB/Weak/WavyWallWeak/PIV_Data/TestDB_scaled_RST/

#RST_scale = (u_tau_theory**2) / (HalfHeight)# Assuming Half height is characteristic lenght for bulk #* (HalfHeight**2)/(NU**2) #* (0.05)**2/(1.562e-5)#1.0/3201.02432778
RST_scale = (u_tau_theory**2) #(u_tau_theory**2) #wavelength_real * (u_tau_theory**2)#(u_tau_theory**2) # * HalfHeight#(u_tau_theory**2) * HalfHeight
# print(RST_scale)
# vel_scale = 

uv = jnp.nan_to_num(jnp.load(uv_file))*RST_scale
uu = jnp.nan_to_num(jnp.load(uu_file))*RST_scale

# Re_bulk = HalfHeight/NU
Re_bulk = 1/(NU)

print(f'Re_bulk: {Re_bulk}')

# each pixel (or cell) is a rectangle with the horizontal length being res_x and the vertical length being res_y
# if using the coordinates, assume that the respective velocity value is at the cell center, 
#  i.e., the first velocity value above the wall is at res_y/2 (and the most left value is at res_x/2)
res_x = 4.22119038e-5 # resolution of the data in horizontal direction [m/pixel]
res_y = 4.18410042e-5 # in vertical direction

# -----------------------------------------------------------------------------------------------------
#  load wall contour data
# mask
mask = cv2.imread(mask_dir)[:,:,0]/255
mask = mask[:,8:-8] # evaluation excludes 8 px on each side of the image in x
mask[mask == 0] = np.nan # (nan=wall)

wall_RAFT = np.zeros(np.shape(mask)[-1]).astype(int)
for i in range(np.shape(mask)[1]):
    # print(np.argwhere(np.isnan(mask[:,i]) == True))
    idx = np.argwhere(np.isnan(mask[:,i]) == True)[-1][0]
    mask[idx,i] = 1.0
    wall_RAFT[i] = (idx - 1)

vel_u = np.load(input_dir)['u_mean'] # mean over all samples
vel_v = np.load(input_dir)['v_mean'] # mean over all samples

'''
Okay... how do we get this on the Weak DB
Workflow:

1.) Normally I would need to interpolate onto a uniform grid, but bc dy ~= dx, I'll only do this if necessary.
Do need to truncate lower half of the wavy wall.

2.) Convolve using "same", so I can manually delete the error that arises from zero-padding the boundaries

3.) Create a mask to truncate both the error from the zero-padding on the boundaries, and the errorneous integration where the wall fell in the domain of the test-function.
This will prob be the hardest part.

4.) Feed truncated weak fields into DB and see what happens!
'''

vel_u_trunc = jnp.nan_to_num(vel_u[:uu.shape[0], :])
vel_v_trunc = jnp.nan_to_num(vel_v[:uu.shape[0], :])
mask_trunc = mask[:uu.shape[0], :]

xx = np.linspace(0, uu.shape[1], uu.shape[1])
yy = np.linspace(0, uu.shape[0], uu.shape[0])



# Calculate Derivatives
compute_derivatives = False
if compute_derivatives:
    # dx = float(x[1]-x[0])
    # print(y.shape)
    # print(y[1:]-y[:-1])
    # dy = float(y[1]-y[0])

    # print(dy)
    # raise Exception('stop')

    nx = len(xx)
    ny = len(yy)

    dx = res_x
    dy = res_y

    Dy = sparse.diags([-1, 1], [-1, 1], shape=(ny, ny)).toarray()
    Dy[0, :3] = np.array([-3, 4, -1])
    Dy[-1, -3:] = np.array([1, -4, 3])
    Dy = Dy / (2*dy)
    Dy = sparse.block_diag([Dy for j in range(nx)])
    Dy = sparse.csr_matrix(Dy)

    Dx = sparse.diags( [-1, 1], [-ny, ny], shape=(nx*ny, nx*ny))
    Dx = sparse.lil_matrix(Dx)
    # Second-order forwards/backwards at boundaries
    for j in range(ny):
        Dx[j, j] = -3
        Dx[j, ny+j] = 4
        Dx[j, 2*ny+j] = -1
        Dx[-(j+1), -(j+1)] = 3
        Dx[-(j+1), -(ny+j+1)] = -4
        Dx[-(j+1), -(2*ny+j+1)] = 1
    Dx = Dx/(2*dx)

    Dx = sparse.csr_matrix(Dx)
    Dy = sparse.csr_matrix(Dy)

    Dxx = 2*(Dx @ Dx)
    Dyy = 2*(Dy @ Dy)
    uy = Dy @ vel_u_trunc.flatten(order='F')
    np.save('results/terms/uy.npy', uy)
    print('uy got')
    ux = Dx @ vel_u_trunc.flatten(order='F')
    np.save('results/terms/ux.npy', ux)
    print('ux got')

    lap_u = (Dxx + Dyy) @ vel_u_trunc.flatten(order='F')
    Ruux = Dx @ uu.flatten(order='F')
    Ruvy = Dy @ uv.flatten(order='F')

    
    # np.save('results/terms/ux.npy', ux)
    # np.save('results/terms/px.npy', px)
    np.save('results/terms/lap_u.npy', lap_u)
    np.save('results/terms/Ruux.npy', Ruux)
    np.save('results/terms/Ruvy.npy', Ruvy)


ux = np.load('results/terms/ux.npy').reshape(vel_u_trunc.shape,order='F')
uy = np.load('results/terms/uy.npy').reshape(vel_u_trunc.shape,order='F')
lap_u = np.load('results/terms/lap_u.npy').reshape(vel_u_trunc.shape,order='F')
Ruux = np.load('results/terms/Ruux.npy').reshape(vel_u_trunc.shape,order='F')
Ruvy = np.load('results/terms/Ruvy.npy').reshape(vel_u_trunc.shape,order='F')

UU_x = vel_u_trunc * ux
VU_y = vel_v_trunc * uy



# Generate a mask to remove erroneous boundaries from the convolved weak fields
mask_weak = np.nan_to_num(mask_trunc).astype(int)

# np.save('mask_weak.npy',mask_weak)

# add a support bound lim to truncate error from finite differencing near boundaries
# grid_spacing = dx # note dx = dy for our FD grid
supp_size_x = supp_size_y = 3
# support_bound_x = support_bound_y = grid_spacing*supp_size_x

y_upper_bound_cutoff = 500

for i in range(mask_weak.shape[0]):
    for j in range(mask_weak.shape[1]):
        # Handle y BCS
        if i <= supp_size_y or i >= (mask_weak.shape[0]-1)-y_upper_bound_cutoff:
            mask_weak[i,j] = 0
            continue

        # Handle x BCS
        if j <= supp_size_x or j >= (mask_weak.shape[1]-1)-supp_size_x:
            mask_weak[i,j] = 0
            continue

        # check for nan in the support of the TF centered at the given point in mask
        local_support = mask[i-supp_size_y:i+supp_size_y+1,j-supp_size_x:j+supp_size_x+1]
        # print(local_support)
        # raise Exception('stop')
        if np.isnan(np.sum(local_support)):
            mask_weak[i,j] = 0

res = UU_x + VU_y + NU * (lap_u) + Ruux + Ruvy

UU_x = jnp.where(mask_weak==0, np.nan, UU_x)

VU_y = jnp.where(mask_weak==0, np.nan, VU_y)

nu_lap_U = (1/Re_bulk)*jnp.where(mask_weak==0, np.nan, lap_u)

Ruux = jnp.where(mask_weak==0, np.nan, Ruux)

Ruvy = jnp.where(mask_weak==0, np.nan, Ruvy)

res = jnp.where(mask_weak==0, np.nan, res)

mask_weak_bool = np.array(mask_weak.astype(bool))

features = np.array([UU_x[mask_weak_bool],
                    VU_y[mask_weak_bool],
                    nu_lap_U[mask_weak_bool],
                    Ruux[mask_weak_bool],
                    Ruvy[mask_weak_bool]]).T

save_dir = 'results/'
nc_arr = [7,8,9,10,11,12,13,14]
nfeatures = 5
no_trials = 5

np.save('results/terms/xx.npy',xx)
np.save('results/terms/yy.npy',yy)
np.save('mask_weak_bool.npy',mask_weak_bool)
np.save('results/terms/vel_u_trunc.npy',vel_u_trunc)
np.save('results/terms/vel_v_trunc.npy',vel_v_trunc)

raise Exception('stop for time trial')

for nc in nc_arr:
    nc_save_dir = save_dir + f'nc{nc}/'
    os.mkdir(nc_save_dir)

    for trial in range(no_trials):
        trial_save_dir = nc_save_dir + f'trial{trial}/'
        os.mkdir(trial_save_dir)

        model = train_gmm_model(nc,features,sample_pct=0.1)
        cluster_idx = model.predict(features)+1

        np.save(trial_save_dir + 'cluster_idx', cluster_idx)

        
        cluster_idx_im = np.zeros_like(mask_weak.flatten())

        # No need the idx?
        for i, idx in enumerate(np.nonzero(mask_weak.flatten())[0]):
            cluster_idx_im[idx] = cluster_idx[i]

        cluster_idx_im = jnp.where(mask_weak==0, np.nan, jnp.reshape(cluster_idx_im, mask_weak.shape))

        np.save(trial_save_dir + 'cluster_idx_im', cluster_idx_im)

        plt.figure(figsize = (7,4))
        plt.pcolormesh(xx,yy,cluster_idx_im, cmap = cm, vmin=-0.5, vmax=cm.N-0.5)
        plt.colorbar(boundaries=jnp.arange(0.5, nc+1.5), ticks=jnp.arange(0, nc+1))
        plt.savefig(trial_save_dir + f'ClusterDomain')
        plt.close()

        if nc > 12:

            C_list = []
            plt.figure(figsize=(16, 16))
            for j in range(nc):
                plt.subplot(4, 4, j+1)
                # get CVs
                jth_cluster = (cluster_idx == j+1)
                cluster_mask = (np.array([k for k, x in enumerate(jth_cluster) if x])).astype(int)
                C = np.cov(features[cluster_mask].T)
                C_list.append(C)
                plt.pcolor(C, vmin=-max(abs(C.flatten())), vmax=max(abs(C.flatten())), cmap='RdBu')
                plt.gca().set_xticks(np.arange(0.5, nfeatures+0.5))
                plt.gca().set_xticklabels(labels, fontsize=8)
                plt.gca().set_yticks(np.arange(0.5, nfeatures+0.5))
                plt.gca().set_yticklabels(labels, fontsize=8)
                plt.gca().set_title('Cluster {0}'.format(j+1), fontsize=14)

            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.3)
            plt.suptitle("CV of CFD Data in Ensembled Clusters")
            plt.savefig(trial_save_dir + f'VanillaDataCV,nc{nc}', dpi=1080)
            plt.close()

        elif nc > 9:

            C_list = []
            plt.figure(figsize=(13, 16))
            for j in range(nc):
                plt.subplot(4, 3, j+1)
                # get CVs
                jth_cluster = (cluster_idx == j+1)
                cluster_mask = (np.array([k for k, x in enumerate(jth_cluster) if x])).astype(int)
                C = np.cov(features[cluster_mask].T)
                C_list.append(C)
                plt.pcolor(C, vmin=-max(abs(C.flatten())), vmax=max(abs(C.flatten())), cmap='RdBu')
                plt.gca().set_xticks(np.arange(0.5, nfeatures+0.5))
                plt.gca().set_xticklabels(labels, fontsize=8)
                plt.gca().set_yticks(np.arange(0.5, nfeatures+0.5))
                plt.gca().set_yticklabels(labels, fontsize=8)
                plt.gca().set_title('Cluster {0}'.format(j+1), fontsize=14)

            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.3)
            plt.suptitle("CV of CFD Data in Ensembled Clusters")
            plt.savefig(trial_save_dir + f'VanillaDataCV,nc{nc}', dpi=1080)
            plt.close()

        else:
            C_list = []
            plt.figure(figsize=(9, 11))
            for j in range(nc):
                plt.subplot(3, 3, j+1)
                # get CVs
                jth_cluster = (cluster_idx == j+1)
                cluster_mask = np.array([k for k, x in enumerate(jth_cluster) if x])
                C = np.cov(features[cluster_mask].T)
                C_list.append(C)
                plt.pcolor(C, vmin=-max(abs(C.flatten())), vmax=max(abs(C.flatten())), cmap='RdBu')
                plt.gca().set_xticks(np.arange(0.5, nfeatures+0.5))
                plt.gca().set_xticklabels(labels, fontsize=8)
                plt.gca().set_yticks(np.arange(0.5, nfeatures+0.5))
                plt.gca().set_yticklabels(labels, fontsize=8)
                plt.gca().set_title('Cluster {0}'.format(j+1), fontsize=14)

            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.3)
            plt.suptitle("CV of CFD Data in Ensembled Clusters")
            plt.savefig(trial_save_dir + f'VanillaDataCV,nc{nc}')
            plt.close()
