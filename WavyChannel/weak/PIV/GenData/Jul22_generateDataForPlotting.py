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

import os
# import cv2
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
sns_list.insert(0, '#ffffff')  # Insert white at zero position
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


'''
TF_x
'''
def TF_x(degree, supp_size_x, supp_size_y, dx, dy):
    # save grid spacing (must be equal for FFT)
    support_bound_x = supp_size_x * dx
    support_bound_y = supp_size_y * dy

    x_domain_TF = jnp.linspace(-support_bound_x, support_bound_x, (supp_size_x * 2) + 1)
    y_domain_TF = jnp.linspace(-support_bound_y, support_bound_y, (supp_size_y * 2) + 1)

    x_domain_full_TF, y_domain_full_TF = jnp.meshgrid(x_domain_TF, y_domain_TF)

    a = -support_bound_x
    b = support_bound_x
    c = -support_bound_y
    d = support_bound_y
    q = p = degree

    C_x = 1 / (p**p * q**q) * ((p+q)/(b-a))**(p+q)
    C_y = 1 / (p**p * q**q) * ((p+q)/(d-c))**(p+q)

    non_func_x = C_x*C_y  * (y_domain_full_TF-c)**p * (d-y_domain_full_TF)**q
    return non_func_x * ( (p*(x_domain_full_TF-a)**(p-1) * (b-(x_domain_full_TF))**q) + ((x_domain_full_TF-a)**p * -q * (b-(x_domain_full_TF))**(q-1)) )

'''
TF_y
'''
def TF_y(degree, supp_size_x, supp_size_y, dx, dy):
    # save grid spacing (must be equal for FFT)
    support_bound_x = supp_size_x * dx
    support_bound_y = supp_size_y * dy

    x_domain_TF = jnp.linspace(-support_bound_x, support_bound_x, (supp_size_x * 2) + 1)
    y_domain_TF = jnp.linspace(-support_bound_y, support_bound_y, (supp_size_y * 2) + 1)

    x_domain_full_TF, y_domain_full_TF = jnp.meshgrid(x_domain_TF, y_domain_TF)

    a = -support_bound_x
    b = support_bound_x
    c = -support_bound_y
    d = support_bound_y
    q = p = degree

    C_x = 1 / (p**p * q**q) * ((p+q)/(b-a))**(p+q)
    C_y = 1 / (p**p * q**q) * ((p+q)/(d-c))**(p+q)

    non_func_y = C_x*C_y  * (x_domain_full_TF-a)**p * (b-x_domain_full_TF)**q
    return non_func_y * ( (p*(y_domain_full_TF-c)**(p-1) * (d-y_domain_full_TF)**q) + ((y_domain_full_TF-c)**p * -q * (d-y_domain_full_TF)**(q-1)) )

'''
TF_xx
'''
def TF_xx(degree, supp_size_x, supp_size_y, dx, dy):
    # save grid spacing (must be equal for FFT)
    support_bound_x = supp_size_x * dx
    support_bound_y = supp_size_y * dy

    x_domain_TF = jnp.linspace(-support_bound_x, support_bound_x, (supp_size_x * 2) + 1)
    y_domain_TF = jnp.linspace(-support_bound_y, support_bound_y, (supp_size_y * 2) + 1)

    x_domain_full_TF, y_domain_full_TF = jnp.meshgrid(x_domain_TF, y_domain_TF)

    a = -support_bound_x
    b = support_bound_x
    c = -support_bound_y
    d = support_bound_y
    q = p = degree

    C_x = 1 / (p**p * q**q) * ((p+q)/(b-a))**(p+q)
    C_y = 1 / (p**p * q**q) * ((p+q)/(d-c))**(p+q)

    non_func_x = C_x*C_y  * (y_domain_full_TF-c)**p * (d-y_domain_full_TF)**q
    return non_func_x * ( (p*(p-1)*(x_domain_full_TF-a)**(p-2) * (b-x_domain_full_TF)**q) - (p*(x_domain_full_TF-a)**(p-1) * q * (b-x_domain_full_TF)**(q-1)) - \
                        (p * (x_domain_full_TF-a)**(p-1) * q * (b-x_domain_full_TF)**(q-1)) + ((x_domain_full_TF-a)**p * q * (q-1) * (b-x_domain_full_TF)**(q-2)) )

'''
TF_yy
'''
def TF_yy(degree, supp_size_x, supp_size_y, dx, dy):
    # save grid spacing (must be equal for FFT)
    support_bound_x = supp_size_x * dx
    support_bound_y = supp_size_y * dy

    x_domain_TF = jnp.linspace(-support_bound_x, support_bound_x, (supp_size_x * 2) + 1)
    y_domain_TF = jnp.linspace(-support_bound_y, support_bound_y, (supp_size_y * 2) + 1)

    x_domain_full_TF, y_domain_full_TF = jnp.meshgrid(x_domain_TF, y_domain_TF)

    a = -support_bound_x
    b = support_bound_x
    c = -support_bound_y
    d = support_bound_y
    q = p = degree

    C_x = 1 / (p**p * q**q) * ((p+q)/(b-a))**(p+q)
    C_y = 1 / (p**p * q**q) * ((p+q)/(d-c))**(p+q)

    non_func_y = C_x * C_y  * (x_domain_full_TF-a)**p * (b-x_domain_full_TF)**q
    return non_func_y * ( (p*(p-1)*(y_domain_full_TF-c)**(p-2) * (d-y_domain_full_TF)**q) - (p*(y_domain_full_TF-c)**(p-1) * q * (d-y_domain_full_TF)**(q-1)) - \
                        (p * (y_domain_full_TF-c)**(p-1) * q * (d-y_domain_full_TF)**(q-1)) + ((y_domain_full_TF-c)**p * q * (q-1) * (d-y_domain_full_TF)**(q-2)) )

labels = [r'$\bar{u} \bar{u}_x$', r'$\bar{v}\bar{u}_y$',
        r'$\nu \nabla \bar{\bf{u}}$',
        r'$\overline{({u^\prime} ^2)}_x$', r'$\overline{(u^\prime v^\prime)}_y$']

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

RST_scale = (u_tau_theory**2) 

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
# mask = cv2.imread(mask_dir)[:,:,0]/255
# mask = mask[:,8:-8] # evaluation excludes 8 px on each side of the image in x
# mask[mask == 0] = np.nan # (nan=wall)

# wall_RAFT = np.zeros(np.shape(mask)[-1]).astype(int)
# for i in range(np.shape(mask)[1]):
#     # print(np.argwhere(np.isnan(mask[:,i]) == True))
#     idx = np.argwhere(np.isnan(mask[:,i]) == True)[-1][0]
#     mask[idx,i] = 1.0
#     wall_RAFT[i] = (idx - 1)

wall_RAFT = np.load('data/wall_Raft.npy')
mask = np.load('data/mask.npy')

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

# support_arr = [(20,20),(20,15),(20,25),(25,20),(25,15),(25,25),(25,30)]
support_arr = [(15,20),(15,15),(30,25),(30,30),(30,15),(30,20)]

# 

for support in support_arr:
    supp_size_x = support[0]
    supp_size_y = support[1]
    degree = 8

    UU_x_weak = 0.5 * fftconvolve(vel_u_trunc * vel_u_trunc, 
                                TF_x(degree, supp_size_x, supp_size_y, res_x,res_y),
                                mode='same')

    # print(UU_x_weak.size)
    # print(np.count_nonzero(np.isnan(UU_x_weak)))
    # raise Exception('stop')

    VU_y_weak = fftconvolve(vel_v_trunc*vel_u_trunc, 
                            TF_y(degree, supp_size_x, supp_size_y, res_x,res_y),
                            mode='same') + UU_x_weak

    # p_x_weak = fftconvolve(p_interp, 
    #                         TF_x(degree, supp_size_x, supp_size_y, res_x,res_y),
    #                         mode='same')

    U_xx_weak = fftconvolve(vel_u_trunc, 
                            TF_xx(degree, supp_size_x, supp_size_y, res_x,res_y),
                            mode='same')

    U_yy_weak = fftconvolve(vel_u_trunc, 
                            TF_yy(degree, supp_size_x, supp_size_y, res_x,res_y),
                            mode='same')

    Ruu_x_weak = fftconvolve(uu, 
                            TF_x(degree, supp_size_x, supp_size_y, res_x,res_y),
                            mode='same')

    Ruv_y_weak = fftconvolve(uv, 
                            TF_y(degree, supp_size_x, supp_size_y, res_x,res_y),
                            mode='same')

    U_x =   fftconvolve(vel_u_trunc, 
                            TF_x(degree, supp_size_x, supp_size_y, res_x,res_y),
                            mode='same')

    # plt.imshow(UU_x_weak)
    # plt.savefig('UU_x_weak')

    # Generate a mask to remove erroneous boundaries from the convolved weak fields
    mask_weak = np.nan_to_num(mask_trunc).astype(int)

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

    # TODO: Now I plot the weak fields, their residuals, and try some DB!!

    res = UU_x_weak + VU_y_weak + NU * (U_xx_weak + U_yy_weak) + Ruu_x_weak + Ruv_y_weak

    UU_x_weak = jnp.where(mask_weak==0, np.nan, UU_x_weak)

    VU_y_weak = jnp.where(mask_weak==0, np.nan, VU_y_weak)

    nu_lap_U = (1/Re_bulk)*jnp.where(mask_weak==0, np.nan, U_xx_weak + U_yy_weak)

    Ruu_x_weak = jnp.where(mask_weak==0, np.nan, Ruu_x_weak)

    Ruv_y_weak = jnp.where(mask_weak==0, np.nan, Ruv_y_weak)

    res = jnp.where(mask_weak==0, np.nan, res)

    clim = 1000


    mask_weak_bool = np.array(mask_weak.astype(bool))

    # print(UU_x_weak[mask_weak_bool].shape)

    # raise Exception('stop')

    features = 1e-3 * np.array([UU_x_weak[mask_weak_bool],
                        VU_y_weak[mask_weak_bool],
                        nu_lap_U[mask_weak_bool],
                        Ruu_x_weak[mask_weak_bool],
                        Ruv_y_weak[mask_weak_bool]]).T


    nc_arr = [13]
    no_trials = 3
    
    nfeatures = 5
    for nc in nc_arr:
        save_dir = f'results/nc{nc}/TFDegree_{degree}/'
        nc_save_dir_1 = save_dir + f'support_x_{supp_size_x}_support_y_{supp_size_y}/'
        os.mkdir(nc_save_dir_1)
        for trial in range(no_trials):

            nc_save_dir = nc_save_dir_1 + f'trial_{trial}/'
            os.mkdir(nc_save_dir)

            model = train_gmm_model(nc,features,sample_pct=0.1)
            cluster_idx = model.predict(features)+1

            np.save(nc_save_dir + 'cluster_idx', cluster_idx)

            
            cluster_idx_im = np.zeros_like(mask_weak.flatten())

            # No need the idx?
            for i, idx in enumerate(np.nonzero(mask_weak.flatten())[0]):
                cluster_idx_im[idx] = cluster_idx[i]

            cluster_idx_im = jnp.where(mask_weak==0, np.nan, jnp.reshape(cluster_idx_im, mask_weak.shape))

            np.save(nc_save_dir + 'cluster_idx_im', cluster_idx_im)

            plt.figure(figsize = (7,4))
            plt.pcolormesh(xx,yy,cluster_idx_im, cmap = cm, vmin=-0.5, vmax=cm.N-0.5)
            plt.colorbar(boundaries=jnp.arange(0.5, nc+1.5), ticks=jnp.arange(0, nc+1))
            plt.savefig(nc_save_dir + f'ClusterDomain')
            plt.close()


            if nc > 20:
                continue

            elif nc > 16:
                C_list = []
                plt.figure(figsize=(16, 19))
                for j in range(nc):
                    plt.subplot(5, 4, j+1)
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
                plt.savefig(nc_save_dir + f'VanillaDataCV,nc{nc}', dpi=480)
                plt.close()
            elif nc > 12:

                C_list = []
                plt.figure(figsize=(16, 16))
                for j in range(nc):
                    plt.subplot(4, 4, j+1)
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
                plt.savefig(nc_save_dir + f'VanillaDataCV,nc{nc}', dpi=480)
                plt.close()
                
            elif nc > 9:

                C_list = []
                plt.figure(figsize=(13, 16))
                for j in range(nc):
                    plt.subplot(4, 3, j+1)
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
                plt.savefig(nc_save_dir + f'VanillaDataCV,nc{nc}', dpi=480)
                plt.close()
            elif nc > 6:
                C_list = []
                plt.figure(figsize=(13, 13))
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
                plt.savefig(nc_save_dir + f'VanillaDataCV,nc{nc}', dpi=480)
                plt.close()
            else:
                C_list = []
                plt.figure(figsize=(13, 10))
                for j in range(nc):
                    plt.subplot(2, 3, j+1)
                    # get CVs
                    jth_cluster = (cluster_idx == j+1)
                    cluster_mask = np.array([k for k, x in enumerate(jth_cluster) if x])
                    C = np.cov(features[cluster_mask].T)
                    C_list.append(C)
                    plt.pcolor(C, vmin=-max(abs(C.flatten())), vmax=max(abs(C.flatten())), cmap='RdBu')
                    plt.gca().set_xticks(np.arange(0.5, nfeatures+0.5))
                    plt.gca().set_xticklabels(labels, fontsize=14)
                    plt.gca().set_yticks(np.arange(0.5, nfeatures+0.5))
                    plt.gca().set_yticklabels(labels, fontsize=14)
                    plt.gca().set_title('Cluster {0}'.format(j+1), fontsize=14)

                plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.3)
                plt.suptitle("CV of CFD Data in Ensembled Clusters")
                plt.savefig(nc_save_dir + f'VanillaDataCV,nc{nc}', dpi=480)
                plt.close()
