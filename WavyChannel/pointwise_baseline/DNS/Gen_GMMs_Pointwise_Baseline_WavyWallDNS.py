'''
This code contains all necessary implementation of the pointwise method, appropriate pipeline for loading the wavy channel data, and appropriate 
hyperparameters to obtain baseline GMM clusters under the RANS equation. application of sPCA can be done in the 
"DNS_Pointwise_WavyChannel_Baseline.ipynb" file under the "/plotting/" folder to generate all components of our results figure.

Warning: Running this MAY OVERWRITE the current stored results from the GMM used in the paper! 
(although likely you will encounter "folder already exists" errors before it is able to override current results).

Important: It is not necessary to run this code to recreate plots for results, "DNS_Pointwise_WavyChannel_Baseline.ipynb" may be run 
independent of/prior to running this.
'''

import h5py
import jax
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp
from jax import jit ,vmap
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
# from scipy.signal import convolve2d, fftconvolve # check jax
from jax.scipy.signal import fftconvolve
# from jax.scipy.interpolate import RegularGridInterpolator
from jax.scipy.ndimage import map_coordinates
import time
from scipy import sparse, linalg

import os
import numpy
import numpy as np

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
    model = GaussianMixture(n_components=nc, random_state=seed, n_init = 3, init_params=mode)

    # PERMUTATION
    mask = numpy.random.permutation(features.shape[0])[:int(sample_pct*features.shape[0])]
    model.fit(features[mask, :])

    return model


labels = [r'$\bar{u} \bar{u}_x$', r'$\bar{v}\bar{u}_y$',
        r'$\nu \nabla \bar{\bf{u}}$', r'$\rho^{-1} \bar{p}_x$', 
        r'$\overline{({u^\prime} ^2)}_x$', r'$\overline{(u^\prime v^\prime)}_y$']
'''
Import Data
'''

referenceDNS = 'data/boxOutput500500.hdf5'
mean_file = 'data/Mean_1300000-1500000.hdf5'
with h5py.File(referenceDNS, "r") as file:
    list_keys = list(file.keys())[0]
    x = jnp.array(file[list_keys].get('x')).squeeze()
    y = jnp.array(file[list_keys].get('y')).squeeze()

with h5py.File(mean_file, "r") as file:
    list_keys = list(file.keys())[0]
    b = list(file[list_keys].keys()) 

    um = jnp.array(file[list_keys].get('um')[0,:,:]).squeeze()
    vm = jnp.array(file[list_keys].get('vm')[0,:,:]).squeeze()
    # wm = jnp.array(file[list_keys].get('wm')[0,:,:]).squeeze() # In theory, these should go to zero
    pm = jnp.array(file[list_keys].get('pm')[0,:,:]).squeeze()

    uu = jnp.array(file[list_keys].get('uu')[0,:,:]).squeeze()
    uv = jnp.array(file[list_keys].get('uv')[0,:,:]).squeeze()
    rhom = jnp.array(file[list_keys].get('rhom')[0,:,:]).squeeze()

# DNS params
ETA = 1.849e-5
RHO = 1.1839
u_tau_theory = 0.0629

'''
Import Data
'''

nu = ETA/RHO

# Define boundaries
def wavyWallCurve(x_input):
    return 0.1*(1 - jnp.cos(jnp.pi * x_input))

'''
pre-process data
'''

# apply symmetry over 3 waves
x_symm = x[:,48:152]
y_symm = y[:,48:152]
xy_symm = jnp.vstack([x_symm.flatten(), y_symm.flatten()]).T

um_symm = (um[:, 48:152] + um[:, 148:252] + um[:, 248:352])/3
vm_symm = (vm[:, 48:152] + vm[:, 148:252] + vm[:, 248:352])/3
pm_symm = (pm[:, 48:152] + pm[:, 148:252] + pm[:, 248:352])/3
uu_symm = (uu[:, 48:152] + uu[:, 148:252] + uu[:, 248:352])/3
uv_symm = (uv[:, 48:152] + uv[:, 148:252] + uv[:, 248:352])/3
rhom_symm = (rhom[:, 48:152] + rhom[:, 148:252] + rhom[:, 248:352])/3

'''
Setup Test Functions
'''
# TF Range:
x0 = 0.99
xF = 3.01
y0 = 0.0
yF = 0.8

dx = dy = 0.001 # Could reasonably go down to 0.001 when deploying... but should I? at that point maybe we're not getting anything useful.

# Define x & y grid spacing for wavy wall
x_range = jnp.linspace(x0,xF,int((xF-x0)/dx) +1)
y_range = jnp.linspace(y0,yF,int((yF-y0)/dy) +1)

x_centers, y_centers = jnp.meshgrid(x_range,y_range)

# Interpolate DNS data onto evenly spaced grid
xy = jnp.vstack([x.flatten(), y.flatten()]).T # store native coords of DNS data
# Interpolate data
print('start interp')
t0 = time.time()
u_interp = jnp.nan_to_num(griddata(xy_symm, um_symm.flatten(), (x_centers, y_centers), method='linear'))#_symm.flatten()#np.zeros(X_interp.shape)
v_interp = jnp.nan_to_num(griddata(xy_symm, vm_symm.flatten(), (x_centers, y_centers), method='linear'))#
p_interp = jnp.nan_to_num(griddata(xy_symm, pm_symm.flatten(), (x_centers, y_centers), method='linear'))#
uu_interp = jnp.nan_to_num(griddata(xy_symm, uu_symm.flatten(), (x_centers, y_centers), method='linear'))#
uv_interp = jnp.nan_to_num(griddata(xy_symm, uv_symm.flatten(), (x_centers, y_centers), method='linear'))#
rhom_interp = jnp.nan_to_num(griddata(xy_symm, rhom_symm.flatten(), (x_centers, y_centers), method='linear'))#
print('time to interpolate: ' + str(time.time() - t0))

# Calculate Derivatives
compute_derivatives = False
if compute_derivatives:
    nx = len(x_range)
    ny = len(y_range)

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
    uy = Dy @ u_interp.flatten(order='F')
    ux = Dx @ u_interp.flatten(order='F')
    px = Dx @ p_interp.flatten(order='F')

    lap_u = (Dxx + Dyy) @ u_interp.flatten(order='F')
    Ruux = Dx @ uu_interp.flatten(order='F')
    Ruvy = Dy @ uv_interp.flatten(order='F')

    np.save('data/uy.npy', uy)
    np.save('data/ux.npy', ux)
    np.save('data/px.npy', px)
    np.save('data/lap_u.npy', lap_u)
    np.save('data/Ruux.npy', Ruux)
    np.save('data/Ruvy.npy', Ruvy)

'''load computed fields'''
ux = np.load('data/ux.npy').reshape(x_centers.shape[0],x_centers.shape[1], order = 'F')
uy = np.load('data/uy.npy').reshape(x_centers.shape[0],x_centers.shape[1], order = 'F')
px = np.load('data/px.npy').reshape(x_centers.shape[0],x_centers.shape[1], order = 'F')
lap_u = np.load('data/lap_u.npy').reshape(x_centers.shape[0],x_centers.shape[1], order = 'F')
Ruux = np.load('data/Ruux.npy').reshape(x_centers.shape[0],x_centers.shape[1], order = 'F')
Ruvy = np.load('data/Ruvy.npy').reshape(x_centers.shape[0],x_centers.shape[1], order = 'F')

UU_x = u_interp * ux
VU_y = v_interp * uy


# add a support bound lim to truncate error from finite differencing near boundaries
grid_spacing = dx # note dx = dy for our FD grid
support_bound_x = support_bound_y = grid_spacing*3

TF_Centers = jnp.vstack([x_centers.flatten(),y_centers.flatten()]).T

'''
Make the mask to runcate erroneous conolved data
'''
mask = numpy.ones(TF_Centers.shape[0])

# new mask, probably faster?
t0 = time.time()

cutoff = x_range.size # I can change the size of the domain of interest for quick debugging

x_TF_0 = TF_Centers[:cutoff,0] - support_bound_x
x_TF_F = TF_Centers[:cutoff,0] + support_bound_x

ranges = []
idx = 0
for start,end in zip(x_TF_0, x_TF_F):
    ranges.append(jnp.linspace(start,end,int((2*support_bound_x)/grid_spacing + 1)))
    idx+=1

ranges = jnp.array(ranges)

max_wall_boundary = jnp.max(wavyWallCurve(ranges), axis=1)

full_grid_of_max_wall_boundary_for_a_given_TF_center = np.tile(max_wall_boundary, (y_range.size,1))

mask = jnp.where((TF_Centers[:,1]-support_bound_y) <= full_grid_of_max_wall_boundary_for_a_given_TF_center.flatten(), 0, mask)

mask = jnp.reshape(mask, ux.shape)

mask = numpy.array(mask)[0:int(yF/grid_spacing + 1),:]
erroneous_support_buffer_x = int(support_bound_x/dx) + 1
erroneous_support_buffer_y = int(support_bound_y/dy) + 1

# Truncate errors on the bounds
mask[:,0:erroneous_support_buffer_x] = 0
mask[:,(-erroneous_support_buffer_x-1):] = 0
mask[(-erroneous_support_buffer_y-1):,:] = 0

mask = jnp.array(mask.astype(bool))


features = jnp.array([UU_x[mask].flatten(),
                    VU_y[mask].flatten(),
                    nu*(lap_u[mask].flatten()),
                    px[mask].flatten(),
                    Ruux[mask].flatten(),
                    Ruvy[mask].flatten()
                    ]).T


masked_x_coords_DNS_grid = x_centers[mask].flatten()
masked_y_coords_DNS_grid = y_centers[mask].flatten()

save_dir = 'results/'

plot_terms = False
cmax = 0.01
cmin=-0.01
if plot_terms:
    os.mkdir(save_dir + 'terms/')

    np.save(save_dir + 'terms/features', features)

    plt.figure(figsize = (15,4))
    plt.scatter(masked_x_coords_DNS_grid,masked_y_coords_DNS_grid,2,UU_x[mask], vmin=cmin, vmax=cmax)
    plt.colorbar()
    plt.savefig(save_dir + 'terms/' + f'UU_x', dpi=480)
    plt.close()

    plt.figure(figsize = (15,4))
    plt.scatter(masked_x_coords_DNS_grid,masked_y_coords_DNS_grid,2,VU_y[mask], vmin=cmin, vmax=cmax)
    plt.colorbar()
    plt.savefig(save_dir + 'terms/' + f'VU_y',dpi=480)
    plt.close()

    plt.figure(figsize = (15,4))
    plt.scatter(masked_x_coords_DNS_grid,masked_y_coords_DNS_grid,2,nu*(lap_u)[mask], vmin=cmin, vmax=cmax)
    plt.colorbar()
    plt.savefig(save_dir + 'terms/' + f'nu_Lap_U',dpi=480)
    plt.close()

    plt.figure(figsize = (15,4))
    plt.scatter(masked_x_coords_DNS_grid,masked_y_coords_DNS_grid,2,px[mask], vmin=cmin, vmax=cmax)
    plt.colorbar()
    plt.savefig(save_dir + 'terms/' + f'px',dpi=480)
    plt.close()

    plt.figure(figsize = (15,4))
    plt.scatter(masked_x_coords_DNS_grid,masked_y_coords_DNS_grid,2,Ruux[mask], vmin=cmin, vmax=cmax)
    plt.colorbar()
    plt.savefig(save_dir + 'terms/' + f'Ruux',dpi=480)
    plt.close()

    plt.figure(figsize = (15,4))
    plt.scatter(masked_x_coords_DNS_grid,masked_y_coords_DNS_grid,2,Ruvy[mask], vmin=cmin, vmax=cmax)
    plt.colorbar()
    plt.savefig(save_dir + 'terms/' + f'Ruvy',dpi=480)
    plt.close()

nfeatures = 6
no_trials = 10

nc_arr = [12]

for nc in nc_arr:
    nc_save_dir = save_dir + f'nc{nc}/'
    os.mkdir(nc_save_dir)

    for trial in range(no_trials):
        save_dir_trial = nc_save_dir + f'trial_{trial}/' 
        os.mkdir(save_dir_trial)

        model = train_gmm_model(nc,features,sample_pct=0.25)
        cluster_idx = (model.predict(features)+1).astype(int)

        plt.figure(figsize = (10,4))
        plt.scatter(masked_x_coords_DNS_grid,masked_y_coords_DNS_grid,2,cluster_idx, cmap = cm, vmin=-0.5, vmax=cm.N-0.5)
        plt.colorbar(boundaries=jnp.arange(0.5, nc+1.5), ticks=jnp.arange(0, nc+1))
        plt.savefig(save_dir_trial + f'ClusterDomain')
        plt.close()

        np.save(save_dir_trial + f'masked_x_coords_DNS_grid', masked_x_coords_DNS_grid)
        np.save(save_dir_trial + f'masked_y_coords_DNS_grid', masked_y_coords_DNS_grid)
        np.save(save_dir_trial + f'cluster_idx', cluster_idx)


        if nc > 20:
            continue

        elif nc > 16:
            C_list = []
            plt.figure(figsize=(16, 19))
            for j in range(nc):
                plt.subplot(5, 4, j+1)
                # get CVs
                jth_cluster = (cluster_idx == j+1)
                cluster_mask = (np.array([k for k, x in enumerate(jth_cluster) if x])).astype(int)
                # print(cluster_mask[:20])
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
            plt.savefig(save_dir_trial + f'VanillaDataCV,nc{nc}', dpi=1080)
            plt.close()
        elif nc > 12:

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
            plt.savefig(save_dir_trial + f'VanillaDataCV,nc{nc}', dpi=1080)
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
            plt.savefig(save_dir_trial + f'VanillaDataCV,nc{nc}', dpi=1080)
            plt.close()
        elif nc > 6:
            C_list = []
            plt.figure(figsize=(13, 13))
            for j in range(nc):
                plt.subplot(3, 3, j+1)
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
            plt.savefig(save_dir_trial + f'VanillaDataCV,nc{nc}', dpi=1080)
            plt.close()
        else:
            C_list = []
            plt.figure(figsize=(13, 10))
            for j in range(nc):
                plt.subplot(2, 3, j+1)
                # get CVs
                jth_cluster = (cluster_idx == j+1)
                cluster_mask = (np.array([k for k, x in enumerate(jth_cluster) if x])).astype(int)
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
            plt.savefig(save_dir_trial + f'VanillaDataCV,nc{nc}', dpi=1080)
            plt.close()