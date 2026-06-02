'''
This code contains all necessary implementation of the weak method, appropriate pipeline for loading the Wavy Channel DNS data, and appropriate hyperparameters
to run weak dominant balance on the Wavy Channel flow under the RANS equation to recreate our final GMMs. application of sPCA can be done in the 
"DNS_Weak_WavyChannel_sPCA_Reduction.ipynb" file under the "/DNS/plotting/" folder, along with "DNS_Weak_WavyChannel_EqnSpace.ipynb" to generate other
components of our results figure.

Warning: Running this MAY OVERWRITE the current stored results from the GMM used in the paper! 
(although likely you will encounter "folder already exists" errors before it is able to override current results).

Important: It is not necessary to run this code to recreate plots for results, "DNS_Weak_WavyChannel_sPCA_Reduction.ipynb" may be run independent of/prior 
to this.
'''

import h5py
import jax
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp
from jax import jit ,vmap
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
# from scipy.signal import convolve2d, fftconvolve # check jax
from jax.scipy.signal import fftconvolve, correlate
# from jax.scipy.interpolate import RegularGridInterpolator
from jax.scipy.ndimage import map_coordinates
import time

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
    model = GaussianMixture(n_components=nc, random_state=seed, n_init = 3, init_params=mode)

    # PERMUTATION
    mask = numpy.random.permutation(features.shape[0])[:int(sample_pct*features.shape[0])]
    model.fit(features[mask, :])

    return model


'''
TF
'''

def TF_ref(x_ref, y_ref, degree):
    """Reference test function"""
    p = degree
    return (1 - x_ref**2)**p * (1 - y_ref**2)**p

def TF_old_scaled(degree, support_bound_x, support_bound_y, dx):
    """Base test function scaled from reference domain [-1,1]^2"""
    Lx = 2 * support_bound_x
    Ly = 2 * support_bound_y

    x_ref = jnp.linspace(-1, 1, round(Lx / dx) + 1)
    y_ref = jnp.linspace(-1, 1, round(Ly / dx) + 1)
    x_ref_full, y_ref_full = jnp.meshgrid(x_ref, y_ref)

    TF = TF_ref(x_ref_full, y_ref_full, degree)

    return TF

def TF_x_scaled(degree, support_bound_x, support_bound_y, dx):
    """X-Derivative in physical coordinates"""
    Lx = 2 * support_bound_x
    Ly = 2 * support_bound_y
    p = degree

    x_ref = jnp.linspace(-1, 1, round(Lx / dx) + 1)
    y_ref = jnp.linspace(-1, 1, round(Ly / dx) + 1)
    x_ref_full, y_ref_full = jnp.meshgrid(x_ref, y_ref)

    # Reference derivatives
    dphi_dx_ref = -2 * p * x_ref_full * (1 - x_ref_full**2)**(p-1) * (1 - y_ref_full**2)**p
    # Physical scaling
    TF_x = dphi_dx_ref * (2 / Lx) 
    return TF_x

def TF_y_scaled(degree, support_bound_x, support_bound_y, dx):
    """Y-Derivative in physical coordinates"""
    Lx = 2 * support_bound_x
    Ly = 2 * support_bound_y
    p = degree

    x_ref = jnp.linspace(-1, 1, round(Lx / dx) + 1)
    y_ref = jnp.linspace(-1, 1, round(Ly / dx) + 1)
    x_ref_full, y_ref_full = jnp.meshgrid(x_ref, y_ref)

    dphi_dy_ref = -2 * p * y_ref_full * (1 - y_ref_full**2)**(p-1) * (1 - x_ref_full**2)**p
    TF_y = (2 / Ly) * dphi_dy_ref
    return TF_y

def TF_xx_scaled(degree, support_bound_x, support_bound_y, dx):
    """Second X-derivative"""
    Lx = 2 * support_bound_x
    Ly = 2 * support_bound_y
    p = degree

    x_ref = jnp.linspace(-1, 1, round(Lx / dx) + 1)
    y_ref = jnp.linspace(-1, 1, round(Ly / dx) + 1)
    x_ref_full, y_ref_full = jnp.meshgrid(x_ref, y_ref)

    # Second derivative in reference coords
    d2phi_dx2_ref = (
        -2 * p * (1 - y_ref_full**2)**p *
        ((1 - x_ref_full**2)**(p-2)) *
        ((1 - x_ref_full**2) - 2 * (p-1) * x_ref_full**2)
    )

    TF_xx = (2 / Lx)**2 * d2phi_dx2_ref
    return TF_xx

def TF_yy_scaled(degree, support_bound_x, support_bound_y, dx):
    """Second Y-derivative"""
    Lx = 2 * support_bound_x
    Ly = 2 * support_bound_y
    p = degree

    x_ref = jnp.linspace(-1, 1, round(Lx / dx) + 1)
    y_ref = jnp.linspace(-1, 1, round(Ly / dx) + 1)
    x_ref_full, y_ref_full = jnp.meshgrid(x_ref, y_ref)

    d2phi_dy2_ref = (
        -2 * p * (1 - x_ref_full**2)**p *
        ((1 - y_ref_full**2)**(p-2)) *
        ((1 - y_ref_full**2) - 2 * (p-1) * y_ref_full**2)
    )

    TF_yy = (2 / Ly)**2 * d2phi_dy2_ref
    return TF_yy

def TF_xy_scaled(degree, support_bound_x, support_bound_y, dx):
    """Mixed X/Y-derivative"""
    Lx = 2 * support_bound_x
    Ly = 2 * support_bound_y
    p = degree

    x_ref = jnp.linspace(-1, 1, round(Lx / dx) + 1)
    y_ref = jnp.linspace(-1, 1, round(Ly / dx) + 1)
    x_ref_full, y_ref_full = jnp.meshgrid(x_ref, y_ref)

    d2phi_dxdy_ref = (
        (4 * p**2) * x_ref_full * y_ref_full *
        (1 - x_ref_full**2)**(p-1) * (1 - y_ref_full**2)**(p-1)
    )

    TF_xy = (2 / Lx) * (2 / Ly) * d2phi_dxdy_ref
    return TF_xy


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
pre-process data to average over symmetric waves
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

dx = dy = 0.001 # Try to match the pixel count in PIV

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

# # Converged candidate, large enough to let the numerical integral converge, small enough to focus on dynamics in a small locale
support_arr_x = np.array([0.02]) #
support_arr_y = np.array([0.015]) #

'''
NOTE: Here we use the correlation function to give a more intuitive basis for how the fftconvolve is working (so we don't bring the negatives out of nowhere)

But it still gives the same results, just the code now mirrors the math derivation of the weak form, and mirroring that is probably more intuitive for someone trying to understand the code along with the paper...
'''

for support_bound_y in support_arr_y:
    for support_bound_x in support_arr_x:
        grid_spacing = dx
        degree = 5 # defintely can boot this higher if wanting other equations in the future

        print('start convolutions')
        # Expand the term and note it pops out of itself. use this to fully remove the derivative from the field
        UU_x_weak = -0.5 * correlate(u_interp*u_interp, 
                                TF_x_scaled(degree, support_bound_x, support_bound_y, grid_spacing),
                                mode='same',
                                method='fft')
        # Expand the term and apply the incompressibility constraint (here in 2D) to fully remove derivative from any data field.
        VU_y_weak = -correlate(v_interp*u_interp, 
                                TF_y_scaled(degree, support_bound_x, support_bound_y, grid_spacing),
                                mode='same',
                                method='fft') + UU_x_weak

        p_x_weak = -correlate(p_interp, 
                                TF_x_scaled(degree, support_bound_x, support_bound_y, grid_spacing),
                                mode='same',
                                method='fft')

        U_xx_weak = correlate(u_interp, 
                                TF_xx_scaled(degree, support_bound_x, support_bound_y, grid_spacing),
                                mode='same',
                                method='fft')

        U_yy_weak = correlate(u_interp, 
                                TF_yy_scaled(degree, support_bound_x, support_bound_y, grid_spacing),
                                mode='same',
                                method='fft')

        Ruu_x_weak = -correlate(uu_interp, 
                                TF_x_scaled(degree, support_bound_x, support_bound_y, grid_spacing),
                                mode='same',
                                method='fft')

        Ruv_y_weak = -correlate(uv_interp, 
                                TF_y_scaled(degree, support_bound_x, support_bound_y, grid_spacing),
                                mode='same',
                                method='fft')

        U_x_weak =   -correlate(u_interp, 
                                TF_x_scaled(degree, support_bound_x, support_bound_y, grid_spacing),
                                mode='same',
                                method='fft')

        # print('time to convolve: ' + str(time.time() - t1))

        TF_Centers = jnp.vstack([x_centers.flatten(),y_centers.flatten()]).T

        '''
        Make the mask to runcate erroneous conolved data
        '''
        mask = numpy.ones(TF_Centers.shape[0])

        # new mask, probably faster?
        t0 = time.time()

        cutoff = x_range.size # I can hardcode this to change the size of the domain of interest for quick debugging

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

        mask = jnp.reshape(mask, UU_x_weak.shape)

        mask = numpy.array(mask)[0:int(yF/grid_spacing + 1),:]
        erroneous_support_buffer_x = int(support_bound_x/dx) + 1
        erroneous_support_buffer_y = int(support_bound_y/dy) + 1

        # Truncate errors on the bounds
        mask[:,0:erroneous_support_buffer_x] = 0
        mask[:,(-erroneous_support_buffer_x-1):] = 0
        mask[(-erroneous_support_buffer_y-1):,:] = 0

        mask = jnp.array(mask.astype(bool))

        features = jnp.array([UU_x_weak[mask].flatten(),
                            VU_y_weak[mask].flatten(),
                            nu*(U_xx_weak[mask].flatten() + U_yy_weak[mask].flatten()),
                            p_x_weak[mask].flatten(),
                            Ruu_x_weak[mask].flatten(),
                            Ruv_y_weak[mask].flatten()
                            ]).T


        print('full features made')

        nc_arr = [11]

        no_x_support_pts = round(support_bound_x/grid_spacing)
        no_y_support_pts = round(support_bound_y/grid_spacing)

        masked_x_coords_DNS_grid = x_centers[mask].flatten()
        masked_y_coords_DNS_grid = y_centers[mask].flatten()

        plot_terms = False
        if plot_terms:
            os.mkdir(save_dir + 'terms/')

            np.save(save_dir + 'terms/features', features)

            clim = 0.2 #np.max(np.abs(features))

            plt.figure(figsize = (6,4))
            plt.scatter(masked_x_coords_DNS_grid,masked_y_coords_DNS_grid,2,UU_x_weak[mask], vmin=-clim, vmax = clim,cmap='RdBu')
            plt.colorbar()
            plt.savefig(save_dir + 'terms/' + f'UU_x_weak', dpi=480)
            plt.close()

            plt.figure(figsize = (6,4))
            plt.scatter(masked_x_coords_DNS_grid,masked_y_coords_DNS_grid,2,VU_y_weak[mask], vmin=-clim, vmax = clim,cmap='RdBu')
            plt.colorbar()
            plt.savefig(save_dir + 'terms/' + f'VU_y_weak',dpi=480)
            plt.close()

            plt.figure(figsize = (6,4))
            plt.scatter(masked_x_coords_DNS_grid,masked_y_coords_DNS_grid,2,nu*(U_xx_weak + U_yy_weak)[mask], vmin=-clim, vmax = clim,cmap='RdBu')
            plt.colorbar()
            plt.savefig(save_dir + 'terms/' + f'nu_Lap_U',dpi=480)
            plt.close()

            plt.figure(figsize = (6,4))
            plt.scatter(masked_x_coords_DNS_grid,masked_y_coords_DNS_grid,2,nu*U_xx_weak[mask], vmin=-clim, vmax = clim,cmap='RdBu')
            plt.colorbar()
            plt.savefig(save_dir + 'terms/' + f'nu_U_xx_weak',dpi=480)
            plt.close()

            plt.figure(figsize = (6,4))
            plt.scatter(masked_x_coords_DNS_grid,masked_y_coords_DNS_grid,2,nu*U_yy_weak[mask], vmin=-clim, vmax = clim,cmap='RdBu')
            plt.colorbar()
            plt.savefig(save_dir + 'terms/' + f'nu_U_yy_weak',dpi=480)
            plt.close()

            plt.figure(figsize = (6,4))
            plt.scatter(masked_x_coords_DNS_grid,masked_y_coords_DNS_grid,2,p_x_weak[mask], vmin=-clim, vmax = clim,cmap='RdBu')
            plt.colorbar()
            plt.savefig(save_dir + 'terms/' + f'p_x_weak',dpi=480)
            plt.close()

            plt.figure(figsize = (6,4))
            plt.scatter(masked_x_coords_DNS_grid,masked_y_coords_DNS_grid,2,Ruu_x_weak[mask], vmin=-clim, vmax = clim,cmap='RdBu')
            plt.colorbar()
            plt.savefig(save_dir + 'terms/' + f'Ruu_x_weak',dpi=480)
            plt.close()

            plt.figure(figsize = (6,4))
            plt.scatter(masked_x_coords_DNS_grid,masked_y_coords_DNS_grid,2,Ruv_y_weak[mask], vmin=-clim, vmax = clim,cmap='RdBu')
            plt.colorbar()
            plt.savefig(save_dir + 'terms/' + f'Ruv_y_weak',dpi=480)
            plt.close()

            plt.figure(figsize = (6,4))
            plt.scatter(masked_x_coords_DNS_grid,masked_y_coords_DNS_grid,2, UU_x_weak[mask] + VU_y_weak[mask] + Ruv_y_weak[mask] + Ruv_y_weak[mask] - nu*(U_xx_weak + U_yy_weak)[mask] + p_x_weak[mask], vmin=-clim, vmax = clim,cmap='RdBu')
            plt.colorbar()
            plt.savefig(save_dir + 'terms/' + f'Res',dpi=480)
            plt.close()




        nfeatures = 6
        no_trials = 3
        for nc in nc_arr:
            save_dir = f'plots/SuppStudy_Post_Correlate_Fix/nc{nc}/support_x_{no_x_support_pts}_support_y_{no_y_support_pts}_grid_spacing_{grid_spacing}_TFDegree_{degree}/'

            nc_save_dir = save_dir
            os.mkdir(nc_save_dir)

            for trial in range(no_trials):
                # nc_save_dir = save_dir_trial + f'nc{nc}/'
                save_dir_trial = nc_save_dir + f'trial_{trial}/' 
                os.mkdir(save_dir_trial)

                model = train_gmm_model(nc,features,sample_pct=0.25)
                cluster_idx = model.predict(features)+1

                plt.figure(figsize = (10,4))
                plt.scatter(masked_x_coords_DNS_grid,masked_y_coords_DNS_grid,2,cluster_idx, cmap = cm, vmin=-0.5, vmax=cm.N-0.5)
                plt.colorbar(boundaries=jnp.arange(0.5, nc+1.5), ticks=jnp.arange(0, nc+1))
                plt.savefig(save_dir_trial + f'ClusterDomain')
                plt.close()

                np.save(save_dir_trial + f'masked_x_coords_DNS_grid', masked_x_coords_DNS_grid)
                np.save(save_dir_trial + f'masked_y_coords_DNS_grid', masked_y_coords_DNS_grid)
                np.save(save_dir_trial + f'cluster_idx', cluster_idx)                

                # print(np.min(cluster_idx))
                # print(np.max(cluster_idx))

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
                    plt.savefig(save_dir_trial + f'VanillaDataCV,nc{nc}', dpi=1080)
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
                    plt.savefig(save_dir_trial + f'VanillaDataCV,nc{nc}', dpi=1080)
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
                    plt.savefig(save_dir_trial + f'VanillaDataCV,nc{nc}', dpi=1080)
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
                    plt.savefig(save_dir_trial + f'VanillaDataCV,nc{nc}', dpi=1080)
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
                    plt.savefig(save_dir_trial + f'VanillaDataCV,nc{nc}', dpi=1080)
                    plt.close()