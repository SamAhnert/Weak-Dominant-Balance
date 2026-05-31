'''
Process ensemble of noisy Pointwise DB, data stored in plots/Derivatives_On_DNS_Grid/

Run through sPCA models and calcualte which gives the least error when compared to a no noise baseline.

'''
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint
import sklearn as sk
from sklearn.mixture import GaussianMixture # jaxxx??? # Hyakkk
from sklearn.decomposition import SparsePCA
from scipy.io import loadmat
import h5py
from scipy import sparse, linalg
from scipy.optimize import curve_fit, root
from scipy.integrate import odeint
from scipy.interpolate import interp1d
#from scipy.signal import convolve2d, fftconvolve # check jax
import os
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

from jax.scipy.signal import fftconvolve
import math
import time

import pickle

import matplotlib as mpl
from matplotlib.colors import ListedColormap

import jax
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp
from jax import jit ,vmap
from jax.scipy.integrate import trapezoid as jtrap

# Seaborn colormap
import seaborn as sns
sns_list = sns.color_palette('deep').as_hex()
sns_list.insert(0, '#ffffff')  # Insert white at zero position

sns_cmap = ListedColormap(sns_list)
cm = sns_cmap


mpl_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
            '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
            '#bcbd22', '#17becf']

def train_sPCA_model(alpha_opt, features, nfeatures, nc, cluster_idx):
    labels = [r'$\bar{u} \bar{u}_x$', r'$\bar{v}\bar{u}_y$',
        r'$\nu \nabla \bar{\bf{u}}$', r'$\rho^{-1} \bar{p}_x$', 
        r'$\overline{({u^\prime} ^2)}_x$', r'$\overline{(u^\prime v^\prime)}_y$']
    spca_model = np.zeros([nc, nfeatures])
    for i in range(nc):
        feature_idx = np.nonzero(cluster_idx==i)[0]
        cluster_features = features[feature_idx, :]
        spca = SparsePCA(n_components=1, alpha=alpha_opt)
        spca.fit(cluster_features)
        # print('GMM Cluster {0}:'.format(i))
        active_terms = np.nonzero(spca.components_[0])[0]
        if len(active_terms)>0:
            # print([labels[k] for k in active_terms])
            spca_model[i, active_terms] = 1  # Set to 1 for active terms in model
        else: print('None')

    return spca_model

def get_JH_DNS_BL_Data(flatten_protocol = 'C'):#x_min, x_max, y_min, y_max, noisy=False, noise_pct=0.0, Nx=0,Ny=0
    file = h5py.File('../data/Transition_BL_Time_Averaged_Profiles.h5', 'r')

    x_JH_DNS = jnp.array(file['x_coor'])
    y_JH_DNS = jnp.array(file['y_coor'])
    # flatten data fields
    u = jnp.array(file['um']).flatten(flatten_protocol)
    v = jnp.array(file['vm']).flatten(flatten_protocol)
    p = jnp.array(file['pm']).flatten(flatten_protocol)
    Ruu = (jnp.array(file['uum'])).flatten(flatten_protocol) - u**2
    Ruv = (jnp.array(file['uvm'])).flatten(flatten_protocol) - u*v
    # Rvv = (jnp.array(file['uvm'])).flatten() - v**2

    data = [u,v,p,Ruu,Ruv]

    return data,x_JH_DNS,y_JH_DNS

def get_JH_DNS_BL_Data_flattened_data(flatten_protocol='C'):#x_min, x_max, y_min, y_max, noisy=False, noise_pct=0.0, Nx=0,Ny=0
    file = h5py.File('../data/Transition_BL_Time_Averaged_Profiles.h5', 'r')

    x_JH_DNS = np.array(file['x_coor'])
    y_JH_DNS = np.array(file['y_coor'])
    # flatten data fields
    u = np.array(file['um']).flatten(flatten_protocol)
    v = np.array(file['vm']).flatten(flatten_protocol)
    p = np.array(file['pm']).flatten(flatten_protocol)
    Ruu = (np.array(file['uum'])).flatten(flatten_protocol) - u**2
    Ruv = (np.array(file['uvm'])).flatten(flatten_protocol) - u*v
    # Rvv = (np.array(file['uvm'])).flatten() - v**2

    # data = np.array([u,v,p,Ruu,Ruv])
    data = [u,v,p,Ruu,Ruv]

    return data,x_JH_DNS,y_JH_DNS

def train_gmm_model(nc, features, sample_pct=0.95, mode='kmeans'):
    seed = randint(2**32)
    # print(seed)
    model = GaussianMixture(n_components=nc, random_state=seed, n_init=5, init_params=mode)

    # PERMUTATION
    mask = np.random.permutation(features.shape[0])[:int(sample_pct*features.shape[0])]
    model.fit(features[mask, :])

    return model

# Do I do a db here? yes, get baseline CVs
def pointwiseDB_no_truncating_terms(pointwise_save_dir):
    labels = [r'$\bar{u} \bar{u}_x$', r'$\bar{v}\bar{u}_y$',
        r'$\nu \nabla \bar{\bf{u}}$', r'$\rho^{-1} \bar{p}_x$', 
        r'$\overline{({u^\prime} ^2)}_x$', r'$\overline{(u^\prime v^\prime)}_y$']
    nfeatures = 6
    nc = 6
    data, x, y = get_JH_DNS_BL_Data_flattened_data(flatten_protocol='F')

    # params for data
    U_inf = 1
    nu = 1/800

    # Calculate Derivatives
    if True:
        dx = float(x[1]-x[0])
        # print(y.shape)
        # print(y[1:]-y[:-1])
        dy = np.array(y[1:]-y[:-1])

        # print(dy)
        # raise Exception('stop')

        nx = len(x)
        ny = len(y)

        Dy = sparse.diags( [-1, 1], [-1, 1], shape=(ny, ny) ).toarray()
        # Second-order forward/backwards at boundaries
        Dy[0, :3] = np.array([-3, 4, -1])
        Dy[-1, -3:] = np.array([1, -4, 3])
        for j in range(ny-1):
            Dy[j, :] = Dy[j, :]/(2*dy[j])
        Dy[-1, :] = Dy[-1, :]/(2*dy[-1])

        # Repeat for each x-location
        Dy = sparse.block_diag([Dy for j in range(nx)])

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

        u = data[0]
        v = data[1]
        p = data[2]
        Ruu = data[3]
        Ruv = data[4]

        ux = Dx @ u
        uy = Dy @ u
        vx = Dx @ v
        vy = Dy @ v
        px = Dx @ p
        py = Dy @ p
        lap_u = (Dxx + Dyy) @ u
        Ruux = Dx @ Ruu
        Ruvy = Dy @ Ruv

    nu = 1/800
    features = 1e3 * np.vstack([u*ux,
                            v*uy,
                            -nu*(lap_u),
                            px,
                            Ruux,
                            Ruvy]).T

    model = train_gmm_model(nc, features, sample_pct=0.25)

    cluster_idx = model.predict(features)
    cluster_map = np.reshape(cluster_idx, (y.size, x.size), order='F')
    # print('plotting')

    plt.figure(figsize = (15,4))
    plt.pcolormesh(x, y, cluster_map+1, cmap = cm, vmin=-0.5, vmax=cm.N-0.5)
    plt.colorbar(boundaries=np.arange(0.5, int(nfeatures)+1.5), ticks=np.arange(1, int(nfeatures)+2))
    plt.savefig(pointwise_save_dir + f'PointwiseClusterDomain,nc{nc}')
    plt.close()

    C_list = []
    plt.figure(figsize=(9, 11))
    for j in range(nc):
        plt.subplot(3, 3, j+1)
        # get CVs
        jth_cluster = (cluster_idx == j)
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
    plt.savefig(pointwise_save_dir + f'PointwiseDataCV')
    plt.close()

    '''NOW DO THE sPCA AND SAVE THOSE TOO'''
    optimal_alpha = 10 # According to Callaham et. al. 2021, Nat. Comm.

    spca_model = train_sPCA_model(optimal_alpha, features, nfeatures, nc, cluster_idx)
    balance_models, model_index = np.unique(spca_model, axis=0, return_inverse=True)
    nmodels = balance_models.shape[0]

    balance_idx = np.array([model_index[i] for i in cluster_idx])
    balancemap = np.reshape(balance_idx, [ny, nx], order='F')

    # Save balance_model and balance maps & plot them too
    plt.figure(figsize = (15,4))
    plt.pcolormesh(x,y, balancemap+1, cmap = cm, vmin=-0.5, vmax=cm.N-0.5)
    plt.colorbar(boundaries=np.arange(0.5, int(nmodels)+1.5), ticks=np.arange(1, int(nmodels)+2))
    plt.savefig(pointwise_save_dir + f'Pointwise_sPCAClusters')
    plt.clf()

    # Plot a grid with active terms in each cluster
    gridmap = balance_models.copy()
    gridmask = gridmap==0
    gridmap = (gridmap.T*np.arange(nmodels)).T+1  # Scale map so that active terms can be color-coded
    gridmap[gridmask] = 0

    # plot spca_model matrix
    # # Delete zero terms
    # grid_mask = np.nonzero( np.all(gridmap==0, axis=0) )[0]
    # gridmap = np.delete(gridmap, grid_mask, axis=1)
    # grid_labels = np.delete(labels, grid_mask)

    grid_labels = labels

    # print(gridmap.shape)

    plt.figure(figsize=(6, 4))
    plt.pcolor(gridmap, vmin=-0.5, vmax=cm.N-0.5, cmap=cm, edgecolors='k', linewidth=1)
    plt.gca().set_xticks(np.arange(0.5, gridmap.shape[1]+0.5))
    plt.gca().set_xticklabels(grid_labels, fontsize=14)
    plt.gca().set_yticklabels([])
    #plt.gca().set_yticks(np.arange(0.5, nmodels+0.5))
    #plt.gca().set_yticklabels(range(nc), fontsize=20
    #plt.ylabel('Balance Model')
    for axis in ['top','bottom','left','right']:
        plt.gca().spines[axis].set_linewidth(2)

    plt.gca().tick_params(axis='both', width=0)
    plt.savefig(pointwise_save_dir + f'Pointwise_sPCA_Matrix_Model')
    plt.clf()

    # save cluster_idx np array and spca_model np
    np.save(pointwise_save_dir + f'spca_model.npy', spca_model)
    np.save(pointwise_save_dir + f'cluster_idx.npy', cluster_idx)
    np.save(pointwise_save_dir + f'cluster_map.npy', cluster_map)

    # TODO: what do we return

    return cluster_map, balancemap, gridmap, cluster_idx, spca_model

def main():
    # noises = np.array([0.25, 0.5, 1.0, 0.01, 0.02, 0.025, 0.03, 0.04, 0.05, 0.1]) # 0.01, 0.02, 0.025, 0.03, 0.04, 0.05, 0.1, 0.25,
    noises = np.array([0.04])
    # Test all alphas, overkill across most alphas, but whatever, should only run once
    # computed_alphas = [1.0,1.2,1.4,1.5,1.7,2.0,2.5,3.0,3.5,4.0,4.5,5.0,7.5,10.0,11.0,12.0,13.0,15.0,17.5,20.0,22.5,25.0,30.0,40.0,50.0,62.5,75.0,87.5,100.0,112.5,125.0,137.5,150.0,162.5,175.0,187.5,200.0,212.5,225.0,237.5,250.0,262.5,275.0,287.5,300.0,312.5, 325.0, 337.5, 350.0, 362.5, 375.0, 400.0]
    # computed_alphas = [0.05,0.1,0.2,0.4,0.6,0.8,1.0,1.2]
    computed_alphas = [0.05,0.1,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.5,1.7,2.0,2.5,3.0,3.5,4.0,4.5,5.0,7.5,10.0,11.0,12.0,13.0,15.0,17.5,20.0,22.5,25.0,30.0,40.0,50.0,62.5,75.0,87.5,100.0,112.5,125.0,137.5,150.0,162.5,175.0,187.5,200.0,212.5,225.0,237.5,250.0,262.5,275.0,287.5,300.0,312.5, 325.0, 337.5, 350.0, 362.5, 375.0, 400.0]#np.array([187.5, 1.2, 7.5, 5.0, 175.0, 250.0, 187.5])
    labels = [r'$\bar{u} \bar{u}_x$', r'$\bar{v}\bar{u}_y$',
        r'$\nu \nabla \bar{\bf{u}}$', r'$\rho^{-1} \bar{p}_x$', 
        r'$\overline{({u^\prime} ^2)}_x$', r'$\overline{(u^\prime v^\prime)}_y$']

    data, x, y = get_JH_DNS_BL_Data(flatten_protocol='F')
    X, Y = np.meshgrid(x,y)

    FD_2ndOrder = True
    FD_6thOrder = False

    if FD_6thOrder:
        idx_support_bound_x = 50 #Calculated by truncating the x vals according to logic in Aug20_NoisyVan... file
        idx_support_bound_y = 10 #Calculated by truncating the y vals according to logic in Aug20_NoisyVan... file

        x_for_derivatives = (X[idx_support_bound_y:-idx_support_bound_y, idx_support_bound_x:-idx_support_bound_x])[0,:]
        y_for_derivatives = (Y[idx_support_bound_y:-idx_support_bound_y, idx_support_bound_x:-idx_support_bound_x])[:,0]

        X_trunc, Y_trunc = np.meshgrid(x_for_derivatives,y_for_derivatives)
    elif FD_2ndOrder:
        x_for_derivatives = (X)[0,:]
        y_for_derivatives = (Y)[:,0]

        X_trunc, Y_trunc = np.meshgrid(x_for_derivatives,y_for_derivatives)
    else:
        raise Exception("No FD Scheme selected")

    
    # load pointwise no noise baseline, arbitrariliy chosse the baseline for 0.01 Noise, and keep for comparison across all alphas for consistency sake
    sPCA_identified_relevant_terms_No_Noise_baseline = np.zeros((X_trunc.shape[0],X_trunc.shape[1],6))

    # os.mkdir(f'plots/2ndOrderFD_Derivatives_On_DNS_Grid/alpha_testing/pointwiseBaseline/')
    # cluster_map_ptwsBase, balancemap_ptwsBase, gridmap_ptwsBase, cluster_idx_ptwsBase, spca_model_ptwsBase = pointwiseDB_no_truncating_terms(f'plots/2ndOrderFD_Derivatives_On_DNS_Grid/alpha_testing/pointwiseBaseline/')
    # os.mkdir(f'plots/2ndOrderFD_Derivatives_On_DNS_Grid/alpha_testing/pointwiseBaseline/')

    cluster_map_ptwsBase = np.load(f'plots/2ndOrderFD_Derivatives_On_DNS_Grid/alpha_testing/pointwiseBaseline/cluster_map.npy')
    spca_model_ptwsBase = np.load(f'plots/2ndOrderFD_Derivatives_On_DNS_Grid/alpha_testing/pointwiseBaseline/spca_model.npy')

    for ii in range(sPCA_identified_relevant_terms_No_Noise_baseline.shape[0]):
        for jj in range(sPCA_identified_relevant_terms_No_Noise_baseline.shape[1]):
            map_idx = cluster_map_ptwsBase[ii,jj]
            sPCA_identified_relevant_terms_No_Noise_baseline[ii,jj,:] += spca_model_ptwsBase[map_idx]

    trunc_No_Noise_baseline = sPCA_identified_relevant_terms_No_Noise_baseline#[idx_support_bound_y:-idx_support_bound_y, idx_support_bound_x:-idx_support_bound_x]


    for i,noise in enumerate(noises):
        no_runs = 100
        nfeatures = 6
        nc = 6

        optimal_alpha = 0.0
        min_err = np.inf
        min_err_arr = np.zeros(no_runs)

        optimal_alpha_Q1 = 0.0
        min_err_Q1 = np.inf
        min_err_arr_Q1 = np.zeros(no_runs)

        for test_alpha in computed_alphas:
            # load dir for the sPCA trials on noisy Pointwise DB
            load_dir = f'plots/2ndOrderFD_Derivatives_On_DNS_Grid/alpha_testing/{noise}_Noise/'

            sPCA_identified_relevant_terms = np.zeros((X_trunc.shape[0],X_trunc.shape[1],6))
            pct_err_arr = []

            for j in range(no_runs): # choose the first no_runs trials to run sPCA
                this_sPCA_identified_relevant_terms = np.zeros((X_trunc.shape[0],X_trunc.shape[1],6))

                # Load necessary info from selected trial
                trial_load_dir = load_dir + f'alpha_{test_alpha}/trial{j}/'
                cluster_idx = np.load(trial_load_dir+ f'cluster_idx.npy')
                spca_model = np.load(trial_load_dir+ f'spca_model.npy')

                cluster_map = np.reshape(cluster_idx, X_trunc.shape, order='F')

                for ii in range(this_sPCA_identified_relevant_terms.shape[0]):
                    for jj in range(this_sPCA_identified_relevant_terms.shape[1]):
                        map_idx = cluster_map[ii,jj]
                        this_sPCA_identified_relevant_terms[ii,jj,:] += spca_model[map_idx]

                # weight based off area
                take_weighted_sum = True
                if take_weighted_sum:
                    Y_diff = np.diff(Y_trunc, axis=0)
                    Y_diff = np.vstack((Y_diff[0,:],Y_diff))
                    dx = x[1]-x[0]
                    area = Y_diff * dx
                    weighted_area = area/(np.sum(area)/area.size)

                    #multiply by weights
                    this_pct_err = np.sum(np.sum(np.abs(this_sPCA_identified_relevant_terms - trunc_No_Noise_baseline),axis=2)*weighted_area) / trunc_No_Noise_baseline.size
                    pct_err_arr.append(this_pct_err)

                    # Keep track for old, unweighted error method, just in case
                    sPCA_identified_relevant_terms += this_sPCA_identified_relevant_terms
                else:
                    #ignore weights
                    this_pct_err = np.sum(np.abs(this_sPCA_identified_relevant_terms - trunc_No_Noise_baseline)) / trunc_No_Noise_baseline.size
                    pct_err_arr.append(this_pct_err)

                    sPCA_identified_relevant_terms += this_sPCA_identified_relevant_terms

            pct_err_arr= np.array(pct_err_arr)

            if np.median(pct_err_arr) < min_err:
                min_err_arr = pct_err_arr
                min_err = np.median(pct_err_arr)
                optimal_alpha = test_alpha
            if np.quantile(pct_err_arr, 0.1) < min_err_Q1:
                min_err_arr_Q1 = pct_err_arr
                min_err_Q1 = np.median(pct_err_arr)
                optimal_alpha_Q1 = test_alpha

        print(f'For Noise {noise} and optimal alpha {optimal_alpha}, best case error for Pointwise DB is {100 * min_err}% (median)')
        print(f'Q1: {np.quantile(min_err_arr, 0.25)} and Q3: {np.quantile(min_err_arr, 0.75)}')

        print(f'For Noise {noise} and optimal alpha {optimal_alpha_Q1}, best case error for Pointwise DB is {100 * min_err_Q1}% (median)')
        print(f'Q10: {np.quantile(min_err_arr_Q1, 0.1)} and Q1: {np.quantile(min_err_arr_Q1, 0.25)} and Q3: {np.quantile(min_err_arr_Q1, 0.75)}')

    ''' TODO Add Weak Stuff Here Too!'''


if __name__ == '__main__':
    main()