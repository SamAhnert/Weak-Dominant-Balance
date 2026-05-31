'''
Process ensemble of noisy Pointwise DB, data stored in directory specified in save_dir in file 1_GeneratePointwiseEnsembleData.py

Specifically, take GMM clusters generated in file 1_, and run candidate sPCA reductions of each of those
This code varies the hyperpara, alpha which controls how likely sPCA is to idenify an axis as relevant for the variance of a particular cluster.
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

def train_gmm_model(nc, features, sample_pct=0.95, mode='kmeans'):
    seed = randint(2**32)
    # print(seed)
    model = GaussianMixture(n_components=nc, random_state=seed, n_init=1, init_params=mode)

    # PERMUTATION
    mask = np.random.permutation(features.shape[0])[:int(sample_pct*features.shape[0])]
    model.fit(features[mask, :])

    return model

def test_alpha_sPCA_GMM_model(alphas, features, nfeatures, nc, model):
    cluster_idx = model.predict(features)
    err = np.zeros([len(alphas)])  # Error defined as norm of inactive terms
    sparsity = np.zeros([len(alphas)])
    for k in range(len(alphas)):
        for i in range(nc):
            # Identify points in the field corresponding to each cluster
            feature_idx = np.nonzero(cluster_idx==i)[0]
            cluster_features = features[feature_idx, :]
            spca = SparsePCA(n_components=1, alpha=alphas[k])
            spca.fit(cluster_features)
            active_terms = np.nonzero(spca.components_[0])[0]
            inactive_terms = [feat for feat in range(nfeatures) if feat not in active_terms ]

            err[k] += np.linalg.norm(cluster_features[:, inactive_terms])

    return err

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

# Do I do a db here? yes, get baseline CVs
def vanillaDB_no_truncating_terms(pointwise_save_dir):
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

def test_alpha_sPCA_GMM_model(alphas, features, model):
    err = np.zeros([len(alphas)])  # Error defined as norm of inactive terms
    sparsity = np.zeros([len(alphas)])
    for k in range(len(alphas)):
        for i in range(nc):
            # Identify points in the field corresponding to each cluster
            feature_idx = np.nonzero(cluster_idx==i)[0]
            cluster_features = features[feature_idx, :]
            spca = SparsePCA(n_components=1, alpha=alphas[k], normalize_components=True)
            spca.fit(cluster_features)
            active_terms = np.nonzero(spca.components_[0])[0]
            inactive_terms = [feat for feat in range(nfeatures) if feat not in active_terms ]

            err[k] += np.linalg.norm(cluster_features[:, inactive_terms])

    return err

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


def main():
    noises = np.array([0.25, 0.5, 1.0]) 

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

    for noise in noises:
        #noise = 0.0 #as a percentage of the std of the V-velocity data
        save_dir = f'plots/2ndOrderFD_Derivatives_On_DNS_Grid/alpha_testing/{noise}_Noise/'
        os.mkdir(save_dir)

        no_runs = 100
        nfeatures = 6
        nc = 6

        test_alphas = [0.05,0.1,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.5,1.7,2.0,2.5,3.0,3.5,4.0,4.5,5.0,7.5,10.0,11.0,12.0,13.0,15.0,17.5,20.0,22.5,25.0,30.0,40.0,50.0,62.5,75.0,87.5,100.0,112.5,125.0,137.5,150.0,162.5,175.0,187.5,200.0,212.5,225.0,237.5,250.0,262.5,275.0,287.5,300.0,312.5, 325.0, 337.5, 350.0, 362.5, 375.0, 400.0]

        for i,optimal_alpha in enumerate(test_alphas):
            os.mkdir(save_dir + f'alpha_{optimal_alpha}/')
            for j in range(no_runs):
                trial_save_dir = save_dir + f'alpha_{optimal_alpha}/trial{j}/'
                os.mkdir(trial_save_dir)

                load_dir = f'plots/2ndOrderFD_Derivatives_On_DNS_Grid/{noise}_Noise/'
                cluster_idx = np.load(load_dir + f'trial{j}/cluster_idx.npy')
                features = np.load(load_dir + f'trial{j}/features.npy')
                spca_model = train_sPCA_model(optimal_alpha, features, nfeatures, nc, cluster_idx-1)

                balance_models, model_index = np.unique(spca_model, axis=0, return_inverse=True)
                # print(balance_models)
                nmodels = balance_models.shape[0]

                balance_idx = np.array([model_index[i] for i in cluster_idx-1])
                balancemap = np.reshape(balance_idx, X_trunc.shape, order='F')
                # print(balance_models)

                # Save balance_model and balance maps & plot them too
                plt.figure(figsize = (15,4))
                plt.pcolormesh(X_trunc,Y_trunc, balancemap+1, cmap = cm, vmin=-0.5, vmax=cm.N-0.5)
                plt.colorbar(boundaries=jnp.arange(0.5, int(nmodels)+1.5), ticks=jnp.arange(1, int(nmodels)+2))
                plt.savefig(trial_save_dir + f'sPCA_ClusterDomain,nc{nc}')
                plt.clf()

                # Plot a grid with active terms in each cluster
                gridmap = balance_models.copy()
                gridmask = gridmap==0
                gridmap = (gridmap.T*np.arange(nmodels)).T+1  # Scale map so that active terms can be color-coded
                gridmap[gridmask] = 0

                # plot spca_model matrix
                # Delete zero terms
                # grid_mask = np.nonzero( np.all(gridmap==0, axis=0) )[0]
                # gridmap = np.delete(gridmap, grid_mask, axis=1)
                grid_labels = labels

                plt.figure(figsize=(6, 4))
                plt.pcolor(gridmap, vmin=-0.5, vmax=cm.N-0.5, cmap=cm, edgecolors='k', linewidth=1)
                plt.gca().set_xticks(np.arange(0.5, gridmap.shape[1]+0.5))
                # print(grid_labels)
                # print(grid_mask)
                # print(nmodels)
                plt.gca().set_xticklabels(grid_labels, fontsize=14)
                plt.gca().set_yticklabels([])

                for axis in ['top','bottom','left','right']:
                    plt.gca().spines[axis].set_linewidth(2)

                plt.gca().tick_params(axis='both', width=0)
                plt.savefig(trial_save_dir + f'sPCA_Matrix_Model')
                plt.clf()

                # save cluster_idx np array and spca_model np
                np.save(trial_save_dir + f'spca_model.npy', balance_models)
                np.save(trial_save_dir + f'cluster_idx.npy', balancemap)


        '''

        Results from 2pt5_...:

        alphas = [50.0, 11.0, 112.5, 200.0, 1.0, 1.4, 4.0]

        We will run these test one noise level at a time. Thus, after each trial we can delete all the test sPCAs for each eta before running the next to ensure we don't run out of space.
        optim_alph
        Results:
        eta = 0.01 , optim_alpha = ___
                    For Noise 0.01 and optimal alpha 50.0, best case error for Pointwise DB is 17.015040414530468% (median)
                    Q1: 0.16962744006430017 and Q3: 0.1865019606951034 

        NOTE: it may be suprising to see 0.01 have such high and then immediately low error.
              UU_x is a much better noise amplifier that the pressure or RST. thus, noise that might not affect reconstruction of the P_x or uv_y terms, may have an outsized impact on the UU_x term.
              That, paired with the fact that UU_x is already probably on a similar level of magnitude in the inertial & freestream regions, means P_x and uv_y are likely much less important than UU_x, 
              causing some overlap, especially in the far downstream and far from wall subsections of the inertial region 

        eta = 0.025, optim_alpha = ___
                    For Noise 0.025 and optimal alpha 11.0, best case error for Pointwise DB is 3.92561547213178% (median)
                    Q1: 0.03898032619248403 and Q3: 0.22847588170653804
        eta = 0.05 , optim_alpha = ___
                    For Noise 0.05 and optimal alpha 112.5, best case error for Pointwise DB is 30.012054344878415% (median)
                    Q1: 0.2986156205642013 and Q3: 0.30023426845673934
        eta = 0.1  , optim_alpha = ___
                    For Noise 0.1 and optimal alpha 200.0, best case error for Pointwise DB is 30.447233633458517% (median)
                    Q1: 0.3017141160690522 and Q3: 0.30454759838463696
        eta = 0.25 , optim_alpha = ___
                    For Noise 0.25 and optimal alpha 1.0, best case error for Pointwise DB is 33.80503225688213% (median)
                    Q1: 0.2580335918161628 and Q3: 0.39856081079827876
                    
        NOTE: more tests were rerun to include params below alpha=1.0, and alpha=1.0 remained the best case param for 25% noise.
        
        eta = 0.5  , optim_alpha = ___
                    For Noise 0.5 and optimal alpha 1.4, best case error for Pointwise DB is 37.3655133675129% (median)
                    Q1: 0.3193930355876384 and Q3: 0.444081372172342

        eta = 1.0  , optim_alpha = ___

                    For Noise 1.0 and optimal alpha 4.0, best case error for Pointwise DB is 36.202663172740934% (median)
                    Q1: 0.2947376304027683 and Q3: 0.4420831282780418       

        BONUS:

        For Noise 0.02 and optimal alpha 17.5, best case error for Pointwise DB is 22.18258045383248% (median)
        Q1: 0.03142167162790051 and Q3: 0.22266584943544476
        For Noise 0.02 and optimal alpha 30.0, best case error for Pointwise DB is 23.579724443944126% (median)
        Q10: 0.18168441020379314 and Q1: 0.18201296203855208 and Q3: 0.2364509763830858

        For Noise 0.03 and optimal alpha 75.0, best case error for Pointwise DB is 29.605273286465806% (median)
        Q1: 0.281866858289421 and Q3: 0.3871368408377136
        For Noise 0.03 and optimal alpha 150.0, best case error for Pointwise DB is 29.682519712967714% (median)
        Q10: 0.28735134135125256 and Q1: 0.2874531368905115 and Q3: 0.32516524427714444

        For Noise 0.04 and optimal alpha 87.5, best case error for Pointwise DB is 29.80961618121706% (median)
        Q1: 0.29771796905140757 and Q3: 0.38930735035802105
        For Noise 0.04 and optimal alpha 400.0, best case error for Pointwise DB is 34.522194927841404% (median)
        Q10: 0.34336459494530486 and Q1: 0.343613653418136 and Q3: 0.3464427639391383

        '''


if __name__ == '__main__':
    main()