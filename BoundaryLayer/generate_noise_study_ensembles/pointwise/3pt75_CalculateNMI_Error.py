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
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score

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


def main():
    # noises = np.array([0.02, 0.03, 0.04]) # 
    noises = np.array([0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]) # 
    # These optimal alphas calculated in 2pt5_ ... and correspond to the optimal alpha for each noise listed above
    # optimal_alpha_arr = np.array([17.5, 75.0, 87.5])
    optimal_alpha_arr = np.array([50.0, 11.0, 112.5, 200.0, 1.0, 1.4, 4.0])
    labels = [r'$\bar{u} \bar{u}_x$', r'$\bar{v}\bar{u}_y$',
        r'$\nu \nabla \bar{\bf{u}}$', r'$\rho^{-1} \bar{p}_x$', 
        r'$\overline{({u^\prime} ^2)}_x$', r'$\overline{(u^\prime v^\prime)}_y$']

    data, x, y = get_JH_DNS_BL_Data(flatten_protocol='F')
    X, Y = np.meshgrid(x,y)

    FD_2ndOrder = True
    FD_6thOrder = False

    if FD_6thOrder:
        idx_support_bound_x = 50 #Calculated by truncating the x vals according to Aug20_Noisy... file
        idx_support_bound_y = 10 #Calculated by truncating the y vals according to Aug20_Noisy... file

        x_for_derivatives = (X[idx_support_bound_y:-idx_support_bound_y, idx_support_bound_x:-idx_support_bound_x])[0,:]
        y_for_derivatives = (Y[idx_support_bound_y:-idx_support_bound_y, idx_support_bound_x:-idx_support_bound_x])[:,0]

        X_trunc, Y_trunc = np.meshgrid(x_for_derivatives,y_for_derivatives)
    elif FD_2ndOrder:
        x_for_derivatives = (X)[0,:]
        y_for_derivatives = (Y)[:,0]

        X_trunc, Y_trunc = np.meshgrid(x_for_derivatives,y_for_derivatives)
    else:
        raise Exception("No FD Scheme selected")

    # load pointwise no noise baseline, arbitrariliy chose the no noise baseline generated for 0.01 Noise, confirmed it is accurate and keep for comparison across all alphas for consistency sake
    sPCA_identified_relevant_terms_No_Noise_baseline = np.zeros((X_trunc.shape[0],X_trunc.shape[1],6))

    cluster_map_ptwsBase = np.load(f'plots/2ndOrderFD_Derivatives_On_DNS_Grid/alpha_testing/pointwiseBaseline/cluster_map.npy')
    spca_model_ptwsBase = np.load(f'plots/2ndOrderFD_Derivatives_On_DNS_Grid/alpha_testing/pointwiseBaseline/spca_model.npy')

    for ii in range(sPCA_identified_relevant_terms_No_Noise_baseline.shape[0]):
        for jj in range(sPCA_identified_relevant_terms_No_Noise_baseline.shape[1]):
            map_idx = cluster_map_ptwsBase[ii,jj]
            sPCA_identified_relevant_terms_No_Noise_baseline[ii,jj,:] += spca_model_ptwsBase[map_idx]

    trunc_No_Noise_baseline = sPCA_identified_relevant_terms_No_Noise_baseline#[idx_support_bound_y:-idx_support_bound_y, idx_support_bound_x:-idx_support_bound_x]

    for i,noise in enumerate(noises):
        #noise = 0.0 #as a percentage of the std of the V-velocity data

        no_runs = 100
        nfeatures = 6
        nc = 6
        optimal_alpha = optimal_alpha_arr[i]

        # load dir for the full ensemble of data
        load_dir = f'plots/2ndOrderFD_Derivatives_On_DNS_Grid/alpha_testing/{noise}_Noise/'

        sPCA_identified_relevant_terms = np.zeros((X_trunc.shape[0],X_trunc.shape[1],6))
        nmi_err_arr = []

        for j in range(no_runs): # choose the first no_runs trials to run sPCA
            # Load necessary info from selected trial
            trial_load_dir = load_dir + f'alpha_{optimal_alpha}/trial{j}/'
            cluster_idx = np.load(trial_load_dir+ f'cluster_idx.npy')
            spca_model = np.load(trial_load_dir+ f'spca_model.npy')

            cluster_map = np.reshape(cluster_idx, X_trunc.shape, order='F')

            nmi = normalized_mutual_info_score(cluster_map.flatten(), cluster_map_ptwsBase.flatten())

            nmi_err_arr.append(nmi)

            
        print()
        print(f'avg NMI for Pointwise DB with zero noise weak baseline for a noise level of {noise}: {np.mean(nmi_err_arr)}')
        print(f'min_nmi: {np.min(nmi_err_arr)}, max_nmi: {np.max(nmi_err_arr)}, Q1: {np.percentile(nmi_err_arr, 25)}, Q2: {np.percentile(nmi_err_arr, 50)}, Q3: {np.percentile(nmi_err_arr, 75)}')
        print()

        ''' TODO Add Saves for pct_err array! '''

        np.save(f'Error_Data/NMI_Error_arr_{noise}_Noise_{optimal_alpha}_alpha.npy', nmi_err_arr)



if __name__ == '__main__':
    main()