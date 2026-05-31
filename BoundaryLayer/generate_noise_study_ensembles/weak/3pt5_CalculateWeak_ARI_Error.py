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
from sklearn.metrics import adjusted_rand_score
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
    noises = np.array([0.01, 1.0, 0.025, 0.05, 0.1, 0.25, 0.5])
    corresponding_selected_supports = np.array([(0.3,0.2), (12.0,0.3), (0.8,0.15), (1.2,0.12), (4.0,0.12), (2.0,0.4),(12.0,0.3)])
    corresponding_selected_alphas = np.array([5.5, 220.0, 10.0, 7.5, 40.0, 55.0, 200.0])

    # noises = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
    # corresponding_selected_supports = np.array([(1.0,0.15),(1.5,0.12),(8.0,0.15),(12.0,0.15),(10.0,0.15),(10.0,0.14),(4.0,0.14),(6.0,0.14),(8.0,0.14),(12.0,0.14),(15.0,0.14)])
    # corresponding_selected_alphas = np.array([7.5, 10.0, 64.0, 100.0, 80.0, 80.0, 30.0, 45.0, 65.0, 90.0, 115.0])

    labels = [r'$\bar{u} \bar{u}_x$', r'$\bar{v}\bar{u}_y$',
            r'$\nu \nabla \bar{\bf{u}}$', r'$\rho^{-1} \bar{p}_x$', 
            r'$\overline{({u^\prime} ^2)}_x$', r'$\overline{(u^\prime v^\prime)}_y$']

    # Use 0.0 noise with weakDB as baseline for noisy weak DB

    # weak_No_Noise_baseline = np.zeros((ptws_sPCA_balance_map.shape[0], ptws_sPCA_balance_map.shape[1], ptws_sPCA_model.shape[1]))
    data, X_JH_DNS_OG, Y_JH_DNS_OG = get_JH_DNS_BL_Data()
    noise = 0.0
    alpha = 2.0 #5.0
    # support = corresponding_selected_supports[i]
    support_bound_x = 0.2 #1.0
    support_bound_y = 0.1

    # NOTE: weak no-noise baseline stored in this location on my machine, would be in plots/ExpandedConvectionTerm/sPCA_models/Noise_{noise}/... locally after running 1_GenWeak... file
    load_dir = f'/home/sahnert/Dominant_Balance_Fine/BoundaryLayer/sym_plots/WeakDB/sPCA_models/Noise_{noise}/support_{support_bound_x}_{support_bound_y}/alph{alpha}/TF_degree_5/'

    x0 = 30.5
    xF = 999.7
    y0 = 0.01
    yF = 26.45
    grid_spacing = 0.01

    x0_TF = x0 + support_bound_x
    xF_TF = xF - support_bound_x

    y0_TF = y0 + support_bound_y
    yF_TF = yF - support_bound_y

    # get indices of this accurate range in order to truncate erroneous boundaries from domain when interpolating onto JH DNS Grid later
    idx_support_bound_x = round(support_bound_x / grid_spacing)
    idx_support_bound_y = round(support_bound_y / grid_spacing)

    baseline_bottom_idx = (Y_JH_DNS_OG < (y0 + support_bound_y)).sum()
    baseline_left_idx = (X_JH_DNS_OG < (x0 + support_bound_x)).sum()

    baseline_top_idx = (Y_JH_DNS_OG > (yF - support_bound_y)).sum()
    baseline_right_idx = (X_JH_DNS_OG > (xF - support_bound_x)).sum()

    y_trunc = Y_JH_DNS_OG[baseline_bottom_idx:-baseline_top_idx]
    x_trunc = X_JH_DNS_OG[baseline_left_idx:-baseline_right_idx]

    X_JH_DNS_trunc, Y_JH_DNS_trunc = np.meshgrid(x_trunc, y_trunc)

    sPCA_identified_relevant_terms = np.zeros((X_JH_DNS_trunc.shape[0],X_JH_DNS_trunc.shape[1],6))

    t0 = time.time()

    # Set number of trials available to use per noise level
    no_runs_Weak = 100

    '''Grab Baseline'''
    for j in range(0,no_runs_Weak):
        spca_model = np.where(np.load(load_dir + f'trial{j}/spca_model.npy')!=0, 1, 0)
        # cluster_idx is a flat array encoding points in the spatial domain to the appropriate model of sparse dynamics
        cluster_idx = np.load(load_dir + f'trial{j}/balancemap.npy')

        for ii in range(sPCA_identified_relevant_terms.shape[0]):
            for jj in range(sPCA_identified_relevant_terms.shape[1]):
                map_idx = cluster_idx[ii,jj]
                sPCA_identified_relevant_terms[ii,jj,:] += spca_model[map_idx]
    
    final_cluster_idx_NoNoise = cluster_idx.copy() # could try and back out an aggregate idx from the weak_No_NOise_Baseline below, but it probably won't be substantially different...

    weak_No_Noise_baseline = np.round(sPCA_identified_relevant_terms / no_runs_Weak) # round <0.5 down to 0, round >=0.5 to 1

    '''Run through noise levels tested in Weak DB'''
    for i,noise in enumerate(noises):
        alpha = corresponding_selected_alphas[i]
        support = corresponding_selected_supports[i]
        support_bound_x = support[0]
        support_bound_y = support[1]

        load_dir = f'plots/ExpandedConvectionTerm/sPCA_models/Noise_{noise}/support_{support_bound_x}_{support_bound_y}/alph{alpha}/TF_degree_5/'

        x0 = 30.5
        xF = 999.7
        y0 = 0.01
        yF = 26.45
        grid_spacing = 0.01

        x0_TF = x0 + support_bound_x
        xF_TF = xF - support_bound_x

        y0_TF = y0 + support_bound_y
        yF_TF = yF - support_bound_y

        # get indices of this accurate range in order to truncate erroneous boundaries from domain when interpolating onto JH DNS Grid later
        idx_support_bound_x = round(support_bound_x / grid_spacing)
        idx_support_bound_y = round(support_bound_y / grid_spacing)

        bottom_idx = (Y_JH_DNS_OG < (y0 + support_bound_y)).sum()
        left_idx = (X_JH_DNS_OG < (x0 + support_bound_x)).sum()

        top_idx = (Y_JH_DNS_OG > (yF - support_bound_y)).sum()
        right_idx = (X_JH_DNS_OG > (xF - support_bound_x)).sum()

        # adjusted the truncation indices to account for the fact that the baseline weak method is already truncated based on the (1.0,0.1) supports
        adjusted_bottom_idx = bottom_idx - baseline_bottom_idx
        adjusted_top_idx = top_idx - baseline_top_idx
        adjusted_left_idx = left_idx - baseline_left_idx
        adjusted_right_idx = right_idx - baseline_right_idx

        # This little "or None" trick just changed my life
        trunc_No_Noise_baseline = weak_No_Noise_baseline[adjusted_bottom_idx:-adjusted_top_idx or None, adjusted_left_idx:-adjusted_right_idx or None]

        trunc_final_cluster_idx_NoNoise  = final_cluster_idx_NoNoise[adjusted_bottom_idx:-adjusted_top_idx or None, adjusted_left_idx:-adjusted_right_idx or None]

        y_trunc = Y_JH_DNS_OG[bottom_idx:-top_idx]
        x_trunc = X_JH_DNS_OG[left_idx:-right_idx]

        
        X_JH_DNS_trunc, Y_JH_DNS_trunc = np.meshgrid(x_trunc, y_trunc)

        # pct_err_arr = []
        ari_err_arr = []

        for j in range(0,no_runs_Weak):
            spca_model = np.where(np.load(load_dir + f'trial{j}/spca_model.npy')!=0, 1, 0)
            # cluster_idx is a flat array encoding points in the spatial domain to the appropriate model of sparse dynamics
            cluster_idx = np.load(load_dir + f'trial{j}/balancemap.npy')

            ari = adjusted_rand_score(cluster_idx.flatten(), trunc_final_cluster_idx_NoNoise.flatten())

            ari_err_arr.append(ari)
                
        ari_err_arr = np.array(ari_err_arr)

        np.save(f'Error_Data/Noise_{noise}_support_{support_bound_x}_{support_bound_y}_alpha_{alpha}_noRunsWeak_{no_runs_Weak}_ARI_Error.npy',ari_err_arr)

        print()
        print(f'avg ARI for weak DB with zero noise weak baseline for a noise level of {noise} and support of {support}: {np.mean(ari_err_arr)}')
        print(f'min_ari: {np.min(ari_err_arr)}, max_ari: {np.max(ari_err_arr)}, Q1: {np.percentile(ari_err_arr, 25)}, Q2: {np.percentile(ari_err_arr, 50)}, Q3: {np.percentile(ari_err_arr, 75)}')
        print()


if __name__ == '__main__':
    main()