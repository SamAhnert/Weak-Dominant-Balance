#ProofOfConceptFigure_VanVsWeakBLComparison
'''
Generate an ensemble of noisy boundary layer data with these post grid search hyperparams deployed to determine a wholistic picture of the errror mitigation provided by the weak form.
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
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

from jax.scipy.signal import fftconvolve
from jax.scipy.interpolate import RegularGridInterpolator
import math
import time

import pickle

import matplotlib as mpl
from matplotlib.colors import ListedColormap

import jax
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp
from jax import jit ,vmap, pmap
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


from scipy.interpolate import griddata

# export XLA_PYTHON_CLIENT_PREALLOCATE=false
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

def train_gmm_model(nc, features, seed=-1, sample_pct=1.0, mode='kmeans'):
    # use random seed if not specified
    if seed == -1:
        seed = randint(2**32)
    else:
        pass
    
    print(seed)
    model = GaussianMixture(n_components=nc, random_state=seed, n_init = 3, init_params=mode)

    # PERMUTATION
    mask = np.random.permutation(features.shape[0])[:int(sample_pct*features.shape[0])]
    model.fit(features[mask, :])

    return model

'''
TF_x
'''
def TF_x(degree, support_bound_x, support_bound_y,dx):
    # save grid spacing (must be equal for FFT)
    dy = dx
    x_domain_TF = jnp.linspace(-support_bound_x, support_bound_x, round((support_bound_x*2)/dx) + 1)
    y_domain_TF = jnp.linspace(-support_bound_y, support_bound_y, round((support_bound_y*2)/dy) + 1)

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
def TF_y(degree, support_bound_x, support_bound_y,dx):
    # save grid spacing (must be equal for FFT)
    dy = dx
    x_domain_TF = jnp.linspace(-support_bound_x, support_bound_x, round((support_bound_x*2)/dx) + 1)
    y_domain_TF = jnp.linspace(-support_bound_y, support_bound_y, round((support_bound_y*2)/dy) + 1)

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
def TF_xx(degree, support_bound_x, support_bound_y,dx):
    # save grid spacing (must be equal for FFT)
    dy = dx
    x_domain_TF = jnp.linspace(-support_bound_x, support_bound_x, round((support_bound_x*2)/dx) + 1)
    y_domain_TF = jnp.linspace(-support_bound_y, support_bound_y, round((support_bound_y*2)/dy) + 1)

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
def TF_yy(degree, support_bound_x, support_bound_y,dx):
    # save grid spacing (must be equal for FFT)
    dy = dx
    x_domain_TF = jnp.linspace(-support_bound_x, support_bound_x, round((support_bound_x*2)/dx) + 1)
    y_domain_TF = jnp.linspace(-support_bound_y, support_bound_y, round((support_bound_y*2)/dy) + 1)

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

# Import DNS data
def get_JH_DNS_BL_Data():#x_min, x_max, y_min, y_max, noisy=False, noise_pct=0.0, Nx=0,Ny=0
    file = h5py.File('../data/Transition_BL_Time_Averaged_Profiles.h5', 'r')

    x_JH_DNS = jnp.array(file['x_coor'])
    y_JH_DNS = jnp.array(file['y_coor'])
    # flatten data fields
    u = jnp.array(file['um'])#.flatten()
    v = jnp.array(file['vm'])#.flatten()
    p = jnp.array(file['pm'])#.flatten()
    Ruu = (jnp.array(file['uum'])) - u**2#.flatten()
    Ruv = (jnp.array(file['uvm'])) - u*v#.flatten()
    # Rvv = (jnp.array(file['uvm'])).flatten() - v**2

    # data = jnp.array([u,v,p,Ruu,Ruv])
    data = [u,v,p,Ruu,Ruv]

    return data,x_JH_DNS,y_JH_DNS

def interpolateBLDataForFFTConvolve(data, X_interp, Y_interp, xy, noise_std=0.0):
    # Interpolate data
    if noise_std > 0.0:
        data_interp = griddata(xy, data, (X_interp, Y_interp), method='cubic') + np.random.normal(scale=noise_std, size=X_interp.shape)#.flatten()#np.zeros(X_interp.shape)
    else:
        data_interp = griddata(xy, data, (X_interp, Y_interp), method='cubic')

    return data_interp

def JAX_interpolateBLDataForFFTConvolve(data, X_interp, Y_interp, x_JH_DNS, y_JH_DNS, method='linear', noise_std=0.0):
    if method == 'cubic':
        raise Exception('cubic interpolation not supported by JAX, must use slower code using scipy.interpolate backend')
    # Interpolate data
    # print(method)
    if noise_std > 0.0:
        data_interpolator = RegularGridInterpolator((x_JH_DNS,y_JH_DNS), data.T, method)
        data_interp = data_interpolator((X_interp, Y_interp)) + jnp.array(np.random.normal(scale=noise_std, size=X_interp.shape))

        #(xy, data, (X_interp, Y_interp), method) + np.random.normal(scale=noise_std, size=X_interp.shape)#.flatten()#np.zeros(X_interp.shape)
    else:
        data_interpolator = RegularGridInterpolator((x_JH_DNS,y_JH_DNS), data.T, method)
        data_interp = data_interpolator((X_interp, Y_interp))
        #(xy, data, (X_interp, Y_interp), method)

    return data_interp

# Convolve the data with appropriate TFs to setup Weak Navier-Stokes
def FFTConvolveForNS(data_interp, TF):
    return fftconvolve(data_interp, TF, mode='same')

def interpolateWeakFieldsOntoJH_DNS_Grid(X_JH_DNS,Y_JH_DNS,data_weak,idx_support_bound_x,idx_support_bound_y,x_centers,y_centers):
    # truncate the data which is generated by TFs with partial
    data_weak_JH_DNS =  griddata(
                            jnp.vstack([x_centers[idx_support_bound_y:-idx_support_bound_y, idx_support_bound_x:-idx_support_bound_x].flatten(),
                                        y_centers[idx_support_bound_y:-idx_support_bound_y, idx_support_bound_x:-idx_support_bound_x].flatten()]).T,
                                data_weak[idx_support_bound_y:-idx_support_bound_y, idx_support_bound_x:-idx_support_bound_x].flatten(),
                                (X_JH_DNS, Y_JH_DNS),
                                'nearest')
    return data_weak_JH_DNS

def JAX_interpolateWeakFieldsOntoJH_DNS_Grid(x_JH_DNS_chunk,y_JH_DNS_chunk,data_weak,idx_support_bound_x,idx_support_bound_y,x_centers,y_centers):
    # truncate the data which is generated by TFs with partial
    x_centers_trunc = x_centers[idx_support_bound_x:-idx_support_bound_x]
    y_centers_trunc = y_centers[idx_support_bound_y:-idx_support_bound_y]
    # print(data_weak.shape)
    data_weak_trunc = data_weak[idx_support_bound_y:-idx_support_bound_y, idx_support_bound_x:-idx_support_bound_x]

    weak_interpolator_on_JH_DNS_Grid = RegularGridInterpolator((x_centers_trunc, y_centers_trunc), data_weak_trunc.T, method='nearest')# NOTE that the JAX interpolator expects data laid out "ij" style or "row major" layout, i.e. the first index is the x and the second is the y. Thus, we take the transpose of the data here.
    data_weak_JH_DNS = weak_interpolator_on_JH_DNS_Grid((x_JH_DNS_chunk, y_JH_DNS_chunk))

    # plt.pcolormesh(x_centers_trunc,y_centers_trunc, data_weak_trunc)
    # plt.colorbar()
    # plt.savefig('data_weak_pre_interp')
    # plt.clf()

    # print(x_centers_trunc)
    # print(x_JH_DNS_chunk)

    # plt.pcolormesh(x_JH_DNS_chunk,y_JH_DNS_chunk, data_weak_JH_DNS)
    # plt.colorbar()
    # plt.savefig('data_weak_post_interp')
    # plt.clf()

    # raise Exception('stop')

    return data_weak_JH_DNS

def calculate_weak_fields(X_JH_DNS_chunk, no_chunks, indicies_per_chunk, y_trunc, support_bound_x, support_bound_y, grid_spacing, yF, y0, data, xy, noise_std, TF_degree, X_JH_DNS_OG, Y_JH_DNS_OG, X):
    # get coords to start and finish the FFT grid
    x0_interp_chunk = int(math.floor((X_JH_DNS_chunk[0] - support_bound_x) * 100.0)) / 100
    xF_interp_chunk = int(math.ceil((X_JH_DNS_chunk[-1] + support_bound_x) * 100.0)) / 100

    # Interpolate data onto a fine, uniform grid for FFT
    # define uniform grid for FFT
    Nx = round((xF_interp_chunk - x0_interp_chunk)/grid_spacing) + 1
    Ny = round((yF - y0)/grid_spacing) + 1
    xx = jnp.linspace(x0_interp_chunk, xF_interp_chunk, Nx,dtype=jnp.float32)
    yy = jnp.linspace(y0, yF, Ny,dtype=jnp.float32)
    X_interp, Y_interp = jnp.meshgrid(xx, yy)
    X_JH_DNS_trunc_chunk, Y_JH_DNS_trunc_chunk = jnp.meshgrid(X_JH_DNS_chunk, y_trunc) 

    # Define x & y grid spacing for portion of the FFT domain that is actually accurate
    # Define amount we will be cutting off to maintain accuracy
    idx_support_bound_y = round(support_bound_y / grid_spacing)
    idx_support_bound_x = round(support_bound_x / grid_spacing)

    for j,field in enumerate(data):
        # print(field.type)
        # field = jnp.reshape(field, shape=X.shape) # reshape for JAX compatibility
        # UUx
        if j == 0:
            data_interp = JAX_interpolateBLDataForFFTConvolve(field, X_interp, Y_interp, X_JH_DNS_OG, Y_JH_DNS_OG, noise_std=noise_std)

            UUx_weak = 0.5 * JAX_interpolateWeakFieldsOntoJH_DNS_Grid(X_JH_DNS_trunc_chunk,Y_JH_DNS_trunc_chunk,
                                            FFTConvolveForNS(data_interp**2, TF_x(TF_degree, support_bound_x,support_bound_y, grid_spacing)),
                                            idx_support_bound_x,idx_support_bound_y, xx, yy)


            UVy_weak = JAX_interpolateWeakFieldsOntoJH_DNS_Grid(X_JH_DNS_trunc_chunk,Y_JH_DNS_trunc_chunk,
                                            FFTConvolveForNS(data_interp * JAX_interpolateBLDataForFFTConvolve(data[1], X_interp, Y_interp, X_JH_DNS_OG, Y_JH_DNS_OG, noise_std=noise_std),
                                            TF_y(TF_degree, support_bound_x,support_bound_y, grid_spacing)),
                                            idx_support_bound_x,idx_support_bound_y, xx, yy) + UUx_weak


            LapU_weak = JAX_interpolateWeakFieldsOntoJH_DNS_Grid(X_JH_DNS_trunc_chunk,Y_JH_DNS_trunc_chunk,
                                            FFTConvolveForNS(data_interp, TF_xx(TF_degree, support_bound_x,support_bound_y, grid_spacing)),
                                            idx_support_bound_x,idx_support_bound_y, xx, yy) + \
                        JAX_interpolateWeakFieldsOntoJH_DNS_Grid(X_JH_DNS_trunc_chunk,Y_JH_DNS_trunc_chunk,
                                            FFTConvolveForNS(data_interp, TF_yy(TF_degree, support_bound_x,support_bound_y, grid_spacing)),
                                            idx_support_bound_x,idx_support_bound_y, xx, yy)
        elif j==1:
            # Do nothing since V-vel is only used with U-vel
            continue
        elif j == 2:
            # field = data[0]
            # TF = TF_x(TF_degree, support_bound_x, support_bound_y, grid_spacing)
            Px_weak = JAX_interpolateWeakFieldsOntoJH_DNS_Grid(X_JH_DNS_trunc_chunk,Y_JH_DNS_trunc_chunk,
                                            FFTConvolveForNS(JAX_interpolateBLDataForFFTConvolve(field, X_interp, Y_interp, X_JH_DNS_OG, Y_JH_DNS_OG, noise_std=noise_std),
                                            TF_x(TF_degree, support_bound_x, support_bound_y, grid_spacing)),
                                            idx_support_bound_x,idx_support_bound_y, xx, yy)
        elif j == 3:
            # TF = TF_x(TF_degree, support_bound_x, support_bound_y, grid_spacing)
            Ruu_x_weak = JAX_interpolateWeakFieldsOntoJH_DNS_Grid(X_JH_DNS_trunc_chunk,Y_JH_DNS_trunc_chunk,
                                            FFTConvolveForNS(JAX_interpolateBLDataForFFTConvolve(field, X_interp, Y_interp, X_JH_DNS_OG, Y_JH_DNS_OG, noise_std=noise_std),
                                            TF_x(TF_degree, support_bound_x, support_bound_y, grid_spacing)),
                                            idx_support_bound_x,idx_support_bound_y, xx, yy)
        elif j == 4:
            # TF = TF_y(TF_degree, support_bound_x, support_bound_y, grid_spacing)
            Ruv_y_weak = JAX_interpolateWeakFieldsOntoJH_DNS_Grid(X_JH_DNS_trunc_chunk,Y_JH_DNS_trunc_chunk,
                                            FFTConvolveForNS(JAX_interpolateBLDataForFFTConvolve(field, X_interp, Y_interp, X_JH_DNS_OG, Y_JH_DNS_OG, noise_std=noise_std), 
                                            TF_y(TF_degree, support_bound_x, support_bound_y, grid_spacing)),
                                            idx_support_bound_x,idx_support_bound_y, xx, yy)
        else:
            print('Indices out of range of current RANS terms')
            raise Exception('stop')

    # return calculated_fields
    return UUx_weak, UVy_weak, LapU_weak, Px_weak, Ruu_x_weak, Ruv_y_weak


def get_weak_features(noise, support, nfeatures, no_chunks, TF_degree = 5):
    labels = [r'$\bar{u} \bar{u}_x$', r'$\bar{v}\bar{u}_y$',
        r'$\nu \nabla \bar{\bf{u}}$', r'$\rho^{-1} \bar{p}_x$', 
        r'$\overline{({u^\prime} ^2)}_x$', r'$\overline{(u^\prime v^\prime)}_y$']

    data, X_JH_DNS_OG, Y_JH_DNS_OG = get_JH_DNS_BL_Data()
    X, Y = jnp.meshgrid(X_JH_DNS_OG,Y_JH_DNS_OG)
    xy = jnp.vstack([X.flatten(),Y.flatten()]).T
    noise_std = noise * jnp.std(data[1]) # get noisy as a % of the std of V-vel
    # Setup bounds for domain of interest
    x0 = 30.5
    xF = 999.7
    y0 = 0.01
    yF = 26.45
    grid_spacing = 0.01
    # TF_degree = 5

    support_bound_x = support[0]
    support_bound_y = support[1]

    # os.mkdir(save_dir + f'{noise}_Noise/support_{support_bound_x}_{support_bound_y}')
    # os.mkdir(save_dir + f'{noise}_Noise/support_{support_bound_x}_{support_bound_y}')

    if support_bound_x < grid_spacing:
        print('warning, x support is smaller than x grid spacing, accuracy and success may vary')

    if support_bound_y < grid_spacing:
        print('warning, y support is smaller than y grid spacing, accuracy and success may vary')
    time_0 = time.time()

    # os.mkdir(save_dir + f'{noise}_Noise/support_{support_bound_x}_{support_bound_y}trial{i}')

    # Calculate full truncated dimensions
    # Define the max and min ranges for which the TF integration remains accurate
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

    y_trunc = Y_JH_DNS_OG[bottom_idx:-top_idx]
    x_trunc = X_JH_DNS_OG[left_idx:-right_idx]

    X_JH_DNS_trunc, Y_JH_DNS_trunc = jnp.meshgrid(x_trunc, y_trunc)

    indicies_per_chunk = x_trunc.size // no_chunks

    # Now we have to save everything
    UUx_weak_OG   = np.zeros_like(X_JH_DNS_trunc)
    UVy_weak_OG   = np.zeros_like(X_JH_DNS_trunc)
    LapU_weak_OG  = np.zeros_like(X_JH_DNS_trunc)
    Px_weak_OG    = np.zeros_like(X_JH_DNS_trunc)
    Ruu_x_weak_OG = np.zeros_like(X_JH_DNS_trunc)
    Ruv_y_weak_OG = np.zeros_like(X_JH_DNS_trunc)


    # for chunk_idx, X_JH_DNS_chunk in enumerate(jnp.array_split(x_trunc,no_chunks)):
        # calculate_weak_fields
    for chunk_idx in range(no_chunks):
        if chunk_idx == 0:
            chunk_idx_end   = (chunk_idx + 1) * indicies_per_chunk
            # print(chunk_idx_end)
            X_JH_DNS_chunk = x_trunc[:chunk_idx_end]
        elif chunk_idx == no_chunks-1:
            chunk_idx_start = chunk_idx * indicies_per_chunk
            # chunk_idx_end   = right_idx
            X_JH_DNS_chunk = x_trunc[chunk_idx_start:] # target grid points on JH DNS to approx via weak form

        else:
            chunk_idx_start = chunk_idx * indicies_per_chunk
            chunk_idx_end   = (chunk_idx + 1) * indicies_per_chunk
            X_JH_DNS_chunk = x_trunc[chunk_idx_start:chunk_idx_end] # target grid points on JH DNS to approx via weak form


        UUx_weak, UVy_weak, LapU_weak, Px_weak, Ruu_x_weak, Ruv_y_weak = calculate_weak_fields(X_JH_DNS_chunk, 
                                                                                                no_chunks, 
                                                                                                indicies_per_chunk, 
                                                                                                y_trunc, 
                                                                                                support_bound_x, 
                                                                                                support_bound_y, 
                                                                                                grid_spacing, 
                                                                                                yF, 
                                                                                                y0, 
                                                                                                data, 
                                                                                                xy, 
                                                                                                noise_std, 
                                                                                                TF_degree, 
                                                                                                X_JH_DNS_OG, 
                                                                                                Y_JH_DNS_OG,
                                                                                                X)
        
        if chunk_idx == 0:
            chunk_idx_end = (chunk_idx + 1) * indicies_per_chunk
            # print(chunk_idx_end)
            UUx_weak_OG[:,:chunk_idx_end]   = UUx_weak
            UVy_weak_OG[:,:chunk_idx_end]   = UVy_weak
            LapU_weak_OG[:,:chunk_idx_end]  = LapU_weak
            Px_weak_OG[:,:chunk_idx_end]    = Px_weak
            Ruu_x_weak_OG[:,:chunk_idx_end] = Ruu_x_weak
            Ruv_y_weak_OG[:,:chunk_idx_end] = Ruv_y_weak

        elif chunk_idx == no_chunks-1:
            chunk_idx_start = chunk_idx * indicies_per_chunk
            # print(chunk_idx_start)
            UUx_weak_OG[:,chunk_idx_start:]   = UUx_weak
            UVy_weak_OG[:,chunk_idx_start:]   = UVy_weak
            LapU_weak_OG[:,chunk_idx_start:]  = LapU_weak
            Px_weak_OG[:,chunk_idx_start:]    = Px_weak
            Ruu_x_weak_OG[:,chunk_idx_start:] = Ruu_x_weak
            Ruv_y_weak_OG[:,chunk_idx_start:] = Ruv_y_weak

        else:
            chunk_idx_start = chunk_idx * indicies_per_chunk
            chunk_idx_end   = (chunk_idx + 1) * indicies_per_chunk
            # print(chunk_idx_start)
            # print(chunk_idx_end)
            UUx_weak_OG[:,chunk_idx_start:chunk_idx_end]   = UUx_weak
            UVy_weak_OG[:,chunk_idx_start:chunk_idx_end]   = UVy_weak
            LapU_weak_OG[:,chunk_idx_start:chunk_idx_end]  = LapU_weak
            Px_weak_OG[:,chunk_idx_start:chunk_idx_end]    = Px_weak
            Ruu_x_weak_OG[:,chunk_idx_start:chunk_idx_end] = Ruu_x_weak
            Ruv_y_weak_OG[:,chunk_idx_start:chunk_idx_end] = Ruv_y_weak
    
    print('done')

    nu = 1/800
    features = jnp.vstack([UUx_weak_OG.flatten(),
                            UVy_weak_OG.flatten(),
                            -nu*(LapU_weak_OG).flatten(),
                            Px_weak_OG.flatten(),
                            Ruu_x_weak_OG.flatten(),
                            Ruv_y_weak_OG.flatten()]).T

    return features, X_JH_DNS_trunc, Y_JH_DNS_trunc

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

    # split noises to distribute data generation across a few GPUs
    # noises = np.array([0.01, 0.1]) # run 1
    # noises = np.array([1.0, 0.025, 0.05]) # run 2
    # noises = np.array([0.0, 0.25, 0.5]) # run 3

    # corresponding_selected_supports = np.array([(0.3,0.2), (10.0,0.1)]) # run 1
    # # corresponding_selected_supports = np.array([(12.0,0.3), (0.8,0.15), (10.0,0.14)]) # run 2
    # # corresponding_selected_supports = np.array([(0.2,0.1), (2.0,0.4),(12.0,0.3)]) # run 3

    # corresponding_selected_alphas = np.array([5.5, 55.0]) # run 1
    # # corresponding_selected_alphas = np.array([220.0, 10.0, 80.0]) # run 2
    # # corresponding_selected_alphas = np.array([2.0, 55.0, 200.0]) # run 3

    noises = np.array([0.0, 0.25, 0.01, 0.1]) # run 1
    noises = np.array([1.0, 0.025, 0.05, 0.5]) # run 2

    corresponding_selected_supports = np.array([(0.2,0.1), (2.0,0.4), (0.3,0.2), (10.0,0.1)]) # run 1
    corresponding_selected_supports = np.array([(12.0,0.3), (0.8,0.15), (10.0,0.14), (12.0,0.3)]) # run 2

    corresponding_selected_alphas = np.array([2.0, 55.0, 5.5, 55.0]) # run 1
    corresponding_selected_alphas = np.array([220.0, 10.0, 80.0, 200.0]) # run 2

    TF_degree = 5

    for noise_index,noise in enumerate(noises):

        support_of_interest = corresponding_selected_supports[noise_index]
        optimal_alpha = corresponding_selected_alphas[noise_index]

        plot_terms = True
        print(support_of_interest)


        support_bound_x = support_of_interest[0]
        support_bound_y = support_of_interest[1]

        labels = [r'$\bar{u} \bar{u}_x$', r'$\bar{v}\bar{u}_y$',
            r'$\nu \nabla \bar{\bf{u}}$', r'$\rho^{-1} \bar{p}_x$', 
            r'$\overline{({u^\prime} ^2)}_x$', r'$\overline{(u^\prime v^\prime)}_y$']

        from jax.lib import xla_bridge 
        print(xla_bridge.get_backend().platform)

        no_runs = 100
        nfeatures = 6
        nc = 6

        no_chunks = 10
        # NOTE: Must change these directories to fit your local file structure if interested in recreating the data used to compute the Error in Weak vsd Vanilla DB comparison figure
        os.mkdir(f'plots/ExpandedConvectionTerm/sPCA_models/Noise_{noise}/')
        os.mkdir(f'plots/ExpandedConvectionTerm/sPCA_models/Noise_{noise}/support_{support_bound_x}_{support_bound_y}/')
        os.mkdir(f'plots/ExpandedConvectionTerm/sPCA_models/Noise_{noise}/support_{support_bound_x}_{support_bound_y}/alph{optimal_alpha}/')
        save_dir = f'plots/ExpandedConvectionTerm/sPCA_models/Noise_{noise}/support_{support_bound_x}_{support_bound_y}/alph{optimal_alpha}/TF_degree_{TF_degree}/'
        os.mkdir(save_dir)

        for i in range(no_runs):
            os.mkdir(save_dir + f'trial{i}/')

            print(f'trial_{i}')

            features, X_JH_DNS_trunc, Y_JH_DNS_trunc = get_weak_features(noise, support_of_interest, nfeatures, no_chunks,TF_degree)
            # features = features*100
            model = train_gmm_model(nc, features)

            cluster_idx = model.predict(features)
            clustermap = np.reshape(cluster_idx, X_JH_DNS_trunc.shape)

            np.save(save_dir + f'trial{i}/cluster_idx.npy', cluster_idx)

            np.save(save_dir + f'trial{i}/clustermap.npy', clustermap)

            # Plot OG DB domain
            plt.figure(figsize = (15,4))
            plt.pcolormesh(X_JH_DNS_trunc,Y_JH_DNS_trunc, clustermap+1, cmap = cm, vmin=-0.5, vmax=cm.N-0.5)
            plt.colorbar(boundaries=jnp.arange(0.5, int(nc)+1.5), ticks=jnp.arange(1, int(nc)+2))
            plt.savefig(save_dir + f'trial{i}/ClusterDomain,nc{nc}')
            plt.clf()

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
            plt.savefig(save_dir  + f'trial{i}/CVForDomBal')
            plt.close()


            spca_model = train_sPCA_model(optimal_alpha, features, nfeatures, nc, cluster_idx)
            balance_models, model_index = np.unique(spca_model, axis=0, return_inverse=True)
            # print(balance_models)
            nmodels = balance_models.shape[0]

            balance_idx = np.array([model_index[i] for i in cluster_idx])
            balancemap = np.reshape(balance_idx, X_JH_DNS_trunc.shape)
            # print(balance_models)

            # Save balance_model and balance maps & plot them too
            plt.figure(figsize = (15,4))
            plt.pcolormesh(X_JH_DNS_trunc,Y_JH_DNS_trunc, balancemap+1, cmap = cm, vmin=-0.5, vmax=cm.N-0.5)
            plt.colorbar(boundaries=jnp.arange(0.5, int(nmodels)+1.5), ticks=jnp.arange(1, int(nmodels)+2))
            plt.savefig(save_dir + f'trial{i}/sPCA_ClusterDomain,nc{nc}')
            plt.clf()

            # Plot a grid with active terms in each cluster
            gridmap = balance_models.copy()
            gridmask = gridmap==0
            gridmap = (gridmap.T*np.arange(nmodels)).T+1  # Scale map so that active terms can be color-coded
            gridmap[gridmask] = 0

            # NOTE We're skipping the deleting of unidentified terms because I need to count terms correctly identified as 0 in my quantitative metric
            # plot spca_model matrix
            # Delete zero terms
            # grid_mask = np.nonzero( np.all(gridmap==0, axis=0) )[0]
            # gridmap = np.delete(gridmap, grid_mask, axis=1)
            # grid_labels = np.delete(labels, grid_mask)

            grid_labels = labels

            plt.figure(figsize=(6, 4))
            plt.pcolor(gridmap, vmin=-0.5, vmax=cm.N-0.5, cmap=cm, edgecolors='k', linewidth=1)
            plt.gca().set_xticks(np.arange(0.5, gridmap.shape[1]+0.5))
            plt.gca().set_xticklabels(grid_labels, fontsize=14)
            plt.gca().set_yticklabels([])

            for axis in ['top','bottom','left','right']:
                plt.gca().spines[axis].set_linewidth(2)

            plt.gca().tick_params(axis='both', width=0)
            plt.savefig(save_dir + f'trial{i}/sPCA_Matrix_Model')
            plt.clf()

            # save cluster_idx np array and spca_model np
            np.save(save_dir + f'trial{i}/spca_model.npy', balance_models)
            np.save(save_dir + f'trial{i}/balancemap.npy', balancemap)

            np.save(save_dir + f'trial{i}/features.npy', features)
            np.save(save_dir + f'trial{i}/X_JH_DNS_trunc.npy', X_JH_DNS_trunc)
            np.save(save_dir + f'trial{i}/Y_JH_DNS_trunc.npy', Y_JH_DNS_trunc)

                    
            if plot_terms:
                os.mkdir(save_dir + 'terms/')

                plt.figure(figsize = (15,4))
                plt.pcolormesh(X_JH_DNS_trunc,Y_JH_DNS_trunc, np.reshape(features[:,0], X_JH_DNS_trunc.shape), cmap = 'RdBu', vmin=-0.5, vmax=0.5)
                plt.colorbar()
                plt.savefig(save_dir + 'terms/' + f'UUx_weak')
                plt.clf()

                plt.figure(figsize = (15,4))
                plt.pcolormesh(X_JH_DNS_trunc,Y_JH_DNS_trunc, np.reshape(features[:,1], X_JH_DNS_trunc.shape), cmap = 'RdBu', vmin=-0.5, vmax=0.5)
                plt.colorbar()
                plt.savefig(save_dir + 'terms/' + f'VUy_weak')
                plt.clf()

                plt.figure(figsize = (15,4))
                plt.pcolormesh(X_JH_DNS_trunc,Y_JH_DNS_trunc, np.reshape(features[:,2], X_JH_DNS_trunc.shape), cmap = 'RdBu', vmin=-0.5, vmax=0.5)
                plt.colorbar()
                plt.savefig(save_dir + 'terms/' + f'nu_LapU_weak')
                plt.clf()

                plt.figure(figsize = (15,4))
                plt.pcolormesh(X_JH_DNS_trunc,Y_JH_DNS_trunc, np.reshape(features[:,3], X_JH_DNS_trunc.shape), cmap = 'RdBu', vmin=-0.5, vmax=0.5)
                plt.colorbar()
                plt.savefig(save_dir + 'terms/' + f'P_x_weak')
                plt.clf()

                plt.figure(figsize = (15,4))
                plt.pcolormesh(X_JH_DNS_trunc,Y_JH_DNS_trunc, np.reshape(features[:,4], X_JH_DNS_trunc.shape), cmap = 'RdBu', vmin=-0.5, vmax=0.5)
                plt.colorbar()
                plt.savefig(save_dir + 'terms/' + f'Ruu_x_weak')
                plt.clf()

                plt.figure(figsize = (15,4))
                plt.pcolormesh(X_JH_DNS_trunc,Y_JH_DNS_trunc, np.reshape(features[:,5], X_JH_DNS_trunc.shape), cmap = 'RdBu', vmin=-0.5, vmax=0.5)
                plt.colorbar()
                plt.savefig(save_dir + 'terms/' + f'Ruv_y_weak')
                plt.clf()

                plot_terms = False

if __name__ == "__main__":
    main()