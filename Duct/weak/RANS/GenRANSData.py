'''
Maybe the issue comes from integrating th TF across [-1,1] insteadt of the true support bound.

I wonder if something like TF_[-1,1] = TF[-supp,supp]/dx or /dx*dy?

I'm sure I could plot that

NO

Lets try changing the range of the data to -> [-100,100]
'''


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy import sparse, linalg
from scipy.io import loadmat,savemat
from math import factorial

from numpy.random import randint
import sklearn as sk
from sklearn.mixture import GaussianMixture # jaxxx??? # Hyakkk
from sklearn.decomposition import SparsePCA
# from scipy.io import loadmat
import matplotlib as mpl
from matplotlib.colors import ListedColormap

# Seaborn colormap
import seaborn as sns
import colorcet as cc
sns_list = sns.color_palette(cc.glasbey,n_colors=20).as_hex()

# print(sns_list)
# raise Exception('stop')

# sns_list.insert(0, '#ffffff')  # Insert white at zero position
sns_cmap = ListedColormap(sns_list)

import jax.numpy as jnp
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
from jax.scipy.signal import fftconvolve
from jax.scipy.interpolate import RegularGridInterpolator
import math
import time
import pickle

cm = sns_cmap
  
mpl_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
            '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
            '#bcbd22', '#17becf']

# from matplotlib import cm
# from matplotlib.ticker import LinearLocator
# Import Vinuesa derivative operator that was specifically given as 6th order accurate for the duct
from RicardoHelper import diff6

def Dx(f, x):
    ny,nx = f.shape
    dfdx = np.ndarray(f.shape)
    for i in range(ny):
        dfdx[i,:] = diff6(f[i,:], x)

    return dfdx

def Dy(f, y):
    ny,nx = f.shape
    dfdx = np.ndarray(f.shape)
    for i in range(nx):
        dfdx[:,i] = diff6(f[:,i], y)

    return dfdx


def train_gmm_model(nc, features, seed=-1, sample_pct=0.95, mode='kmeans'):
    # use random seed if not specified
    if seed == -1:
        seed = randint(2**32)
    else:
        pass
    
    print(seed)
    model = GaussianMixture(n_components=nc, random_state=seed, n_init = 10, init_params=mode)

    # PERMUTATION
    mask = np.random.permutation(features.shape[0])[:int(sample_pct*features.shape[0])]
    model.fit(features[mask, :])

    return model

'''
TF
'''
def TF(degree, support_bound_x, support_bound_y,dx):
    # save grid spacing (must be equal for FFT)
    dy = dx
    x_domain_TF = jnp.linspace(-1, 1, round((2*support_bound_x)/dx) + 1)
    y_domain_TF = jnp.linspace(-1, 1, round((2*support_bound_y)/dy) + 1)

    x_domain_full_TF, y_domain_full_TF = jnp.meshgrid(x_domain_TF, y_domain_TF)

    a = -1
    b = 1
    c = -1
    d = 1
    q = p = degree

    C_x = 1 / (p**p * q**q) * ((p+q)/(b-a))**(p+q)
    C_y = 1 / (p**p * q**q) * ((p+q)/(d-c))**(p+q)

    print(C_x)
    print(C_y)

    TF = C_x * C_y * (x_domain_full_TF-a)**p * (b-x_domain_full_TF)**q * (y_domain_full_TF-c)**p * (d-y_domain_full_TF)**q
    return TF


'''
TF
'''
def TF_old(degree, support_bound_x, support_bound_y,dx):
    # save grid spacing (must be equal for FFT)
    dy = dx
    x_domain_TF = jnp.linspace(-support_bound_x, support_bound_x, round((2*support_bound_x)/dx) + 1)
    y_domain_TF = jnp.linspace(-support_bound_y, support_bound_y, round((2*support_bound_y)/dy) + 1)

    x_domain_full_TF, y_domain_full_TF = jnp.meshgrid(x_domain_TF, y_domain_TF)

    a = -support_bound_x
    b = support_bound_x
    c = -support_bound_y
    d = support_bound_y
    q = p = degree

    C_x = 1 / (p**p * q**q) * ((p+q)/(b-a))**(p+q)
    C_y = 1 / (p**p * q**q) * ((p+q)/(d-c))**(p+q)

    print(C_x)
    print(C_y)

    TF = C_x * C_y * (x_domain_full_TF-a)**p * (b-x_domain_full_TF)**q * (y_domain_full_TF-c)**p * (d-y_domain_full_TF)**q
    return TF

'''
TF_x
'''
def TF_x_old(degree, support_bound_x, support_bound_y,dx):
    # save grid spacing (must be equal for FFT)
    dy = dx
    x_domain_TF = jnp.linspace(-support_bound_x, support_bound_x, round((2*support_bound_x)/dx) + 1)
    y_domain_TF = jnp.linspace(-support_bound_y, support_bound_y, round((2*support_bound_y)/dy) + 1)

    x_domain_full_TF, y_domain_full_TF = jnp.meshgrid(x_domain_TF, y_domain_TF)

    a = -support_bound_x
    b = support_bound_x
    c = -support_bound_y
    d = support_bound_y
    q = p = degree

    C_x = 1 / (p**p * q**q) * ((p+q)/(b-a))**(p+q)
    C_y = 1 / (p**p * q**q) * ((p+q)/(d-c))**(p+q)

    # TODO: prob need to look into error of translating variables to the normalized [-1,1 domain]... not sure where there's this error except that something probably isn't getting normalized
    # could be work trying the Jacobi polynomial implementation since I think these polynomials are equal to the n=0 case
    non_func_x = C_x*C_y  * (y_domain_full_TF-c)**p * (d-y_domain_full_TF)**q
    return non_func_x * ( (p*(x_domain_full_TF-a)**(p-1) * (b-(x_domain_full_TF))**q) + ((x_domain_full_TF-a)**p * -q * (b-(x_domain_full_TF))**(q-1)) )

'''
TF_y
'''
def TF_y_old(degree, support_bound_x, support_bound_y,dx):
    # save grid spacing (must be equal for FFT)
    dy = dx
    x_domain_TF = jnp.linspace(-support_bound_x, support_bound_x, round((2*support_bound_x)/dx) + 1)
    y_domain_TF = jnp.linspace(-support_bound_y, support_bound_y, round((2*support_bound_y)/dy) + 1)

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
TF_x
'''
def TF_x(degree, support_bound_x, support_bound_y,dx):
    # save grid spacing (must be equal for FFT)
    dy = dx
    x_domain_TF = jnp.linspace(-1, 1, round((2*support_bound_x)/dx) + 1)
    y_domain_TF = jnp.linspace(-1, 1, round((2*support_bound_y)/dy) + 1)

    x_domain_full_TF, y_domain_full_TF = jnp.meshgrid(x_domain_TF, y_domain_TF)

    a = -1
    b = 1
    c = -1
    d = 1
    q = p = degree

    C_x = 1 / (p**p * q**q) * ((p+q)/(b-a))**(p+q)
    C_y = 1 / (p**p * q**q) * ((p+q)/(d-c))**(p+q)

    # TODO: prob need to look into error of translating variables to the normalized [-1,1 domain]... not sure where there's this error except that something probably isn't getting normalized
    # could be work trying the Jacobi polynomial implementation since I think these polynomials are equal to the n=0 case
    non_func_x = C_x*C_y  * (y_domain_full_TF-c)**p * (d-y_domain_full_TF)**q
    return non_func_x * ( (p*(x_domain_full_TF-a)**(p-1) * (b-(x_domain_full_TF))**q) + ((x_domain_full_TF-a)**p * -q * (b-(x_domain_full_TF))**(q-1)) )

'''
TF_y
'''
def TF_y(degree, support_bound_x, support_bound_y,dx):
    # save grid spacing (must be equal for FFT)
    dy = dx
    x_domain_TF = jnp.linspace(-1, 1, round((2*support_bound_x)/dx) + 1)
    y_domain_TF = jnp.linspace(-1, 1, round((2*support_bound_y)/dy) + 1)

    x_domain_full_TF, y_domain_full_TF = jnp.meshgrid(x_domain_TF, y_domain_TF)

    a = -1
    b = 1
    c = -1
    d = 1
    q = p = degree

    C_x = 1 / (p**p * q**q) * ((p+q)/(b-a))**(p+q)
    C_y = 1 / (p**p * q**q) * ((p+q)/(d-c))**(p+q)

    non_func_y = C_x*C_y  * (x_domain_full_TF-a)**p * (b-x_domain_full_TF)**q
    return non_func_y * ( (p*(y_domain_full_TF-c)**(p-1) * (d-y_domain_full_TF)**q) + ((y_domain_full_TF-c)**p * -q * (d-y_domain_full_TF)**(q-1)) )

def interpolateBLDataForFFTConvolve(data, X_interp, Y_interp, xy, noise_std=0.0):
    # Interpolate data
    if noise_std > 0.0:
        data_interp = griddata(xy, data, (X_interp, Y_interp), method='cubic') + np.random.normal(scale=noise_std, size=X_interp.shape)#.flatten()#np.zeros(X_interp.shape)
    else:
        data_interp = griddata(xy, data, (X_interp, Y_interp), method='cubic')

    return data_interp

def JAX_interpolateBLDataForFFTConvolve(data, X_interp, Y_interp, x_DNS, y_DNS, method='linear', noise_std=0.0):
    if method == 'cubic':
        raise Exception('cubic interpolation not supported by JAX, must use slower code using scipy.interpolate backend')
    # Interpolate data
    # print(method)
    if noise_std > 0.0:
        data_interpolator = RegularGridInterpolator((x_DNS,y_DNS), data.T, method)
        data_interp = data_interpolator((X_interp, Y_interp)) + jnp.array(np.random.normal(scale=noise_std, size=X_interp.shape))

        #(xy, data, (X_interp, Y_interp), method) + np.random.normal(scale=noise_std, size=X_interp.shape)#.flatten()#np.zeros(X_interp.shape)
    else:
        data_interpolator = RegularGridInterpolator((x_DNS,y_DNS), data.T, method)
        data_interp = data_interpolator((X_interp, Y_interp))
        #(xy, data, (X_interp, Y_interp), method)

    return data_interp

# Convolve the data with appropriate TFs to setup Weak Navier-Stokes
def FFTConvolveForNS(data_interp, TF):
    return fftconvolve(data_interp, TF, mode='same')

def interpolateWeakFieldsOntoJH_DNS_Grid(X_DNS,Y_DNS,data_weak,idx_support_bound_x,idx_support_bound_y,x_centers,y_centers):
    # truncate the data which is generated by TFs with partial
    data_weak_JH_DNS =  griddata(
                            jnp.vstack([x_centers[idx_support_bound_y:-idx_support_bound_y, idx_support_bound_x:-idx_support_bound_x].flatten(),
                                        y_centers[idx_support_bound_y:-idx_support_bound_y, idx_support_bound_x:-idx_support_bound_x].flatten()]).T,
                                data_weak[idx_support_bound_y:-idx_support_bound_y, idx_support_bound_x:-idx_support_bound_x].flatten(),
                                (X_DNS, Y_DNS),
                                'nearest')
    return data_weak_JH_DNS

def JAX_interpolateWeakFieldsOntoJH_DNS_Grid(x_DNS_chunk,y_DNS_chunk,data_weak,idx_support_bound_x,idx_support_bound_y,x_centers,y_centers):
    # truncate the data which is generated by TFs with partial
    x_centers_trunc = x_centers[idx_support_bound_x:-idx_support_bound_x]
    y_centers_trunc = y_centers[idx_support_bound_y:-idx_support_bound_y]
    # print(data_weak.shape)
    data_weak_trunc = data_weak[idx_support_bound_y:-idx_support_bound_y, idx_support_bound_x:-idx_support_bound_x]

    weak_interpolator_on_JH_DNS_Grid = RegularGridInterpolator((x_centers_trunc, y_centers_trunc), data_weak_trunc.T, method='nearest')# NOTE that the JAX interpolator expects data laid out "ij" style or "row major" layout, i.e. the first index is the x and the second is the y. Thus, we take the transpose of the data here.
    data_weak_JH_DNS = weak_interpolator_on_JH_DNS_Grid((x_DNS_chunk, y_DNS_chunk))

    return data_weak_JH_DNS

def calculate_weak_fields(x_trunc, y_trunc, support_bound_x, support_bound_y, grid_spacing, yF, y0, x0, xF, data, xy, TF_degree, X_DNS_OG, Y_DNS_OG):

    # Calculate first gradients for use in weak form
    W_x = Dx(data[0],X_DNS_OG)
    W_y = Dy(data[0],Y_DNS_OG)

    plt.pcolormesh(X_DNS_OG, Y_DNS_OG, W_x,cmap='RdBu')
    plt.colorbar()
    plt.savefig('W_x')
    plt.clf()
    plt.close()

    plt.pcolormesh(X_DNS_OG, Y_DNS_OG, W_y,cmap='RdBu')
    plt.colorbar()
    plt.savefig('W_y')
    plt.clf()
    plt.close()

    Nx = round((xF - x0)/grid_spacing) + 1
    Ny = round((yF - y0)/grid_spacing) + 1

    xx = jnp.linspace(x0, xF, Nx,dtype=jnp.float32)
    yy = jnp.linspace(y0, yF, Ny,dtype=jnp.float32)
    X_interp, Y_interp = jnp.meshgrid(xx, yy)
    X_DNS_trunc_chunk, Y_DNS_trunc_chunk = jnp.meshgrid(x_trunc, y_trunc) 

    # Define x & y grid spacing for portion of the FFT domain that is actually accurate
    # Define amount we will be cutting off to maintain accuracy
    idx_support_bound_y = round(support_bound_y / grid_spacing)
    idx_support_bound_x = round(support_bound_x / grid_spacing)



    for j,field in enumerate(data):
        # print(field.type)
        # field = jnp.reshape(field, shape=X.shape) # reshape for JAX compatibility
        # UW_x
        # VW_y
        # W_xx + W_yy

        print(field.shape)

        # print(TF_x(4, support_bound_x,support_bound_y, grid_spacing))

        if j == 0:
            print(TF_old(TF_degree, support_bound_x,support_bound_y, grid_spacing))
            # TODO: do I figure out the gradient -> interpolate or interpolate -> gradient?
            data_interp = JAX_interpolateBLDataForFFTConvolve(field, X_interp, Y_interp, X_DNS_OG, Y_DNS_OG)

            UW_x_weak = JAX_interpolateWeakFieldsOntoJH_DNS_Grid(X_DNS_trunc_chunk,Y_DNS_trunc_chunk,
                                            FFTConvolveForNS(JAX_interpolateBLDataForFFTConvolve(W_x, X_interp, Y_interp, X_DNS_OG, Y_DNS_OG) * JAX_interpolateBLDataForFFTConvolve(data[1], X_interp, Y_interp, X_DNS_OG, Y_DNS_OG),
                                            TF_old(TF_degree, support_bound_x,support_bound_y, grid_spacing)),
                                            idx_support_bound_x,idx_support_bound_y, xx, yy)

            VW_y_weak = JAX_interpolateWeakFieldsOntoJH_DNS_Grid(X_DNS_trunc_chunk,Y_DNS_trunc_chunk,
                                            FFTConvolveForNS(JAX_interpolateBLDataForFFTConvolve(W_y, X_interp, Y_interp, X_DNS_OG, Y_DNS_OG) * JAX_interpolateBLDataForFFTConvolve(data[2], X_interp, Y_interp, X_DNS_OG, Y_DNS_OG),
                                            TF_old(TF_degree, support_bound_x,support_bound_y, grid_spacing)),
                                            idx_support_bound_x,idx_support_bound_y, xx, yy)

            W_xx_weak = JAX_interpolateWeakFieldsOntoJH_DNS_Grid(X_DNS_trunc_chunk,Y_DNS_trunc_chunk,
                                            FFTConvolveForNS(JAX_interpolateBLDataForFFTConvolve(W_x, X_interp, Y_interp, X_DNS_OG, Y_DNS_OG), TF_x_old(TF_degree, support_bound_x,support_bound_y, grid_spacing)),
                                            idx_support_bound_x,idx_support_bound_y, xx, yy)
            W_yy_weak = JAX_interpolateWeakFieldsOntoJH_DNS_Grid(X_DNS_trunc_chunk,Y_DNS_trunc_chunk,
                                            FFTConvolveForNS(JAX_interpolateBLDataForFFTConvolve(W_y, X_interp, Y_interp, X_DNS_OG, Y_DNS_OG) , TF_y_old(TF_degree, support_bound_x,support_bound_y, grid_spacing)),
                                            idx_support_bound_x,idx_support_bound_y, xx, yy)
            
        elif j==1:
            # Do nothing since U-vel is only used with W-vel
            continue
        elif j==2:
            # Do nothing since V-vel is only used with W-vel
            continue
        # elif j == 3:
        #     # Do nothing since pressure gradient is constant throughout the duct in the time-averaged data
        #     continue
        elif j == 3:
            # TF = TF_x(TF_degree, support_bound_x, support_bound_y, grid_spacing)
            Ruw_x_weak = JAX_interpolateWeakFieldsOntoJH_DNS_Grid(X_DNS_trunc_chunk,Y_DNS_trunc_chunk,
                                            FFTConvolveForNS(JAX_interpolateBLDataForFFTConvolve(field, X_interp, Y_interp, X_DNS_OG, Y_DNS_OG),
                                            TF_x_old(TF_degree, support_bound_x, support_bound_y, grid_spacing)),
                                            idx_support_bound_x,idx_support_bound_y, xx, yy)
        elif j == 4:
            # TF = TF_y(TF_degree, support_bound_x, support_bound_y, grid_spacing)
            Rvw_y_weak = JAX_interpolateWeakFieldsOntoJH_DNS_Grid(X_DNS_trunc_chunk,Y_DNS_trunc_chunk,
                                            FFTConvolveForNS(JAX_interpolateBLDataForFFTConvolve(field, X_interp, Y_interp, X_DNS_OG, Y_DNS_OG), 
                                            TF_y_old(TF_degree, support_bound_x, support_bound_y, grid_spacing)),
                                            idx_support_bound_x,idx_support_bound_y, xx, yy)
        elif j == 5:
            # print(np.mean(field))
            # print(field[0,0])
            press_grad_weak = JAX_interpolateWeakFieldsOntoJH_DNS_Grid(X_DNS_trunc_chunk,Y_DNS_trunc_chunk,
                                            FFTConvolveForNS(field[0,0]*np.ones_like(X_interp),
                                            TF_old(TF_degree, support_bound_x, support_bound_y, grid_spacing)),
                                            idx_support_bound_x,idx_support_bound_y, xx, yy)
        else:
            print('Indices out of range of current RANS terms')
            raise Exception('stop')



        # UUx
        # if j == 0:
        #     data_interp = JAX_interpolateBLDataForFFTConvolve(field, X_interp, Y_interp, X_DNS_OG, Y_DNS_OG, noise_std=0.0)

        #     # print(data_interp[idx_support_bound_y:-idx_support_bound_y, idx_support_bound_x:-idx_support_bound_x].shape)
        #     #UV term'
        #     #TODO: is idx_support_bound wwrong for this because I'm working with a different chunk now???
        #     # X_interp, Y_interp, X_DNS_OG, Y_DNS_OG, X_DNS_OG, Y_DNS_OG, X_DNS_OG, Y_DNS_OG
        #     UVy_weak = JAX_interpolateWeakFieldsOntoJH_DNS_Grid(X_DNS_trunc_chunk,Y_DNS_trunc_chunk,
        #                                     FFTConvolveForNS(data_interp * JAX_interpolateBLDataForFFTConvolve(data[1], X_interp, Y_interp, X_DNS_OG, Y_DNS_OG, noise_std=noise_std),
        #                                     TF_y(TF_degree, support_bound_x,support_bound_y, grid_spacing)),
        #                                     idx_support_bound_x,idx_support_bound_y, xx, yy)

        #     # raise Exception('stop')

        #     UUx_weak = JAX_interpolateWeakFieldsOntoJH_DNS_Grid(X_DNS_trunc_chunk,Y_DNS_trunc_chunk,
        #                                     FFTConvolveForNS(data_interp, TF_x(TF_degree, support_bound_x,support_bound_y, grid_spacing)),
        #                                     idx_support_bound_x,idx_support_bound_y, xx, yy)

        #     LapU_weak = JAX_interpolateWeakFieldsOntoJH_DNS_Grid(X_DNS_trunc_chunk,Y_DNS_trunc_chunk,
        #                                     FFTConvolveForNS(data_interp, TF_xx(TF_degree, support_bound_x,support_bound_y, grid_spacing)),
        #                                     idx_support_bound_x,idx_support_bound_y, xx, yy) + \
        #                 JAX_interpolateWeakFieldsOntoJH_DNS_Grid(X_DNS_trunc_chunk,Y_DNS_trunc_chunk,
        #                                     FFTConvolveForNS(data_interp, TF_yy(TF_degree, support_bound_x,support_bound_y, grid_spacing)),
        #                                     idx_support_bound_x,idx_support_bound_y, xx, yy)
        # elif j==1:
        #     # Do nothing since V-vel is only used with U-vel
        #     continue
        # elif j == 2:
        #     # field = data[0]
        #     # TF = TF_x(TF_degree, support_bound_x, support_bound_y, grid_spacing)
        #     Px_weak = JAX_interpolateWeakFieldsOntoJH_DNS_Grid(X_DNS_trunc_chunk,Y_DNS_trunc_chunk,
        #                                     FFTConvolveForNS(JAX_interpolateBLDataForFFTConvolve(field, X_interp, Y_interp, X_DNS_OG, Y_DNS_OG, noise_std=noise_std),
        #                                     TF_x(TF_degree, support_bound_x, support_bound_y, grid_spacing)),
        #                                     idx_support_bound_x,idx_support_bound_y, xx, yy)
        # elif j == 3:
        #     # TF = TF_x(TF_degree, support_bound_x, support_bound_y, grid_spacing)
        #     Ruu_x_weak = JAX_interpolateWeakFieldsOntoJH_DNS_Grid(X_DNS_trunc_chunk,Y_DNS_trunc_chunk,
        #                                     FFTConvolveForNS(JAX_interpolateBLDataForFFTConvolve(field, X_interp, Y_interp, X_DNS_OG, Y_DNS_OG, noise_std=noise_std),
        #                                     TF_x(TF_degree, support_bound_x, support_bound_y, grid_spacing)),
        #                                     idx_support_bound_x,idx_support_bound_y, xx, yy)
        # elif j == 4:
        #     # TF = TF_y(TF_degree, support_bound_x, support_bound_y, grid_spacing)
        #     Ruv_y_weak = JAX_interpolateWeakFieldsOntoJH_DNS_Grid(X_DNS_trunc_chunk,Y_DNS_trunc_chunk,
        #                                     FFTConvolveForNS(JAX_interpolateBLDataForFFTConvolve(field, X_interp, Y_interp, X_DNS_OG, Y_DNS_OG, noise_std=noise_std), 
        #                                     TF_y(TF_degree, support_bound_x, support_bound_y, grid_spacing)),
        #                                     idx_support_bound_x,idx_support_bound_y, xx, yy)
        # else:
        #     print('Indices out of range of current RANS terms')
        #     raise Exception('stop')

    # return calculated_fields
    print('return')
    print(UW_x_weak.shape)
    print(np.min(UW_x_weak))
    print(np.max(UW_x_weak))
    print(TF(TF_degree, support_bound_x,support_bound_y, grid_spacing))
    return UW_x_weak, VW_y_weak, W_xx_weak, W_yy_weak, Ruw_x_weak, Rvw_y_weak, press_grad_weak

def apply_symmetry(array):
    # assuming it's the full 194 x 193 array
    arr_top = array[97:, :]
    arr_topFlipped = np.flip(arr_top, axis = 0)
    arr_bottom = (arr_topFlipped + array[:97, :])/2

    arr_right = arr_bottom[:, 97:]
    arr_left = np.zeros_like(arr_bottom[:, 0:97])

    arr_rightFlipped = np.flip(arr_right, axis = 1)

    arr_left[:,:-1] = (arr_bottom[:, 0:96] + arr_rightFlipped)/2
    arr_left[:,-1] = arr_bottom[:,96]

    return arr_left

def apply_midline_symmetry(array):
    # assuming it's the full 194 x 193 array
    arr_top = array[97:, :]
    arr_topFlipped = np.flip(arr_top, axis = 0)
    arr_bottom = (arr_topFlipped + array[:97, :])/2

    arr_right = arr_bottom[:, 97:]
    arr_left = np.zeros_like(arr_bottom[:, 0:97])

    arr_rightFlipped = np.flip(arr_right, axis = 1)

    arr_left[:,:-1] = (arr_bottom[:, 0:96] - arr_rightFlipped)/2
    arr_left[:,-1] = arr_bottom[:,96]

    return arr_left

def apply_centerline_symmetry(array):
    # assuming it's the full 194 x 193 array
    arr_top = array[97:, :]
    arr_topFlipped = np.flip(arr_top, axis = 0)
    arr_bottom = (array[:97, :] - arr_topFlipped)/2

    arr_right = arr_bottom[:, 97:]
    arr_left = np.zeros_like(arr_bottom[:, 0:97])

    arr_rightFlipped = np.flip(arr_right, axis = 1)

    arr_left[:,:-1] = (arr_bottom[:, 0:96] + arr_rightFlipped)/2
    arr_left[:,-1] = arr_bottom[:,96]

    return arr_left

def apply_diagonal_symmetry(array):
    # assuming it's the full 194 x 193 array
    arr_top = array[97:, :]
    arr_topFlipped = np.flip(arr_top, axis = 0)
    arr_bottom = (array[:97, :] - arr_topFlipped)/2

    arr_right = arr_bottom[:, 97:]
    arr_left = np.zeros_like(arr_bottom[:, 0:97])

    arr_rightFlipped = np.flip(arr_right, axis = 1)

    arr_left[:,:-1] = (arr_bottom[:, 0:96] - arr_rightFlipped)/2
    arr_left[:,-1] = arr_bottom[:,96]

    return arr_left


def main():
    raw_data = loadmat('../data/duct180.mat',squeeze_me=True, struct_as_record=False)['duct180']
    nu = 1/2500 # 1/Re_b
    x = raw_data.xx
    y = raw_data.yy

    xx,yy = jnp.meshgrid(x,y)

    U = raw_data.time.U
    V = raw_data.time.V
    W = raw_data.time.W
    uw = raw_data.time.uw
    vw = raw_data.time.vw

    ''' Apply symmetries of the duct! '''
    # Take bottom left quad
    xx_symm = xx[:97, :97]
    yy_symm = yy[:97, :97]

    # apply different symmetries based on what the full duct plots look like
    U_symm = jnp.array(apply_midline_symmetry(U)) # U is only symmetric across the midline
    V_symm = jnp.array(apply_centerline_symmetry(V)) # V is only symmetric across the centerline
    W_symm = jnp.array(apply_symmetry(W)) # W is symmetric across all 4 quadrants

    uw_symm = jnp.array(apply_midline_symmetry(uw))
    vw_symm = jnp.array(apply_centerline_symmetry(vw))


    U = U_symm / 100
    V = V_symm / 100
    W = W_symm / 100
    uw = uw_symm / 100**2
    vw = vw_symm / 100**2

    X_DNS_OG = jnp.array(100 * x[:97])
    Y_DNS_OG = jnp.array(100 * y[:97])

    print(jnp.max(X_DNS_OG))
    print(jnp.max(Y_DNS_OG))

    # raise Exception('stop')

    X = xx_symm
    Y = yy_symm
    xy = jnp.vstack([X.flatten(),Y.flatten()]).T

    support_arr = np.array([(0.2,0.2)])# (0.15,0.15),,(0.25,0.25),(0.3,0.3),(0.35,0.35),(0.4,0.4),(0.45,0.45),(0.5,0.5)

    for support in support_arr:

        grid_spacing = 0.025
        x0 = -100.0
        xF = 0.0 - grid_spacing # shrink the domain just a tiny bit
        y0 = -100.0
        yF = 0.0 - grid_spacing # shrink the domain just a tiny bit
        TF_degree = 5
        # os.mkdir(f'../plots/RANS/support_tests/grid_spacing_{grid_spacing}_TF_{TF_degree}/')

        support_bound_x = support[0]
        support_bound_y = support[1]


        x0_TF = x0 + support_bound_x
        xF_TF = xF - support_bound_x

        y0_TF = y0 + support_bound_y
        yF_TF = yF - support_bound_y

        # get indices of this accurate range in order to truncate erroneous boundaries from domain when interpolating onto JH DNS Grid later
        idx_support_bound_x = round(support_bound_x / grid_spacing)
        idx_support_bound_y = round(support_bound_y / grid_spacing)

        bottom_idx = (Y_DNS_OG < (y0 + support_bound_y)).sum()
        left_idx = (X_DNS_OG < (x0 + support_bound_x)).sum()

        top_idx = (Y_DNS_OG > (yF - support_bound_y)).sum()
        right_idx = (X_DNS_OG > (xF - support_bound_x)).sum()

        y_trunc = Y_DNS_OG[bottom_idx:-top_idx]
        x_trunc = X_DNS_OG[left_idx:-right_idx]

        X_DNS_trunc, Y_DNS_trunc = jnp.meshgrid(x_trunc, y_trunc)

        # Now we have to save everything
        UUx_weak_OG   = np.zeros_like(X_DNS_trunc)
        UVy_weak_OG   = np.zeros_like(X_DNS_trunc)
        LapU_weak_OG  = np.zeros_like(X_DNS_trunc)
        Px_weak_OG    = np.zeros_like(X_DNS_trunc)
        Ruu_x_weak_OG = np.zeros_like(X_DNS_trunc)
        Ruv_y_weak_OG = np.zeros_like(X_DNS_trunc)

        data_matrix = np.array([W,U,V,uw,vw,0.008666135084163201*np.ones_like(U)])# XXX

        UW_x_weak, VW_y_weak, W_xx_weak, W_yy_weak, Ruw_x_weak, Rvw_y_weak, press_grad_weak = calculate_weak_fields(x_trunc,
                                                                                                y_trunc, 
                                                                                                support_bound_x, 
                                                                                                support_bound_y, 
                                                                                                grid_spacing, 
                                                                                                yF, 
                                                                                                y0,
                                                                                                x0,
                                                                                                xF,
                                                                                                data_matrix, 
                                                                                                xy,
                                                                                                TF_degree, 
                                                                                                X_DNS_OG, 
                                                                                                Y_DNS_OG
                                                                                                )

        print(x_trunc.size)
        print(y_trunc.size)

        
        save_dir = f'plots/results/grid_spacing_{grid_spacing}_TF_{TF_degree}/support_x_{support_bound_x}_support_y_{support_bound_y}/'
        os.mkdir(save_dir)

        plot_terms = False

        if plot_terms:

            os.mkdir(save_dir + 'terms/')

            x_plot, y_plot = np.meshgrid(x_trunc,y_trunc)

            plt.pcolormesh(x_plot, y_plot, UW_x_weak,cmap='RdBu')
            plt.colorbar()
            plt.savefig(save_dir + 'terms/UW_x')
            plt.clf()
            plt.close()

            plt.pcolormesh(x_plot, y_plot, VW_y_weak,cmap='RdBu')
            plt.colorbar()
            plt.savefig(save_dir + 'terms/VW_y')
            plt.clf()
            plt.close()

            plt.pcolormesh(x_plot, y_plot, nu*(W_xx_weak + W_yy_weak),cmap='RdBu')
            plt.colorbar()
            plt.savefig(save_dir + 'terms/LapU')
            plt.clf()
            plt.close()

            plt.pcolormesh(x_plot, y_plot, Ruw_x_weak,cmap='RdBu')
            plt.colorbar()
            plt.savefig(save_dir + 'terms/Ruw_x')
            plt.clf()
            plt.close()

            plt.pcolormesh(x_plot, y_plot, Rvw_y_weak,cmap='RdBu')
            plt.colorbar()
            plt.savefig(save_dir + 'terms/Rvw_y')
            plt.clf()
            plt.close()

            plt.pcolormesh(x_plot, y_plot, press_grad_weak,cmap='RdBu')
            plt.colorbar()
            plt.savefig(save_dir + 'terms/press_grad_weak')
            plt.clf()
            plt.close()

            plt.pcolormesh(x_plot, y_plot, -UW_x_weak - VW_y_weak + nu*(W_xx_weak + W_yy_weak) - Ruw_x_weak - Rvw_y_weak + press_grad_weak)
            plt.colorbar()
            plt.savefig(save_dir + 'terms/RES')
            plt.clf()
            plt.close()

        features = 1e5 * jnp.array([UW_x_weak.flatten(),
                        VW_y_weak.flatten(),
                        nu*(W_xx_weak).flatten(),
                        nu*(W_yy_weak).flatten(),
                        Ruw_x_weak.flatten(),
                        Rvw_y_weak.flatten()
                        ]).T

        print('full features made')

        masked_x_coords_DNS_grid,masked_y_coords_DNS_grid = np.meshgrid(x_trunc,y_trunc)
        labels = [r'$\bar{u} \bar{w}_x$', r'$\bar{v}\bar{w}_y$',
            r'$\nu \bar{w_{xx}}$', r'$\nu \bar{w_{yy}}$', 
            r'$\overline{({u^\prime w^\prime})}_x$', r'$\overline{(v^\prime w^\prime)}_y$']

        nc_arr = [9]

        np.save(save_dir + 'terms/features', features)
        np.save(save_dir + f'terms/masked_x_coords_DNS_grid', masked_x_coords_DNS_grid)
        np.save(save_dir + f'terms/masked_y_coords_DNS_grid', masked_y_coords_DNS_grid)

        nfeatures = 6
        for nc in nc_arr:
            nc_save_dir = save_dir + f'nc{nc}/'
            os.mkdir(nc_save_dir)

            no_runs = 10

            for trial in range(no_runs):

                nc_save_dir = save_dir + f'nc{nc}/trial_{trial}/'
                os.mkdir(nc_save_dir)

                model = train_gmm_model(nc,features, sample_pct=0.5)
                cluster_idx = model.predict(features)+1

                np.save(nc_save_dir + f'cluster_idx', cluster_idx)

                plt.figure(figsize = (5,4))
                plt.pcolormesh(masked_x_coords_DNS_grid,masked_y_coords_DNS_grid,cluster_idx.reshape(masked_y_coords_DNS_grid.shape), cmap = cm, vmin=-0.5, vmax=cm.N-0.5)
                plt.colorbar(boundaries=jnp.arange(0.5, nc+1.5), ticks=jnp.arange(0, nc+1))
                plt.savefig(nc_save_dir + f'ClusterDomain')
                plt.close()



                # print(np.min(cluster_idx))
                # print(np.max(cluster_idx))

                if nc > 12:
                    continue
                elif nc > 9:

                    C_list = []
                    plt.figure(figsize=(9, 14))
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
                    plt.savefig(nc_save_dir + f'VanillaDataCV,nc{nc}')
                    plt.close()
                elif nc > 6:
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
                    plt.savefig(nc_save_dir + f'VanillaDataCV,nc{nc}')
                    plt.close()
                else:
                    C_list = []
                    plt.figure(figsize=(9, 8))
                    for j in range(nc):
                        plt.subplot(2, 3, j+1)
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
                    plt.savefig(nc_save_dir + f'VanillaDataCV,nc{nc}')
                    plt.close()


if __name__ == "__main__":
    main()

    '''
    Looks good.
    Potential ways to make this better
    - Try new TFs (Jacobi Polynomials)
    - Try interpolating back onto DNS grid
    - Variable TF supports, this feels most likely to work because I bet I'm averaging over TONS of data near the edges of the domain where the grid size is ~5e-4
        - Still think differen TFs might help with the nan/inf problem, but maybe not so much with getting the most accurate fields
    '''