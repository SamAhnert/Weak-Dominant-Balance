'''
This code contains all necessary implementation of the weak method, appropriate pipeline for loading the duct data, and appropriate hyperparameters
to run weak dominant balance on the Duct data under the VortTrans equation to recreate our final GMMs. application of sPCA can be done in the 
"plot_results_errBased_sPCA_Weak_VortTransDuct.ipynb" file. Equation space and representative vorticity fields can be ploted using the same
files in the RANS case, but with our using the appropriate fields described in this code.

Warning: Running this MAY OVERWRITE the current stored results from the GMM used in the paper! 
(although likely you will encounter "folder already exists" errors before it is able to override current results).

Important: It is not necessary to run this code to recreate plots for results, "plot_results_errBased_sPCA_Weak_VortTransDuct.ipynb" may be run 
independent of/prior to this.
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
# Get lot's of colors
sns_list = sns.color_palette(cc.glasbey,n_colors=20).as_hex()

# sns_list.insert(0, '#ffffff')  # Insert white at zero position? probably only for plotting
sns_cmap = ListedColormap(sns_list)

cm = sns_cmap

import jax.numpy as jnp
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
from jax.scipy.signal import fftconvolve, correlate
from jax.scipy.interpolate import RegularGridInterpolator
import math
import time
import pickle


  
mpl_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
            '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
            '#bcbd22', '#17becf']

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
    model = GaussianMixture(n_components=nc, random_state=seed, n_init = 3, init_params=mode)

    # PERMUTATION
    mask = np.random.permutation(features.shape[0])[:int(sample_pct*features.shape[0])]
    model.fit(features[mask, :])

    return model

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

    TF = C_x * C_y * (x_domain_full_TF-a)**p * (b-x_domain_full_TF)**q * (y_domain_full_TF-c)**p * (d-y_domain_full_TF)**q 
    
    return TF

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

def TF_xy(degree, support_bound_x, support_bound_y,dx):
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

    TF_xy_eval = (C_x * C_y * ( (p*(x_domain_full_TF-a)**(p-1) * (b-(x_domain_full_TF))**q) + ((x_domain_full_TF-a)**p * -q * (b-(x_domain_full_TF))**(q-1)) ) *
                              ( (p*(y_domain_full_TF-c)**(p-1) * (d-y_domain_full_TF)**q) + ((y_domain_full_TF-c)**p * -q * (d-y_domain_full_TF)**(q-1)) ))

    return TF_xy_eval

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
    # NOTE: the literature surrounding weak data-driven system identification has established the convolution as the method to use to reconstruct terms...
    # However, when the convolution is meant to literally represent an integral operator,not a matched filter, the convolution will flip the kernel, which requires a lot of book-keeping, and it is better to simply compute the correlation.
    return correlate(data_interp, TF, mode='same', method='fft')

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

    weak_interpolator_on_JH_DNS_Grid = RegularGridInterpolator((x_centers_trunc, y_centers_trunc), data_weak_trunc.T, method='linear')# NOTE that the JAX interpolator expects data laid out "ij" style or "row major" layout, i.e. the first index is the x and the second is the y. Thus, we take the transpose of the data here.
    data_weak_JH_DNS = weak_interpolator_on_JH_DNS_Grid((x_DNS_chunk, y_DNS_chunk))

    return data_weak_JH_DNS

def calculate_weak_fields(x_trunc, y_trunc, support_bound_x, support_bound_y, grid_spacing, yF, y0, x0, xF, data, xy, TF_degree, X_DNS_OG, Y_DNS_OG):
    # data = [W,U,V, , , ,uu,vv,uv]
    Nx = round((xF - x0)/grid_spacing) + 1
    Ny = round((yF - y0)/grid_spacing) + 1

    xx = jnp.linspace(x0, xF, Nx,dtype=jnp.float32)
    yy = jnp.linspace(y0, yF, Ny,dtype=jnp.float32)
    X_interp, Y_interp = jnp.meshgrid(xx, yy)
    X_DNS_trunc_chunk, Y_DNS_trunc_chunk = jnp.meshgrid(x_trunc, y_trunc) 

    # Define x & y grid spacing for portion of the FFT domain that is actually accurate
    # Define border size we will be truncating to maintain accuracy
    idx_support_bound_y = round(support_bound_y / grid_spacing)
    idx_support_bound_x = round(support_bound_x / grid_spacing)

    # Calculate first gradients for use in weak form
    # TODO: Calculate Gradients pre-interpolation? Otherwise it takes wayyyyy too long?
    # or just stop recreating the derivative operator...
    W_x = Dx(data[0],X_DNS_OG) # Dx_Fast(JAX_interpolateBLDataForFFTConvolve(data[0], X_interp, Y_interp, X_DNS_OG, Y_DNS_OG), X_interp,grid_spacing)
    W_y = Dy(data[0],Y_DNS_OG)# Dy_Fast(JAX_interpolateBLDataForFFTConvolve(data[0], X_interp, Y_interp, X_DNS_OG, Y_DNS_OG), Y_interp,grid_spacing)

    U_x = Dx(data[1],X_DNS_OG) #np.reshape(Dx @ data[1].flatten('F'), data[1].shape, order='F')# np.reshape(Dx @ JAX_interpolateBLDataForFFTConvolve(data[1], X_interp, Y_interp, X_DNS_OG, Y_DNS_OG).flatten(order='F'), X_interp.shape, order='F')
    U_y = Dy(data[1],Y_DNS_OG) #np.reshape(Dy @ data[1].flatten('F'), data[1].shape, order='F')# np.reshape(Dy @ JAX_interpolateBLDataForFFTConvolve(data[1], X_interp, Y_interp, X_DNS_OG, Y_DNS_OG).flatten(order='F'), X_interp.shape, order='F')

    V_x = Dx(data[2],X_DNS_OG) #np.reshape(Dx @ data[2].flatten('F'), data[2].shape, order='F')# np.reshape(Dx @ JAX_interpolateBLDataForFFTConvolve(data[2], X_interp, Y_interp, X_DNS_OG, Y_DNS_OG).flatten(order='F'), X_interp.shape, order='F')
    V_y = Dy(data[2],Y_DNS_OG) #np.reshape(Dy @ data[2].flatten('F'), data[2].shape, order='F')#np.reshape(Dy @ JAX_interpolateBLDataForFFTConvolve(data[2], X_interp, Y_interp, X_DNS_OG, Y_DNS_OG).flatten(order='F'), X_interp.shape, order='F')

    OmegaZ = V_x - U_y

    UOmegaZ_x = -JAX_interpolateWeakFieldsOntoJH_DNS_Grid(
        X_DNS_trunc_chunk, Y_DNS_trunc_chunk,
        FFTConvolveForNS(
            JAX_interpolateBLDataForFFTConvolve(OmegaZ * U_x, X_interp, Y_interp, X_DNS_OG, Y_DNS_OG),
            TF_old(TF_degree, support_bound_x, support_bound_y, grid_spacing)
        )
        + FFTConvolveForNS(
            JAX_interpolateBLDataForFFTConvolve(OmegaZ, X_interp, Y_interp, X_DNS_OG, Y_DNS_OG)
            * JAX_interpolateBLDataForFFTConvolve(data[1], X_interp, Y_interp, X_DNS_OG, Y_DNS_OG),
            TF_x(TF_degree, support_bound_x, support_bound_y, grid_spacing)
        ),
        idx_support_bound_x, idx_support_bound_y, xx, yy
    )

    VOmegaZ_y = -JAX_interpolateWeakFieldsOntoJH_DNS_Grid(
        X_DNS_trunc_chunk,
        Y_DNS_trunc_chunk,
        FFTConvolveForNS(
            JAX_interpolateBLDataForFFTConvolve(OmegaZ * V_y, X_interp, Y_interp, X_DNS_OG, Y_DNS_OG),
            TF_old(TF_degree, support_bound_x, support_bound_y, grid_spacing)  # ⟵ scaled TF here
        )
        + FFTConvolveForNS(
            JAX_interpolateBLDataForFFTConvolve(OmegaZ, X_interp, Y_interp, X_DNS_OG, Y_DNS_OG)
            * JAX_interpolateBLDataForFFTConvolve(data[2], X_interp, Y_interp, X_DNS_OG, Y_DNS_OG),
            TF_y(TF_degree, support_bound_x, support_bound_y, grid_spacing)    # ⟵ scaled derivative TF here
        ),
        idx_support_bound_x,
        idx_support_bound_y,
        xx,
        yy,
    )

    OmegaZ_xx = JAX_interpolateWeakFieldsOntoJH_DNS_Grid(
        X_DNS_trunc_chunk, Y_DNS_trunc_chunk,
        FFTConvolveForNS(
            JAX_interpolateBLDataForFFTConvolve(OmegaZ, X_interp, Y_interp, X_DNS_OG, Y_DNS_OG),
            TF_xx(TF_degree, support_bound_x, support_bound_y, grid_spacing)
        ),
        idx_support_bound_x, idx_support_bound_y, xx, yy
    )

    OmegaZ_yy = JAX_interpolateWeakFieldsOntoJH_DNS_Grid(
        X_DNS_trunc_chunk, Y_DNS_trunc_chunk,
        FFTConvolveForNS(
            JAX_interpolateBLDataForFFTConvolve(OmegaZ, X_interp, Y_interp, X_DNS_OG, Y_DNS_OG),
            TF_yy(TF_degree, support_bound_x, support_bound_y, grid_spacing)
        ),
        idx_support_bound_x, idx_support_bound_y, xx, yy
    )

    OmegaXW_x = JAX_interpolateWeakFieldsOntoJH_DNS_Grid(
        X_DNS_trunc_chunk, Y_DNS_trunc_chunk,
        FFTConvolveForNS(
            JAX_interpolateBLDataForFFTConvolve(-W_y * W_x, X_interp, Y_interp, X_DNS_OG, Y_DNS_OG),
            TF_old(TF_degree, support_bound_x, support_bound_y, grid_spacing)
        ),
        idx_support_bound_x, idx_support_bound_y, xx, yy
    )

    OmegaYW_y = JAX_interpolateWeakFieldsOntoJH_DNS_Grid(
        X_DNS_trunc_chunk, Y_DNS_trunc_chunk,
        FFTConvolveForNS(
            JAX_interpolateBLDataForFFTConvolve(W_x * W_y, X_interp, Y_interp, X_DNS_OG, Y_DNS_OG),
            TF_old(TF_degree, support_bound_x, support_bound_y, grid_spacing)
        ),
        idx_support_bound_x, idx_support_bound_y, xx, yy
    )

    RS_shear = JAX_interpolateWeakFieldsOntoJH_DNS_Grid(
        X_DNS_trunc_chunk, Y_DNS_trunc_chunk,
        FFTConvolveForNS(
            JAX_interpolateBLDataForFFTConvolve(data[8], X_interp, Y_interp, X_DNS_OG, Y_DNS_OG),
            TF_yy(TF_degree, support_bound_x, support_bound_y, grid_spacing)
        )
        - FFTConvolveForNS(
            JAX_interpolateBLDataForFFTConvolve(data[8], X_interp, Y_interp, X_DNS_OG, Y_DNS_OG),
            TF_xx(TF_degree, support_bound_x, support_bound_y, grid_spacing)
        ),
        idx_support_bound_x, idx_support_bound_y, xx, yy
    )

    RS_normal = JAX_interpolateWeakFieldsOntoJH_DNS_Grid(
        X_DNS_trunc_chunk, Y_DNS_trunc_chunk,
        FFTConvolveForNS(
            JAX_interpolateBLDataForFFTConvolve(data[6], X_interp, Y_interp, X_DNS_OG, Y_DNS_OG),
            TF_xy(TF_degree, support_bound_x, support_bound_y, grid_spacing)
        )
        - FFTConvolveForNS(
            JAX_interpolateBLDataForFFTConvolve(data[7], X_interp, Y_interp, X_DNS_OG, Y_DNS_OG),
            TF_xy(TF_degree, support_bound_x, support_bound_y, grid_spacing)
        ),
        idx_support_bound_x, idx_support_bound_y, xx, yy
    )

    return UOmegaZ_x, VOmegaZ_y, OmegaZ_xx, OmegaZ_yy, OmegaXW_x, OmegaYW_y, RS_shear, RS_normal, OmegaZ

def main():
    raw_data = loadmat('../Vinuesa_Duct_Data_HiFi/duct180.mat',squeeze_me=True, struct_as_record=False)['duct180']
    nu = 1/2500 # 1/Re_b
    x = raw_data.xx
    y = raw_data.yy

    print(np.min(np.diff(x)))
    print(np.max(np.diff(x)))

    X_DNS_OG = 100 * x
    Y_DNS_OG = 100 * y

    X, Y = jnp.meshgrid(X_DNS_OG,Y_DNS_OG)
    xy = jnp.vstack([X.flatten(),Y.flatten()]).T

    # Scaling data for more managable support sizes (no effect on outcome, just easier to keep track of)
    U = raw_data.time.U / 100
    V = raw_data.time.V / 100
    W = raw_data.time.W / 100

    uu = raw_data.time.uu / 100**2
    vv = raw_data.time.vv / 100**2
    uv = raw_data.time.uv / 100**2
    uw = raw_data.time.uw / 100**2
    vw = raw_data.time.vw / 100**2

    support_arr = np.array([(0.3,0.3),(0.15,0.15),(0.2,0.2),(0.35,0.35),(0.4,0.4),(0.45,0.45),(0.5,0.5)])#,(0.3,0.3),(0.15,0.15),(0.2,0.2),(0.1,0.1),

    TF_degree = 5
    # grid_spacing = 0.025
    # grid_spacing = 0.025
    grid_spacing = 0.0125
    x0 = -100.0
    y0 = -100.0

    xF = 0.0
    yF = 0.0

    # Make save folder
    os.mkdir(f'results/GMM_data/grid_spacing_{grid_spacing}_TF_{TF_degree}/')

    for support in support_arr:

        support_bound_x = support[0]
        support_bound_y = support[1]

        x0_TF = x0 + support_bound_x
        xF_TF = xF - support_bound_x

        y0_TF = y0 + support_bound_y
        yF_TF = yF - support_bound_y

        # get indices of this accurate range in order to truncate erroneous boundaries from domain when interpolating onto JH DNS Grid later
        bottom_idx = (Y_DNS_OG < (y0_TF)).sum()
        left_idx = (X_DNS_OG < (x0_TF)).sum()

        top_idx = (Y_DNS_OG > (yF_TF)).sum()
        right_idx = (X_DNS_OG > (xF_TF)).sum()

        y_trunc = Y_DNS_OG[bottom_idx:-top_idx]
        x_trunc = X_DNS_OG[left_idx:-right_idx]

        X_DNS_trunc, Y_DNS_trunc = jnp.meshgrid(x_trunc, y_trunc)

        # NOTE: pressure grad (0.008666135084163201*) & uw & vw are not needed here so they are replaced with ones arrays 
        data_matrices = np.array([[W,U,V,np.ones_like(U),np.ones_like(U),np.ones_like(U),uu,vv,uv],
                                  [W,-U,-V,np.ones_like(U),np.ones_like(U),np.ones_like(U),-uu,-vv,-uv],
                                  [W,V,U,np.ones_like(U),np.ones_like(U),np.ones_like(U),vv,uu,uv]])# XXX

        # 8 orientations, swap x & y, + & - x, + & - y
        # for i in range(8):

        # data_matrix = data_matrices[i]
        # Save the whole thing in VortTransTerms_Shifted_Coords or save by term?
        O6_TOmegaZ_t, O6_NOmegaZ_n, O6_OmegaZ_tt, O6_OmegaZ_nn, O6_OmegaTT_t, O6_OmegaNN_n, O6_RS_shear, O6_RS_normal,O6_Omega_z = calculate_weak_fields(x_trunc,
                                                                                                                                y_trunc, 
                                                                                                                                support_bound_x, 
                                                                                                                                support_bound_y, 
                                                                                                                                grid_spacing, 
                                                                                                                                yF, 
                                                                                                                                y0,
                                                                                                                                x0,
                                                                                                                                xF,
                                                                                                                                data_matrices[0], 
                                                                                                                                xy,
                                                                                                                                TF_degree, 
                                                                                                                                X_DNS_OG, 
                                                                                                                                Y_DNS_OG
                                                                                                                                )

        O5_NOmegaZ_n, O5_TOmegaZ_t, O5_OmegaZ_nn, O5_OmegaZ_tt, O5_OmegaNN_n, O5_OmegaTT_t, O5_RS_shear, O5_RS_normal,O5_Omega_z = calculate_weak_fields(x_trunc,
                                                                                                                                y_trunc, 
                                                                                                                                support_bound_x, 
                                                                                                                                support_bound_y, 
                                                                                                                                grid_spacing, 
                                                                                                                                yF, 
                                                                                                                                y0,
                                                                                                                                x0,
                                                                                                                                xF,
                                                                                                                                data_matrices[0], 
                                                                                                                                xy,
                                                                                                                                TF_degree, 
                                                                                                                                X_DNS_OG, 
                                                                                                                                Y_DNS_OG
                                                                                                                                )

        # print(x_trunc.size)
        # print(y_trunc.size)


        # raise Exception('stop, gradients calculated')


        # Now we have to save everything to one grid
        TOmegaZ_t = np.zeros_like(X_DNS_trunc)
        NOmegaZ_n = np.zeros_like(X_DNS_trunc)
        OmegaZ_tt = np.zeros_like(X_DNS_trunc)
        OmegaZ_nn = np.zeros_like(X_DNS_trunc)
        OmegaTT_t = np.zeros_like(X_DNS_trunc)
        OmegaNN_n = np.zeros_like(X_DNS_trunc)
        RS_shear  = np.zeros_like(X_DNS_trunc)
        RS_normal = np.zeros_like(X_DNS_trunc)
        Omega_z = np.zeros_like(X_DNS_trunc)

        for i,x_coord in enumerate(x_trunc):
            for j,y_coord in enumerate(y_trunc):
                if x_coord > y_coord:
                    TOmegaZ_t[j,i] = O6_TOmegaZ_t[j,i]
                    NOmegaZ_n[j,i] = O6_NOmegaZ_n[j,i]
                    OmegaZ_tt[j,i] = O6_OmegaZ_tt[j,i]
                    OmegaZ_nn[j,i] = O6_OmegaZ_nn[j,i]
                    OmegaTT_t[j,i] = O6_OmegaTT_t[j,i]
                    OmegaNN_n[j,i] = O6_OmegaNN_n[j,i]
                    RS_shear[j,i]  = O6_RS_shear[j,i]
                    RS_normal[j,i] = O6_RS_normal[j,i]
                    Omega_z[j,i]   = O6_Omega_z[j,i]
                else:
                    TOmegaZ_t[j,i] = -O5_TOmegaZ_t[j,i]
                    NOmegaZ_n[j,i] = -O5_NOmegaZ_n[j,i]
                    OmegaZ_tt[j,i] = -O5_OmegaZ_tt[j,i]
                    OmegaZ_nn[j,i] = -O5_OmegaZ_nn[j,i]
                    OmegaTT_t[j,i] = O5_OmegaTT_t[j,i]
                    OmegaNN_n[j,i] = O5_OmegaNN_n[j,i]
                    RS_shear[j,i]  = -O5_RS_shear[j,i]
                    RS_normal[j,i] = -O5_RS_normal[j,i]
                    Omega_z[j,i]   = -O5_Omega_z[j,i]


        
        save_dir = f'results/GMM_data/grid_spacing_{grid_spacing}_TF_{TF_degree}/support_x_{support_bound_x}_support_y_{support_bound_y}/'
        os.mkdir(save_dir)
        # Create save dir for flow data, storing features, stc.
        # Save 
        os.mkdir(save_dir + 'features/')

        plot_terms = True

        if plot_terms:

            clim = 3e-8

            plt.pcolormesh(X_DNS_trunc, Y_DNS_trunc, TOmegaZ_t,cmap='RdBu')
            plt.colorbar()
            plt.savefig(save_dir + 'features/' + 'TOmegaZ_t')
            plt.clf()
            plt.close()

            plt.pcolormesh(X_DNS_trunc, Y_DNS_trunc, NOmegaZ_n,cmap='RdBu')
            plt.colorbar()
            plt.savefig(save_dir + 'features/' + 'NOmegaZ_n')
            plt.clf()
            plt.close()

            plt.pcolormesh(X_DNS_trunc, Y_DNS_trunc, nu*(OmegaZ_tt + OmegaZ_nn),cmap='RdBu')
            plt.colorbar()
            plt.savefig(save_dir + 'features/' + 'LapOmegaZ')
            plt.clf()
            plt.close()

            plt.pcolormesh(X_DNS_trunc, Y_DNS_trunc, OmegaTT_t,cmap='RdBu')
            plt.colorbar()
            plt.savefig(save_dir + 'features/' + 'OmegaTT_t')
            plt.clf()
            plt.close()

            plt.pcolormesh(X_DNS_trunc, Y_DNS_trunc, OmegaNN_n,cmap='RdBu')
            plt.colorbar()
            plt.savefig(save_dir + 'features/' + 'OmegaNN_n')
            plt.clf()
            plt.close()

            plt.pcolormesh(X_DNS_trunc, Y_DNS_trunc, RS_shear,cmap='RdBu')
            plt.colorbar()
            plt.savefig(save_dir + 'features/' + 'RS_shear')
            plt.clf()
            plt.close()

            plt.pcolormesh(X_DNS_trunc, Y_DNS_trunc, RS_normal,cmap='RdBu')
            plt.colorbar()
            plt.savefig(save_dir + 'features/' + 'RS_normal')
            plt.clf()
            plt.close()

            # plt.pcolormesh(X_DNS_trunc, Y_DNS_trunc, -UOmegaZ_x - VOmegaZ_y + nu*(OmegaZ_xx + OmegaZ_yy) + OmegaXU_x + OmegaYW_y - RS_shear + RS_normal)
            # plt.colorbar()
            # plt.savefig(save_dir + 'RES')
            # plt.clf()
            # plt.close()

            plt.pcolormesh(X_DNS_trunc, Y_DNS_trunc, TOmegaZ_t,cmap='RdBu', vmin=-clim, vmax=clim)
            plt.colorbar()
            plt.savefig(save_dir + 'features/' + 'TOmegaZ_t_clim')
            plt.clf()
            plt.close()

            plt.pcolormesh(X_DNS_trunc, Y_DNS_trunc, NOmegaZ_n,cmap='RdBu', vmin=-clim, vmax=clim)
            plt.colorbar()
            plt.savefig(save_dir + 'features/' + 'NOmegaZ_n_clim')
            plt.clf()
            plt.close()

            plt.pcolormesh(X_DNS_trunc, Y_DNS_trunc, nu*(OmegaZ_tt + OmegaZ_nn),cmap='RdBu', vmin=-clim, vmax=clim)
            plt.colorbar()
            plt.savefig(save_dir + 'features/' + 'LapOmegaZ_clim')
            plt.clf()
            plt.close()

            plt.pcolormesh(X_DNS_trunc, Y_DNS_trunc, OmegaTT_t,cmap='RdBu', vmin=-clim, vmax=clim)
            plt.colorbar()
            plt.savefig(save_dir + 'features/' + 'OmegaTT_t_clim')
            plt.clf()
            plt.close()

            plt.pcolormesh(X_DNS_trunc, Y_DNS_trunc, OmegaNN_n,cmap='RdBu', vmin=-clim, vmax=clim)
            plt.colorbar()
            plt.savefig(save_dir + 'features/' + 'OmegaNN_n_clim')
            plt.clf()
            plt.close()

            plt.pcolormesh(X_DNS_trunc, Y_DNS_trunc, RS_shear,cmap='RdBu', vmin=-clim, vmax=clim)
            plt.colorbar()
            plt.savefig(save_dir + 'features/' + 'RS_shear_clim')
            plt.clf()
            plt.close()

            plt.pcolormesh(X_DNS_trunc, Y_DNS_trunc, RS_normal,cmap='RdBu', vmin=-clim, vmax=clim)
            plt.colorbar()
            plt.savefig(save_dir + 'features/' + 'RS_normal_clim')
            plt.clf()
            plt.close()

        np.save(save_dir + 'features/Omega_z.npy', Omega_z)

        '''With Weak features created, perform GMM to cluster domain'''

        feature_scaling = 1e6 # Necessary due to magnitude of sthe eed of GMMs as implemented by scipy

        features = feature_scaling * jnp.array([TOmegaZ_t.flatten(),
                        NOmegaZ_n.flatten(),
                        nu*(OmegaZ_tt).flatten(),
                        nu*(OmegaZ_nn).flatten(),
                        RS_shear.flatten(),
                        RS_normal.flatten()
                        ]).T

      

        print('full features made')

        masked_x_coords_DNS_grid,masked_y_coords_DNS_grid = np.meshgrid(x_trunc,y_trunc)

        np.save(save_dir + 'features/features.npy', features)  
        np.save(save_dir + 'features/masked_x_coords_DNS_grid.npy', masked_x_coords_DNS_grid)
        np.save(save_dir + 'features/masked_y_coords_DNS_grid.npy', masked_y_coords_DNS_grid)

        labels = [r'$T \frac{\partial \Omega_z}{\partial t}$', r'$N \frac{\partial \Omega_z}{\partial n}$',
            r'$\frac{\partial^2 \Omega_z}{\partial t^2}$', r'$\frac{\partial^2 \Omega_z}{\partial n^2}$', 
            #r'$\Omega_x\frac{\partial U}{\partial x}$', r'$\Omega_y\frac{\partial V}{\partial y}$',
            r'$(RS_{shear})$', r'$(RS_{norm})$']

        no_trials = 5

        nc_arr = [4,5,6,7,8,9,10]
        for nc in nc_arr:

            # save_dir_trial = save_dir + f'trial_{trial}/'
            nc_save_dir = save_dir + f'nc{nc}/' 

            os.mkdir(nc_save_dir)

            nfeatures = 6
            # for nc in nc_arr:
            for trial in range(no_trials):
                # nc_save_dir = save_dir_trial + f'nc{nc}/'
                save_dir_trial = nc_save_dir + f'trial_{trial}/' 
                os.mkdir(save_dir_trial)

                model = train_gmm_model(nc,features)
                cluster_idx = model.predict(features)+1

                np.save(save_dir_trial + 'cluster_idx',cluster_idx)

                plt.figure(figsize = (5,4))
                plt.pcolormesh(masked_x_coords_DNS_grid,masked_y_coords_DNS_grid,cluster_idx.reshape(masked_y_coords_DNS_grid.shape), cmap = cm, vmin=-0.5, vmax=cm.N-0.5)
                plt.colorbar(boundaries=jnp.arange(0.5, nc+1.5), ticks=jnp.arange(0, nc+1))
                plt.savefig(save_dir_trial + f'ClusterDomain')
                plt.close()

                # print(np.min(cluster_idx))
                # print(np.max(cluster_idx)) 

                if nc > 20:
                    continue

                if nc > 16:
                    C_list = []
                    plt.figure(figsize=(19, 16))
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


if __name__ == "__main__":
    main()