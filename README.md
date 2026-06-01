# Weak Dominant Balance

We include all code required to implement the methodology and recreate the results of the paper "Weak Dominant Balance for Robust Identification of Dynamically Consistent Fluid Flow Structure". Our code is not built out to the point of a regularly maintaned package, although any updates will be made accordingly and all feedback, discussions, or inquiries are welcome and appreciated from the community.

# Code

While the applied method remains consistent across examples, each specific application requires a unique pipeline to load the various challenge data sets, and the repo is partitioned as such. We note that we have tested several variations of the method across our challenge cases including weak vs. pointwise equation-space construction, RANS vs. Vorticity Transport equations, and PIV vs DNS data. Thus, in order to remove redundant storage of non-trivially sized datasets in several folders across the code, we have centralized the data storage in a single ```data``` folder, from which files may be copied into the appropriate data folders throughout a downloaded repo as necessary. The DNS fields for the wavy channel exceed githubs maximum files size and are available upon request, although we have included post-method run fields in order to recreate our results.

# Examples

## Transitional Boundary Layer Flow

The ```BoundaryLayer``` folder is currently divided into ```data```, ```pointwise```, and ```weak```. 

```data``` in this case contains a file with the mean fields, since it only needs to be stored in one location.

```pointwise``` contains all code that, when run in numbered order, will generate an ensemble of dominant balance trials across each noise level. The 100 trial files per noise level are too large to store in the folder, but all hyperparameters used to recreate our distributions are provided. 

```weak``` follows the same workflow as in pointwise, now with the proper implementation and hyperparameters required for the weak dominant balance method.

## Turbulent Duct Flow

The ```Duct``` folder is currently divided into ```pointwise``` and ```weak```. The duct data is stored in the general ```data``` folder in the repo to avoid redundant storage of the same data in multiple locations. 

```pointwise``` contains all code for RANS and Vorticity Transport cases, in separate folders. Each notebook in each of the two folders can currently be run with our final clustering models (starting with the line that loads those .npy files) to obtain the figures in the paper, and the data may be moved into the AR_1_180(HiFi) Folder, and the rest of the file may be run to generate novel results with the appropriate hyperparameters (ours are of course included).

```weak``` has the same folder structure, being split by the RANS and Vorticity Transport cases. The included code follows a slightly more involved workflow, where we include the code for our hyperparameter search in the first level of ```RANS``` and ```VortTrans``` for transparency. We have included the final clustering labels to recreate our results without re-running the full code from scratch (see ```Aug7_DuctRANS_errBased_sPCA.ipynb``` for RANS results and ```2_errBased_sPCA_Weak_VortTransDuct.ipynb``` for VortTrans).

## Turbulent Wavy Channel Flow

The ```WavyChannel``` folder is again divided into ```pointwise``` and ```weak``` and PIV data is again stored in the general ```data``` folder to avoid redundant storage. 

```pointwise``` contains PIV and DNS examples, both under the streamwise RANS equation. The notebooks directly in ```WavyChannel``` are used to reproduce our Quantitative comparison of the two methods in ```Weak_WavyWall_QuantitativeComparison.ipynb```.

### pointwise DNS
In ```plotting```, the code to generate our results can be found. The results post GMM step can be found in ```results``` and after the sPCA is run on the GMM cluster, the results are placed in ```post_sPCA_arrays```, and a user may run their own GMMs using cod ein ```Oct_22_VanDBWavyWallDNS.py```, and load those results into the ```plotting/DNS_Pointwise_WavyChannel_Baseline.ipynb``` notebook to run the sPCA on their GMMs.

### pointwise PIV

Code to generate GMMs can be found in ```GenDBTrialData/Oct24_VanDBWavyWallPIV.py```, and ```plotting/PIV_Pointwise_WavyChannel_Baseline.ipynb``` will load results from ```post_sPCA_arrays``` to recreate our results. A practitioner may generate their own GMMs and sPCA reductions using the same workflow.

### weak DNS & PIV

Weak dominant balance implementation for both these examples can be found in the ```GenData``` directory of either of the ```DNS``` and ```PIV``` folders, and ```plotting``` will recreate the results of the paper.

# Data

The data used in our example of turbulent duct flow has been generously provided by the authors of:
R. Vinuesa, A. Noorani, A. Lozano-Durán, G. K. El Khoury, P. Schlatter, P. F. Fischer, and H. M. Nagib. Aspect ratio effects in turbulent duct flows studied through direct numerical simulation. *Journal of Turbulence*, 15(10):677-706, 2014.

The data used in our example of turbulent wavy channel flow has been generously provided by the authors of:
E. Lagemann, S. L. Brunton, and C. Lagemann. Uncovering wall-shear stress dynamics from neural-network enhanced fluid flow measurements. *Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences*, 480(2292):20230798, 2024.

The data used in our example of Transitional Boundary Layer Flow flow is obtained from the Johns Hopkins Turbulence Database, and we are appreciative of the authors of the original DNS:
T. A. Zaki. From streaks to spots and on to turbulence: Exploring the dynamics of boundary layer transition. *Flow, Turbulence and Combustion*, 91(3):451-473, 2013.