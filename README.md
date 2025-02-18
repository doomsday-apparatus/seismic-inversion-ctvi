# Constrained Total Variation Seismic inversion

This repository contains the Python code of a prototype of Constrained Total Variation Seismic Inversion algorithm. Depth-to-Time transform code is also provided for the user to be able to generate synthetic seismic images in time domain.  
The optimization of the cost function is done via Scipy **minimize** function using the **L-BFGS-B** method.


# Contents
All binary files written in column-order (Fortran style) with single precision.
- **data** folder. Contains necessary data to run test case.
    - **input** folder. Contains initial impedance model and seismic image in time domain for inversion.
        - **img_synt_time_nt782_noise30** file. Binary file with synthetic seismic image generated with convolutional modeling with 30% random gaussian noise in L2 norm for each seismic track.
        - **imp_init_time_nt782** file. Binary file with initial acoustic impedance model.
    - **other** folder. Contains data for visualization and comparison with true model.
        - **c_true_nz234nx284dz15dx60** file. True velocity model of the P-wave in depth domain (Marmousi).
        - **c0_start3_nz234nx284dz15dx60** file. Smooth initial velocity model of the P-wave in depth domain(Marmousi).
        - **imp_true_time_nt782** file. True acoustic impedance model in time domain (Marmousi).
        - **rho_true_nz234nx284dz15dx60** file. True density model in depth domain (Marmousi).
    - **output** folder. Contains inversion results.
        - **img_rec_time_nt782** file. Reconstructed seismic image in time domain.
        - **imp_rec_time_nt782** file. Reconstructed acoustic impedance in time domain.
        - **rpp_rec_time_nt782** file. Reconstructed reflection coefficient in time domain.
- **scripts** folder. Contains Python scripts for 1D convolutional modeling, optimization procedure, Depth-to-Time transform and other useful stuff.
    - **depth_to_time.py** script. Code for Depth-to-Time transform and Time-to-Depth transform. Uses cubic spline interpolation for data (seismic image) and linear interpolation for model (acoustic impedance). To obtain reflection coefficients in time domain from reflection coefficients in depth domain and vice versa one should first compute the corresponding acoustic impedance model using reflection coefficients, transform it to other domain and compute reflection coefficients using interpolated acoustic impedance.
    - **impulse.py** script. Used to create convolution matrix for Ricker's impulse or its derivative.
    - **inversion.py** script. Performs CTVI cost function minimization using provided initial impedance track, seismic track and regularization parameters.
    - **objective.py** script. Contains implementation of cost function for minimization.
    - **utils.py** script. Auxiliary functions to read and write binary files, compute impedance using reflection coefficient and vice versa and to apply operators in cost function stabilizers.
- **main.py** script. Main script with code for seismic inversion.
- **plot_result.py** script. Visualization script of the inversion results.
- **prepare_synt_data.py** Auxiliary script for generation of noisy synthetic seismic images via 1D convolutional modeling.
- **README.md** file. Contains file description and guide on how to setup Python Virtual Environment.
- **requirements.txt** file. Contains necessary Python libraries used in presented codes.


# Installation

## Python virtual environment
For Windows, run
```
python.exe -m venv env
env\Scripts\Activate.ps1
pip install -r requirements.txt
```

For Ubuntu, run
```
sudo apt install python3.12-dev python3.12-venv
python3.12 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

# Running the program
- To generate the seismic image one can run **prepare_synt_data.py** script. By default, it will create synthetic seismic image with 30% gaussian noise in L2 norm via 1D convolutional modeling and write created image to **data/input/img_synt_time_nt782_noise30**. This file is already in **data** folder, so the script will overwrite it. The script provides with the example on how to use Depth-to-Time transform and Time-to-Depth transform.
- To apply the seismic inversion one can run **main.py** script. The progress bar will show how much time is left for inversion to be done. One may also change the default values of regularization parameters to explore their influence on the solution. The inversion results will be written in **data/output** folder. These files are already there, so the script will overwrite them. Because the noisy seismic data is already in **data/input** folder one do not need to run **prepare_synt_data.py** script to generate input data for inversion but may want to.
- To visualize the inversion results one can run **plot_result.py** script. Three plots will appear consecutively. First plot shows the comparison of true, initial and reconstructed by inversion acoustic impedance models. Second plot shows the comparison of input seismic image, synthetic seismic image obtained by 1D convolutional modeling with reconstructed impedance model and reconstructed reflection coefficients model. Third plot will show the same comparison but for one chosen track with the number ix set by user. Because the inversion results are already present in **data/output** folder one do not need to run **main.py** script to apply the seismic inversion but may want to.