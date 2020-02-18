# rspt to spectra

This acts as an interface from the RSPt software to the [impurityModel](https://github.com/JohanSchott/impurityModel) spectra software.

This software contains a few Python modules that can be useful for parsing and analyzing RSPt output, and constructing an Anderson impurity Hamiltonian, including bath orbitals.
A few scripts are stored in the `scripts` folder.

Separately, in the module plotSpectra, a few functions for reading and plotting output from the Quanty software is provided.

## Preparations
- Execute the bash-script `setup.sh`:
```bash
./setup.sh
```
This will create a Python virtual environment and install the required Python packages.

- Activate the virtual environment and set the PYTHONPATH by sourcing the bash-script `env.sh`:
```bash
source env.sh
```

- Optionally, for convienience add the absolute path of the sub directory `rspt2spectra/scripts` to the `PATH` environment variable. This enables the Python scripts to be found, without having to specify the absolute path to the scripts. If this is desired, add the following to the `~/.bashrc`:
 ```bash
 export PATH=$PATH:path/to/folder/rspt2spectra/scripts
 ```

## Usage 
- Move to the directory with the RSPt simulation of interest.
- Use Python scripts in the `scripts` folder to parse and analyze the RSPt simulation output data.
- Note that values generated by RSPt are in Rydberg but output from the scripts in this software are in eV.
- Obtained results can easily be passed on and used by the impurityModel software. 

Below follows steps for how to extract different parameters from an RSPt simulation.
In the impurityModel software there are a few example scripts. 
Please copy one of these scripts and replace the parameter values inside with the ones obtained using this software.

### 1) Choose "convenient" impurity orbitals in RSPt
For the hybridization and the on-site energies of the impurity, the choice of the impurity orbitals matter. 
#### 1a) Diagonal basis
This section is only useful if the system has high symmetry, e.g. Oh symmetry. For a system with low symmetry, see next section.
For a system of high symmetry, there will be a basis for the impurity orbitals which diagonalizes both the local non-interacting local Hamiltonian as well as the hybridization function.
Using such a basis simplifies the discretization of the hybridzation functions.
- For an atom with cubic environment (Oh symmetry), the orbitals of choice are the cubic harmonics. 
Cubic harmonics are selected in RSPt by using the basis index `3` in the `green.inp`-file.

- For an atom with high symmtry, but not Oh symmetry, one can choose the impurity basis by diagonalizing the non-interacting local Hamiltonina. This can be done by using RSPt's projection files. 
If one does not know which orbitals will generate a diagonal non-interacting local Hamiltonian, 
one can start with a spherical harmonics basis (basis index `0`) in `green.inp` and run one iteration with `rspt`. 
Then execute the Python script from this software:
```bash
diagonalize_local_H0.py 
```
If the RSPt calculations are spin-polarized, type instead:
```bash
diagonalize_local_H0.py spinpol
```
This will generate a projection file for each cluster in the `green.inp`-file.
Now add a new cluster in `green.inp` which will read the projection-file for the orbitals you want to rotate. 
For example, if the d-orbitals in atom type 1 is of interest, the new cluster may look like this:
```bash
cluster
1 0 Irr1    ! ntot udef [nsites]
1 2 1 1 0   ! t l e site basis, U J or F0 F2 F4 (F6)
```

#### 1b) Spherical harmonics
RSPt's default basis is spherical harmonics and for low symmetry cases this is the basis of choice to use, from a convinient/lazyness point of view.


### 2) Generate PDOS from RSPt
To generate projected density of states (PDOS) and hybridization functions from RSPt, add the block
```bash
spectrum
Dos Hyb
```
to the `green.inp`-file and run `rspt` once. 

If one selected a diagonal basis in the step before, in the `out`-file, the cluster should have a diagonal local non-interacting Hamiltonian (grep for `Local hamiltonian`).


### 3) Discretize hybridization functions and generate non-interacting Hamiltonian

The are two scripts that can discretize the hybrdization functions. 
The script `finiteH0.py` can handle general cases of systems also with off-diagonal hybridization functions.
The script `finiteH0_diag.py` assumes the studied system do not have off-diagonal onsite energies (and off-diagonal hybridization functions).
The two scripts use different input data (see in the `input_settings` folder).

The only good reasons to use the `finiteH0_diag.py` script is that is uses a better algorithm for determining the strength of the hopping parameters.
This means that one can get a reasonable hybridization discretization with a few bath states, if using the `finiteH0_diag.py` script.
But if one uses the `finiteH0.py` script, more bath states are at the moment needed, but the algorithm here can easily be improved.

Both scripts will read the local Hamiltonian and the hybridization function and print the non-interacting Hamiltonian to disk.
Move the created file to an empty simulation folder.
Use an impurityModel script which reads this file, and which also contains the other relevant information (e.g. Slater-Condon parameters). How to extract Slater-Condon parameters and SOC-values from RSPt are described below in this README file. 
Finally, execute your impurityModel script. E.g. if the script is called `Ni_NiO_30bath.py`, type:
```bash
Ni_NiO_30bath.py
```
After this, you should have the spectra you wanted.

#### 3a) `finiteH0.py` 
Copy the input file `rspt2spectra_parameters.py` from the `input_settings` folder to the simulation folder.
Edit it to your purpose, e.g. select which cluster to study (with the `basis_tag` variable).
Execute the script `finiteH0.py`:
```bash
finiteH0.py
```

#### 3b) `finiteH0_diag.py` 
Copy the input file `rspt2spectra_parameters_diag.py` from the `input_settings` folder to the simulation folder.
Edit it to your purpose, e.g. select which cluster to study (with the `basis_tag` variable).
Execute the script `finiteH0_diag.py`:
```bash
finiteH0_diag.py
```

### 4) Extract Slater-Condon integrals from RSPt
In case the some of the correlated orbitals of interest are treated as core states in RSPt, 
first create a subdirectory in the simulation folder, containing a copy of the RSPt simulation directory.
In the subdiretory, move the core states into the valence energy set by modifiying the `data`-file.
Then, to make RSPt print Slater-Condon integrals, add a cluster in the `green.inp` containing the correlated orbitals of interest, e.g.:
```bash
# 3d and 2p of type 1
cluster
2 1 2 eV
1 2 1 1 0 -1.0 
1 1 3 1 0 -1.0
0 0 0.30
```
Also add the following to the `green.inp`:
```bash
verbose
Umatrix

debug
Noscreening
```
In the subdirectory, the correlated orbitals of interest should now be in RSPt's valence basis. 
Now run RSPt only once. 
The Slater-Condon intergral values are printed to the `out`-file and can be found by grepping for `Slater parameters`. 
Convert them to eV and insert them in your impurityModel script.


### 5) Extract SOC parameters from RSPt
The SOC parameters are printed in the RSPt generated `out`-file.

For orbitals which are treated as core states in RSPt, the orbital energy difference is found by grepping for the keyword `core states`.
From this difference we can extract the SOC parameter. E.g. for 2p orbitals, the energy difference is equal to 3/2 times the SOC parameter.
Convert the value to eV and insert it in your impurityModel script.

For orbitals which are treated as valence states in RSPt, the SOC parameter can be extracted if the `data`-file has the keyword `f-rel` set to `t` (true). 
If `f-rel` is set to `f` (false) in your simulation folder, create a subdirectory, copy the simulations files there, and in the subdirectory change the `f-rel` flag to `t`.

Also a `green.inp` is needed with a cluster of the orbitals of interest. 
The non-interacting local Hamiltonian is found by grepping for the keyword `Local hamiltonian`.
The SOC parameter is equal to the (2,6)-element if spherical harmonics are used.
Again, convert the value to eV and insert it in your impurityModel script.



### Documentation
The documentation of this package is found in the directory `docs`.

To update the manual, go to directory `docs` and simply type:

```
make html
```
to generate a html-page.
To instead generate a pdf-file, type:
```
make latex
```
and follow the instructions.

Note:
- package `numpydoc` is required. If missing, e.g. type `conda install numpydoc` 
- If a new module or subpackage is created, this information needs to be added to `docs/index.rst`. 

