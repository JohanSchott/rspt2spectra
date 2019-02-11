# rspt to spectra

Interface from the RSPt software to the [impurityModel](https://github.com/JohanSchott/impurityModel) spectra repository.

This repository contains a few Python modules that can be useful for parsing and analyzing RSPt output, and constructing a crystal-field Hamiltonian including bath orbitals.
A few examples scripts are stored in the `scripts` folder.

Separately, in the module plotQuanty, functions for reading and plotting output from the Quanty software is provided.

## Usage 
### Preparations
- Add the path of the folder to the `PYTHONPATH` variable:
```bash
export PYTHONPATH=${PYTHONPATH}:path/to/folder/rspt2spectra
```

- Optionally, for convienience add the absolute path of the sub directories `rspt2spectra/scripts` to the `PATH` environment variable. This enables the Python scripts to be found, without having to specify the absolute path to the scripts. If this is desired, add the following to the `~/.bashrc`:
 ```bash
 export PATH=$PATH:path/to/folder/rspt2spectra/scripts
 ```

### Example usage
- Move to the directory with the RSPt simulation of interest.
- Use Python scripts in the `scripts` folder to parse and analyze the RSPt simulation output data.

#### Choose convenient orbitals

#### Discretize hybridization functions

#### Adjust on-site energies

#### Slater-Condon integrals

#### SOC parameters

#### Extract Hamiltonian or Hamiltonian parameters
- Parameters: Run or modify script `getHamiltonianParameters.py`.
- Hamiltonian: Run or modify script `getHamiltonian.py`.

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

