# NNMF analysis

Python and Jupyter workflow for preprocessing spectroscopy measurements and fitting
shared component shapes with nonnegative matrix factorization (NNMF).

## Files

- `MATLAB/nnmf_full_fixed_jh_V3.py` contains the analysis functions and setup window.
- `MATLAB/nnmf_full_fixed_jh_V3_schrenkv.ipynb` guides the user through the analysis.

Measurement data, reference workbooks, and generated exports are not included in this
repository. Keep them locally and select the appropriate data folder in the setup
window. If `INIT_COMP` is greater than zero, select a reference workbook containing a
wavelength column followed by the reference spectra. Set `INIT_COMP` to zero to use
random initialization without a workbook.

## Setup

Use Python 3.9 or newer, then install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

Open `MATLAB/nnmf_full_fixed_jh_V3_schrenkv.ipynb` in Jupyter or VS Code and run
the cells in order. The setup window lets you choose the working folder, data
subfolders, and analysis parameters. Do not run the Python module directly if you
want to use the notebook's visible, cell-by-cell workflow.

The notebook uses the data-processing and plotting functions from
`nnmf_full_fixed_jh_V3.py`; keep both files together in the `MATLAB` folder.
