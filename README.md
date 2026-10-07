# NNMF analysis

Python and Jupyter workflow for preprocessing spectroscopy measurements and fitting
shared component shapes with nonnegative matrix factorization (NNMF).

## Files

- `Scripts/nnmf_full_fixed_jh_V3.py` contains the analysis functions and setup window.
- `Scripts/nnmf_full_fixed_jh_V3_schrenkv.ipynb` guides the user through the analysis.

Measurement data, reference workbooks, and generated exports are not included in this
repository. Keep them locally and select the appropriate data folder in the setup
window. If `INIT_COMP` is greater than zero, select a reference workbook containing a
wavelength column followed by the reference spectra. Set `INIT_COMP` to zero to use
random initialization without a workbook.

## Setup

Use Python 3.9 or newer. From the repository root, create a virtual environment
and install the requirements:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

To run the Python script:

```powershell
.\.venv\Scripts\python.exe Scripts\nnmf_full_fixed_jh_V3.py
```

To run the notebook from Jupyter:

```powershell
.\.venv\Scripts\python.exe -m notebook
```

Then open `Scripts/nnmf_full_fixed_jh_V3_schrenkv.ipynb` and select the `.venv`
Python kernel if prompted. In VS Code, open the notebook and select the
repository's `.venv` as its kernel.

The script and notebook use the same analysis code; keep both files together in
the `Scripts` folder. The script opens the analysis setup window directly, while
the notebook provides a cell-by-cell workflow. Measurement data and reference
workbooks are not included in the repository; select your own files in the setup.
