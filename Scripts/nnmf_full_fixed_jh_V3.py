"""Python transcription of Matlab script nnmf_full_fixed_jh_V3.m
Install dependencies with: pip install numpy pandas scipy scikit-learn matplotlib openpyxl
When run as a script, popup dialogs ask for the folder containing the configured
data subfolders and the reference spectra workbook. Input spectra are selected
as .txt or .csv files with X and Y columns. The notebook can pass its settings
through configure_analysis() before calling the helper functions.
"""
from __future__ import annotations
import json
from itertools import combinations
import math
import shutil
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib.figure import Figure
from scipy.integrate import trapezoid
from scipy.optimize import least_squares, nnls
from scipy.signal import find_peaks
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning

# User configuration
DATA_FOLDER = Path.cwd()  # Used only as the starting location for the folder picker.
LIST_OF_DATA_FOLDERS: list[str] = []
INPUT_FORMAT = "txt"
TRAPZ_DATA_FOLDERS = [""]
ARIT_DATA_FOLDERS: list[str] = []

# Reference workbook component columns, in order after the wavelength column.
COMPONENT_NAMES = ["NV0", "NV-", "H3", "laser"]
COMPONENTS = 2
DEF_SPECTRA = True # Initialize the NNMF using reference spectra ref.xlsx
USE_GAUSSIAN_REFERENCE = False # If true and INIT_COMP == 0, fit Gaussian reference curves and use them for initialization.
INIT_COMP = COMPONENTS # Number of reference spectra to use for initialization (must be <= COMPONENTS)

# Each source spectrum has its own mean in this range subtracted before folder averaging.
BASEMEAN_LOWER = 300
BASEMEAN_UPPER = 510

# Wavelength range for the NNMF fit; the spectra are cropped to this range before fitting
THRESHOLD_LOWER = 377
THRESHOLD_UPPER = 920

# Component plotting colors are sampled from the prism colormap.
COLOURS = list(plt.get_cmap("prism")(np.linspace(0.0, 1.0, 4, endpoint=False)))
NV0 = 575 
NVN = 637 #NV minus
SHOW_COMPARED = True
CHANGE_NAME = True
OWN_AXIS = False
X_AXIS_FILE = "X.txt"
REFERENCE_FILE = None  # Selected by the user when the analysis starts.
EXPORT_FOLDER = "export_pokus"
FIGURE_FOLDER = "figures_pokus"
UPDATE_FILES = True # If TRUE, existing files in the all_spectra folder are deleted before copying new spectra
NNMF_REPLICATES = 20 # Run NNMF X times
MAX_ITERATIONS = 100 # Each fit up to 100 iterations
RANDOM_SEED = 21 # NNMF can start with different random guesses and may find different answers; random_state=RANDOM_SEED + replicate
NNMF_NEGATIVE_TOLERANCE = 1e-5 # Values in [-tolerance, 0) are treated as numerical noise.

ANALYSIS_SETTINGS = {
    "DATA_FOLDER",
    "LIST_OF_DATA_FOLDERS",
    "INPUT_FORMAT",
    "TRAPZ_DATA_FOLDERS",
    "ARIT_DATA_FOLDERS",
    "COMPONENT_NAMES",
    "COMPONENTS",
    "DEF_SPECTRA",
    "USE_GAUSSIAN_REFERENCE",
    "INIT_COMP",
    "BASEMEAN_LOWER",
    "BASEMEAN_UPPER",
    "THRESHOLD_LOWER",
    "THRESHOLD_UPPER",
    "COLOURS",
    "NV0",
    "NVN",
    "SHOW_COMPARED",
    "CHANGE_NAME",
    "OWN_AXIS",
    "X_AXIS_FILE",
    "REFERENCE_FILE",
    "EXPORT_FOLDER",
    "FIGURE_FOLDER",
    "UPDATE_FILES",
    "NNMF_REPLICATES",
    "MAX_ITERATIONS",
    "RANDOM_SEED",
    "NNMF_NEGATIVE_TOLERANCE",
}


def configure_analysis(settings: dict[str, object]) -> None:
    """Apply settings supplied by the notebook to this helper module."""
    unknown = settings.keys() - ANALYSIS_SETTINGS
    if unknown:
        raise ValueError(f"Unknown analysis settings: {', '.join(sorted(unknown))}")
    if "INPUT_FORMAT" in settings:
        _validate_input_format(settings["INPUT_FORMAT"])
    globals().update(settings)


def display_component_names() -> list[str]:
    """Return short labels for component plots and contribution summaries."""
    display_names = {"NV0": "C1", "NV-": "C2", "H3": "C3"}
    return [display_names.get(name, name) for name in COMPONENT_NAMES]


def prism_colours(count: int) -> list[tuple[float, float, float, float]]:
    if count < 0:
        raise ValueError("Color count cannot be negative")
    if count == 0:
        return []
    return list(
        plt.get_cmap("prism")(np.linspace(0.0, 1.0, count, endpoint=False))
    )


def pastel_colours(count: int) -> tuple[tuple[float, float, float], ...]:
    palette = plt.get_cmap("Pastel1").colors
    if count < 0 or count > len(palette):
        raise ValueError(f"Pastel1 supports between zero and {len(palette)} component colors")
    return palette[:count]


def _validate_input_format(value: object) -> str:
    if not isinstance(value, str) or value.lower() not in {"txt", "csv"}:
        raise ValueError("Input format must be 'txt' or 'csv'")
    return value.lower()


def _list_spectrum_files(folder: Path, input_format: str | None = None) -> list[Path]:
    extension = f".{_validate_input_format(input_format or INPUT_FORMAT)}"
    return sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() == extension
    )


PARAMETER_FILE = ".nnmf_settings.json"
PARAMETER_FIELDS = {
    "COMPONENTS": int,
    "INIT_COMP": int,
    "BASEMEAN_LOWER": float,
    "BASEMEAN_UPPER": float,
    "THRESHOLD_LOWER": float,
    "THRESHOLD_UPPER": float,
    "NNMF_REPLICATES": int,
    "MAX_ITERATIONS": int,
    "RANDOM_SEED": int,
    "NNMF_NEGATIVE_TOLERANCE": float,
    "NV0": float,
    "NVN": float,
    "DEF_SPECTRA": bool,
    "USE_GAUSSIAN_REFERENCE": bool,
    "SHOW_COMPARED": bool,
}


def _validate_analysis_parameters(parameters: dict[str, object]) -> dict[str, object]:
    """Validate and normalize values editable in the parameter dialog."""
    legacy_output_fields = {"THRESHOLD_LOWER2", "THRESHOLD_UPPER2"}
    unknown = parameters.keys() - PARAMETER_FIELDS.keys() - legacy_output_fields
    if unknown:
        raise ValueError(f"Unknown saved analysis parameters: {', '.join(sorted(unknown))}")

    validated: dict[str, object] = {}
    for name, expected_type in PARAMETER_FIELDS.items():
        if name not in parameters:
            continue
        value = parameters[name]
        if expected_type is bool:
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be true or false")
        elif expected_type is int:
            try:
                is_whole_number = not isinstance(value, bool) and int(value) == float(value)
            except (TypeError, ValueError, OverflowError):
                is_whole_number = False
            if not is_whole_number:
                raise ValueError(f"{name} must be a whole number")
            value = int(value)
        else:
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        validated[name] = value

    components = int(validated.get("COMPONENTS", COMPONENTS))
    init_comp = int(validated.get("INIT_COMP", INIT_COMP))
    if validated.get("DEF_SPECTRA") is False:
        init_comp = 0
    if not 1 <= components <= len(COMPONENT_NAMES):
        raise ValueError(f"COMPONENTS must be between 1 and {len(COMPONENT_NAMES)}")
    if not 0 <= init_comp <= components:
        raise ValueError("INIT_COMP must be between zero and COMPONENTS")
    validated["INIT_COMP"] = init_comp
    validated["DEF_SPECTRA"] = init_comp > 0
    if int(validated.get("NNMF_REPLICATES", NNMF_REPLICATES)) < 1:
        raise ValueError("NNMF_REPLICATES must be at least 1")
    if int(validated.get("MAX_ITERATIONS", MAX_ITERATIONS)) < 1:
        raise ValueError("MAX_ITERATIONS must be at least 1")
    if float(validated.get("NNMF_NEGATIVE_TOLERANCE", NNMF_NEGATIVE_TOLERANCE)) < 0:
        raise ValueError("NNMF_NEGATIVE_TOLERANCE cannot be negative")

    baseline_lower = float(validated.get("BASEMEAN_LOWER", BASEMEAN_LOWER))
    baseline_upper = float(validated.get("BASEMEAN_UPPER", BASEMEAN_UPPER))
    fit_lower = float(validated.get("THRESHOLD_LOWER", THRESHOLD_LOWER))
    fit_upper = float(validated.get("THRESHOLD_UPPER", THRESHOLD_UPPER))
    if baseline_lower >= baseline_upper:
        raise ValueError("Baseline lower wavelength must be less than the upper wavelength")
    if fit_lower >= fit_upper:
        raise ValueError("Fit lower wavelength must be less than the upper wavelength")
    return validated


def edit_analysis_parameters(data_folder: Path,
                             defaults: dict[str, object]) -> dict[str, object]:
    """Show editable settings loaded from the selected folder's latest saved run."""
    defaults = _validate_analysis_parameters(defaults)
    settings_path = data_folder / PARAMETER_FILE
    parameters = defaults.copy()
    if settings_path.exists():
        try:
            saved = json.loads(settings_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Cannot load saved settings from {settings_path}: {error}") from error
        if not isinstance(saved, dict):
            raise ValueError(f"Saved settings in {settings_path} must be a JSON object")
        raw_parameters = saved.get("parameters", saved)
        if not isinstance(raw_parameters, dict):
            raise ValueError(f"Saved parameters in {settings_path} must be a JSON object")
        parameters.update(_validate_analysis_parameters(raw_parameters))

    import tkinter as tk
    from tkinter import messagebox, ttk

    root = tk.Tk()
    root.title("NNMF analysis parameters")
    root.resizable(False, False)
    root.attributes("-topmost", True)

    entries: dict[str, tk.Entry] = {}
    booleans: dict[str, tk.BooleanVar] = {}
    fields = [
        ("COMPONENTS", "Number of components"),
        ("INIT_COMP", "Reference components for initialization"),
        ("BASEMEAN_LOWER", "Baseline lower wavelength (nm)"),
        ("BASEMEAN_UPPER", "Baseline upper wavelength (nm)"),
        ("THRESHOLD_LOWER", "Fit lower wavelength (nm)"),
        ("THRESHOLD_UPPER", "Fit upper wavelength (nm)"),
        ("NNMF_REPLICATES", "NNMF replicates"),
        ("MAX_ITERATIONS", "Maximum iterations per replicate"),
        ("RANDOM_SEED", "Random seed"),
        ("NNMF_NEGATIVE_TOLERANCE", "Negative-value tolerance"),
        ("NV0", "Plot marker 1 (nm)"),
        ("NVN", "Plot marker 2 (nm)"),
    ]
    for row, (name, label) in enumerate(fields):
        ttk.Label(root, text=label).grid(row=row, column=0, sticky="w", padx=10, pady=3)
        entry = ttk.Entry(root, width=18)
        entry.insert(0, str(parameters[name]))
        entry.grid(row=row, column=1, sticky="ew", padx=10, pady=3)
        entries[name] = entry

    parameters.setdefault("USE_GAUSSIAN_REFERENCE", USE_GAUSSIAN_REFERENCE)
    bool_row = len(fields)
    for offset, (name, label) in enumerate((
        ("USE_GAUSSIAN_REFERENCE", "Fit Gaussian reference curves when INIT_COMP = 0"),
        ("SHOW_COMPARED", "Keep plots open after saving"),
    )):
        variable = tk.BooleanVar(value=bool(parameters[name]))
        ttk.Checkbutton(root, text=label, variable=variable).grid(
            row=bool_row + offset, column=0, columnspan=2, sticky="w", padx=10, pady=3
        )
        booleans[name] = variable

    ttk.Label(
        root,
        text="Folder averages remain arithmetic; individual spectra use area normalization.",
        wraplength=390,
    ).grid(row=bool_row + 2, column=0, columnspan=2, sticky="w", padx=10, pady=(8, 4))

    result: dict[str, object] = {}

    def save_and_close() -> None:
        try:
            candidate: dict[str, object] = {
                name: entry.get().strip() for name, entry in entries.items()
            }
            candidate.update({name: variable.get() for name, variable in booleans.items()})
            result.update(_validate_analysis_parameters(candidate))
            temporary_path = settings_path.with_name(settings_path.name + ".tmp")
            temporary_path.write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8"
            )
            temporary_path.replace(settings_path)
        except (OSError, ValueError, TypeError, OverflowError) as error:
            messagebox.showerror("Invalid NNMF parameters", str(error), parent=root)
            return
        root.destroy()

    def cancel() -> None:
        root.destroy()

    buttons = ttk.Frame(root)
    buttons.grid(row=bool_row + 3, column=0, columnspan=2, sticky="e", padx=10, pady=10)
    ttk.Button(buttons, text="Cancel", command=cancel).pack(side="right", padx=(6, 0))
    ttk.Button(buttons, text="Use these parameters", command=save_and_close).pack(side="right")
    root.protocol("WM_DELETE_WINDOW", cancel)
    root.mainloop()

    if not result:
        raise RuntimeError("NNMF parameter selection was cancelled.")
    return result


def _validate_reference_file(init_comp: int, reference_text: str) -> Path | None:
    if init_comp == 0:
        return None
    reference_file = Path(reference_text).expanduser() if reference_text else None
    if reference_file is None or not reference_file.is_file():
        raise ValueError(
            "Choose an existing reference workbook when reference components "
            "for initialization is greater than 0."
        )
    if reference_file.suffix.lower() not in {".xls", ".xlsx"}:
        raise ValueError("Reference workbook must be an .xls or .xlsx file.")
    return reference_file


def select_analysis_setup(default_data_folder: Path,
                          default_reference_file: Path | None,
                          component_names: list[str],
                          defaults: dict[str, object]) -> dict[str, object]:
    """Choose data, reference, folders, and fit settings in one saved setup window."""
    global COMPONENT_NAMES
    COMPONENT_NAMES = component_names.copy()
    defaults = _validate_analysis_parameters(defaults)

    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.widgets import SpanSelector

    root = tk.Tk()
    root.title("NNMF analysis setup")
    root.geometry("1200x850")
    root.minsize(1000, 700)
    root.attributes("-topmost", True)

    data_folder_var = tk.StringVar(value=str(default_data_folder))
    input_format_var = tk.StringVar(value=_validate_input_format(INPUT_FORMAT))
    reference_var = tk.StringVar(
        value="" if default_reference_file is None else str(default_reference_file)
    )
    entries: dict[str, tk.Entry] = {}
    boolean_vars: dict[str, tk.BooleanVar] = {}
    boolean_controls: dict[str, ttk.Checkbutton] = {}
    folder_names: list[str] = []
    selected_parameters = defaults.copy()
    selected_parameters.setdefault("USE_GAUSSIAN_REFERENCE", USE_GAUSSIAN_REFERENCE)
    setup: dict[str, object] = {}

    ttk.Label(root, text="Data folder").grid(row=0, column=0, sticky="w", padx=10, pady=(10, 3))
    ttk.Entry(root, textvariable=data_folder_var).grid(
        row=1, column=0, sticky="ew", padx=10, pady=(0, 6)
    )
    ttk.Button(root, text="Browse...", command=lambda: browse_data_folder()).grid(
        row=1, column=1, sticky="ew", padx=(0, 10), pady=(0, 6)
    )
    ttk.Label(root, text="Reference workbook (needed only if INIT_COMP > 0)").grid(
        row=2, column=0, sticky="w", padx=10, pady=3
    )
    reference_entry = ttk.Entry(root, textvariable=reference_var)
    reference_entry.grid(
        row=3, column=0, sticky="ew", padx=10, pady=(0, 10)
    )
    reference_button = ttk.Button(root, text="Browse...", command=lambda: browse_reference())
    reference_button.grid(
        row=3, column=1, sticky="ew", padx=(0, 10), pady=(0, 10)
    )

    content = ttk.Frame(root)
    content.grid(row=4, column=0, columnspan=2, sticky="nsew", padx=10, pady=4)
    content.columnconfigure(0, weight=1)
    content.columnconfigure(1, weight=2)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(5, weight=1)

    folder_frame = ttk.LabelFrame(content, text="Data subfolders to include")
    folder_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
    folder_frame.rowconfigure(1, weight=1)
    folder_frame.columnconfigure(0, weight=1)
    ttk.Label(folder_frame, text="Input file format").grid(
        row=0, column=0, sticky="w", padx=8, pady=(8, 2)
    )
    ttk.Combobox(
        folder_frame, textvariable=input_format_var, values=("txt", "csv"),
        state="readonly", width=10,
    ).grid(row=0, column=1, sticky="ew", padx=8, pady=(8, 2))
    folder_list = tk.Listbox(folder_frame, selectmode=tk.EXTENDED, exportselection=False)
    folder_list.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)
    folder_scroll = ttk.Scrollbar(folder_frame, orient="vertical", command=folder_list.yview)
    folder_scroll.grid(row=1, column=1, sticky="ns", pady=8)
    folder_list.configure(yscrollcommand=folder_scroll.set)
    ttk.Label(folder_frame, text="Use Ctrl/Shift-click to select multiple folders.").grid(
        row=2, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 8)
    )

    parameter_frame = ttk.LabelFrame(content, text="Fit and preprocessing parameters")
    parameter_frame.grid(row=0, column=1, sticky="nsew")
    parameter_frame.columnconfigure(1, weight=1)
    fields = [
        ("COMPONENTS", "Number of components"),
        ("INIT_COMP", "Reference components for initialization"),
        ("BASEMEAN_LOWER", "Baseline lower wavelength (nm)"),
        ("BASEMEAN_UPPER", "Baseline upper wavelength (nm)"),
        ("THRESHOLD_LOWER", "Fit lower wavelength (nm)"),
        ("THRESHOLD_UPPER", "Fit upper wavelength (nm)"),
        ("NNMF_REPLICATES", "NNMF replicates"),
        ("MAX_ITERATIONS", "Maximum iterations per replicate"),
        ("RANDOM_SEED", "Random seed"),
        ("NNMF_NEGATIVE_TOLERANCE", "Negative-value tolerance"),
        ("NV0", "Plot marker 1 (nm)"),
        ("NVN", "Plot marker 2 (nm)"),
    ]
    for row, (name, label) in enumerate(fields):
        ttk.Label(parameter_frame, text=label).grid(
            row=row, column=0, sticky="w", padx=8, pady=2
        )
        entry = ttk.Entry(parameter_frame, width=16)
        entry.grid(row=row, column=1, sticky="ew", padx=8, pady=2)
        entries[name] = entry

    def update_reference_controls(_event: object = None) -> None:
        try:
            reference_needed = int(entries["INIT_COMP"].get()) > 0
        except ValueError:
            reference_needed = True
        state = "normal" if reference_needed else "disabled"
        reference_entry.configure(state=state)
        reference_button.configure(state=state)
        if "USE_GAUSSIAN_REFERENCE" in boolean_controls:
            boolean_controls["USE_GAUSSIAN_REFERENCE"].configure(
                state="disabled" if reference_needed else "normal"
            )

    entries["INIT_COMP"].bind("<KeyRelease>", update_reference_controls)
    selected_parameters.setdefault("USE_GAUSSIAN_REFERENCE", USE_GAUSSIAN_REFERENCE)
    bool_row = len(fields)
    for offset, (name, label) in enumerate((
        ("USE_GAUSSIAN_REFERENCE", "Fit Gaussian reference curves when INIT_COMP = 0"),
        ("SHOW_COMPARED", "Keep plots open after saving"),
    )):
        variable = tk.BooleanVar(value=bool(selected_parameters[name]))
        control = ttk.Checkbutton(parameter_frame, text=label, variable=variable)
        control.grid(
            row=bool_row + offset, column=0, columnspan=2, sticky="w", padx=8, pady=4
        )
        boolean_vars[name] = variable
        boolean_controls[name] = control

    preview_frame = ttk.LabelFrame(
        root, text="Raw spectra preview - drag to select a wavelength range"
    )
    preview_frame.grid(row=5, column=0, columnspan=2, sticky="nsew", padx=10, pady=4)
    preview_frame.columnconfigure(0, weight=1)
    preview_frame.rowconfigure(1, weight=1)
    preview_controls = ttk.Frame(preview_frame)
    preview_controls.grid(row=0, column=0, sticky="ew", padx=8, pady=4)
    selection_mode_var = tk.StringVar(value="baseline")
    ttk.Label(preview_controls, text="Mouse drag selects:").pack(side="left", padx=(0, 8))
    ttk.Radiobutton(
        preview_controls, text="Baseline band", variable=selection_mode_var,
        value="baseline",
    ).pack(side="left")
    ttk.Radiobutton(
        preview_controls, text="Fit / output range", variable=selection_mode_var,
        value="fit",
    ).pack(side="left", padx=(8, 16))
    preview_status_var = tk.StringVar(value="Select data folders to preview their spectra.")
    ttk.Label(preview_controls, textvariable=preview_status_var).pack(side="left")

    preview_figure = Figure(figsize=(10, 3.5))
    preview_axis = preview_figure.add_subplot(111)
    preview_canvas = FigureCanvasTkAgg(preview_figure, master=preview_frame)
    preview_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=6, pady=4)
    preview_range_artists: list[object] = []

    def update_preview_ranges() -> None:
        x_limits = preview_axis.get_xlim()
        for artist in preview_range_artists:
            artist.remove()
        preview_range_artists.clear()
        for lower_name, upper_name, label, color in (
            ("BASEMEAN_LOWER", "BASEMEAN_UPPER", "Baseline band", 0.08),
            ("THRESHOLD_LOWER", "THRESHOLD_UPPER", "Fit / output", 0.62),
        ):
            try:
                lower = float(entries[lower_name].get())
                upper = float(entries[upper_name].get())
            except ValueError:
                continue
            if lower < upper:
                artist = preview_axis.axvspan(
                    lower, upper, color=plt.get_cmap("prism")(color), alpha=0.16,
                    label=label,
                )
                preview_range_artists.append(artist)
        preview_axis.set_xlim(x_limits)
        preview_canvas.draw_idle()

    def refresh_preview(*_callback_args: object) -> None:
        preview_range_artists.clear()
        preview_axis.clear()
        selected_indices = folder_list.curselection()
        selected_names = [folder_names[int(index)] for index in selected_indices]
        if not selected_names:
            preview_status_var.set("Select one or more data folders to preview spectra.")
            preview_axis.text(0.5, 0.5, "No data folders selected",
                              ha="center", va="center", transform=preview_axis.transAxes)
            preview_canvas.draw_idle()
            return

        data_folder = Path(data_folder_var.get()).expanduser()
        input_format = _validate_input_format(input_format_var.get())
        source_files = [
            path
            for folder_name in selected_names
            for path in _list_spectrum_files(data_folder / folder_name, input_format)
        ]
        if not source_files:
            preview_status_var.set(f"No .{input_format} spectra found in selected folders.")
            preview_axis.text(0.5, 0.5, "No matching spectra found",
                              ha="center", va="center", transform=preview_axis.transAxes)
            preview_canvas.draw_idle()
            return

        count = min(10, len(source_files))
        random_seed = RANDOM_SEED
        try:
            random_seed = int(entries["RANDOM_SEED"].get())
        except ValueError:
            pass
        generator = np.random.default_rng(random_seed)
        chosen_indices = generator.choice(len(source_files), size=count, replace=False)
        colors = plt.get_cmap("prism")(np.linspace(0.0, 1.0, count, endpoint=False))
        try:
            for color, index in zip(colors, chosen_indices):
                path = source_files[int(index)]
                raw_spectrum = read_spectrum(path)
                relative_name = path.relative_to(data_folder).as_posix()
                preview_axis.plot(
                    raw_spectrum[:, 0], raw_spectrum[:, 1], color=color,
                    linewidth=1, label=relative_name,
                )
        except (OSError, ValueError) as error:
            preview_axis.clear()
            preview_axis.text(0.5, 0.5, f"Cannot preview spectra:\n{error}",
                              ha="center", va="center", transform=preview_axis.transAxes)
            preview_status_var.set("Preview failed; inspect the message in the plot.")
            preview_canvas.draw_idle()
            return

        preview_axis.set(
            title="Random raw spectra (not baseline-corrected or normalized)",
            xlabel="Wavelength (nm)", ylabel="Raw intensity",
        )
        preview_status_var.set(
            f"Showing {count} of {len(source_files)} spectra. "
            "Choose a range type, then drag across the plot."
        )
        update_preview_ranges()
        preview_axis.legend(fontsize="x-small", loc="best", ncols=2)
        preview_canvas.draw_idle()

    def select_preview_range(lower: float, upper: float) -> None:
        if not lower < upper:
            return
        if selection_mode_var.get() == "baseline":
            lower_name, upper_name = "BASEMEAN_LOWER", "BASEMEAN_UPPER"
        else:
            lower_name, upper_name = "THRESHOLD_LOWER", "THRESHOLD_UPPER"
        entries[lower_name].delete(0, tk.END)
        entries[lower_name].insert(0, f"{lower:.6g}")
        entries[upper_name].delete(0, tk.END)
        entries[upper_name].insert(0, f"{upper:.6g}")
        update_preview_ranges()

    span_selector = SpanSelector(
        preview_axis, select_preview_range, "horizontal",
        useblit=True, props={"alpha": 0.2, "facecolor": "tab:blue"},
    )
    span_selector.set_active(True)
    for name in (
        "BASEMEAN_LOWER", "BASEMEAN_UPPER", "THRESHOLD_LOWER", "THRESHOLD_UPPER",
    ):
        entries[name].bind("<KeyRelease>", lambda _event: update_preview_ranges())
    entries["RANDOM_SEED"].bind("<FocusOut>", refresh_preview)
    folder_list.bind("<<ListboxSelect>>", refresh_preview)
    input_format_var.trace_add("write", refresh_preview)

    ttk.Label(
        root,
        text=(
            "Normalization follows MATLAB: subtract each spectrum's baseline, scale it "
            "at NV0, crop to the selected fit/output range, then divide by its own trapezoidal area. "
            "Folder averages are arithmetic and are not used for NNMF. Set reference "
            "components to 0 for random-only initialization; use the Gaussian-reference option to "
            "fit one curve per configured component and use those as NNMF starting spectra."
        ),
        wraplength=940,
        justify="left",
    ).grid(row=6, column=0, columnspan=2, sticky="ew", padx=12, pady=8)

    def load_folder_settings(folder: Path) -> None:
        nonlocal folder_names, selected_parameters
        if not folder.is_dir():
            folder_names = []
            folder_list.delete(0, tk.END)
            return
        settings_path = folder / PARAMETER_FILE
        saved: dict[str, object] = {}
        if settings_path.exists():
            try:
                decoded = json.loads(settings_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(f"Cannot load saved settings from {settings_path}: {error}") from error
            if not isinstance(decoded, dict):
                raise ValueError(f"Saved settings in {settings_path} must be a JSON object")
            saved = decoded
        raw_parameters = saved.get("parameters", saved)
        if not isinstance(raw_parameters, dict):
            raise ValueError(f"Saved parameters in {settings_path} must be a JSON object")
        selected_parameters = defaults.copy()
        selected_parameters.update(_validate_analysis_parameters(raw_parameters))
        selected_parameters.setdefault("USE_GAUSSIAN_REFERENCE", USE_GAUSSIAN_REFERENCE)
        input_format_var.set(_validate_input_format(saved.get("input_format", INPUT_FORMAT)))
        for name, entry in entries.items():
            entry.delete(0, tk.END)
            entry.insert(0, str(selected_parameters[name]))
        update_reference_controls()
        for name, variable in boolean_vars.items():
            variable.set(bool(selected_parameters[name]))

        saved_reference = saved.get("reference_file")
        if saved_reference:
            reference_path = Path(str(saved_reference))
            if not reference_path.is_absolute():
                reference_path = folder / reference_path
            reference_var.set(str(reference_path))
        else:
            reference_var.set("")

        all_folders = sorted(
            child.name for child in folder.iterdir()
            if child.is_dir() and child.name != EXPORT_FOLDER
        )
        saved_folders = saved.get("data_folders", all_folders)
        if not isinstance(saved_folders, list) or any(
            not isinstance(name, str) for name in saved_folders
        ):
            raise ValueError(f"Saved data_folders in {settings_path} must be a list of folder names")
        selected_names = set(saved_folders)
        folder_names = all_folders
        folder_list.delete(0, tk.END)
        for index, name in enumerate(folder_names):
            folder_list.insert(tk.END, name)
            if name in selected_names:
                folder_list.selection_set(index)
        refresh_preview()

    def browse_data_folder() -> None:
        start = Path(data_folder_var.get()).expanduser()
        chosen = filedialog.askdirectory(
            parent=root, title="Choose the working folder containing data subfolders",
            initialdir=str(start if start.is_dir() else Path.cwd()),
        )
        if chosen:
            data_folder_var.set(chosen)
            try:
                load_folder_settings(Path(chosen))
            except (OSError, ValueError) as error:
                messagebox.showerror("Cannot load folder settings", str(error), parent=root)

    def browse_reference() -> None:
        start = Path(data_folder_var.get()).expanduser()
        chosen = filedialog.askopenfilename(
            parent=root, title="Choose the reference spectra workbook",
            initialdir=str(start if start.is_dir() else Path.cwd()),
            filetypes=[("Excel workbooks", "*.xlsx *.xls"), ("All files", "*.*")],
        )
        if chosen:
            reference_var.set(chosen)

    def accept() -> None:
        try:
            data_folder = Path(data_folder_var.get()).expanduser()
            if not data_folder.is_dir():
                raise ValueError("Choose an existing data folder.")
            selected_indices = folder_list.curselection()
            selected_folders = [folder_names[int(index)] for index in selected_indices]
            if not selected_folders:
                raise ValueError("Select at least one data subfolder.")
            candidate: dict[str, object] = {
                name: entry.get().strip() for name, entry in entries.items()
            }
            candidate.update({name: variable.get() for name, variable in boolean_vars.items()})
            parameters = _validate_analysis_parameters(candidate)
            input_format = _validate_input_format(input_format_var.get())
            reference_text = reference_var.get().strip()
            reference_file = _validate_reference_file(
                int(parameters["INIT_COMP"]), reference_text
            )
            settings_path = data_folder / PARAMETER_FILE
            saved_setup = {
                "parameters": parameters,
                "reference_file": str(reference_file) if reference_file is not None else None,
                "data_folders": selected_folders,
                "input_format": input_format,
            }
            temporary_path = settings_path.with_name(settings_path.name + ".tmp")
            temporary_path.write_text(
                json.dumps(saved_setup, indent=2) + "\n", encoding="utf-8"
            )
            temporary_path.replace(settings_path)
        except (OSError, ValueError, TypeError, OverflowError) as error:
            messagebox.showerror("Invalid NNMF setup", str(error), parent=root)
            return
        setup.update({
            "DATA_FOLDER": data_folder,
            "REFERENCE_FILE": reference_file,
            "LIST_OF_DATA_FOLDERS": selected_folders,
            "INPUT_FORMAT": input_format,
            "parameters": parameters,
        })
        root.destroy()

    try:
        initial_folder = Path(default_data_folder).expanduser()
        data_folder_var.set(str(initial_folder))
        if initial_folder.is_dir():
            load_folder_settings(initial_folder)
        elif initial_folder != Path.cwd() and Path.cwd().is_dir():
            data_folder_var.set(str(Path.cwd()))
            load_folder_settings(Path.cwd())
    except (OSError, ValueError) as error:
        root.destroy()
        raise ValueError(f"Cannot initialize NNMF setup window: {error}") from error

    update_reference_controls()
    buttons = ttk.Frame(root)
    buttons.grid(row=7, column=0, columnspan=2, sticky="e", padx=10, pady=(0, 10))
    ttk.Button(buttons, text="Cancel", command=root.destroy).pack(side="right", padx=(6, 0))
    ttk.Button(buttons, text="Start analysis", command=accept).pack(side="right")
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()
    if not setup:
        raise RuntimeError("NNMF setup was cancelled.")
    return setup


# Read the first two numeric columns of a delimited text spectrum
def read_spectrum(path: Path) -> np.ndarray:
    frame = pd.read_csv(path, sep=None, engine="python", header=None, comment="#")
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=frame.columns[:2])
    if frame.shape[1] < 2 or frame.empty:
        raise ValueError(f"Expected at least two numeric columns in {path}")
    return frame.iloc[:, :2].to_numpy(dtype=float)

# Save NumPy array to a tab separated text file
def write_spectrum(path: Path, values: np.ndarray) -> None:
    np.savetxt(path, values, delimiter="\t")


def subtract_spectrum_baseline(spectrum: np.ndarray) -> np.ndarray:
    """Subtract this spectrum's own mean intensity in the configured baseline band."""
    lower_index = int(np.argmin(np.abs(spectrum[:, 0] - BASEMEAN_LOWER)))
    upper_index = int(np.argmin(np.abs(spectrum[:, 0] - BASEMEAN_UPPER)))
    if lower_index > upper_index:
        lower_index, upper_index = upper_index, lower_index

    corrected = spectrum.copy()
    baseline = np.mean(spectrum[lower_index : upper_index + 1, 1])
    corrected[:, 1] -= baseline
    return corrected


# Create selected-folder averages and copy spectra into all_spectra.
def prepare_spectra(export_dir: Path) -> list[Path]:
    all_spectra_dir = export_dir / "all_spectra"
    all_spectra_dir.mkdir(parents=True, exist_ok=True)
    # Remove existing files in the all_spectra directory if UPDATE_FILES is True
    if UPDATE_FILES:
        for path in all_spectra_dir.iterdir():
            if path.is_file():
                path.unlink()
    # Process each data folder; skip it if empty or missing.
    for folder_name in LIST_OF_DATA_FOLDERS:
        folder = DATA_FOLDER / folder_name
        files = _list_spectrum_files(folder) if folder.is_dir() else []
        if not files:
            continue
        # Correct each source spectrum separately before averaging the folder.
        spectra = [
            subtract_spectrum_baseline(read_spectrum(path)) for path in files
        ]
        # For folders in TRAPZ_DATA_FOLDERS, normalize corrected spectra by area
        # and average those normalized spectra.
        if folder_name in TRAPZ_DATA_FOLDERS:
            reference_x = spectra[0][:, 0]
            normalized = []
            for spectrum in spectra:
                y_values = spectrum[:, 1] - np.min(spectrum[:, 1])
                area = trapezoid(y_values)
                normalized.append(y_values / area if area != 0 else y_values)
            average = np.column_stack((reference_x, np.mean(normalized, axis=0)))
            write_spectrum(
                all_spectra_dir / f"{folder_name}_trapz_avg.txt", average
            )
        # For folders in ARIT_DATA_FOLDERS, average corrected spectra arithmetically.
        if folder_name in ARIT_DATA_FOLDERS:
            reference_x = spectra[0][:, 0]
            average_y = np.mean([spectrum[:, 1] for spectrum in spectra], axis=0)
            write_spectrum(
                all_spectra_dir / f"{folder_name}_arit_avg.txt",
                np.column_stack((reference_x, average_y)),
            )
            if folder_name not in TRAPZ_DATA_FOLDERS:
                continue
        # Save corrected individual spectra when the folder is not averaged away.
        for index, (source, spectrum) in enumerate(zip(files, spectra), start=1):
            target_name = (
                f"{folder_name}_{index}.txt"
                if CHANGE_NAME
                else f"{folder_name}{source.name}"
            )
            write_spectrum(all_spectra_dir / target_name, spectrum)
    # Copy the X-axis (X.txt) file if OWN_AXIS is True, return sorted list of all .txt spectra in the all_spectra directory, excluding X.txt if OWN_AXIS is True
    if OWN_AXIS:
        shutil.copy2(DATA_FOLDER / X_AXIS_FILE, all_spectra_dir / "X.txt")
    filelist = sorted(
        path
        for path in all_spectra_dir.rglob("*.txt")
        if not (OWN_AXIS and path.name == X_AXIS_FILE)
    )
    if not filelist:
        raise FileNotFoundError(f"No processed .txt spectra found under {DATA_FOLDER}")
    return filelist

# Assemble tables from the processed spectra, save them in xlsx format, and return the wavelength axis, aligned spectra, and filenames
def assemble_tables(filelist: list[Path], export_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    loaded = [(path, read_spectrum(path)) for path in filelist] # list of tuples (path, values)
    row_count = max(len(values) for _, values in loaded)
    raw_data = np.zeros((row_count, 1 + 2 * len(loaded)))
    raw_headers = ["X"]
    # Fill the raw_data array with the loaded spectra, interleaving X and Y values, and create headers for each spectrum
    for index, (path, values) in enumerate(loaded):
        raw_data[: len(values), 1 + 2 * index] = values[:, 0]
        raw_data[: len(values), 2 + 2 * index] = values[:, 1]
        raw_headers.extend([f"X_{path.name}", f"Y_{path.name}"])
        if index == 0 and not OWN_AXIS:
            raw_data[: len(values), 0] = values[:, 0]
    # Determine the common X-axis for interpolation, either from the provided X.txt file or from the first spectrum
    if OWN_AXIS:
        axis_values = pd.read_csv(DATA_FOLDER / X_AXIS_FILE, header=None).iloc[:, 0]
        x_axis = pd.to_numeric(axis_values, errors="coerce").dropna().to_numpy(float)
    else:
        x_axis = loaded[0][1][:, 0].copy()

    aligned = np.zeros((len(x_axis), len(loaded)))
    for index, (_, values) in enumerate(loaded):
        source_x, source_y = values[:, 0], values[:, 1]
        inside = (x_axis >= source_x[0]) & (x_axis <= source_x[-1])
        aligned[inside, index] = np.interp(x_axis[inside], source_x, source_y)

    raw_table = pd.DataFrame(raw_data, columns=raw_headers)
    raw_table.to_excel(export_dir / "all_data.xlsx", index=False)
    aligned_table = pd.DataFrame(
        aligned, columns=[path.name for path, _ in loaded]
    )
    aligned_table.insert(0, "X", x_axis)
    aligned_table.to_excel(export_dir / "all_on_same_axis.xlsx", index=False)
    return x_axis, aligned, [path.name for path, _ in loaded]

# Load reference spectra from ref.xlsx and prepare the initial W matrix for NNMF
def load_reference_spectra(wavelength: np.ndarray, count: int) -> np.ndarray:
    """Read and interpolate workbook spectra without changing their shapes."""
    if REFERENCE_FILE is None:
        raise RuntimeError("Choose a reference spectra workbook before fitting.")
    reference_path = Path(REFERENCE_FILE)
    if not reference_path.is_absolute():
        reference_path = DATA_FOLDER / reference_path
    reference = pd.read_excel(reference_path, header=None)
    reference = reference.apply(pd.to_numeric, errors="coerce").dropna(
        subset=[0]
    )
    reference_values = reference.to_numpy(dtype=float)
    reference_values = reference_values[np.argsort(reference_values[:, 0])]
    ref_wavelength = reference_values[:, 0]
    if reference_values.shape[1] < count + 1:
        raise ValueError(f"Reference workbook needs wavelength plus {count} spectra columns")
    spectra = np.column_stack([
        np.interp(wavelength, ref_wavelength, reference_values[:, component + 1],
                  left=0.0, right=0.0)
        for component in range(count)
    ])
    if not np.all(np.isfinite(spectra)) or np.any(spectra < 0):
        raise ValueError("Reference spectra must contain finite, nonnegative intensities")
    return spectra


def save_reference_workbook(reference_path: Path, wavelength: np.ndarray,
                            references: np.ndarray) -> None:
    """Write a workbook with wavelength in column 0 and one reference spectrum per remaining column."""
    if references.ndim != 2 or references.shape[0] != len(wavelength):
        raise ValueError("references must have shape (wavelength, component)")
    if references.shape[1] == 0:
        raise ValueError("At least one reference spectrum is required")
    frame = pd.DataFrame(np.column_stack([wavelength, references]))
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_excel(reference_path, index=False, header=False)


def gaussian_reference_curve(x: np.ndarray, amplitude: float, center: float,
                             sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-6)
    return float(amplitude) * np.exp(-0.5 * ((x - float(center)) / sigma) ** 2)


def fit_gaussian_reference_curves(wavelength: np.ndarray, spectra: np.ndarray,
                                  count: int,
                                  reference_path: Path | None = None
                                  ) -> tuple[np.ndarray, np.ndarray]:
    """Jointly fit shared Gaussian centers and widths across all spectra."""
    if (wavelength.ndim != 1 or spectra.ndim != 2
            or spectra.shape[0] != len(wavelength)):
        raise ValueError("spectra must have shape (wavelength, sample)")
    if not 1 <= count <= max(1, COMPONENTS):
        raise ValueError("A Gaussian reference fit requires at least one component")
    if len(wavelength) < 3 or np.any(~np.isfinite(wavelength)) or np.any(np.diff(wavelength) <= 0):
        raise ValueError("Gaussian fitting needs at least three increasing finite wavelengths")
    if spectra.shape[1] < 1 or not np.all(np.isfinite(spectra)):
        raise ValueError("Gaussian fitting needs at least one spectrum with finite intensities")

    wavelength_span = float(wavelength[-1] - wavelength[0])
    minimum_sigma = float(np.median(np.diff(wavelength))) / 2.0
    maximum_sigma = wavelength_span / 2.0
    if minimum_sigma >= maximum_sigma:
        raise ValueError("The fit wavelength range is too narrow for Gaussian width bounds")
    mean_spectrum = np.maximum(np.mean(spectra, axis=1), 0.0)
    peak_indices, _ = find_peaks(mean_spectrum)
    ranked_peaks = sorted(
        peak_indices.tolist(), key=lambda index: mean_spectrum[index], reverse=True
    )
    candidate_peaks = ranked_peaks[:max(3 * count, count)]

    center_starts: list[np.ndarray] = []
    if len(candidate_peaks) >= count:
        for selected_peaks in combinations(candidate_peaks, count):
            centers = np.sort(wavelength[np.asarray(selected_peaks, dtype=int)])
            if not any(np.allclose(centers, previous) for previous in center_starts):
                center_starts.append(centers)
            if len(center_starts) >= 6:
                break

    evenly_spaced = wavelength[0] + wavelength_span * (
        np.arange(1, count + 1) / (count + 1)
    )
    center_starts.append(evenly_spaced)
    generator = np.random.default_rng(RANDOM_SEED)
    while len(center_starts) < 10:
        centers = np.sort(generator.uniform(wavelength[0], wavelength[-1], size=count))
        center_starts.append(centers)

    def fit_amplitudes(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        centers = parameters[:count]
        sigmas = parameters[count:]
        basis = np.column_stack([
            gaussian_reference_curve(wavelength, 1.0, center, sigma)
            for center, sigma in zip(centers, sigmas)
        ])
        amplitudes = np.empty((count, spectra.shape[1]), dtype=float)
        fitted = np.empty_like(spectra)
        for sample_index in range(spectra.shape[1]):
            amplitudes[:, sample_index], _ = nnls(basis, spectra[:, sample_index])
            fitted[:, sample_index] = basis @ amplitudes[:, sample_index]
        return amplitudes, fitted

    def residual(parameters: np.ndarray) -> np.ndarray:
        _, fitted = fit_amplitudes(parameters)
        return (fitted - spectra).ravel()

    lower_bounds = np.r_[
        np.full(count, wavelength[0]),
        np.full(count, minimum_sigma),
    ]
    upper_bounds = np.r_[
        np.full(count, wavelength[-1]),
        np.full(count, maximum_sigma),
    ]
    best_result = None
    best_cost = np.inf
    sigma_starts = (
        wavelength_span / (4.0 * count),
        wavelength_span / (2.0 * count),
        wavelength_span / count,
    )
    for start_index, centers in enumerate(center_starts):
        sigma_guess = np.clip(
            sigma_starts[start_index % len(sigma_starts)],
            minimum_sigma * 1.01,
            maximum_sigma * 0.99,
        )
        initial = np.r_[centers, np.full(count, sigma_guess)]
        result = least_squares(
            residual,
            x0=initial,
            bounds=(lower_bounds, upper_bounds),
            max_nfev=300,
        )
        if result.cost < best_cost:
            best_result = result
            best_cost = float(result.cost)
    if best_result is None:
        raise RuntimeError("Joint Gaussian reference fitting failed")

    centers = best_result.x[:count]
    sigmas = best_result.x[count:]
    amplitudes, _ = fit_amplitudes(best_result.x)
    order = np.argsort(centers)
    centers = centers[order]
    sigmas = sigmas[order]
    amplitudes = amplitudes[order]
    references = np.column_stack([
        gaussian_reference_curve(wavelength, 1.0, center, sigma)
        for center, sigma in zip(centers, sigmas)
    ])
    areas = trapezoid(references, x=wavelength, axis=0)
    if np.any(~np.isfinite(areas)) or np.any(areas <= 0):
        raise ValueError("Every Gaussian reference component must have positive area in the fit range")
    mean_component_areas = areas * np.mean(amplitudes, axis=1)
    total_component_area = mean_component_areas.sum()
    if not np.isfinite(total_component_area) or total_component_area <= 0:
        raise ValueError("Joint Gaussian fit has no positive fitted component area")
    area_fractions = mean_component_areas / total_component_area
    references = references / areas
    if reference_path is not None:
        save_reference_workbook(reference_path, wavelength, references)
    return references, area_fractions


def load_reference_initialization(wavelength: np.ndarray) -> np.ndarray:
    if not 0 <= INIT_COMP <= COMPONENTS:
        raise ValueError("INIT_COMP must be between zero and COMPONENTS")
    initial_w = np.random.default_rng(RANDOM_SEED).random(
        (len(wavelength), COMPONENTS)
    )
    references = load_reference_spectra(wavelength, INIT_COMP) if INIT_COMP else None
    for component in range(INIT_COMP):
        spectrum = references[:, component]
        norm = np.linalg.norm(spectrum)
        if norm == 0:
            raise ValueError(
                f"Reference component {component + 1} has zero norm in the selected range"
            )
        initial_w[:, component] = spectrum / norm
    return initial_w # shape (number of wavelengths, COMPONENTS)

# Run NNMF with multiple replicates, return the best W and H matrices (lowest reconstruction error)
# Spectra of samples D = W*H
# W: rows = wavelengths, columns = components
# H: rows = components, columns = samples
def fit_nmf(values: np.ndarray, initial_w: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    if COMPONENTS > len(COMPONENT_NAMES):
        raise ValueError(
            f"COMPONENTS={COMPONENTS}, but only {len(COMPONENT_NAMES)} component names are configured."
        )
    values = values.copy()
    tiny_negative = (values < 0) & (values >= -NNMF_NEGATIVE_TOLERANCE) # set values in [-tolerance, 0) to zero
    values[tiny_negative] = 0.0

    if np.any(values < 0): # values must be nonnegative for NNMF
        minimum = float(np.min(values))
        negative_count = int(np.count_nonzero(values < 0))
        raise ValueError(
            f"NNMF input still contains {negative_count} negative values after "
            f"zeroing values down to -{NNMF_NEGATIVE_TOLERANCE:g} "
            f"(minimum {minimum:.6g}); inspect the baseline correction or tolerance."
        )
    
    best_w: np.ndarray | None = None
    best_h: np.ndarray | None = None
    best_error = np.inf
    epsilon = np.finfo(float).eps

    for replicate in range(NNMF_REPLICATES):
        model = NMF(
            n_components=COMPONENTS,
            init="custom" if replicate == 0 and initial_w is not None else "random",
            solver="cd", # Coordinate descent algorithm (updates one component at a time while keeping the others fixed)
            max_iter=MAX_ITERATIONS,
            random_state=RANDOM_SEED + replicate,
        )
        if replicate == 0 and initial_w is not None: # run NNMF with initial matrices
            w_init = initial_w.copy()
            h_init = np.maximum(w_init.T @ values, epsilon)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                w_fit = model.fit_transform(values, W=w_init, H=h_init) 
        else: # otherwise, run NNMF with random initialization
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                w_fit = model.fit_transform(values)
        h_fit = model.components_
        error = float(np.linalg.norm(values - w_fit @ h_fit))
        if error < best_error:
            best_w, best_h, best_error = w_fit, h_fit, error

    if best_w is None or best_h is None:
        raise RuntimeError("NNMF did not produce a fit")
    return best_w, best_h

# Calc the Pearson correlation coefficient between two 1D arrays
def correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def fit_fixed_reference_spectra(values: np.ndarray, references: np.ndarray,
                                filenames: list[str]) -> np.ndarray:
    """Fit nonnegative weights per spectrum without modifying reference shapes.

    Measured data may contain negative baseline noise; NNLS constrains the
    coefficients, not the measurements. No clipping or random starts are used.
    """
    if values.ndim != 2 or values.shape[1] != len(filenames):
        raise ValueError("values must have axes (wavelength, spectrum), matching filenames")
    if (references.ndim != 2 or references.shape[0] != values.shape[0]
            or references.shape[1] == 0):
        raise ValueError("references must have axes (wavelength, component), matching values")
    if not np.all(np.isfinite(values)):
        raise ValueError("Measured spectra must contain finite intensities")
    if not np.all(np.isfinite(references)) or np.any(references < 0):
        raise ValueError("Reference spectra must contain finite, nonnegative intensities")
    if np.linalg.matrix_rank(references) < references.shape[1]:
        raise ValueError("Reference spectra must be linearly independent in the fit range")
    weights = np.empty((references.shape[1], values.shape[1]))
    for sample, filename in enumerate(filenames):
        try:
            weights[:, sample], _ = nnls(references, values[:, sample])
        except RuntimeError as error:
            raise RuntimeError(f"NNLS did not converge for {filename}: {error}") from error
    return weights


def component_percentages(wavelength: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Integrated signal fractions over the selected fit range; undefined for zero fit."""
    areas = trapezoid(components, x=wavelength, axis=0)
    totals = areas.sum(axis=1, keepdims=True)
    return np.divide(100 * areas, totals, out=np.full_like(areas, np.nan), where=totals > 0)


def select_data_folder() -> Path:
    """Ask the user to choose the folder containing the data subfolders."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        data_folder = filedialog.askdirectory(
            title="Choose the working folder containing the data subfolders",
            initialdir=str(Path.cwd()),
        )
        if not data_folder:
            raise RuntimeError("No working folder was selected.")
    finally:
        root.destroy()
    return Path(data_folder)


def select_reference_file(data_folder: Path) -> Path:
    """Ask the user to choose the reference spectra workbook."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        reference_file = filedialog.askopenfilename(
            title="Choose the reference spectra workbook",
            initialdir=str(data_folder),
            filetypes=[("Excel workbooks", "*.xlsx *.xls"), ("All files", "*.*")],
        )
        if not reference_file:
            raise RuntimeError("No reference spectra workbook was selected.")
    finally:
        root.destroy()
    return Path(reference_file)


def select_data_folders(data_folder: Path) -> list[str]:
    """Let the user select multiple immediate child folders for analysis."""
    import tkinter as tk
    from tkinter import Listbox

    folder_names = sorted(
        path.name
        for path in data_folder.iterdir()
        if path.is_dir() and path.name != EXPORT_FOLDER
    )
    if not folder_names:
        raise FileNotFoundError(f"No data subfolders found under {data_folder}")

    root = tk.Tk()
    root.title("Choose data subfolders")
    root.attributes("-topmost", True)
    selected_folders: list[str] = []

    tk.Label(
        root,
        text="Select one or more folders (Ctrl/Shift-click), then click Continue.",
    ).pack(padx=12, pady=(12, 6))
    folder_list = Listbox(root, selectmode=tk.EXTENDED, exportselection=False)
    folder_list.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)
    for folder_name in folder_names:
        folder_list.insert(tk.END, folder_name)

    def accept_selection() -> None:
        selected_folders.extend(
            folder_names[index] for index in folder_list.curselection()
        )
        root.destroy()

    tk.Button(root, text="Continue", command=accept_selection).pack(pady=(6, 12))
    root.mainloop()

    if not selected_folders:
        raise RuntimeError("No data subfolders were selected.")
    return selected_folders


def prepare_analysis_spectra(export_dir: Path) -> tuple[list[Path], list[Path], list[str], list[str]]:
    """Baseline-correct originals once; retain originals and arithmetic folder averages."""
    average_dir = export_dir / "folder_averages" / "spectra"
    individual_dir = export_dir / "individual_spectra"
    average_dir.mkdir(parents=True, exist_ok=True)
    individual_dir.mkdir(parents=True, exist_ok=True)
    average_files, individual_files, sample_folders, sample_names = [], [], [], []
    for folder_name in LIST_OF_DATA_FOLDERS:
        folder = DATA_FOLDER / folder_name
        files = _list_spectrum_files(folder) if folder.is_dir() else []
        files = [path for path in files if not (OWN_AXIS and path.name == X_AXIS_FILE)]
        if not files:
            continue
        corrected = [subtract_spectrum_baseline(read_spectrum(path)) for path in files]
        # Average on the first original's grid when source grids differ.
        reference_x = corrected[0][:, 0]
        aligned_y = []
        for index, (source, spectrum) in enumerate(zip(files, corrected), start=1):
            if np.any(np.diff(spectrum[:, 0]) <= 0):
                raise ValueError(f"Wavelengths must be strictly increasing: {source}")
            aligned_y.append(np.interp(reference_x, spectrum[:, 0], spectrum[:, 1],
                                       left=0.0, right=0.0))
            target = individual_dir / folder_name / f"{index:04d}_{source.name}"
            target.parent.mkdir(parents=True, exist_ok=True)
            write_spectrum(target, spectrum)
            individual_files.append(target)
            sample_folders.append(folder_name)
            sample_names.append(f"{folder_name}/{source.relative_to(folder).as_posix()}")
        average_file = average_dir / f"{folder_name}_arit_avg.txt"
        write_spectrum(average_file, np.column_stack((reference_x, np.mean(aligned_y, axis=0))))
        average_files.append(average_file)
    if not individual_files:
        raise FileNotFoundError(
            f"No input .{_validate_input_format(INPUT_FORMAT)} spectra found under {DATA_FOLDER}"
        )
    # Only files from this selection are returned, so old outputs cannot enter a fit.
    return average_files, individual_files, sample_folders, sample_names


def normalize_spectra(wavelength: np.ndarray, spectra: np.ndarray,
                      filenames: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Apply the same NV0 scaling, fit crop, and area normalization to either branch."""
    nv0_index = int(np.argmin(np.abs(wavelength - NV0)))
    scale = spectra[nv0_index]
    invalid = ~np.isfinite(scale) | (scale <= 0)
    if np.any(invalid):
        names = [filenames[i] for i in np.flatnonzero(invalid)]
        raise ValueError(f"Nonpositive or invalid intensity near {NV0} nm: {names}")
    selected = (wavelength > THRESHOLD_LOWER) & (wavelength <= THRESHOLD_UPPER)
    fit_wavelength = wavelength[selected]
    if len(fit_wavelength) < 2:
        raise ValueError("The fit range must contain at least two wavelength points")
    normalized = (spectra / scale)[selected]
    areas = trapezoid(normalized, axis=0)
    invalid = ~np.isfinite(areas) | (areas <= 0)
    if np.any(invalid):
        names = [filenames[i] for i in np.flatnonzero(invalid)]
        raise ValueError(f"Nonpositive or invalid normalization area: {names}")
    return fit_wavelength, normalized / areas


def fit_joint_spectra(values: np.ndarray, initial_w: np.ndarray | None,
                      filenames: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Fit all samples jointly with shared shapes. W: wavelength/component; H: component/sample."""
    if values.ndim != 2 or values.shape[1] != len(filenames):
        raise ValueError("values must have axes (wavelength, spectrum), matching filenames")
    try:
        return fit_nmf(values, initial_w)
    except ValueError as error:
        raise ValueError(f"Cannot fit spectra jointly: {error}") from error


def plot_processed_spectra(average_wavelength: np.ndarray, averages: np.ndarray,
                           average_names: list[str], wavelength: np.ndarray,
                           spectra: np.ndarray, filenames: list[str],
                           sample_folders: list[str], figure_dir: Path,
                           reference_spectra: np.ndarray | None = None,
                           reference_names: list[str] | None = None,
                           reference_area_fractions: np.ndarray | None = None) -> None:
    """Show normalized folder averages and one original-spectrum overlay per folder."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 5))
    average_colours = prism_colours(len(average_names))
    for sample, filename in enumerate(average_names):
        axis.plot(average_wavelength, averages[:, sample], color=average_colours[sample],
                  label=Path(filename).stem)
    if reference_spectra is not None:
        if reference_spectra.ndim != 2 or reference_spectra.shape[0] != len(average_wavelength):
            raise ValueError(
                "reference_spectra must have axes (average wavelength, component)"
            )
        labels = reference_names or [
            f"Reference {index + 1}" for index in range(reference_spectra.shape[1])
        ]
        if len(labels) != reference_spectra.shape[1]:
            raise ValueError("reference_names must contain one label per reference spectrum")
        reference_areas = trapezoid(reference_spectra, axis=0)
        if np.any(~np.isfinite(reference_areas)) or np.any(reference_areas <= 0):
            raise ValueError("Every plotted reference spectrum must have positive finite area")
        if reference_area_fractions is None:
            display_fractions = np.full(reference_spectra.shape[1], 1.0)
        else:
            if (reference_area_fractions.shape != (reference_spectra.shape[1],)
                    or not np.all(np.isfinite(reference_area_fractions))
                    or np.any(reference_area_fractions < 0)
                    or reference_area_fractions.sum() <= 0):
                raise ValueError(
                    "reference_area_fractions must contain finite nonnegative weights, "
                    "one per reference spectrum, with a positive sum"
                )
            display_fractions = reference_area_fractions / reference_area_fractions.sum()
        line_styles = ("--", ":", "-.", (0, (5, 2, 1, 2)))
        for index, label in enumerate(labels):
            axis.plot(
                average_wavelength,
                (reference_spectra[:, index] / reference_areas[index])
                * display_fractions[index],
                color="black",
                linestyle=line_styles[index % len(line_styles)],
                linewidth=1.5,
                label=label,
            )
    axis.set(title="Normalized folder averages", xlabel="Wavelength (nm)",
             ylabel="Normalized intensity")
    axis.legend(fontsize="small")
    figure.tight_layout()
    figure.savefig(figure_dir / "folder_averages.png", bbox_inches="tight")
    if not SHOW_COMPARED:
        plt.close(figure)
    for folder_index, folder_name in enumerate(dict.fromkeys(sample_folders), start=1):
        figure, axis = plt.subplots(figsize=(9, 5))
        indices = [i for i, group in enumerate(sample_folders) if group == folder_name]
        sample_colours = prism_colours(len(indices))
        for color, sample in zip(sample_colours, indices):
            axis.plot(wavelength, spectra[:, sample], color=color, label=filenames[sample])
        axis.set(title=f"Individual spectra: {folder_name}", xlabel="Wavelength (nm)",
                 ylabel="Normalized intensity")
        if len(indices) <= 12:
            axis.legend(fontsize="small")
        figure.tight_layout()
        figure.savefig(figure_dir / f"folder_{folder_index:03d}_individual_overlay.png",
                       bbox_inches="tight")
        if not SHOW_COMPARED:
            plt.close(figure)


def plot_individual_fits(wavelength: np.ndarray, measured: np.ndarray,
                        fitted: np.ndarray, components: np.ndarray,
                        filenames: list[str], figure_dir: Path) -> None:
    """Plot every processed measurement, fit, and sample-weighted component.

    components[wavelength, sample, component] must sum to fitted along axis 2.
    Reference spectra alone do not contain the sample-specific fitted weights.
    """
    expected_shape = (len(wavelength), len(filenames))
    for name, values in (("measured", measured), ("fitted", fitted)):
        if values.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}; received {values.shape}.")
    if components.ndim != 3 or components.shape[:2] != expected_shape:
        raise ValueError(
            "Pass the fitted components array as the fourth argument, not "
            "reference shapes. Expected axes (wavelength, spectrum, component) "
            f"with first two dimensions {expected_shape}; received {components.shape}."
        )
    if not np.allclose(components.sum(axis=2), fitted):
        raise ValueError("The weighted components must sum to the fitted spectrum.")
    figure_dir.mkdir(parents=True, exist_ok=True)
    for sample, filename in enumerate(filenames):
        figure, axis = plt.subplots(figsize=(9, 5))
        axis.plot(wavelength, fitted[:, sample], color="blue", label="Fit")
        # A dashed measurement keeps both curves visible when the fit overlaps it.
        axis.plot(wavelength, measured[:, sample], color="red", linestyle="--", label="Measured")
        display_names = display_component_names()
        component_colours = pastel_colours(components.shape[2])
        for component in range(components.shape[2]):
            component_name = (display_names[component] if component < len(display_names)
                              else f"component {component + 1}")
            is_third_component = component == 2
            axis.plot(wavelength, components[:, sample, component],
                      color=component_colours[component],
                      linestyle="-." if is_third_component
                      else (":" if component == 0 else "--"),
                      label=f"Fitted {component_name}")
        axis.set(title=filename, xlabel="Wavelength (nm)", ylabel="Normalized intensity")
        axis.legend(fontsize="small")
        figure.tight_layout()
        figure.savefig(figure_dir / f"spectrum_{sample + 1:04d}_fit.png",
                       bbox_inches="tight")
        if not SHOW_COMPARED:
            plt.close(figure)


def plot_shared_component_shapes(wavelength: np.ndarray, shapes: np.ndarray,
                                 figure_dir: Path) -> None:
    """Plot shared learned component shapes normalized to unit area for comparison."""
    if shapes.shape != (len(wavelength), COMPONENTS):
        raise ValueError(
            f"shapes must have axes (wavelength, component); received {shapes.shape}."
        )
    areas = trapezoid(shapes, x=wavelength, axis=0)
    if np.any(areas <= 0):
        raise ValueError("Every learned component shape must have positive area")
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 5))
    component_colours = pastel_colours(COMPONENTS)
    for component, component_name in enumerate(display_component_names()[:COMPONENTS]):
        axis.plot(wavelength, shapes[:, component] / areas[component],
                  color=component_colours[component], label=component_name)
    axis.axvline(NV0, color="black", linestyle="--", label=f"Plot marker 1 ({NV0} nm)")
    axis.axvline(NVN, color="black", linestyle="--", label=f"Plot marker 2 ({NVN} nm)")
    axis.set(title="Learned NNMF component shapes",
             xlabel="Wavelength (nm)", ylabel="Normalized intensity")
    axis.legend()
    figure.tight_layout()
    figure.savefig(figure_dir / "shared_component_shapes.png", bbox_inches="tight")
    if not SHOW_COMPARED:
        plt.close(figure)


def normalized_reference_spectra(wavelength: np.ndarray, count: int = 2) -> np.ndarray:
    """Interpolate fixed references and normalize each to unit area over wavelength."""
    if not 1 <= count <= len(COMPONENT_NAMES):
        raise ValueError("Reference count must be between one and the number of component names")
    if len(wavelength) < 2 or np.any(np.diff(wavelength) <= 0):
        raise ValueError("Reference normalization needs at least two increasing wavelengths")
    references = load_reference_spectra(wavelength, count)
    areas = trapezoid(references, x=wavelength, axis=0)
    if np.any(areas <= 0):
        raise ValueError("Every reference needs positive area in the fit range")
    return references / areas


def plot_fit_overviews(wavelength: np.ndarray, fitted: np.ndarray,
                       filenames: list[str],
                       figure_dir: Path) -> None:
    """Show reconstruction overview for all original spectra."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 5))
    sample_colours = prism_colours(len(filenames))
    for color, sample, filename in zip(sample_colours, range(len(filenames)), filenames):
        axis.plot(wavelength, fitted[:, sample], color=color, label=filename)
    axis.axvline(NV0, color="black", linestyle="--", label=f"Plot marker 1 ({NV0} nm)")
    axis.axvline(NVN, color="black", linestyle="--", label=f"Plot marker 2 ({NVN} nm)")
    axis.set(title="Spectral reconstructions per spectrum", xlabel="Wavelength (nm)",
             ylabel="Normalized intensity")
    if len(filenames) <= 12:
        axis.legend(fontsize="small")
    figure.tight_layout()
    figure.savefig(figure_dir / "reconstruction_overview.png", bbox_inches="tight")
    if not SHOW_COMPARED:
        plt.close(figure)


def export_fit_results(export_dir: Path, wavelength: np.ndarray,
                       normalized: np.ndarray, components: np.ndarray,
                       fitted: np.ndarray, filenames: list[str],
                       sample_folders: list[str]) -> pd.DataFrame:
    """Export fitted originals; All averages individual results."""
    export_dir.mkdir(parents=True, exist_ok=True)
    contributions = component_percentages(wavelength, components)
    # Assemble per-spectrum columns and save the spectral results workbook.
    sample_count = normalized.shape[1]
    per_spectrum_columns: list[np.ndarray] = []
    per_spectrum_headers: list[str] = []
    for sample in range(sample_count):
        per_spectrum_columns.append(normalized[:, sample])
        per_spectrum_headers.append(f"measured[{sample + 1}]")
        for component in range(COMPONENTS):
            per_spectrum_columns.append(components[:, sample, component])
            per_spectrum_headers.append(
                f"{COMPONENT_NAMES[component]}[{sample + 1}]"
            )
        per_spectrum_columns.append(fitted[:, sample])
        per_spectrum_headers.append(f"fit[{sample + 1}]")

    component_means = np.mean(components, axis=1)
    spectral_values = np.column_stack(
        [
            wavelength,
            np.mean(normalized, axis=1),
            component_means,
            np.mean(fitted, axis=1),
            *per_spectrum_columns,
        ]
    )
    spectral_headers = [
        "WL (nm)",
        "avg measured",
        *(f"avg {component_name}" for component_name in COMPONENT_NAMES[:COMPONENTS]),
        "avg fit",
        *per_spectrum_headers,
    ]
    pd.DataFrame(spectral_values, columns=spectral_headers).to_excel(
        export_dir / "spect.xlsx", index=False
    )

    # Calculate contribution, correlation, and reconstruction-error summaries.
    contribution_table = np.zeros((sample_count + 1, COMPONENTS + 1))
    contribution_table[0, :COMPONENTS] = np.mean(contributions, axis=0)
    contribution_table[1:, :COMPONENTS] = contributions
    contribution_table[:, -1] = np.sum(contribution_table[:, :COMPONENTS], axis=1)

    correlations = np.array(
        [correlation(fitted[:, i], normalized[:, i]) for i in range(sample_count)]
    )
    correlations = np.r_[correlation(np.mean(fitted, axis=1), np.mean(normalized, axis=1)), correlations]
    deviations = np.sqrt(np.mean((fitted - normalized) ** 2, axis=0))
    deviations = np.r_[
        np.sqrt(np.mean((np.mean(fitted, axis=1) - np.mean(normalized, axis=1)) ** 2)),
        deviations,
    ]
    ratio_names = ["All", *filenames]
    ratio_headers = [
        "#",
        "Filename",
        "Folder",
        *(f"{component_name} [%]" for component_name in display_component_names()[:COMPONENTS]),
        "check_sum",
        "CORR",
        "DEV",
    ]
    ratio_rows = []
    for row_index, ratio_name in enumerate(ratio_names):
        contribution_row = contribution_table[0] if row_index == 0 else contribution_table[row_index]
        ratio_rows.append(
            [(-1 if row_index == 0 else row_index), ratio_name,
             ("All" if row_index == 0 else sample_folders[row_index - 1]),
             *contribution_row, correlations[row_index], deviations[row_index]]
        )
    ratios_df = pd.DataFrame(ratio_rows, columns=ratio_headers)
    ratios_df.to_excel(
        export_dir / "ratios.xlsx", index=False
    )

    return ratios_df


def main() -> None:
    global DATA_FOLDER, REFERENCE_FILE, LIST_OF_DATA_FOLDERS
    global INPUT_FORMAT, TRAPZ_DATA_FOLDERS, ARIT_DATA_FOLDERS
    defaults = {name: globals()[name] for name in PARAMETER_FIELDS}
    setup = select_analysis_setup(
        DATA_FOLDER, REFERENCE_FILE, COMPONENT_NAMES, defaults
    )
    DATA_FOLDER = setup["DATA_FOLDER"]
    REFERENCE_FILE = setup["REFERENCE_FILE"]
    LIST_OF_DATA_FOLDERS = setup["LIST_OF_DATA_FOLDERS"]
    INPUT_FORMAT = setup["INPUT_FORMAT"]
    globals().update(setup["parameters"])
    TRAPZ_DATA_FOLDERS = []
    ARIT_DATA_FOLDERS = LIST_OF_DATA_FOLDERS.copy()
    export_dir = DATA_FOLDER / EXPORT_FOLDER
    figure_dir = export_dir / FIGURE_FOLDER
    average_files, individual_files, sample_folders, filenames = prepare_analysis_spectra(export_dir)
    average_wavelength, average_spectra, average_names = assemble_tables(
        average_files, export_dir / "folder_averages")
    individual_wavelength, spectra, _ = assemble_tables(individual_files, export_dir)
    average_fit_wavelength, average_normalized = normalize_spectra(
        average_wavelength, average_spectra, average_names)
    fit_wavelength, normalized = normalize_spectra(individual_wavelength, spectra, filenames)
    plot_processed_spectra(average_fit_wavelength, average_normalized, average_names,
                           fit_wavelength, normalized, filenames, sample_folders, figure_dir)
    if USE_GAUSSIAN_REFERENCE and INIT_COMP == 0:
        gaussian_reference_path = export_dir / "ref.xlsx"
        gaussian_reference, _ = fit_gaussian_reference_curves(
            fit_wavelength,
            normalized,
            COMPONENTS,
            reference_path=gaussian_reference_path,
        )
        REFERENCE_FILE = str(gaussian_reference_path)
        initial_w = gaussian_reference.copy()
    else:
        initial_w = load_reference_initialization(fit_wavelength) if INIT_COMP > 0 else None
    approximations, weighting = fit_joint_spectra(normalized, initial_w, filenames)
    wavelength = fit_wavelength
    components = approximations[:, None, :] * weighting.T[None, :, :]
    fitted = np.sum(components, axis=2)
    export_fit_results(export_dir, wavelength, normalized, components, fitted,
                       filenames, sample_folders)
    plot_shared_component_shapes(wavelength, approximations, figure_dir)
    plot_fit_overviews(wavelength, fitted, filenames, figure_dir)
    plot_individual_fits(wavelength, normalized, fitted, components,
                         filenames, figure_dir)
    print(f"Analysis complete: fitted {len(filenames)} original spectra.")
    print(f"Results: {export_dir / 'spect.xlsx'} and {export_dir / 'ratios.xlsx'}")
    print(f"Plots: {figure_dir}")
    if SHOW_COMPARED:
        plt.show()


if __name__ == "__main__":
    main()
