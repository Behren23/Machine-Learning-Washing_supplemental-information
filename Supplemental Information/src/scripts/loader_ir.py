"""
IR Spectral Data Loader

This module provides functions to load IR spectral data from CSV files,
parse filenames to extract plastic type and contaminant labels, and save
the processed data as NumPy arrays with one-hot encoded labels.

Author: Generated for PhD research on AI-based spectroscopic analysis
"""

import os
import re
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Any, Literal
from pathlib import Path
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial import ConvexHull


# Define known plastic types and contaminants
PLASTIC_TYPES = ['PP', 'LDPE', 'LLDPE', 'HDPE']
CONTAMINANTS = ['BSA', 'Oil', 'Starch', 'CMC', 'Guar', 'Casein']


def parse_filename(filename: str) -> Tuple[str, List[str]]:
    """
    Extract plastic type and contaminants from a spectral data filename.
    
    Parameters
    ----------
    filename : str
        The filename to parse (without path).
    
    Returns
    -------
    Tuple[str, List[str]]
        A tuple containing:
        - plastic_type: str (one of PP, LDPE, LLDPE, HDPE)
        - contaminants: List[str] (list of detected contaminants)
    
    Examples
    --------
    >>> parse_filename("1203-54-2 pp-1.CSV")
    ('PP', [])
    >>> parse_filename("1203-55-1 PP in 10% Oil + 1% Starch + 0.5% Guar-1.CSV")
    ('PP', ['Oil', 'Starch', 'Guar'])
    >>> parse_filename("HDPE contaminated with BSA and Oil.CSV")
    ('HDPE', ['BSA', 'Oil'])
    """
    # Remove file extension and convert to uppercase for matching
    name = os.path.splitext(filename)[0]
    name_upper = name.upper()
    
    # Identify plastic type
    plastic_type = None
    
    # Check for plastic types - order matters (LLDPE before LDPE)
    for plastic in ['LLDPE', 'LDPE', 'HDPE', 'PP']:
        if plastic in name_upper:
            plastic_type = plastic
            break
    
    if plastic_type is None:
        raise ValueError(f"Could not identify plastic type in filename: {filename}")
    
    # Identify contaminants
    detected_contaminants = []
    
    # Define patterns to search for each contaminant
    contaminant_patterns = {
        'BSA': [r'\bBSA\b', r'\bprotein\b'],
        'Oil': [r'\bOil\b', r'\bOlive\s*Oil\b'],
        'Starch': [r'\bStarch\b'],
        'CMC': [r'\bCMC\b'],
        'Guar': [r'\bGuar\b'],
        'Casein': [r'\bCasein\b']
    }
    
    for contaminant, patterns in contaminant_patterns.items():
        for pattern in patterns:
            if re.search(pattern, name, re.IGNORECASE):
                detected_contaminants.append(contaminant)
                break  # Found this contaminant, move to next
    
    return plastic_type, detected_contaminants


def load_spectrum(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a single spectrum from a CSV file.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file containing spectral data.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        wavenumber: 1D array of wavenumber values (cm^-1)
        absorbance: 1D array of absorbance values
    """
    df = pd.read_csv(filepath, header=None, names=['wavenumber', 'absorbance'])
    return df['wavenumber'].to_numpy(), df['absorbance'].to_numpy()


def create_one_hot_plastic(plastic_type: str) -> np.ndarray:
    """
    Create one-hot encoded vector for plastic type.
    
    Parameters
    ----------
    plastic_type : str
        The plastic type (PP, LDPE, LLDPE, or HDPE).
    
    Returns
    -------
    np.ndarray
        One-hot encoded vector of shape (4,).
        Order: [PP, LDPE, LLDPE, HDPE]
    """
    one_hot = np.zeros(len(PLASTIC_TYPES), dtype=np.float32)
    if plastic_type in PLASTIC_TYPES:
        idx = PLASTIC_TYPES.index(plastic_type)
        one_hot[idx] = 1.0
    return one_hot


def create_one_hot_contaminants(contaminants: List[str]) -> np.ndarray:
    """
    Create multi-hot encoded vector for contaminants.
    
    Parameters
    ----------
    contaminants : List[str]
        List of contaminants present in the sample.
    
    Returns
    -------
    np.ndarray
        Multi-hot encoded vector of shape (6,).
        Order: [BSA, Oil, Starch, CMC, Guar, Casein]
    """
    multi_hot = np.zeros(len(CONTAMINANTS), dtype=np.float32)
    for contaminant in contaminants:
        if contaminant in CONTAMINANTS:
            idx = CONTAMINANTS.index(contaminant)
            multi_hot[idx] = 1.0
    return multi_hot


def baseline_correction(
    wavenumber: np.ndarray,
    absorbance: np.ndarray,
    method: Literal['rubberband', 'polynomial', 'als'] = 'rubberband',
    **kwargs
) -> np.ndarray:
    """
    Apply baseline correction to a spectrum.
    
    Parameters
    ----------
    wavenumber : np.ndarray
        1D array of wavenumber values (cm^-1).
    absorbance : np.ndarray
        1D array of absorbance/intensity values.
    method : str, optional
        Baseline correction method. Options:
        - 'rubberband': Rubberband (convex hull) baseline correction.
          Good for spectra with broad, curved baselines.
        - 'polynomial': Polynomial fit baseline correction.
          kwargs: degree (int, default=3)
        - 'als': Asymmetric Least Squares baseline correction.
          Good for spectra with varying baseline shapes.
          kwargs: lam (float, default=1e6), p (float, default=0.01),
                  niter (int, default=10)
        Default is 'rubberband'.
    **kwargs
        Additional parameters for specific methods.
    
    Returns
    -------
    np.ndarray
        Baseline-corrected absorbance values.
    
    Examples
    --------
    >>> wavenumber, absorbance = load_spectrum('spectrum.csv')
    >>> corrected = baseline_correction(wavenumber, absorbance, method='rubberband')
    >>> corrected = baseline_correction(wavenumber, absorbance, method='polynomial', degree=4)
    >>> corrected = baseline_correction(wavenumber, absorbance, method='als', lam=1e5, p=0.01)
    """
    if method == 'rubberband':
        return _rubberband_baseline(wavenumber, absorbance)
    elif method == 'polynomial':
        degree = kwargs.get('degree', 3)
        return _polynomial_baseline(wavenumber, absorbance, degree)
    elif method == 'als':
        lam = kwargs.get('lam', 1e6)
        p = kwargs.get('p', 0.01)
        niter = kwargs.get('niter', 10)
        return _als_baseline(absorbance, lam, p, niter)
    else:
        raise ValueError(f"Unknown baseline correction method: {method}. "
                        f"Choose from 'rubberband', 'polynomial', or 'als'.")


def _rubberband_baseline(wavenumber: np.ndarray, absorbance: np.ndarray) -> np.ndarray:
    """
    Rubberband (convex hull) baseline correction.
    
    This method fits a baseline by finding the convex hull of the inverted
    spectrum and subtracting it. Works well for spectra with broad, curved
    baselines typical of IR spectroscopy.
    
    Parameters
    ----------
    wavenumber : np.ndarray
        1D array of wavenumber values.
    absorbance : np.ndarray
        1D array of absorbance values.
    
    Returns
    -------
    np.ndarray
        Baseline-corrected absorbance values.
    """
    # Create points for convex hull (invert spectrum so hull touches minima)
    points = np.column_stack([wavenumber, absorbance])
    
    # Get convex hull vertices
    try:
        hull = ConvexHull(points)
        hull_vertices = hull.vertices
    except Exception:
        # If convex hull fails (e.g., collinear points), return original
        return absorbance
    
    # Sort hull vertices by wavenumber
    hull_vertices = sorted(hull_vertices, key=lambda i: wavenumber[i])
    
    # Get the lower part of the hull (baseline)
    # Find indices that form the lower envelope
    lower_hull_indices = []
    
    # Start from the leftmost point
    for i, idx in enumerate(hull_vertices):
        lower_hull_indices.append(idx)
    
    # Filter to get only the lower envelope points
    # The lower envelope connects minima, not maxima
    baseline_indices = [hull_vertices[0]]  # Start with first point
    
    for i in range(1, len(hull_vertices)):
        idx = hull_vertices[i]
        # Check if this point is on the lower part of the hull
        # by comparing with linear interpolation between neighbors
        if i < len(hull_vertices) - 1:
            prev_idx = baseline_indices[-1]
            # Always include if it's a local minimum region
            baseline_indices.append(idx)
        else:
            baseline_indices.append(idx)
    
    # Remove points that are clearly on the upper envelope
    # by checking if they are above the line connecting neighbors
    final_baseline_indices = [baseline_indices[0]]
    for i in range(1, len(baseline_indices) - 1):
        idx = baseline_indices[i]
        prev_idx = final_baseline_indices[-1]
        next_idx = baseline_indices[i + 1]
        
        # Linear interpolation between prev and next
        t = (wavenumber[idx] - wavenumber[prev_idx]) / (wavenumber[next_idx] - wavenumber[prev_idx] + 1e-10)
        interp_value = absorbance[prev_idx] + t * (absorbance[next_idx] - absorbance[prev_idx])
        
        # Include point if it's at or below the interpolated line (lower envelope)
        if absorbance[idx] <= interp_value + 0.01 * (absorbance.max() - absorbance.min()):
            final_baseline_indices.append(idx)
    
    final_baseline_indices.append(baseline_indices[-1])
    
    # Interpolate baseline across all wavenumbers
    baseline_wavenumbers = wavenumber[final_baseline_indices]
    baseline_values = absorbance[final_baseline_indices]
    baseline = np.interp(wavenumber, baseline_wavenumbers, baseline_values)
    
    # Subtract baseline
    corrected = absorbance - baseline
    
    return corrected


def _polynomial_baseline(wavenumber: np.ndarray, absorbance: np.ndarray, degree: int = 3) -> np.ndarray:
    """
    Polynomial baseline correction.
    
    Fits a polynomial to the spectrum endpoints and selected minima points,
    then subtracts it as the baseline.
    
    Parameters
    ----------
    wavenumber : np.ndarray
        1D array of wavenumber values.
    absorbance : np.ndarray
        1D array of absorbance values.
    degree : int, optional
        Polynomial degree (default: 3).
    
    Returns
    -------
    np.ndarray
        Baseline-corrected absorbance values.
    """
    # Normalize wavenumber for numerical stability
    wn_min, wn_max = wavenumber.min(), wavenumber.max()
    wn_normalized = (wavenumber - wn_min) / (wn_max - wn_min)
    
    # Find local minima to use as baseline anchor points
    # Use a simple approach: divide spectrum into segments and find minimum in each
    n_segments = max(degree + 2, 5)
    segment_size = len(absorbance) // n_segments
    
    anchor_indices = [0]  # Start with first point
    for i in range(n_segments):
        start = i * segment_size
        end = min((i + 1) * segment_size, len(absorbance))
        segment_min_idx = start + np.argmin(absorbance[start:end])
        anchor_indices.append(segment_min_idx)
    anchor_indices.append(len(absorbance) - 1)  # End with last point
    
    # Remove duplicates and sort
    anchor_indices = sorted(set(anchor_indices))
    
    # Fit polynomial to anchor points
    anchor_wn = wn_normalized[anchor_indices]
    anchor_abs = absorbance[anchor_indices]
    
    # Fit polynomial
    coeffs = np.polyfit(anchor_wn, anchor_abs, degree)
    baseline = np.polyval(coeffs, wn_normalized)
    
    # Subtract baseline
    corrected = absorbance - baseline
    
    return corrected


def _als_baseline(absorbance: np.ndarray, lam: float = 1e6, p: float = 0.01, niter: int = 10) -> np.ndarray:
    """
    Asymmetric Least Squares (ALS) baseline correction.
    
    Also known as "Asymmetric Least Squares Smoothing" by Eilers and Boelens.
    Good for spectra with varying baseline shapes.
    
    Parameters
    ----------
    absorbance : np.ndarray
        1D array of absorbance values.
    lam : float, optional
        Smoothness parameter (default: 1e6). Larger values = smoother baseline.
    p : float, optional
        Asymmetry parameter (default: 0.01). Typically 0.001 to 0.1.
        Smaller values penalize positive deviations more heavily.
    niter : int, optional
        Number of iterations (default: 10).
    
    Returns
    -------
    np.ndarray
        Baseline-corrected absorbance values.
    
    References
    ----------
    Eilers, P.H.C., Boelens, H.F.M. (2005). Baseline Correction with 
    Asymmetric Least Squares Smoothing.
    """
    L = len(absorbance)
    
    # Construct second-order difference matrix
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L))
    D = D.tocsc()
    
    # Initialize weights
    w = np.ones(L)
    W = sparse.diags(w, 0, shape=(L, L))
    
    # Iterative reweighted least squares
    for _ in range(niter):
        W = sparse.diags(w, 0, shape=(L, L))
        Z = W + lam * D.T @ D
        baseline = spsolve(Z, w * absorbance)
        
        # Update weights: asymmetric weighting
        w = p * (absorbance > baseline) + (1 - p) * (absorbance <= baseline)
    
    # Subtract baseline
    corrected = absorbance - baseline
    
    return corrected


def apply_baseline_correction(
    spectra: np.ndarray,
    wavenumbers: np.ndarray,
    method: Literal['rubberband', 'polynomial', 'als'] = 'rubberband',
    verbose: bool = True,
    **kwargs
) -> np.ndarray:
    """
    Apply baseline correction to multiple spectra.
    
    Parameters
    ----------
    spectra : np.ndarray
        2D array of spectral data (n_samples, n_wavenumbers).
    wavenumbers : np.ndarray
        1D array of wavenumber values.
    method : str, optional
        Baseline correction method ('rubberband', 'polynomial', or 'als').
        Default is 'rubberband'.
    verbose : bool, optional
        If True, print progress information.
    **kwargs
        Additional parameters passed to the baseline correction method.
    
    Returns
    -------
    np.ndarray
        Baseline-corrected spectra (n_samples, n_wavenumbers).
    
    Examples
    --------
    >>> data = load_all_ir_data('path/to/data')
    >>> corrected_spectra = apply_baseline_correction(
    ...     data['spectra'], data['wavenumbers'], method='als', lam=1e5
    ... )
    """
    n_samples = spectra.shape[0]
    corrected_spectra = np.zeros_like(spectra)
    
    if verbose:
        print(f"Applying {method} baseline correction to {n_samples} spectra...")
    
    for i in range(n_samples):
        corrected_spectra[i] = baseline_correction(
            wavenumbers, spectra[i], method=method, **kwargs
        )
        
        if verbose and (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{n_samples} spectra")
    
    if verbose:
        print(f"Baseline correction complete.")
    
    return corrected_spectra


def load_all_ir_data(
    data_dir: str,
    target_wavenumbers: Optional[np.ndarray] = None,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Load all IR spectral data from a directory.
    
    This function loads all CSV files from the specified directory,
    extracts labels from filenames, and returns aligned spectral data
    with one-hot encoded labels.
    
    Parameters
    ----------
    data_dir : str
        Path to directory containing CSV files.
    target_wavenumbers : np.ndarray, optional
        If provided, interpolate all spectra to these wavenumbers.
        If None, use the wavenumbers from the first file with the
        most common number of data points.
    verbose : bool
        If True, print progress information.
    
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing:
        - 'wavenumbers': 1D array of wavenumber values (n_wavenumbers,)
        - 'spectra': 2D array of absorbance data (n_samples, n_wavenumbers)
        - 'plastic_labels': 2D array of one-hot encoded plastic types (n_samples, 4)
        - 'contaminant_labels': 2D array of multi-hot encoded contaminants (n_samples, 6)
        - 'filenames': List of processed filenames
        - 'plastic_types': List of string plastic type labels
        - 'contaminants_list': List of contaminant lists per sample
    """
    data_path = Path(data_dir)
    csv_files = list(data_path.glob('*.CSV')) + list(data_path.glob('*.csv'))
    
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    if verbose:
        print(f"Found {len(csv_files)} CSV files in {data_dir}")
    
    # First pass: determine the most common wavenumber length
    wavenumber_lengths = {}
    for csv_file in csv_files:
        try:
            wavenumber, _ = load_spectrum(str(csv_file))
            length = len(wavenumber)
            if length not in wavenumber_lengths:
                wavenumber_lengths[length] = {'count': 0, 'file': csv_file, 'wavenumber': wavenumber}
            wavenumber_lengths[length]['count'] += 1
        except Exception as e:
            if verbose:
                print(f"Warning: Could not read {csv_file.name}: {e}")
    
    # Use the highest resolution (most data points) as reference
    highest_resolution_length = max(wavenumber_lengths.keys())
    reference_wavenumbers = wavenumber_lengths[highest_resolution_length]['wavenumber']
    
    if target_wavenumbers is not None:
        reference_wavenumbers = target_wavenumbers
    
    if verbose:
        print(f"\nWavenumber resolution distribution:")
        for length in sorted(wavenumber_lengths.keys(), reverse=True):
            count = wavenumber_lengths[length]['count']
            marker = " (reference)" if length == highest_resolution_length else ""
            print(f"  - {length} points: {count} files{marker}")
        print(f"\nReference wavenumber range: {reference_wavenumbers.min():.1f} - {reference_wavenumbers.max():.1f} cm^-1")
        print(f"Number of wavenumber points: {len(reference_wavenumbers)}")
    
    # Second pass: load all data and interpolate to common wavenumbers
    spectra_list = []
    plastic_labels_list = []
    contaminant_labels_list = []
    filenames_list = []
    plastic_types_list = []
    contaminants_raw_list = []
    
    skipped_count = 0
    interpolated_count = 0
    interpolated_files = []
    
    for i, csv_file in enumerate(csv_files):
        try:
            # Load spectrum
            wavenumber, absorbance = load_spectrum(str(csv_file))
            
            # Interpolate to reference wavenumbers if needed
            if len(wavenumber) != len(reference_wavenumbers) or not np.allclose(wavenumber, reference_wavenumbers):
                absorbance = np.interp(reference_wavenumbers, wavenumber, absorbance)
                interpolated_count += 1
                interpolated_files.append((csv_file.name, len(wavenumber)))
            
            # Parse filename for labels
            plastic_type, contaminants = parse_filename(csv_file.name)
            
            # Create one-hot encodings
            plastic_one_hot = create_one_hot_plastic(plastic_type)
            contaminant_multi_hot = create_one_hot_contaminants(contaminants)
            
            # Append to lists
            spectra_list.append(absorbance)
            plastic_labels_list.append(plastic_one_hot)
            contaminant_labels_list.append(contaminant_multi_hot)
            filenames_list.append(csv_file.name)
            plastic_types_list.append(plastic_type)
            contaminants_raw_list.append(contaminants)
            
        except Exception as e:
            skipped_count += 1
            if verbose:
                print(f"Warning: Skipping {csv_file.name}: {e}")
    
    if verbose:
        print(f"\nSuccessfully loaded {len(spectra_list)} spectra")
        if interpolated_count > 0:
            print(f"Interpolated {interpolated_count} files to match reference resolution ({len(reference_wavenumbers)} points)")
        if skipped_count > 0:
            print(f"Skipped {skipped_count} files due to errors")
    
    # Convert lists to arrays
    result = {
        'wavenumbers': reference_wavenumbers,
        'spectra': np.array(spectra_list, dtype=np.float32),
        'plastic_labels': np.array(plastic_labels_list, dtype=np.float32),
        'contaminant_labels': np.array(contaminant_labels_list, dtype=np.float32),
        'filenames': filenames_list,
        'plastic_types': plastic_types_list,
        'contaminants_list': contaminants_raw_list,
        'interpolated_count': interpolated_count,
        'interpolated_files': interpolated_files
    }
    
    return result


def save_processed_data(
    data: Dict[str, np.ndarray],
    output_dir: str,
    prefix: str = 'ir_data'
) -> None:
    """
    Save processed spectral data to NumPy files.
    
    Parameters
    ----------
    data : Dict[str, np.ndarray]
        Dictionary containing processed data from load_all_ir_data().
    output_dir : str
        Directory to save the output files.
    prefix : str
        Prefix for output filenames.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save numerical arrays
    np.save(output_path / f'{prefix}_wavenumbers.npy', data['wavenumbers'])
    np.save(output_path / f'{prefix}_spectra.npy', data['spectra'])
    np.save(output_path / f'{prefix}_plastic_labels.npy', data['plastic_labels'])
    np.save(output_path / f'{prefix}_contaminant_labels.npy', data['contaminant_labels'])
    
    # Save metadata as a compressed npz file
    np.savez(
        output_path / f'{prefix}_metadata.npz',
        filenames=np.array(data['filenames'], dtype=object),
        plastic_types=np.array(data['plastic_types'], dtype=object),
        contaminants_list=np.array(data['contaminants_list'], dtype=object),
        plastic_type_order=np.array(PLASTIC_TYPES, dtype=object),
        contaminant_order=np.array(CONTAMINANTS, dtype=object)
    )
    
    print(f"Saved processed data to {output_dir}")
    print(f"  - {prefix}_wavenumbers.npy: shape {data['wavenumbers'].shape}")
    print(f"  - {prefix}_spectra.npy: shape {data['spectra'].shape}")
    print(f"  - {prefix}_plastic_labels.npy: shape {data['plastic_labels'].shape}")
    print(f"  - {prefix}_contaminant_labels.npy: shape {data['contaminant_labels'].shape}")
    print(f"  - {prefix}_metadata.npz: filenames, plastic_types, contaminants_list")


def load_processed_data(
    data_dir: str,
    prefix: str = 'ir_data'
) -> Dict[str, Any]:
    """
    Load previously saved processed data.
    
    Parameters
    ----------
    data_dir : str
        Directory containing the saved files.
    prefix : str
        Prefix used when saving the files.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing all loaded data (arrays and lists).
    """
    data_path = Path(data_dir)
    
    # Load numerical arrays
    wavenumbers = np.load(data_path / f'{prefix}_wavenumbers.npy')
    spectra = np.load(data_path / f'{prefix}_spectra.npy')
    plastic_labels = np.load(data_path / f'{prefix}_plastic_labels.npy')
    contaminant_labels = np.load(data_path / f'{prefix}_contaminant_labels.npy')
    
    # Load metadata
    metadata = np.load(data_path / f'{prefix}_metadata.npz', allow_pickle=True)
    
    return {
        'wavenumbers': wavenumbers,
        'spectra': spectra,
        'plastic_labels': plastic_labels,
        'contaminant_labels': contaminant_labels,
        'filenames': list(metadata['filenames']),
        'plastic_types': list(metadata['plastic_types']),
        'contaminants_list': list(metadata['contaminants_list']),
        'plastic_type_order': list(metadata['plastic_type_order']),
        'contaminant_order': list(metadata['contaminant_order'])
    }


def print_data_summary(data: Dict[str, np.ndarray]) -> None:
    """
    Print a summary of the loaded data.
    
    Parameters
    ----------
    data : Dict[str, np.ndarray]
        Dictionary containing processed data.
    """
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    
    print(f"\nSpectral Data:")
    print(f"  - Number of samples: {data['spectra'].shape[0]}")
    print(f"  - Number of wavenumber points: {data['spectra'].shape[1]}")
    print(f"  - Wavenumber range: {data['wavenumbers'].min():.1f} - {data['wavenumbers'].max():.1f} cm^-1")
    
    print(f"\nPlastic Type Distribution:")
    plastic_counts = data['plastic_labels'].sum(axis=0)
    for i, plastic in enumerate(PLASTIC_TYPES):
        print(f"  - {plastic}: {int(plastic_counts[i])} samples")
    
    print(f"\nContaminant Distribution:")
    contaminant_counts = data['contaminant_labels'].sum(axis=0)
    for i, contaminant in enumerate(CONTAMINANTS):
        print(f"  - {contaminant}: {int(contaminant_counts[i])} samples")
    
    # Count samples by number of contaminants
    n_contaminants = data['contaminant_labels'].sum(axis=1)
    print(f"\nSamples by number of contaminants:")
    for n in range(int(n_contaminants.max()) + 1):
        count = (n_contaminants == n).sum()
        print(f"  - {n} contaminants: {count} samples")
    
    print("=" * 60 + "\n")


# Main execution
if __name__ == '__main__':
    # Define paths relative to project root
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    
    raw_data_dir = project_root / 'data' / 'flattened_IR_data' / 'raw'
    output_dir = project_root / 'data' / 'flattened_IR_data' / 'processed'
    
    print(f"Loading IR spectral data from: {raw_data_dir}")
    print(f"Output will be saved to: {output_dir}")
    print()
    
    # Load and process all data
    data = load_all_ir_data(str(raw_data_dir), verbose=True)
    
    # Print summary
    print_data_summary(data)
    
    # Save processed data
    save_processed_data(data, str(output_dir), prefix='ir_data')
    
    print("\nDone!")
