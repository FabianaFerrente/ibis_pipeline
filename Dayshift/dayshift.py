import argparse
import glob
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.io import readsav
from scipy.optimize import curve_fit

# Ignora avvisi non critici di Astropy
warnings.filterwarnings('ignore', category=fits.verify.VerifyWarning)

# --- Costanti e Configurazione ---

# Parametri specifici per ciascun filtro
FILTER_PARAMS = {
    "8542": {
        "prefilter_key": "prefilt8542_ref_main",
        "jung_range": (8539.0, 8545.0)
    },
    "6173": {
        "prefilter_key": "prefilt6173_ref_main",
        "jung_range": (6170.0, 6175.0)
    }
}

# --- Funzioni di Utilità e Caricamento Dati ---

def read_jung_fits(file_path: Path, w1: float, w2: float, subsample: int = 1):
    """
    Reads the Jungfraujoch atlas spectrum from a FITS file.
    This function is specific to the structure of the jun.fits file as provided.

    Args:
        file_path (Path): Path to the atlas FITS file (e.g., jun.fits).
        w1 (float): Starting wavelength of the range to read.
        w2 (float): Ending wavelength of the range.
        subsample (int): Subsampling factor.

    Returns:
        tuple: (wave, spectrum_norm) - wavelength axis and normalized spectrum.
    """
    if not file_path.is_file():
        raise FileNotFoundError(f"File dell'atlante non trovato: {file_path}")

    skip = round((w1 - 3601) * 1000)
    np_points = int((w2 - w1 + 1) * 500)

    try:
        with fits.open(file_path, memmap=True) as hdul:
            raw_data = hdul[0].data
        
        sp = raw_data.flatten()[skip // 2: skip // 2 + np_points]
        
        if sp.dtype.byteorder != '=':
            sp = sp.byteswap().view(sp.dtype.newbyteorder('S'))
        
        sp = sp.astype(np.float32) * 0.1
        
        if subsample > 1:
            sp = sp.reshape(-1, subsample).mean(axis=1)
        
        wave = np.arange(len(sp), dtype=np.float32) * 0.002 * subsample + w1
        
        spectrum_norm = (sp - sp.min()) / (sp.max() - sp.min())
        print(f"Loaded spectrum from atlas '{file_path.name}'.")
        return wave, spectrum_norm
    except Exception as e:
        raise IOError(f"Error '{file_path.name}': {e}")

def load_prefilter_data(prefilter_dir: Path, prefilter_key: str, selected_filter: str):
    """
    Loads prefilter transmission data from the .sav file.

    Args:
        prefilter_dir (Path): Folder containing the prefilter .sav files.
        prefilter_key (str): Key to extract the prefilter from the .sav file.
        selected_filter (str): Current filter to find the correct file.

    Returns:
        np.ndarray: Prefilter transmission data.
    """
    file_list = list(prefilter_dir.glob(f"*{selected_filter}*.sav"))
    if not file_list:
        raise FileNotFoundError(f"No prefilter .sav file found in {prefilter_dir} for the {selected_filter} filter.")
    
    try:
        prefilter_data = readsav(str(file_list[0]))
        main_filter = prefilter_data[prefilter_key]
        print(f"Loaded prefilter from'{file_list[0].name}'.")
        return main_filter
    except Exception as e:
        raise IOError(f"Error reading prefilter file '{file_list[0].name}': {e}")

def process_scan_files(data_dir: Path, selected_filter: str):
    """
    Processes scan FITS files to extract the average spectrum and relative wavelength axis.

    Args:
        data_dir (Path): Folder containing the scan FITS files.
        selected_filter (str): Filter to find relevant files.

    Returns:
        tuple: (wave_im, central_regions_processed) - relative wavelength axis and spectrum.
    """
    fits_files = sorted(list(data_dir.glob(f"*{selected_filter}.fits")))
    if not fits_files:
        raise FileNotFoundError(f"No scan FITS file found in {data_dir} for the filter {selected_filter}.")

    # Extract REL_WAVE from the fits 
    all_rel_wave_values = []
    for f_path in fits_files:
        with fits.open(f_path) as hdul:
            for hdu in hdul:
                try:
                    all_rel_wave_values.append(hdu.header['REL_WAVE'])
                except KeyError:
                    continue
    
    # We find a valid image
    image_data = None
    for f_path in fits_files:
        with fits.open(f_path) as hdul:
            for hdu in hdul:
                if hdu.data is not None and hdu.data.ndim == 2:
                    image_data = hdu.data
                    break
        if image_data is not None:
            break

    if image_data is None:
        raise ValueError(f"No valid 2D image data found in the FITS files in **{data_dir}**.")

    # We define the central regions
    half_size = 25
    h, w = image_data.shape
    y_center = h // 2
    x_center_1 = w // 2 - 220
    x_center_2 = w // 2 + 220
    x1s, x1e = x_center_1 - half_size, x_center_1 + half_size
    x2s, x2e = x_center_2 - half_size, x_center_2 + half_size
    ys, ye = y_center - half_size, y_center + half_size

    # Mean of the central regions
    central_regions = []
    current_rel_wave_values_for_regions = [] 
    with fits.open(fits_files[0]) as hdul:
        for i in range(1, len(hdul)): # From HDU 1 
            hdu = hdul[i]
            data = hdu.data
            if data is not None and data.ndim >= 2:
                try:
                    region1 = data[ys:ye, x1s:x1e]
                    region2 = data[ys:ye, x2s:x2e]
                    avg_intensity = np.mean([region1.mean(), region2.mean()])
                    central_regions.append(avg_intensity)
                    current_rel_wave_values_for_regions.append(hdu.header['REL_WAVE'])
                except KeyError:
                    # Se REL_WAVE non c'è, ignora questa HDU per la spectral scan
                    continue
    
    if not central_regions:
        raise ValueError(f"No intensity data extracted from the central regions of the FITS files. Check the structure of the HDUs after PRIMARY (HDU 1+).")

    central_regions = np.array(central_regions)

    # Normalize
    central_regions = (central_regions - central_regions.min()) / (central_regions.max() - central_regions.min())

    # Remove zeroes from the edges
    temp_central_regions = list(central_regions)
    temp_rel_wave_values = list(current_rel_wave_values_for_regions)

  
    if temp_central_regions and temp_central_regions[0] == 0 and temp_rel_wave_values:
        temp_central_regions.pop(0)
        temp_rel_wave_values.pop(0)


    if temp_central_regions and temp_central_regions[-1] == 0 and temp_rel_wave_values:
        temp_central_regions.pop()
        temp_rel_wave_values.pop()

    central_regions_processed = np.array(temp_central_regions)
    wave_im_processed = np.array(temp_rel_wave_values)

    print("Scans files processed, average spectrum and relative wavelengths extracted.")
    return wave_im_processed, central_regions_processed

def correct_and_normalize_scan(central_regions: np.ndarray, prefilter_main: np.ndarray):
    """
    Applies prefilter correction and normalizes the scan spectrum.

    Args:
        central_regions (np.ndarray): Raw scan spectrum (mean intensities).
        prefilter_main (np.ndarray): Prefilter transmission data.

    Returns:
        np.ndarray: Corrected and normalized scan spectrum.
    """
    # Interpolation of the prefilter to the scan grid
    old_x = np.linspace(0, 1, len(prefilter_main))
    new_x = np.linspace(0, 1, len(central_regions))
    prefilter_interp = np.interp(new_x, old_x, prefilter_main)

    # Ratio
    prefilter_interp_safe = np.where(prefilter_interp < 0.05, np.nan, prefilter_interp)
    corrected_scan = central_regions / prefilter_interp_safe

    # Replace NaNs generated by division with the median.
    corrected_scan = np.where(np.isnan(corrected_scan), np.nanmedian(corrected_scan), corrected_scan)

    # Normalization with percentiles
    vmin = np.percentile(corrected_scan, 1)
    vmax = np.percentile(corrected_scan, 99)
    corrected_scan = (corrected_scan - vmin) / (vmax - vmin)
    corrected_scan = np.clip(corrected_scan, 0, 1) # Assicura che sia tra 0 e 1

    print("Scan spectrum corrected with prefilter and normalized.")
    return corrected_scan

# --- Plotting ---

def plot_initial_interpolated_comparison(wave_im, atlas_interp, corrected_scan, selected_filter, output_dir: Path, show_plot: bool):
    """
    Create and save an initial comparison plot of the scan spectrum interpolated on the **REL_WAVE** axis and the atlas interpolated on the same scale.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(wave_im, atlas_interp, label="Jungfraujoch Atlas (Interpolated)", color='b', linewidth=2)
    plt.plot(wave_im, corrected_scan, label="Image Spectral Scan / Prefilter", color='r', linewidth=2)
    plt.xlabel(f"Relative Wavelength - {selected_filter} (Å)") # Label modificato in "Relative Wavelength"
    plt.ylabel("Normalized Intensity")
    plt.title("Initial Comparison: Corrected Scan vs Jungfraujoch Atlas (Interpolated)")
    plt.legend()
    plt.grid()
    output_path = output_dir / f"initial_interpolated_scan_vs_atlas_{selected_filter}.png"
    plt.savefig(output_path)
    if show_plot:
        plt.show(block=False)
    else:
        plt.close()
    print(f"Initial comparison plot (interpolated) saved to: **{output_path}**")

def gaussian(x, a, x0, sigma, offset):
    """Gaussian function for the fit"""
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + offset


def fit_gaussian_near_zero(wavelengths, intensities, window_angstrom):
    """
    This function performs a Gaussian fit around the minimum of a spectral line, searching for the minimum closest to zero on the relative wavelength axis.
    """
    # Find the minimum closer to zero
    zero_idx = np.argmin(np.abs(wavelengths))
    
    # Compute pixel scale
    pixel_scale = wavelengths[1] - wavelengths[0] if len(wavelengths) > 1 else 0.01 # Fallback per evitare divisione per zero
    if pixel_scale == 0: pixel_scale = 0.01

    search_window = int(window_angstrom / pixel_scale)  # in pixel
    start = max(0, zero_idx - search_window)
    end = min(len(wavelengths), zero_idx + search_window)

    local_wavelengths = wavelengths[start:end]
    local_intensities = intensities[start:end]

    if not local_wavelengths.size: # check for empty arrays
        print("ATTENTION: Gaussian fit search window is empty.")
        return np.nan, np.array([]), np.array([]), np.array([])

    # Find the local minimum in the selected window
    local_min_idx = np.argmin(local_intensities)
    min_idx = start + local_min_idx

    # Window for the fit (1 Å in pixel)
    fit_window = int(1.0 / pixel_scale)
    fit_start = max(0, min_idx - fit_window // 2)
    fit_end = min(len(wavelengths), min_idx + fit_window // 2 + 1)

    x = wavelengths[fit_start:fit_end]
    y = intensities[fit_start:fit_end]

    if len(x) < 4: # Requires at least 4 points
        print(f"ATTENTION: Not enough points ({len(x)}) for the Gaussian fit. Returning the direct minimum.")
        return x[np.argmin(y)] if x.size > 0 else np.nan, x, y, np.array([])

    a0 = y.min() - y.max()
    x0 = x[np.argmin(y)]
    sigma0 = 0.2
    offset0 = y.max()

    try:
        popt, _ = curve_fit(gaussian, x, y, p0=[a0, x0, sigma0, offset0])
        fitted_min = popt[1]
        fit_curve = gaussian(x, *popt)
    except RuntimeError:
        print("ATTENTION: Gaussian fit did not converge. Returning the direct minimum.")
        fitted_min = x[np.argmin(y)] if x.size > 0 else np.nan
        fit_curve = np.full_like(x, np.nan)
    except ValueError as e: # Aggiunto controllo per ValueError
        print(f"ERROR in the gaussian fit (ValueError): {e}. Returning the direct minimum.")
        fitted_min = x[np.argmin(y)] if x.size > 0 else np.nan
        fit_curve = np.full_like(x, np.nan)
    return fitted_min, x, y, fit_curve

def plot_gaussian_fits(lambda_atlas, x_fit_a, y_fit_a, gauss_fit_a,
                       lambda_scan, x_fit_s, y_fit_s, gauss_fit_s, output_dir: Path, show_plot: bool):
    """
    Create and save the plots of the gaussian fits for the interpolated atlas and the scan. 
    """
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(x_fit_a, y_fit_a, 'k.', label='Atlas data')
    if gauss_fit_a.size > 0:
        plt.plot(x_fit_a, gauss_fit_a, 'b-', label='Gaussian fit')
    plt.axvline(lambda_atlas, color='k', linestyle='--', label=f'Minimum Fit = {lambda_atlas:.4f} Å')
    plt.title("Gaussian fit - Atlas")
    plt.xlabel("Relative Wavelength (Å)") # Label modificato in "Relative Wavelength"
    plt.ylabel("Normalized Intensity")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(x_fit_s, y_fit_s, 'k.', label='Scan data')
    if gauss_fit_s.size > 0:
        plt.plot(x_fit_s, gauss_fit_s, 'r-', label='Gaussian fit')
    plt.axvline(lambda_scan, color='k', linestyle='--', label=f'Minimum Fit = {lambda_scan:.4f} Å')
    plt.title("Gaussian fit - Corrected Scan")
    plt.xlabel("Relative Wavelength (Å)") # Label modificato in "Relative Wavelength"
    plt.ylabel("Normalized intensity")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    output_path = output_dir / "gaussian_fit_minima.png"
    plt.savefig(output_path)
    if show_plot:
        plt.show(block=False)
    else:
        plt.close()
    print(f"Gaussian fit plots saved in: {output_path}")

def plot_shifted_comparison(wave_im_shifted_atlas, atlas_interp, wave_im, corrected_scan, selected_filter, output_dir: Path, show_plot: bool):
    """
    Create and save a comparison plot between the shifted atlas and the scan spectrum.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(wave_im_shifted_atlas, atlas_interp, label="Jungfraujoch Atlas (Shifted)", color='b', linewidth=2)
    plt.plot(wave_im , corrected_scan, label="Image Spectral Scan / Prefilter", color='r', linewidth=2)
    plt.xlabel(f"Relative Wavelength - {selected_filter} (Å)") # Label modificato in "Relative Wavelength"
    plt.ylabel("Normalized Intensity")
    plt.title("Corrected Scan vs Jungfraujoch Atlas (Shifted)")
    plt.legend()
    plt.grid()
    output_path = output_dir / f"corrected_scan_vs_atlas_shifted_{selected_filter}.png"
    plt.savefig(output_path)
    if show_plot:
        plt.show(block=False)
    else:
        plt.close()
    print(f"Shifted spectrum saved in: {output_path}")

def plot_dual_axis_spectrum(wave_im, corrected_scan, lambda_zero_voltage, selected_filter, output_dir: Path, show_plot: bool):
    """
    Create a plot of the corrected spectrum with a double X-axis: relative and absolute.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Primo asse (in basso): wave_im (relativo)
    ax1.plot(wave_im, corrected_scan, color='r', linewidth=2, label="Corrected Scan")
    ax1.set_xlabel(f"Relative Wavelength - {selected_filter} (Å)")
    ax1.set_ylabel("Normalized Intensity")
    ax1.set_title("Corrected Image Spectral Scan with Dual X-Axis")
    ax1.grid()
    ax1.legend(loc='upper right')

    # Second axes (top): wave_im + lambda_zero_voltage 
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())  # sincronia assi
    
    # Get the ticks from the relative axis and convert them to absolute.
    absolute_ticks = ax1.get_xticks()
    ax2.set_xticks(absolute_ticks)
    ax2.set_xticklabels([f"{x + lambda_zero_voltage:.2f}" for x in absolute_ticks])
    ax2.set_xlabel("Absolute Wavelength (Å)")

    plt.tight_layout()
    output_path = output_dir / f"dual_axis_spectrum_{selected_filter}.png"
    plt.savefig(output_path)
    if show_plot:
        plt.show(block=False)
    else:
        plt.close()
    print(f"Plot with double X-axis saved to: {output_path}")

def save_results_to_file(output_dir: Path, selected_filter: str, 
                         shift_angstrom: float, lambda_zero_voltage: float,
                         lambda_min_abs: float, lambda_max_abs: float):
    """
    Save the key results (shift, lambda zero voltage, absolute range) to a text file.
    """
    output_file = output_dir / f"results_{selected_filter}.txt"
    with open(output_file, 'w') as f:
        f.write(f"--- Dayshift Analysis Results for Filter {selected_filter} ---\n")
        f.write(f"Calculated shift (from Gaussian fit on relative scale): {shift_angstrom:.4f} Å\n")
        f.write(f"Wavelength corresponding to the reference point (zero voltage, calculated): {lambda_zero_voltage:.4f} Å\n")
        f.write(f"Estimated absolute wavelength range: {lambda_min_abs:.4f} Å – {lambda_max_abs:.4f} Å\n")
        f.write("-----------------------------------------------------------\n")
    print(f"\nResults saved in: {output_file}")


# --- Funzione Principale ---

def main():
    """Main function to perform the complete dayshift analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze scan FITS files to calculate the dayshift using a FITS atlas and a SAV prefilter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--filter', 
        type=str, 
        required=True, 
        choices=FILTER_PARAMS.keys(), 
        help="The filter to analyze (es. '8542' o '6173')."
    )
    parser.add_argument(
        '--data_dir', 
        type=Path, 
        default=Path('./data/fits'),
        help="Folder containing FITS files with measured spectrum data."
    )
    parser.add_argument(
        '--prefilter_dir', 
        type=Path, 
        default=Path('./data/calib/prefilter'), 
        help="Folder containing the prefilter .sav files."
    )
    parser.add_argument(
        '--atlas_path', 
        type=Path, 
        default=Path('./data/calib/atlas/jun.fits'), 
        help="Full path to the 'jun.fits' atlas FITS file."
    )
    parser.add_argument(
        '--output_dir', 
        type=Path, 
        default=Path('./output'), 
        help="Output folder for saving plots."
    )
    parser.add_argument(
        '--show-plots',
        action='store_true',
        help="Show interactive plots. If not specified, plots will only be saved."
    )
    args = parser.parse_args()

    # Create output folder if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("--- Start Dayshift ---")
    print(f"Selected filter: {args.filter}")
    
    # 1. Carica parametri specifici del filtro
    params = FILTER_PARAMS[args.filter]
    
    # 2. Carica lo spettro dell'atlante da jun.fits
    try:
        wave_atlas_full, spectrum_norm_atlas_full = read_jung_fits(
            args.atlas_path, params["jung_range"][0], params["jung_range"][1]-1
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return
    except IOError as e:
        print(f"ERROR loading atlas: {e}")
        return

    # 3. Carica i dati di trasmissione del prefiltro dal file .sav
    try:
        prefilter_main = load_prefilter_data(
            args.prefilter_dir, params["prefilter_key"], args.filter
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return
    except IOError as e:
        print(f"ERROR loading prefilter: {e}")
        return
    
    # 4. Processa i file di scan per ottenere lo spettro medio e l'asse REL_WAVE
    try:
        wave_im, central_regions = process_scan_files(args.data_dir, args.filter)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR during scan file processing: {e}")
        return

    if central_regions.size == 0 or prefilter_main.size == 0:
        print("ERROR: Scan or prefilter data is empty after loading/processing.")
        return

    # 5. Apply prefilter correction
    corrected_scan = correct_and_normalize_scan(central_regions, prefilter_main)

    # 6. Interpolation of the atlas on the relative wavelength axis (wave_im)
    
    atlas_w_norm = np.linspace(0, 1, len(wave_atlas_full))
    spectr_w_norm = np.linspace(0, 1, len(wave_im))
    atlas_interp = np.interp(spectr_w_norm, atlas_w_norm, spectrum_norm_atlas_full)
    
    # 7. Visual comparison of the spectra (with respect to the same axis wave_im)
    plot_initial_interpolated_comparison(wave_im, atlas_interp, corrected_scan, args.filter, args.output_dir, args.show_plots)

    # 8. Gaussian fit to find line minima (on the relative 'wave_im' scale)
    print("\nGaussian fits...")
    lambda_atlas_relative, x_fit_a, y_fit_a, gauss_fit_a = fit_gaussian_near_zero(wave_im, atlas_interp, 1.0)
    lambda_scan_relative, x_fit_s, y_fit_s, gauss_fit_s = fit_gaussian_near_zero(wave_im, corrected_scan, 1.0)

    # 9. Plot gaussian fits
    plot_gaussian_fits(lambda_atlas_relative, x_fit_a, y_fit_a, gauss_fit_a,
                       lambda_scan_relative, x_fit_s, y_fit_s, gauss_fit_s, args.output_dir, args.show_plots)

    # 10. Shift evaluation (on relative scale)
    shift_angstrom = lambda_scan_relative - lambda_atlas_relative
    print(f"\nShift (from Gaussian fit on relative scale): {shift_angstrom:.4f} Å")

    # 11. Plot spectrum of shifted atlas
    # we shift the atlas
    wave_im_shifted_atlas = wave_im + shift_angstrom 
    plot_shifted_comparison(wave_im_shifted_atlas, atlas_interp, wave_im, corrected_scan, args.filter, args.output_dir, args.show_plots)

    # 12. Compute the zero voltage wavelength
    lambda_zero_voltage = np.nan 
    lambda_min_abs = np.nan
    lambda_max_abs = np.nan

    # Find the index of the point closest to zero in the shifted atlas.
    #This `zero_idx` is relative to `wave_im_shifted_atlas`.
    if wave_im_shifted_atlas.size > 0:
        zero_idx_on_shifted_atlas_axis = np.argmin(np.abs(wave_im_shifted_atlas))
        
        # IBIS grid normalized (wave_im normalized)
        spectr_w = np.linspace(0, 1, len(wave_im))
        
        # Atlas grid normalized (wave_atlas_full normalized)
        atlas_w_orig_norm = np.linspace(0, 1, len(wave_atlas_full))

        # Find the normalized position of the zero voltage point on the IBIS grid corresponding to the zero of the shifted atlas axis.
        if zero_idx_on_shifted_atlas_axis < len(spectr_w):
            spectr_zero = spectr_w[zero_idx_on_shifted_atlas_axis]
            
            # Go back to the absolute wavelength in the original atlas using the normalized position found.
            lambda_zero_voltage = np.interp(spectr_zero, atlas_w_orig_norm, wave_atlas_full)
            
            print(f"Wavelength corresponding to the zero voltage value: {lambda_zero_voltage:.4f} Å")
            
            # Compute the wavelenght range
            if wave_im.size > 0:
                lambda_min_abs = np.min(wave_im) + lambda_zero_voltage
                lambda_max_abs = np.max(wave_im) + lambda_zero_voltage
                print(f"Estimated absolute wavelength range: {lambda_min_abs:.4f} Å – {lambda_max_abs:.4f} Å")
            else:
                print("Unable to calculate the absolute range; wave_im is empty.")
        else:
             print("ATTENTION: `zero_idx_on_shifted_atlas_axis` is out of bounds for `spectr_w`. Unable to accurately calculate `lambda_zero_voltage`.")
    else:
        print("ATTENTION: `wave_im_shifted_atlas` is empty. Unable to calculate `lambda_zero_voltage`.")

    plot_dual_axis_spectrum(wave_im, corrected_scan, lambda_zero_voltage, args.filter, args.output_dir, args.show_plots)
    
    # 13. Save results in a txt
    save_results_to_file(args.output_dir, args.filter, 
                         shift_angstrom, lambda_zero_voltage,
                         lambda_min_abs, lambda_max_abs)
    
    if args.show_plots:
        print("\nClose the plots windows to terminate the script")
        plt.show()

    print("\n--- Completed ---")


if __name__ == '__main__':
    main()
