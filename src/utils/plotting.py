import os
import numpy as np
import matplotlib.pyplot as plt

def plot_statistics(
    stats,
    wavelength,
    calib: str = None,
    show: bool = False,
    save_dir: str = None,
    dpi: int = 150,
    outlier_sigma: float = 3.0
):
    """
    Plot and save dark statistics (mean level and RMS) across all frames,
    concatenated from all input FITS files. Adds legend and threshold lines.

    Parameters
    ----------
    stats : dict
        Output of `extract_dark_frames`.
        Must contain concatenated vectors:
        {
            "mean_level": [...],
            "rms": [...],
            "time": [...],
            "images": [...],
            "n_files": int,
            "frames_per_file": [...]
        }
    wavelength : int or float
        Wavelength (Å) for plot titles.
    show : bool, optional
        If True, display the plot with matplotlib. Default is False.
    save_dir : str, optional
        Directory where the figure should be saved. If None, saves to CWD.
    dpi : int, optional
        Resolution for the saved figure. Default is 150.
    outlier_sigma : float, optional
        Sigma threshold for marking outliers. Default 3.0.

    Returns
    -------
    str
        Path of the saved figure.
    """
    dlevel = np.array(stats["mean_level"])
    drms   = np.array(stats["rms"])

    # Calcola media e sigma globali
    global_mean = np.mean(dlevel)
    global_std  = np.std(dlevel)
    mad = np.median(np.abs(dlevel - global_mean))
    mask = np.abs(dlevel - global_mean) <= outlier_sigma * mad 
        
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left subplot: mean dark counts ---
    frame_indices = np.arange(len(dlevel))
    axes[0].plot(frame_indices, dlevel, 'o-', label='Mean per frame')
    axes[0].plot(frame_indices[~mask], dlevel[~mask], 'rx', label='Outliers')
    axes[0].axhline(global_mean, color='green', linestyle='--', label='Global mean')
    axes[0].axhline(global_mean + outlier_sigma * global_std, color='red', linestyle='--', label=f'+{outlier_sigma}σ')
    axes[0].axhline(global_mean - outlier_sigma * global_std, color='red', linestyle='--', label=f'-{outlier_sigma}σ')

    axes[0].set_title("Mean Dark Counts (all frames)")
    axes[0].set_xlabel("Frame index (concatenated)")
    axes[0].set_ylabel("Mean counts")
    axes[0].grid(True)
    axes[0].legend()

    # --- Right subplot: RMS ---
    axes[1].plot(frame_indices, drms, 'o-', color='red', label='RMS per frame')
    axes[1].set_title("RMS per Frame (all frames)")
    axes[1].set_xlabel("Frame index (concatenated)")
    axes[1].set_ylabel("RMS")
    axes[1].grid(True)
    axes[1].legend()

    fig.suptitle(f"{calib}_statistics (λ={wavelength} Å)", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # Save always
    if save_dir is None:
        save_dir = os.getcwd()
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{calib}_statistics_{wavelength}.png")
    fig.savefig(out_path, dpi=dpi)
    print(f"Grafico salvato in {out_path}")

    # Show optionally
    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_path