import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import argparse
import os
from datetime import datetime, timedelta


def compute_2d_psd(data_2d):
    f_2d = np.fft.fft2(data_2d)
    f_2d_shifted = np.fft.fftshift(f_2d)
    psd_2d = np.abs(f_2d_shifted)**2
    
    freqs_x = np.fft.fftshift(np.fft.fftfreq(data_2d.shape[1]))
    freqs_y = np.fft.fftshift(np.fft.fftfreq(data_2d.shape[0]))
    
    return freqs_x, freqs_y, psd_2d

def compute_radial_psd(freqs_x, freqs_y, psd_2d):
    fx, fy = np.meshgrid(freqs_x, freqs_y)
    freq_radial = np.sqrt(fx**2 + fy**2)
    
    freq_bins = np.linspace(0, np.max(freq_radial), 50)
    psd_radial = np.zeros_like(freq_bins[:-1])
    
    for i in range(len(freq_bins)-1):
        mask = (freq_radial >= freq_bins[i]) & (freq_radial < freq_bins[i+1])
        if mask.any():
            psd_radial[i] = np.mean(psd_2d[mask])
    
    return (freq_bins[:-1] + freq_bins[1:])/2, psd_radial

def compute_psd(data, avg_across_samples=True, sample_indices=None):
    """
    Compute PSD either averaged across all samples or for specific samples.
    """
    N, C, H, W = data.shape
    freq_dict = {}
    psd_dict = {}
    
    if avg_across_samples:
        for c in range(C):
            psds_all_samples = []
            for n in range(N):
                freqs_x, freqs_y, psd_2d = compute_2d_psd(data[n, c])
                freq_radial, psd_radial = compute_radial_psd(freqs_x, freqs_y, psd_2d)
                psds_all_samples.append(psd_radial)
            freq_dict[c] = freq_radial
            psd_dict[c] = np.mean(psds_all_samples, axis=0)
    else:
        for c in range(C):
            freqs_list = []
            psds_list = []
            for idx in sample_indices:
                freqs_x, freqs_y, psd_2d = compute_2d_psd(data[idx, c])
                freq_radial, psd_radial = compute_radial_psd(freqs_x, freqs_y, psd_2d)
                freqs_list.append(freq_radial)
                psds_list.append(psd_radial)
            freq_dict[c] = freqs_list
            psd_dict[c] = psds_list
    
    return freq_dict, psd_dict

def index_to_datetime(index):
    """Convert sample index to datetime in 2020"""
    start_date = datetime(2020, 1, 1)  # Start from Jan 1st 2020
    return start_date + timedelta(hours=index * 6)  # 6 hour intervals

def plot_model_comparison_psd(targets, climax_preds, stormer_preds, unet_preds, 
                            save_path='psd_comparison.png', dpi=300, 
                            avg_across_samples=True, num_samples=3, seed=42,
                            figsize=(15, 5)):
    """
    Plot PSD comparison between different models and ground truth and save to file.
    """
    variable_names = ['T2M', 'Z500', 'T850']
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # If not averaging, select random samples
    if not avg_across_samples:
        N = targets.shape[0]
        sample_indices = np.random.choice(N, size=num_samples, replace=False)
        sample_indices.sort()  # Sort indices for chronological order
        figsize = (figsize[0], figsize[1] * num_samples)  # Adjust figure height for multiple rows
    else:
        sample_indices = None
        
    # Compute PSDs for all datasets
    freq_target, psd_target = compute_psd(targets, avg_across_samples, sample_indices)
    freq_climax, psd_climax = compute_psd(climax_preds, avg_across_samples, sample_indices)
    freq_stormer, psd_stormer = compute_psd(stormer_preds, avg_across_samples, sample_indices)
    freq_unet, psd_unet = compute_psd(unet_preds, avg_across_samples, sample_indices)
    
    # Create figure
    fig, axes = plt.subplots(num_samples if not avg_across_samples else 1, 3, figsize=figsize)
    if avg_across_samples:
        axes = axes.reshape(1, -1)
    
    # Colors and line styles
    styles = {
        'Ground Truth': {'color': 'black', 'linestyle': '-', 'linewidth': 3.5},
        'ClimaX': {'color': 'red', 'linestyle': '--', 'linewidth': 3},
        'Stormer': {'color': 'blue', 'linestyle': '--', 'linewidth': 3},
        'UNet': {'color': 'green', 'linestyle': '--', 'linewidth': 3}
    }
    
    # Plot for each sample and variable
    for row in range(axes.shape[0]):
        for col in range(3):
            ax = axes[row, col]
            
            if avg_across_samples:
                # Plot averaged PSDs
                ax.loglog(freq_target[col], psd_target[col], **styles['Ground Truth'])
                ax.loglog(freq_climax[col], psd_climax[col], **styles['ClimaX'])
                ax.loglog(freq_stormer[col], psd_stormer[col], **styles['Stormer'])
                ax.loglog(freq_unet[col], psd_unet[col], **styles['UNet'])
            else:
                # Plot individual sample PSDs
                ax.loglog(freq_target[col][row], psd_target[col][row], **styles['Ground Truth'])
                ax.loglog(freq_climax[col][row], psd_climax[col][row], **styles['ClimaX'])
                ax.loglog(freq_stormer[col][row], psd_stormer[col][row], **styles['Stormer'])
                ax.loglog(freq_unet[col][row], psd_unet[col][row], **styles['UNet'])
            
            # Increase tick label sizes
            ax.tick_params(axis='both', which='major', labelsize=15)
            ax.tick_params(axis='both', which='minor', labelsize=15)
            
            # Set labels according to position
            if col == 1 and row == axes.shape[0]-1:  # bottom middle subplot
                ax.set_xlabel('Spatial Frequency', fontsize=18)
            if col == 0 and row == axes.shape[0]//2:  # middle left subplot
                ax.set_ylabel('Power Spectral Density', fontsize=18)
            
            # Set title only for top row
            if row == 0:
                ax.set_title(variable_names[col], fontsize=20)
            
            # Add datetime for multiple sample plots
            if not avg_across_samples:
                date_time = index_to_datetime(sample_indices[row].item())
                date_str = date_time.strftime('%Y-%m-%d %H:%M')
                ax.text(0.98, 0.98, date_str, 
                       transform=ax.transAxes, fontsize=14,
                       horizontalalignment='right',
                       verticalalignment='top')
            
            ax.grid(True, which="both", ls="-", alpha=0.2)
    
    # Create a single legend at the bottom center
    fig.legend(
        labels=['Ground Truth', 'ClimaX', 'Stormer', 'UNet'],
        loc='center',
        bbox_to_anchor=(0.5, -0.1 if avg_across_samples else 0.0),
        ncol=4,
        fontsize=18,
    )
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2 if avg_across_samples else 0.1)
    
    # Save the figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Create directory if needed
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate PSD comparison plots for climate models')
    
    # Data paths
    parser.add_argument('--target_path', type=str, default='/eagle/MDClimSim/tungnd/atmost-arena/downscaling/targets.npy',
                        help='Path to target data file')
    parser.add_argument('--climax_path', type=str, default='/eagle/MDClimSim/tungnd/atmost-arena/downscaling/climax_preds.npy',
                        help='Path to ClimaX predictions file')
    parser.add_argument('--stormer_path', type=str, default='/eagle/MDClimSim/tungnd/atmost-arena/downscaling/stormer_preds.npy',
                        help='Path to Stormer predictions file')
    parser.add_argument('--unet_path', type=str, default='/eagle/MDClimSim/tungnd/atmost-arena/downscaling/unet_preds.npy',
                        help='Path to UNet predictions file')
    
    # Plot settings
    parser.add_argument('--save_path', type=str, default='psd_comparison.png',
                        help='Path to save the output figure')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for the output figure')
    parser.add_argument('--figsize_width', type=float, default=15,
                        help='Figure width in inches')
    parser.add_argument('--figsize_height', type=float, default=5,
                        help='Figure height in inches')
    
    # Analysis settings
    parser.add_argument('--avg_samples', action='store_true',
                        help='Average PSD across all samples')
    parser.add_argument('--num_samples', type=int, default=3,
                        help='Number of random samples to plot when not averaging')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sample selection')
    
    args = parser.parse_args()
    
    # Load data
    targets = np.load(args.target_path)
    climax_preds = np.load(args.climax_path)
    stormer_preds = np.load(args.stormer_path)
    unet_preds = np.load(args.unet_path)
    
    # Generate and save plot
    plot_model_comparison_psd(
        targets, 
        climax_preds, 
        stormer_preds, 
        unet_preds,
        save_path=args.save_path,
        dpi=args.dpi,
        avg_across_samples=args.avg_samples,
        num_samples=args.num_samples,
        seed=args.seed,
        figsize=(args.figsize_width, args.figsize_height)
    )

if __name__ == "__main__":
    main()