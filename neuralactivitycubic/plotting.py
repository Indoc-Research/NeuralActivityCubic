import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np


from typing import List
from pathlib import Path

from .analysis import Square


def plot_activity_overview(all_squares: List[Square], preview_image: np.ndarray, window_size: int, minimum_activity_counts: int) -> None:
    filtered_peak_counts = []
    for square in all_squares:
        if hasattr(square, 'peaks_count') == True:
            if square.peaks_count >= minimum_activity_counts:
                filtered_peak_counts.append(square.peaks_count)
            else:
                filtered_peak_counts.append(0)
        else:
            filtered_peak_counts.append(0)
    # creating the plot
    fig, ax = plt.subplots()
    # drawing the circles
    max_peak_count = max(filtered_peak_counts)
    for square, peak_count in zip(all_squares, filtered_peak_counts):
        true_size = (peak_count/max_peak_count)*(window_size/2)
        circle = plt.Circle((square.center_coords[1], square.center_coords[0]), radius=true_size, fill=False, color='red')
        ax.add_patch(circle)
    
    # putting the image in the background
    ax.imshow(preview_image, cmap="gray")
    
    # putting the grid on top of the image
    ax.grid(color = 'gray', linestyle = '--', linewidth = 1)
    
    # setting the grid size to the square size
    ax.xaxis.set_major_locator(MultipleLocator(window_size))
    ax.yaxis.set_major_locator(MultipleLocator(window_size))
    # plt.savefig('overview.png')
    plt.show()



def plot_intensity_trace_with_identified_peaks_for_individual_square(square: Square, filepath: Path) -> None:
    plt.figure(figsize = (9, 2.67), facecolor = 'white')
    plt.plot(square.mean_intensity_over_time, c = 'gray')
    if hasattr(square, 'baseline'):
        plt.plot(square.baseline, c = 'cyan')
    for peak in square.peaks.values():
        if peak.has_neighboring_intersections == True:
            plt.plot(peak.frame_idx, peak.intensity, 'mo')
            start_idx = peak.frame_idxs_of_neighboring_intersections[0] - 1
            if start_idx < 0:
                start_idx = 0
            end_idx = peak.frame_idxs_of_neighboring_intersections[1] + 1
            plt.fill_between(np.arange(start_idx, end_idx, 1), 
                             square.mean_intensity_over_time[start_idx : end_idx], 
                             square.baseline[start_idx : end_idx], 
                             where = square.mean_intensity_over_time[start_idx : end_idx] > square.baseline[start_idx : end_idx], 
                             interpolate = True, 
                             color='yellow',
                             alpha = 0.6)
        else:
            plt.plot(peak.frame_idx, peak.intensity, 'ko')
    plt.title(f'Graph: [{square.idx}]     Total Activity: {square.peaks_count}')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()