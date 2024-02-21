import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from typing import List

from .analysis import Square


def plot_activity_overview(all_squares: List[Square], preview_image: np.ndarray, window_size: int) -> None:
    sizes = [square.peaks_count for square in all_squares]
    # creating the plot
    fig, ax = plt.subplots()
    # drawing the circles
    for square in all_squares:
        true_size = (square.peaks_count/max(sizes))*(window_size/2)
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



def plot_intensity_trace_with_identified_peaks_for_individual_square(square: Square, user, settings, bla, bli, blubb) -> None:
    plt.figure(figsize = (12, 3), facecolor = 'white')
    plt.plot(square.mean_intensity_over_time, zorder = 0, c='black')
    plt.scatter(x = square.frame_idxs_of_peaks, y = square.mean_intensity_over_time[square.frame_idxs_of_peaks], color = 'red')
    plt.ylabel('mean bit value')
    plt.xlabel('frame index')
    plt.title(f'Graph "x-idx", "y-idx"    Total Activity: {square.peaks_count}')
    plt.tight_layout()
    # plt.savefig('asdf.png') # include square coords in grid in here