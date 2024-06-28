import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd

from typing import List, Tuple, Optional
from pathlib import Path
from matplotlib.text import Text
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes 

from .analysis import Square
from .input import ROI


def _get_text_bounding_box_size_in_data_dimensions(text: Text, fig: Figure, ax: Axes) -> Tuple[float, float]:
    renderer = fig.canvas.get_renderer()
    text_bbox_raw_dimensions = text.get_window_extent(renderer=renderer)
    text_bbox_data_dimensions = Bbox(ax.transData.inverted().transform(text_bbox_raw_dimensions))
    return np.abs(text_bbox_data_dimensions.width), np.abs(text_bbox_data_dimensions.height)


def _iteratively_decrease_fontsize_to_fit_text_in_squares(text: Text, max_size: float, fig: Figure, ax: Axes) -> None:
    text.set_fontsize(text.get_fontsize()-1)
    text_width, text_height = _get_text_bounding_box_size_in_data_dimensions(text, fig, ax)
    if (text_width > max_size) or (text_height > max_size): 
        _iteratively_decrease_fontsize_to_fit_text_in_squares(text, max_size, fig, ax)


def _get_adjusted_fontsize(preview_image: np.ndarray, window_size: int, max_peak_count: int, default_fontsize: int=25) -> float:
    max_text_size = window_size*0.75
    tmp_fig, tmp_ax = plt.subplots()
    tmp_ax.imshow(preview_image)
    sample_coord = window_size + 0.5 * window_size
    max_width_text_from_number = '4'*len(str(max_peak_count))
    tmp_text = tmp_ax.text(sample_coord, sample_coord, max_width_text_from_number, fontsize = default_fontsize)
    text_width, text_height = _get_text_bounding_box_size_in_data_dimensions(tmp_text, tmp_fig, tmp_ax)
    if (text_width > max_text_size) or (text_height > max_text_size):
        _iteratively_decrease_fontsize_to_fit_text_in_squares(tmp_text, max_text_size, tmp_fig, tmp_ax)
    adjusted_fontsize = tmp_text.get_fontsize()
    plt.close(tmp_fig)
    return adjusted_fontsize


def plot_window_size_preview(preview_image: np.ndarray, window_size: int, row_cropping_idx: int, col_cropping_idx: int, roi: Optional[ROI]=None) -> Tuple[Figure, Axes]:
    fig, ax = plt.subplots()
    ax.imshow(preview_image, cmap="gray", vmin = np.percentile(preview_image, 2.5), vmax = np.percentile(preview_image, 97.5))
    ax.grid(color = 'gray', linestyle = '--', linewidth = 1)
    if roi == None:
        plt.hlines([0, row_cropping_idx], xmin=0, xmax=col_cropping_idx, color = 'cyan', linewidth = 2)
        plt.vlines([0, col_cropping_idx], ymin=0, ymax=row_cropping_idx, color = 'cyan', linewidth = 2)
    else:
        roi_boundary_coords_for_plotting = np.asarray(roi.boundary_coords)
        plt.plot(roi_boundary_coords_for_plotting[:, 1], roi_boundary_coords_for_plotting[:, 0], c = 'cyan', linewidth = 2)
    ax.set_xticks(np.arange(0, preview_image.shape[1], window_size), labels = [])
    ax.set_xticks(np.arange(window_size/2, col_cropping_idx + window_size/2, window_size), 
                  labels = np.arange(1, col_cropping_idx/window_size + 1, 1, dtype='int'), minor = True)
    ax.xaxis.set_label_text('X')
    ax.set_yticks(np.arange(0,  preview_image.shape[0], window_size), labels = [])
    ax.set_yticks(np.arange(window_size/2, row_cropping_idx + window_size/2, window_size), 
                  labels = np.arange(1, row_cropping_idx/window_size + 1, 1, dtype='int'), minor = True)
    ax.yaxis.set_label_text('Y')
    ax.tick_params(bottom = False, left = False)
    ax.set_title(f'Preview of grid for window size: {window_size}')
    return fig, ax



def plot_activity_overview(squares_with_sufficient_activity: List[Square], 
                           preview_image: np.ndarray, 
                           row_cropping_idx: int, 
                           col_cropping_idx: int, 
                           window_size: int, 
                           indicate_activity: bool=False,
                           roi: Optional[ROI]=None
                          ) -> Tuple[Figure, Axes]:
    all_peak_counts = [square.peaks_count for square in squares_with_sufficient_activity]
    if len(all_peak_counts) > 0:
        max_peak_count = max(all_peak_counts)
    else:
        max_peak_count = 0
    peak_text_fontsize = _get_adjusted_fontsize(preview_image, window_size, max_peak_count)
    fig, ax = plt.subplots()
    ax.imshow(preview_image, cmap="gray", vmin = np.percentile(preview_image, 2.5), vmax = np.percentile(preview_image, 97.5))
    if indicate_activity == True:
        for square in squares_with_sufficient_activity:
            ax.text(square.center_coords[1], 
                    square.center_coords[0], 
                    square.peaks_count, 
                    color = 'magenta', 
                    horizontalalignment='center', 
                    verticalalignment = 'center', 
                    fontsize = peak_text_fontsize)
    ax.grid(color = 'gray', linestyle = '--', linewidth = 1)
    if roi == None:
        plt.hlines([0, row_cropping_idx], xmin=0, xmax=col_cropping_idx, color = 'cyan', linewidth = 2)
        plt.vlines([0, col_cropping_idx], ymin=0, ymax=row_cropping_idx, color = 'cyan', linewidth = 2)
    else:
        roi_boundary_coords_for_plotting = np.asarray(roi.boundary_coords)
        plt.plot(roi_boundary_coords_for_plotting[:, 1], roi_boundary_coords_for_plotting[:, 0], c = 'cyan', linewidth = 2)
    ax.set_xticks(np.arange(0, preview_image.shape[1], window_size), labels = [])
    ax.set_xticks(np.arange(window_size/2, col_cropping_idx + window_size/2, window_size), 
                  labels = np.arange(1, col_cropping_idx/window_size + 1, 1, dtype='int'), 
                  minor = True, 
                  fontsize = min(12, peak_text_fontsize))
    ax.xaxis.set_label_text('X')
    ax.set_yticks(np.arange(0,  preview_image.shape[0], window_size), labels = [])
    ax.set_yticks(np.arange(window_size/2, row_cropping_idx + window_size/2, window_size), 
                  labels = np.arange(1, row_cropping_idx/window_size + 1, 1, dtype='int'), 
                  minor = True, 
                  fontsize = min(12, peak_text_fontsize))
    ax.yaxis.set_label_text('Y')
    ax.tick_params(bottom = False, left = False)
    ax.set_title(f'Total activity: {np.sum(all_peak_counts)}')
    return fig, ax


def plot_intensity_trace_with_identified_peaks_for_individual_square(square: Square) -> Figure:
    fig = plt.figure(figsize = (9, 2.67), facecolor = 'white')
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
    plt.title(f'Graph: [{square.grid_col_label} / {square.grid_row_label}]     Total Activity: {square.peaks_count}')
    plt.tight_layout()
    return fig



def export_peak_results_df_from_square(square: Square) -> pd.DataFrame:
    all_peaks = [peak for peak in square.peaks.values()]
    df_all_peak_results_one_square = pd.DataFrame(all_peaks)
    df_all_peak_results_one_square.drop(['has_neighboring_intersections', 'frame_idxs_of_neighboring_intersections'], axis = 'columns', inplace = True)
    df_all_peak_results_one_square.columns = ['peak frame index', 'peak bit value', 'peak amplitude', 'peak dF/F',  'peak AUC', 'peak classification']
    df_all_peak_results_one_square.insert(loc = 0, column = 'square coordinates [X / Y]', value = f'[{square.grid_col_label} / {square.grid_row_label}]')
    return df_all_peak_results_one_square



def create_single_square_amplitude_and_delta_f_over_f_results(df_all_results_single_square: pd.DataFrame, zfill_factor: int) -> pd.DataFrame:
    rearranged_data = {'square coordinates [X / Y]': [df_all_results_single_square['square coordinates [X / Y]'].iloc[0]],
                       'total peak count': [df_all_results_single_square.shape[0]]}
    for i in range(df_all_results_single_square.shape[0]):
        peak_idx = str(i + 1)
        peak_idx_suffix = peak_idx.zfill(zfill_factor)
        rearranged_data[f'frame index peak #{peak_idx_suffix}'] = [df_all_results_single_square.iloc[i]['peak frame index']]
        rearranged_data[f'amplitude peak #{peak_idx_suffix}'] = [df_all_results_single_square.iloc[i]['peak amplitude']]
        rearranged_data[f'dF/F peak #{peak_idx_suffix}'] = [df_all_results_single_square.iloc[i]['peak dF/F']]
    return pd.DataFrame(rearranged_data)
    

    
def create_single_square_auc_results(df_all_results_single_square: pd.DataFrame, zfill_factor: int) -> pd.DataFrame:
    rearranged_data = {'square coordinates [X / Y]': [df_all_results_single_square['square coordinates [X / Y]'].iloc[0]],
                       'total count all peaks': [df_all_results_single_square.shape[0]],
                       'total count "singular" peaks': [df_all_results_single_square['peak classification'].str.count('singular').sum()],
                       'total count "clustered" peaks': [df_all_results_single_square['peak classification'].str.count('clustered').sum()],
                       'total count "isolated" peaks': [df_all_results_single_square['peak classification'].str.count('isolated').sum()]}
    for i in range(df_all_results_single_square.shape[0]):
        peak_idx = str(i + 1)
        peak_idx_suffix = peak_idx.zfill(zfill_factor)
        rearranged_data[f'frame index peak #{peak_idx_suffix}'] = [df_all_results_single_square.iloc[i]['peak frame index']]
        rearranged_data[f'AUC peak #{peak_idx_suffix}'] = [df_all_results_single_square.iloc[i]['peak AUC']]
        rearranged_data[f'classification peak #{peak_idx_suffix}'] = [df_all_results_single_square.iloc[i]['peak classification']]
    return pd.DataFrame(rearranged_data)

