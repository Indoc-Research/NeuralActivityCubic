import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd


from typing import List
from pathlib import Path

from .analysis import Square


def plot_activity_overview(squares_with_sufficient_activity: List[Square], preview_image: np.ndarray, row_cropping_idx: int, col_cropping_idx: int, window_size: int):
    fig, ax = plt.subplots()
    ax.imshow(preview_image, cmap="gray")
    peak_counts = [square.peaks_count for square in squares_with_sufficient_activity]
    max_peak_count = max(peak_counts)
    for square, peak_count in zip(squares_with_sufficient_activity, peak_counts):
        true_size = (peak_count/max_peak_count)*(window_size/2)
        circle = plt.Circle((square.center_coords[1], square.center_coords[0]), radius=true_size, fill=False, color='cyan', linewidth = 2)
        ax.add_patch(circle)
    ax.grid(color = 'gray', linestyle = '--', linewidth = 1)
    plt.hlines([0, row_cropping_idx], xmin=0, xmax=col_cropping_idx, color = 'magenta', linewidth = 2)
    plt.vlines([0, col_cropping_idx], ymin=0, ymax=row_cropping_idx, color = 'magenta', linewidth = 2)
    ax.set_xticks(np.arange(0, preview_image.shape[1], window_size), labels = [])
    ax.set_xticks(np.arange(window_size/2, col_cropping_idx + window_size/2, window_size), labels = np.arange(1, col_cropping_idx/window_size + 1, 1, dtype='int'), minor = True)
    ax.xaxis.set_label_text('X')
    ax.set_yticks(np.arange(0,  preview_image.shape[0], window_size), labels = [])
    ax.set_yticks(np.arange(window_size/2, row_cropping_idx + window_size/2, window_size), labels = np.arange(1, row_cropping_idx/window_size + 1, 1, dtype='int'), minor = True)
    ax.yaxis.set_label_text('Y')
    ax.tick_params(bottom = False, left = False)
    ax.set_title(f'Total activity: {np.sum(peak_counts)}')
    return fig, ax



def plot_intensity_trace_with_identified_peaks_for_individual_square(square: Square) -> None:
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
    df_all_peak_results_one_square.columns = ['peak frame index', 'peak bit value', 'peak dF/F',  'peak AUC', 'peak classification']
    df_all_peak_results_one_square.insert(loc = 0, column = 'square coordinates [X / Y]', value = f'[{square.grid_col_label} / {square.grid_row_label}]')
    return df_all_peak_results_one_square



def create_single_square_delta_f_over_f_results(df_all_results_single_square: pd.DataFrame, zfill_factor: int) -> pd.DataFrame:
    rearranged_data = {'square coordinates [X / Y]': [df_all_results_single_square['square coordinates [X / Y]'].iloc[0]],
                       'total peak count': [df_all_results_single_square.shape[0]]}
    for i in range(df_all_results_single_square.shape[0]):
        peak_idx = str(i + 1)
        peak_idx_suffix = peak_idx.zfill(zfill_factor)
        rearranged_data[f'frame index peak #{peak_idx_suffix}'] = [df_all_results_single_square.iloc[i]['peak frame index']]
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

