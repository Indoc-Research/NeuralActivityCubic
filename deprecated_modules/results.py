import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
from shapely import get_coordinates
from typing import List, Tuple, Optional, Union, Any, Dict, Never
from pathlib import Path
from matplotlib.text import Text
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes 

from .input import ROI
from .analysis import AnalysisROI



def plot_roi_boundaries(roi: Union[ROI, AnalysisROI], line_color: str, line_style: str, line_width: Union[int, float]) -> None:
    roi_boundary_row_col_coords = np.asarray(roi.boundary_row_col_coords)
    plt.plot(roi_boundary_row_col_coords[:, 1], roi_boundary_row_col_coords[:, 0], c = line_color, linestyle = line_style, linewidth = line_width)


def _use_grid_row_col_labels_as_axis_labels(image_shape: Tuple[int, int], grid_configs: Dict[str, Any], ax: Axes, font_size: int=12) -> None:
    ax.set_xticks(np.arange(0, image_shape[1], grid_configs['window_size']), labels = [])
    ax.set_xticks(np.arange(grid_configs['window_size']/2, grid_configs['col_cropping_idx'] + grid_configs['window_size']/2, grid_configs['window_size']), 
                  labels = np.arange(1, grid_configs['col_cropping_idx']/grid_configs['window_size'] + 1, 1, dtype='int'),
                  minor = True,
                  fontsize = min(12, font_size))
    ax.xaxis.set_label_text('X')
    ax.set_yticks(np.arange(0,  image_shape[0], grid_configs['window_size']), labels = [])
    ax.set_yticks(np.arange(grid_configs['window_size']/2, grid_configs['row_cropping_idx'] + grid_configs['window_size']/2, grid_configs['window_size']), 
                  labels = np.arange(1, grid_configs['row_cropping_idx']/grid_configs['window_size'] + 1, 1, dtype='int'), 
                  minor = True, 
                  fontsize = min(12, font_size))
    ax.yaxis.set_label_text('Y')
    ax.tick_params(bottom = False, left = False)


def _plot_grid(grid_configs: Dict[str, Any], 
               grid_line_color: str,
               grid_line_style: str,
               grid_line_width: int,
               focus_area_roi: Union[ROI, None],
               area_line_color: str,
               area_line_style: str,
               area_line_width: int,
               ax: Axes
              ) -> None:
    ax.grid(color = grid_line_color, linestyle = grid_line_style, linewidth = grid_line_width)
    if focus_area_roi == None:
        plt.hlines([0, grid_configs['row_cropping_idx']], 
                   xmin = 0, 
                   xmax = grid_configs['col_cropping_idx'], 
                   color = area_line_color,
                   linestyle = area_line_style,
                   linewidth = area_line_width)
        plt.vlines([0, grid_configs['col_cropping_idx']],
                   ymin = 0, 
                   ymax = grid_configs['row_cropping_idx'], 
                   color = area_line_color,
                   linestyle = area_line_style,
                   linewidth = area_line_width)
    else:
        plot_roi_boundaries(focus_area_roi, area_line_color, area_line_style, area_line_width)    



def plot_window_size_preview(preview_image: np.ndarray, 
                             grid_configs: Dict[str, Any], 
                             focus_area_roi: Optional[ROI]=None
                            ) ->Tuple[Figure, Axes]:
    fig, ax = plt.subplots()
    ax.imshow(preview_image, cmap="gray") # vmin = np.percentile(preview_image, 2.5), vmax = np.percentile(preview_image, 97.5)
    _plot_grid(grid_configs, 'gray', 'dashed', 1, focus_area_roi, 'cyan', 'solid', 2, ax)
    _use_grid_row_col_labels_as_axis_labels(preview_image.shape, grid_configs, ax) 
    ax.set_title(f'Preview of grid for window size: {grid_configs["window_size"]}')
    return fig, ax


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


def _get_fontsize_adjusted_to_slimmest_roi(preview_image: np.ndarray, min_width: int, max_text_length: int, default_fontsize: int=25) -> float:
    max_text_size = min_width*0.75
    tmp_fig, tmp_ax = plt.subplots()
    tmp_ax.imshow(preview_image)
    sample_coord = min_width + 0.5 * min_width
    max_width_text_from_number = '4'*max_text_length
    tmp_text = tmp_ax.text(sample_coord, sample_coord, max_width_text_from_number, fontsize = default_fontsize)
    text_width, text_height = _get_text_bounding_box_size_in_data_dimensions(tmp_text, tmp_fig, tmp_ax)
    if (text_width > max_text_size) or (text_height > max_text_size):
        _iteratively_decrease_fontsize_to_fit_text_in_squares(tmp_text, max_text_size, tmp_fig, tmp_ax)
    adjusted_fontsize = tmp_text.get_fontsize()
    plt.close(tmp_fig)
    return adjusted_fontsize


def _get_min_width_among_all_rois(analysis_rois: List[AnalysisROI], grid_configs: Optional[Dict[str, Any]]) -> int:
    if grid_configs != None:
        min_width = grid_configs['window_size']
    else:
        all_roi_widths = []
        for roi in analysis_rois:
            bounding_box_col_coords = get_coordinates(roi.as_polygon.envelope)[:, 1]
            min_col_idx = bounding_box_col_coords.min()
            max_col_idx = bounding_box_col_coords.max()
            width = max_col_idx - min_col_idx
            all_roi_widths.append(width)
        min_width = min(all_roi_widths)
    return min_width


def _get_all_peak_counts(analysis_rois: List[AnalysisROI]) -> List[int]:
    all_peak_counts = [roi.peaks_count for roi in analysis_rois]
    return all_peak_counts


def _get_max_peak_count(analysis_rois: List[AnalysisROI]) -> int:
    all_peak_counts = _get_all_peak_counts(analysis_rois)
    if len(all_peak_counts) > 0:
        max_peak_count = max(all_peak_counts)
    else:
        max_peak_count = 0
    return max_peak_count  


def _get_total_peak_count(analysis_rois: List[AnalysisROI]) -> int:
    all_peak_counts = _get_all_peak_counts(analysis_rois)
    if len(all_peak_counts) == 0:
        total_peak_count = 0
    else:
        total_peak_count = sum(all_peak_counts)
    return total_peak_count


def _plot_text_at_roi_centroid_coordinates(analysis_roi: AnalysisROI, text: str, size: int, color: str, ax: Axes) -> None:
        ax.text(analysis_roi.centroid_row_col_coords[1], 
                analysis_roi.centroid_row_col_coords[0], 
                text, 
                color = color, 
                horizontalalignment='center', 
                verticalalignment = 'center', 
                fontsize = size)    


def _plot_peak_count_text_for_each_analysis_roi(analysis_rois: List[AnalysisROI], preview_image: np.ndarray, grid_configs: Optional[Dict[str, Any]], ax: Axes) -> None:
    max_peak_count = _get_max_peak_count(analysis_rois)
    max_len_peak_count_text = len(str(max_peak_count))
    min_width_among_all_rois = _get_min_width_among_all_rois(analysis_rois, grid_configs)
    font_size = _get_fontsize_adjusted_to_slimmest_roi(preview_image, min_width_among_all_rois, max_len_peak_count_text)
    for roi in analysis_rois:
        _plot_text_at_roi_centroid_coordinates(roi, roi.peaks_count, font_size, 'magenta', ax)


def plot_activity_overview(analysis_rois_with_sufficient_activity: Union[List[AnalysisROI], List[Never]],
                           preview_image: np.ndarray,
                           indicate_activity: bool=False,
                           focus_area: Optional[ROI]=None,
                           grid_configs: Optional[Dict[str, Any]]=None
                          ) -> Tuple[Figure, Axes]:
    fig, ax = plt.subplots()
    ax.imshow(preview_image, cmap="gray")
    if len(analysis_rois_with_sufficient_activity) > 0:
        if indicate_activity == True:
            _plot_peak_count_text_for_each_analysis_roi(analysis_rois_with_sufficient_activity, preview_image, grid_configs, ax)
        if grid_configs != None:
            max_peak_count = _get_max_peak_count(analysis_rois_with_sufficient_activity)
            max_len_peak_count_text = len(str(max_peak_count))
            font_size = _get_fontsize_adjusted_to_slimmest_roi(preview_image, grid_configs['window_size'], max_len_peak_count_text)
            _plot_grid(grid_configs, 'gray', 'dashed', 1, focus_area, 'cyan', 'solid', 2, ax)
            _use_grid_row_col_labels_as_axis_labels(preview_image.shape, grid_configs, ax, font_size)
        else:
            for roi in analysis_rois_with_sufficient_activity:
                plot_roi_boundaries(roi, 'magenta', 'solid', 1)
    ax.set_title(f'Total activity: {_get_total_peak_count(analysis_rois_with_sufficient_activity)}')
    return fig, ax


def _get_max_label_id_length(analysis_rois_with_sufficient_activity: List[AnalysisROI], grid_configs: Union[None, Dict[str, Any]]) -> int:
    if grid_configs != None:
        max_label_id_length = 1 + grid_configs['max_len_row_label_id'] + grid_configs['max_len_col_label_id']
    else:
        max_label_id_length = len(analysis_rois_with_sufficient_activity[0].label_id)
    return max_label_id_length


def plot_rois_with_label_id_overview(analysis_rois_with_sufficient_activity: Union[List[AnalysisROI], List[Never]],
                                     preview_image: np.ndarray,
                                     focus_area: Optional[ROI]=None,
                                     grid_configs: Optional[Dict[str, Any]]=None
                                    ) -> Tuple[Figure, Axes]:
    fig, ax = plt.subplots()
    image_row_col_dimensions = (preview_image.shape[0], preview_image.shape[1])
    blank_image = np.zeros(image_row_col_dimensions)
    ax.imshow(blank_image, cmap='gray_r')
    if len(analysis_rois_with_sufficient_activity) > 0:
        max_label_id_length = _get_max_label_id_length(analysis_rois_with_sufficient_activity, grid_configs)
        min_width_among_all_rois = _get_min_width_among_all_rois(analysis_rois_with_sufficient_activity, grid_configs)
        font_size = _get_fontsize_adjusted_to_slimmest_roi(blank_image, min_width_among_all_rois, max_label_id_length)
        for roi in analysis_rois_with_sufficient_activity:
            _plot_text_at_roi_centroid_coordinates(roi, roi.label_id, font_size, 'black', ax)
        if grid_configs != None:
            _plot_grid(grid_configs, 'black', 'solid', 1, focus_area, 'gray', 'dashed', 2, ax)
            _use_grid_row_col_labels_as_axis_labels(preview_image.shape, grid_configs, ax, font_size)
        else:
            for roi in analysis_rois_with_sufficient_activity:
                plot_roi_boundaries(roi, 'black', 'solid', 1)
    ax.set_title('IDs of ROIs with sufficient activity:')
    return fig, ax


def plot_intensity_trace_with_identified_peaks_for_individual_roi(analysis_roi: AnalysisROI) -> Figure:
    fig = plt.figure(figsize = (9, 2.67), facecolor = 'white')
    plt.plot(analysis_roi.mean_intensity_over_time, c = 'gray')
    if hasattr(analysis_roi, 'baseline'):
        plt.plot(analysis_roi.baseline, c = 'cyan')
    for peak in analysis_roi.peaks.values():
        if peak.has_neighboring_intersections == True:
            plt.plot(peak.frame_idx, peak.intensity, 'mo')
            start_idx = peak.frame_idxs_of_neighboring_intersections[0] - 1
            if start_idx < 0:
                start_idx = 0
            end_idx = peak.frame_idxs_of_neighboring_intersections[1] + 1
            plt.fill_between(np.arange(start_idx, end_idx, 1), 
                             analysis_roi.mean_intensity_over_time[start_idx : end_idx], 
                             analysis_roi.baseline[start_idx : end_idx], 
                             where = analysis_roi.mean_intensity_over_time[start_idx : end_idx] > analysis_roi.baseline[start_idx : end_idx], 
                             interpolate = True, 
                             color='yellow',
                             alpha = 0.6)
        else:
            plt.plot(peak.frame_idx, peak.intensity, 'ko')
    plt.title(f'Graph of ROI: {analysis_roi.label_id}   -   Total Activity: {analysis_roi.peaks_count}')
    plt.tight_layout()
    return fig


def export_peak_results_df_from_analysis_roi(analysis_roi: AnalysisROI) -> pd.DataFrame:
    all_peaks = [peak for peak in analysis_roi.peaks.values()]
    df_all_peak_results_single_roi = pd.DataFrame(all_peaks)
    df_all_peak_results_single_roi.drop(['has_neighboring_intersections', 'frame_idxs_of_neighboring_intersections'], axis = 'columns', inplace = True)
    df_all_peak_results_single_roi.columns = ['peak frame index', 'peak bit value', 'peak amplitude', 'peak dF/F',  'peak AUC', 'peak classification']
    df_all_peak_results_single_roi.insert(loc = 0, column = 'ROI label ID', value = f'{analysis_roi.label_id}')
    return df_all_peak_results_single_roi


def create_single_roi_amplitude_and_delta_f_over_f_results(df_all_results_single_roi: pd.DataFrame, zfill_factor: int) -> pd.DataFrame:
    rearranged_data = {'ROI label ID': [df_all_results_single_roi['ROI label ID'].iloc[0]],
                       'total peak count': [df_all_results_single_roi.shape[0]]}
    for i in range(df_all_results_single_roi.shape[0]):
        peak_idx = str(i + 1)
        peak_idx_suffix = peak_idx.zfill(zfill_factor)
        rearranged_data[f'frame index peak #{peak_idx_suffix}'] = [df_all_results_single_roi.iloc[i]['peak frame index']]
        rearranged_data[f'amplitude peak #{peak_idx_suffix}'] = [df_all_results_single_roi.iloc[i]['peak amplitude']]
        rearranged_data[f'dF/F peak #{peak_idx_suffix}'] = [df_all_results_single_roi.iloc[i]['peak dF/F']]
    return pd.DataFrame(rearranged_data)
    
    
def create_single_roi_auc_results(df_all_results_single_roi: pd.DataFrame, zfill_factor: int) -> pd.DataFrame:
    rearranged_data = {'ROI label ID': [df_all_results_single_roi['ROI label ID'].iloc[0]],
                       'total count all peaks': [df_all_results_single_roi.shape[0]],
                       'total count "singular" peaks': [df_all_results_single_roi['peak classification'].str.count('singular').sum()],
                       'total count "clustered" peaks': [df_all_results_single_roi['peak classification'].str.count('clustered').sum()],
                       'total count "isolated" peaks': [df_all_results_single_roi['peak classification'].str.count('isolated').sum()]}
    for i in range(df_all_results_single_roi.shape[0]):
        peak_idx = str(i + 1)
        peak_idx_suffix = peak_idx.zfill(zfill_factor)
        rearranged_data[f'frame index peak #{peak_idx_suffix}'] = [df_all_results_single_roi.iloc[i]['peak frame index']]
        rearranged_data[f'AUC peak #{peak_idx_suffix}'] = [df_all_results_single_roi.iloc[i]['peak AUC']]
        rearranged_data[f'classification peak #{peak_idx_suffix}'] = [df_all_results_single_roi.iloc[i]['peak classification']]
    return pd.DataFrame(rearranged_data)
