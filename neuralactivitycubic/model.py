from pathlib import Path
import multiprocessing
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .analysis import Square, process_squares
from . import results
from .io import RecordingLoaderFactory

from typing import List, Tuple
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes 



class Model:

    def __init__(self) -> None:
        self.num_processes = multiprocessing.cpu_count()
        

    def load_recording(self, recording_filepath: Path) -> None:
        recording_loader = RecordingLoaderFactory().get_loader(recording_filepath)
        self.recording_zstack = recording_loader.load_all_frames()
        self.estimated_bit_depth = self._estimate_bit_depth()
        self.recording_preview = self._create_brightness_and_contrast_enhanced_preview()


    def _estimate_bit_depth(self) -> int:
        max_bit_value = self.recording_zstack.max()
        if max_bit_value <= 255:
            estimated_bit_depth = 255
        elif max_bit_value <= 4095:
            estimated_bit_depth = 4095
        elif max_bit_value <= 65535:
            estimated_bit_depth = 65535
        else:
            raise ValueError(f'Max bit value in recording found to be {max_bit_value}, but NA3 currently only handles up to 16-bit recordings!')
        return estimated_bit_depth


    def _create_brightness_and_contrast_enhanced_preview(self, percentile_for_adjustment: int=1) -> np.ndarray:
        raw_image = self.recording_zstack[0, :, :, :].copy() # ensure that dimensions are the same as for ".get_single_frame_as_preview()"
        lower_percentile_bit_value = np.percentile(raw_image, percentile_for_adjustment)
        upper_percentile_bit_value = np.percentile(raw_image, 100-percentile_for_adjustment)
        contrast_adjustment_factor = self.estimated_bit_depth / (upper_percentile_bit_value - lower_percentile_bit_value)
        brightness_adjustment_factor = -(contrast_adjustment_factor * lower_percentile_bit_value)
        raw_image_clipped_at_percentile_bit_values = self._clip_image_at_bit_values(raw_image, lower_percentile_bit_value, upper_percentile_bit_value)
        brightness_contrast_adjusted_image = contrast_adjustment_factor * raw_image_clipped_at_percentile_bit_values + brightness_adjustment_factor
        return brightness_contrast_adjusted_image
        

    def _compute_contrast_and_brightness_adjustment_factors(self, raw_image: np.ndarray, percentile_for_adjustment: int=1) -> Tuple[float, float]:
        lower_percentile_bit_value = np.percentile(raw_image, percentile_for_adjustment)
        upper_percentile_bit_value = np.percentile(raw_image, 100-percentile_for_adjustment)
        contrast_adjustment = self.estimated_bit_depth / (upper_percentile_bit_value - lower_percentile_bit_value)
        brightness_adjustment = -(contrast_adjustment * lower_percentile_bit_value)
        return contrast_adjustment, brightness_adjustment


    def _clip_image_at_bit_values(self, raw_image: np.ndarray, min_bit_value: float, max_bit_value: float) -> np.ndarray:
        raw_image[raw_image <= min_bit_value] = min_bit_value
        raw_image[raw_image >= max_bit_value] = max_bit_value
        return raw_image

        

    def load_roi(self, roi_filepath: Path) -> None:
        # roi_loader = ROILoaderFactory().get_loader(roi_filepath)
        # self.rois = roi_loader.load_all_rois()
        # self._create_preview_with_superimposed_rois()
        pass


    def preview_window_size(self, window_size: int) -> Tuple[Figure, Axes]:
        self._clear_all_data_from_previous_analyses()
        self._create_squares(window_size)
        preview_fig, preview_ax = results.plot_window_size_preview(self.recording_preview, self.row_cropping_idx, self.col_cropping_idx, window_size)
        return preview_fig, preview_ax


    def run_analysis(self,
                     window_size: int,
                     limit_analysis_to_frame_interval: bool,
                     start_frame_idx: int,
                     end_frame_idx: int,
                     signal_average_threshold: float,
                     signal_to_noise_ratio: float,
                     octaves_ridge_needs_to_spann: float,
                     noise_window_size: int,
                     baseline_estimation_method: str,
                     #include_variance: bool,
                     #variance: float,
                    ) -> None:
        self._clear_all_data_from_previous_analyses()
        self._set_analysis_start_datetime()
        self._create_squares(window_size)
        configs = locals()
        configs.pop('self')
        with multiprocessing.Pool(processes = self.num_processes) as pool:
            # check if following line can instead refer to original list - maybe the individual elements are also directly overwritten? 
            # consider changing into for loop if neccessary & possible
            processed_squares = pool.starmap(process_squares, [(square, configs) for square in self.squares])
        self.processed_squares = processed_squares


    def _clear_all_data_from_previous_analyses(self) -> None:
        for attribute_name in ['squares', 'processed_squares', 'row_cropping_idx', 'col_cropping_idx', 'results_subdir_path']:
            if hasattr(self, attribute_name):
                delattr(self, attribute_name)


    def _set_analysis_start_datetime(self) -> None:
            users_local_timezone = datetime.now().astimezone().tzinfo
            self.analysis_start_datetime = datetime.now(users_local_timezone)      


    def _create_squares(self, window_size: int) -> None:
        self.row_cropping_idx, self.col_cropping_idx = self._get_cropping_indices_to_adjust_for_window_size(window_size)
        upper_left_pixel_idxs_of_squares_in_grid, grid_cell_labels = self._get_positions_for_squares_in_grid(window_size)
        self.squares = []
        for (upper_left_row_pixel_idx, upper_left_col_pixel_idx), grid_cell_label in zip(upper_left_pixel_idxs_of_squares_in_grid, grid_cell_labels):
            square_row_coords_slice = slice(upper_left_row_pixel_idx, upper_left_row_pixel_idx + window_size)
            square_col_coords_slice = slice(upper_left_col_pixel_idx, upper_left_col_pixel_idx + window_size)
            zstack_within_square = self.recording_zstack[:, square_row_coords_slice, square_col_coords_slice, :]
            self.squares.append(Square(grid_cell_label, (upper_left_row_pixel_idx, upper_left_col_pixel_idx), zstack_within_square))

    
    def _get_cropping_indices_to_adjust_for_window_size(self, window_size: int) -> Tuple[int, int]:
        row_cropping_index = (self.recording_preview.shape[0] // window_size) * window_size
        col_cropping_index = (self.recording_preview.shape[1] // window_size) * window_size
        return row_cropping_index, col_cropping_index

    
    def _get_positions_for_squares_in_grid(self, window_size: int) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        pixel_idxs_of_grid_rows = np.arange(start = 0, stop = self.row_cropping_idx, step = window_size)
        pixel_idxs_of_grid_cols = np.arange(start = 0, stop = self.col_cropping_idx, step = window_size)
        grid_row_labels = np.arange(start = 1, stop = self.row_cropping_idx / window_size + 1, step = 1, dtype = 'int')
        grid_col_labels = np.arange(start = 1, stop = self.col_cropping_idx / window_size + 1, step = 1, dtype = 'int')
        upper_left_pixel_idxs_of_squares_in_grid = []
        grid_cell_labels = []
        for row_pixel_idx, row_label in zip(pixel_idxs_of_grid_rows, grid_row_labels):
            for col_pixel_idx, col_label in zip(pixel_idxs_of_grid_cols, grid_col_labels):
                upper_left_pixel_idxs_of_squares_in_grid.append((row_pixel_idx, col_pixel_idx))
                grid_cell_labels.append((row_label, col_label))
        return upper_left_pixel_idxs_of_squares_in_grid, grid_cell_labels


    def create_overview_results(self,
                                minimum_activity_counts: int,
                                window_size: int,
                                save_overview_png: bool,
                                results_filepath: Path
                               ) -> Tuple[Figure, Axes]:
        filtered_squares = [square for square in self.processed_squares if square.peaks_count >= minimum_activity_counts]
        overview_fig, ax = results.plot_activity_overview(filtered_squares, self.recording_preview, self.row_cropping_idx, self.col_cropping_idx, window_size, True)
        if save_overview_png == True:
            self._ensure_results_subdir_for_current_analysis_exists(results_filepath)
            overview_fig.savefig(self.results_subdir_path.joinpath('overview.png'))
        return overview_fig, ax


    def _ensure_results_subdir_for_current_analysis_exists(self, results_filepath: Path) -> None:
        if hasattr(self, 'results_subdir_path') == False:
            self.results_subdir_path = results_filepath.joinpath(self.analysis_start_datetime.strftime('%Y_%m_%d_%H-%M-%S_NA3_results'))
            self.results_subdir_path.mkdir()


    def create_detailed_results(self,
                                save_detailed_results: bool,
                                window_size: int,
                                signal_average_threshold: float,
                                signal_to_noise_ratio: float,
                                minimum_activity_counts: int,
                                results_filepath: Path
                               ) -> None:
        if save_detailed_results == True:
            self._ensure_results_subdir_for_current_analysis_exists(results_filepath)
            self._create_csv_result_files(minimum_activity_counts)
            filename = f'Plots_WS-{window_size}_SNR-{signal_to_noise_ratio}_SAT-{signal_average_threshold}_MAC-{minimum_activity_counts}.pdf'
            filepath = self.results_subdir_path.joinpath(filename)
            with PdfPages(filepath) as pdf:
                filtered_squares = [square for square in self.processed_squares if square.peaks_count >= minimum_activity_counts]
                for indicate_activity in [True, False]:
                    overview_fig, ax = results.plot_activity_overview(filtered_squares, self.recording_preview, self.row_cropping_idx, self.col_cropping_idx, window_size, indicate_activity)
                    pdf.savefig(overview_fig)
                    plt.close()
                for square in filtered_squares:
                    fig = results.plot_intensity_trace_with_identified_peaks_for_individual_square(square)
                    pdf.savefig(fig)
                    plt.close()
        #with multiprocessing.Pool(processes = self.num_processes) as pool:
            # check and align with multiprocessing of square processing above
            #pool.starmap(plot_intensity_trace_with_identified_peaks_for_individual_square, [(square, user, settings) for square in self.processed_squares])


    def _create_csv_result_files(self, minimum_activity_counts: int) -> None:
        peak_results_per_square = [results.export_peak_results_df_from_square(square) for square in self.processed_squares if square.peaks_count >= minimum_activity_counts]
        df_all_peak_results = pd.concat(peak_results_per_square, ignore_index = True)
        max_peak_count_across_all_squares = df_all_peak_results.groupby('square coordinates [X / Y]').count()['peak frame index'].max()
        zfill_factor = int(np.log10(max_peak_count_across_all_squares)) + 1
        amplitude_and_delta_f_over_f_results_all_squares = []
        auc_results_all_squares = []
        for square_coords in df_all_peak_results['square coordinates [X / Y]'].unique():
            tmp_df_single_square = df_all_peak_results[df_all_peak_results['square coordinates [X / Y]'] == square_coords].copy()
            amplitude_and_delta_f_over_f_results_all_squares.append(results.create_single_square_amplitude_and_delta_f_over_f_results(tmp_df_single_square, zfill_factor))
            auc_results_all_squares.append(results.create_single_square_auc_results(tmp_df_single_square, zfill_factor))
        df_all_amplitude_and_delta_f_over_f_results = pd.concat(amplitude_and_delta_f_over_f_results_all_squares, ignore_index = True)
        df_all_auc_results = pd.concat(auc_results_all_squares, ignore_index = True)
        # Once all DataFrames are created successfully, write them to disk 
        df_all_peak_results.to_csv(self.results_subdir_path.joinpath('all_peak_results.csv'), index = False)
        df_all_amplitude_and_delta_f_over_f_results.to_csv(self.results_subdir_path.joinpath('Amplitude_and_dF_over_F_results.csv'), index = False)
        df_all_auc_results.to_csv(self.results_subdir_path.joinpath('AUC_results.csv'), index = False)


    

    def _create_preview_with_superimposed_rois(self) -> None:
        #self.recording_preview_with_superimposed_rois = 
        pass