import multiprocessing
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from datetime import datetime, timezone

from typing import Optional, Tuple, Dict, List, Any

from .input import RecordingLoader, Recording, ROILoader, ROI
from . import results
from .analysis import Square


def process_squares(square: Square, configs: Dict[str, Any]) -> Square:
    square.compute_mean_intensity_timeseries(configs['limit_analysis_to_frame_interval'], configs['start_frame_idx'], configs['end_frame_idx'])
    if np.mean(square.mean_intensity_over_time) >= configs['signal_average_threshold']:
        square.detect_peaks(configs['signal_to_noise_ratio'], configs['octaves_ridge_needs_to_spann'], configs['noise_window_size'])
        square.estimate_baseline(configs['baseline_estimation_method'])
        square.compute_area_under_curve()
        square.compute_amplitude_and_delta_f_over_f()
    return square



class AnalysisJob:

    def __init__(self, 
                 number_of_parallel_processes: int,
                 recording_loader: RecordingLoader, 
                 roi_loader: Optional[ROILoader]=None
                ) -> None:
        self.number_of_parallel_processes = number_of_parallel_processes
        self.recording_loader = recording_loader
        self.parent_dir_path = self.recording_loader.filepath.parent
        self.roi_loader = roi_loader
        self.roi_based = (self.roi_loader != None)


    def load_data_into_memory(self) -> None:
        if hasattr(self, 'recording') == False:
            self.recording = self.recording_loader.load_data()
            if self.roi_based == True:
                self.roi = self.roi_loader.load_data()

    
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
                #variance: float
               ) -> None:
        self._set_analysis_start_datetime()
        self.load_data_into_memory()
        self.squares = self._create_squares(window_size)
        configs = locals()
        configs.pop('self')
        with multiprocessing.Pool(processes = self.number_of_parallel_processes) as pool:
            processed_squares = pool.starmap(process_squares, [(square, configs) for square in self.squares])
        self.processed_squares = processed_squares


    def _set_analysis_start_datetime(self) -> None:
            users_local_timezone = datetime.now().astimezone().tzinfo
            self.analysis_start_datetime = datetime.now(users_local_timezone)      


    def _create_squares(self, window_size: int) -> List[Square]:
        self.row_cropping_idx, self.col_cropping_idx = self._get_cropping_indices_to_adjust_for_window_size(window_size)
        upper_left_pixel_idxs_of_squares_in_grid, grid_cell_labels = self._get_positions_for_squares_in_grid(window_size)
        squares = []
        for (upper_left_row_pixel_idx, upper_left_col_pixel_idx), grid_cell_label in zip(upper_left_pixel_idxs_of_squares_in_grid, grid_cell_labels):
            square_row_coords_slice = slice(upper_left_row_pixel_idx, upper_left_row_pixel_idx + window_size)
            square_col_coords_slice = slice(upper_left_col_pixel_idx, upper_left_col_pixel_idx + window_size)
            zstack_within_square = self.recording.zstack[:, square_row_coords_slice, square_col_coords_slice, :]
            squares.append(Square(grid_cell_label, (upper_left_row_pixel_idx, upper_left_col_pixel_idx), zstack_within_square))
        if self.roi_based == True:
            squares_filtered_by_roi = self._filter_squares_based_on_roi(squares)
            squares = squares_filtered_by_roi
        return squares


    def _filter_squares_based_on_roi(self, squares: List[Square]) -> List[Square]:
        filtered_squares = [square for square in squares if square.as_polygon.intersects(self.roi.as_polygon)]
        return filtered_squares

    
    def _get_cropping_indices_to_adjust_for_window_size(self, window_size: int) -> Tuple[int, int]:
        row_cropping_index = (self.recording.preview.shape[0] // window_size) * window_size
        col_cropping_index = (self.recording.preview.shape[1] // window_size) * window_size
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


    def create_results(self, 
                       save_overview_png: bool,
                       save_detailed_results: bool,
                       minimum_activity_counts: int, 
                       window_size: int,
                       signal_average_threshold: float, 
                       signal_to_noise_ratio: float
                      ) -> None:
        self._ensure_results_dir_exists()
        filtered_squares = [square for square in self.processed_squares if square.peaks_count >= minimum_activity_counts]
        roi = self.roi if self.roi_based == True else None
        self.overview_results = results.plot_activity_overview(squares_with_sufficient_activity = filtered_squares, 
                                                               preview_image = self.recording.preview, 
                                                               row_cropping_idx = self.row_cropping_idx, 
                                                               col_cropping_idx = self.col_cropping_idx, 
                                                               window_size = window_size, 
                                                               indicate_activity = True,
                                                               roi = roi)
        if save_overview_png == True:
            self.overview_results[0].savefig(self.results_dir_path.joinpath('overview.png'))
        if save_detailed_results == True:
            self._create_and_save_csv_result_files(filtered_squares)
            self._create_and_save_individual_traces_pdf_result_file(filtered_squares, window_size)
    
        
    def _ensure_results_dir_exists(self) -> None:
        if hasattr(self, 'results_dir_path') == False:
            self.results_dir_path = self.parent_dir_path.joinpath(self.analysis_start_datetime.strftime('%Y_%m_%d_%H-%M-%S_NA3_results'))
            self.results_dir_path.mkdir()


    def _create_and_save_csv_result_files(self, filtered_squares: List[Square]) -> None:
        peak_results_per_square = [results.export_peak_results_df_from_square(square) for square in filtered_squares]
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
        df_all_peak_results.to_csv(self.results_dir_path.joinpath('all_peak_results.csv'), index = False)
        df_all_amplitude_and_delta_f_over_f_results.to_csv(self.results_dir_path.joinpath('Amplitude_and_dF_over_F_results.csv'), index = False)
        df_all_auc_results.to_csv(self.results_dir_path.joinpath('AUC_results.csv'), index = False)


    def _create_and_save_individual_traces_pdf_result_file(self, filtered_squares: List[Square], window_size: int) -> None:
            filepath = self.results_dir_path.joinpath('Individual_traces_with_identified_events.pdf')
            with PdfPages(filepath) as pdf:
                for indicate_activity in [True, False]:
                    roi = self.roi if self.roi_based == True else None
                    overview_fig, ax = results.plot_activity_overview(squares_with_sufficient_activity = filtered_squares,
                                                                      preview_image = self.recording.preview, 
                                                                      row_cropping_idx = self.row_cropping_idx, 
                                                                      col_cropping_idx = self.col_cropping_idx, 
                                                                      window_size = window_size, 
                                                                      indicate_activity = indicate_activity,
                                                                      roi = roi)
                    pdf.savefig(overview_fig)
                    plt.close()
                for square in filtered_squares:
                    fig = results.plot_intensity_trace_with_identified_peaks_for_individual_square(square)
                    pdf.savefig(fig)
                    plt.close()