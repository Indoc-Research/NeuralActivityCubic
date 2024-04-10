from pathlib import Path
import multiprocessing
import numpy as np
import pandas as pd

from .analysis import Square, process_squares
from . import results
from .io import RecordingLoaderFactory

from typing import List, Tuple



class Model:

    def __init__(self) -> None:
        self.squares = []
        self.num_processes = multiprocessing.cpu_count()
        

    def load_recording(self, recording_filepath: Path) -> None:
        recording_loader = RecordingLoaderFactory().get_loader(recording_filepath)
        self.recording_zstack = recording_loader.load_all_frames()
        self.recording_preview = self.recording_zstack[0, :, :, :] # ensure that dimensions are the same as for ".get_single_frame_as_preview()"


    def load_roi(self, roi_filepath: Path) -> None:
        # roi_loader = ROILoaderFactory().get_loader(roi_filepath)
        # self.rois = roi_loader.load_all_rois()
        # self._create_preview_with_superimposed_rois()
        pass


    def run_analysis(self,
                     window_size: int,
                     limit_analysis_to_frame_interval: bool,
                     start_frame_idx: int,
                     end_frame_idx: int,
                     signal_average_threshold: float,
                     signal_to_noise_ratio: float,
                     #octaves_ridge_needs_to_spann: float,
                     #noise_window_size: int,
                     baseline_estimation_method: str,
                     #interpolate_intersection_frame_idxs: bool,
                     #include_variance: bool,
                     #variance: float,
                    ) -> None:
        self._create_squares(window_size)
        configs = locals()
        configs.pop('self')
        with multiprocessing.Pool(processes = self.num_processes) as pool:
            # check if following line can instead refer to original list - maybe the individual elements are also directly overwritten? 
            # consider changing into for loop if neccessary & possible
            processed_squares = pool.starmap(process_squares, [(square, configs) for square in self.squares])
        self.processed_squares = processed_squares


    def create_overview_results(self,
                                window_size: int,
                                minimum_activity_counts: int,
                               ) -> None:
        results.plot_activity_overview(self.processed_squares, self.recording_preview, window_size, minimum_activity_counts)



    def create_detailed_results(self,
                                include_detailed_results: bool,
                                window_size: int,
                                signal_average_threshold: float,
                                signal_to_noise_ratio: float,
                                minimum_activity_counts: int,
                                results_filepath: Path
                               ) -> None:
        if include_detailed_results == True:
            self._create_csv_result_files(results_filepath, minimum_activity_counts)
            for square in self.processed_squares:
                if hasattr(square, 'peaks_count') == True:
                    if square.peaks_count >= minimum_activity_counts:
                        # modify to aggregate into single PDF here
                        filename = f'Single_trace_graph_{square.idx}_WS-{window_size}_SNR-{signal_to_noise_ratio}_SAT-{signal_average_threshold}_MAC-{minimum_activity_counts}.pdf'
                        filepath = results_filepath.joinpath(filename)
                        results.plot_intensity_trace_with_identified_peaks_for_individual_square(square, filepath)
        #with multiprocessing.Pool(processes = self.num_processes) as pool:
            # check and align with multiprocessing of square processing above
            #pool.starmap(plot_intensity_trace_with_identified_peaks_for_individual_square, [(square, user, settings) for square in self.processed_squares])


    def _create_csv_result_files(self, results_filepath: Path, minimum_activity_counts: int) -> None:
        peak_results_per_square = [results.export_peak_results_df_from_square(square) for square in self.processed_squares if square.peaks_count >= minimum_activity_counts]
        df_all_peak_results = pd.concat(peak_results_per_square, ignore_index = True)
        max_peak_count_across_all_squares = df_all_peak_results.groupby('square coordinates [X / Y]').count()['peak frame index'].max()
        zfill_factor = int(np.log10(max_peak_count_across_all_squares)) + 1
        delta_f_over_f_results_all_squares = []
        auc_results_all_squares = []
        for square_coords in df_all_peak_results['square coordinates [X / Y]'].unique():
            tmp_df_single_square = df_all_peak_results[df_all_peak_results['square coordinates [X / Y]'] == square_coords].copy()
            delta_f_over_f_results_all_squares.append(results.create_single_square_delta_f_over_f_results(tmp_df_single_square, zfill_factor))
            auc_results_all_squares.append(results.create_single_square_auc_results(tmp_df_single_square, zfill_factor))
        df_all_delta_f_over_f_results = pd.concat(delta_f_over_f_results_all_squares, ignore_index = True)
        df_all_auc_results = pd.concat(auc_results_all_squares, ignore_index = True)
        # Once all DataFrames are created successfully, write them to disk 
        df_all_peak_results.to_csv(results_filepath.joinpath('all_peak_results.csv'), index = False)
        df_all_delta_f_over_f_results.to_csv(results_filepath.joinpath('dF_over_F_results.csv'), index = False)
        df_all_auc_results.to_csv(results_filepath.joinpath('AUC_results.csv'), index = False)


    

    def _create_preview_with_superimposed_rois(self) -> None:
        #self.recording_preview_with_superimposed_rois = 
        pass

    
    def _create_squares(self, window_size: int) -> None:
        upper_left_coords_of_squares_in_grid = self._get_upper_left_coords_of_squares_in_grid(window_size)
        for idx, (upper_left_y, upper_left_x) in enumerate(upper_left_coords_of_squares_in_grid):
            square_y_coords_slice = slice(upper_left_y, upper_left_y + window_size)
            square_x_coords_slice = slice(upper_left_x, upper_left_x + window_size)
            zstack_within_square = self.recording_zstack[:, square_y_coords_slice, square_x_coords_slice, :]
            self.squares.append(Square(idx, (upper_left_y, upper_left_x), zstack_within_square))
            

    def _get_upper_left_coords_of_squares_in_grid(self, window_size: int) -> List[Tuple[int, int]]:
        grid_row_idxs = np.arange(start = 0, stop = self.recording_preview.shape[0], step = window_size)
        grid_col_idxs = np.arange(start = 0, stop = self.recording_preview.shape[1], step = window_size)
        upper_left_coords_of_squares_in_grid = []
        for row_idx in grid_row_idxs:
            for col_idx in grid_col_idxs:
                upper_left_coords_of_squares_in_grid.append((row_idx, col_idx))
        return upper_left_coords_of_squares_in_grid