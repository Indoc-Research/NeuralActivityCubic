import numpy as np
from scipy import signal
from pybaselines import Baseline
from collections import Counter

from typing import Optional, Tuple, Dict, List, Callable, Any
from dataclasses import dataclass



class BaselineEstimatorFactory:
        
    @property
    def supported_baseline_estimation_methods(self) -> Dict[str, Callable]:
        supported_baseline_estimation_methods = {'asls': Baseline().asls,
                                                 'fabc': Baseline().fabc,
                                                 'psalsa': Baseline().psalsa,
                                                 'std_distribution': Baseline().std_distribution}
        return supported_baseline_estimation_methods

    def get_baseline_estimation_callable(self, algorithm_acronym: str) -> Callable:
        baseline_estimation_method = self.supported_baseline_estimation_methods[algorithm_acronym]
        return baseline_estimation_method



@dataclass
class Peak:

    frame_idx: int
    intensity: float
    amplitude: Optional[float]=None
    delta_f_over_f: Optional[float]=None
    has_neighboring_intersections: Optional[bool]=None
    frame_idxs_of_neighboring_intersections: Optional[Tuple[int, int]]=None
    area_under_curve: Optional[float]=None
    peak_type: Optional[str]=None



class Square:
    def __init__(self, grid_cell_label: Tuple[int, int], upper_left_corner_coords: Tuple[int, int], frames_zstack: np.ndarray) -> None:
        self.grid_row_label, self.grid_col_label = grid_cell_label
        self.upper_left_corner_coords = upper_left_corner_coords
        self.frames_zstack = frames_zstack
        self.center_coords = self._get_center_coords()
        self.as_polygon = self._create_square_as_polygon()
        self.peaks_count = 0

    
    def _get_center_coords(self) -> Tuple[int, int]:
        square_height = self.frames_zstack.shape[1]
        square_width = self.frames_zstack.shape[2]
        return (self.upper_left_corner_coords[0] + int(square_height/2), self.upper_left_corner_coords[1] + int(square_width/2))

    def _create_square_as_polygon(self) -> Polygon:
        square_rows = self.frames_zstack.shape[1]
        square_cols = self.frames_zstack.shape[2]
        upper_left_corner_row_coord = self.upper_left_corner_coords[0]
        upper_left_corner_col_coord = self.upper_left_corner_coords[1]
        all_corner_coords = [[upper_left_corner_row_coord, upper_left_corner_col_coord],
                             [upper_left_corner_row_coord, upper_left_corner_col_coord + square_cols],
                             [upper_left_corner_row_coord + square_cols, upper_left_corner_col_coord + square_cols],
                             [upper_left_corner_row_coord + square_cols, upper_left_corner_col_coord]]
        square_as_polygon = Polygon(all_corner_coords)
        assert square_as_polygon.is_valid, (
            f'Something went wrong when trying to create a Polygon for Square [{self.grid_col_label}/{self.self.grid_row_label}]!'
        )
        return square_as_polygon

    
    def compute_mean_intensity_timeseries(self, limit_analysis_to_frame_interval: bool, start_frame_idx: int, end_frame_idx: int) -> None:
        if limit_analysis_to_frame_interval == True:
            self.mean_intensity_over_time = np.mean(self.frames_zstack[start_frame_idx:end_frame_idx], axis = (1,2,3))
        else:
            self.mean_intensity_over_time = np.mean(self.frames_zstack, axis = (1,2,3))


    def detect_peaks(self, signal_to_noise_ratio: float, octaves_ridge_needs_to_spann: float, noise_window_size: int) -> None:
        widths = np.logspace(np.log10(1), np.log10(self.mean_intensity_over_time.shape[0]), 100)
        min_length = octaves_ridge_needs_to_spann / np.log2(widths[1] / widths[0])
        n_padded_frames = int(np.median(widths)) + 1
        signal_padded_with_reflection = np.pad(self.mean_intensity_over_time, n_padded_frames, 'reflect')
        frame_idxs_of_peaks_in_padded_signal = signal.find_peaks_cwt(vector = signal_padded_with_reflection, 
                                                         wavelet = signal.ricker, 
                                                         widths = widths, 
                                                         min_length = min_length,
                                                         max_distances = widths / 4, # default
                                                         gap_thresh = 0.0,
                                                         noise_perc = 5, # default: 10
                                                         min_snr = signal_to_noise_ratio,
                                                         window_size = noise_window_size # window size to calculate noise is very narrow (lowest point = noise)
                                                        )
        frame_idxs_of_peaks_in_padded_signal = frame_idxs_of_peaks_in_padded_signal[((frame_idxs_of_peaks_in_padded_signal >= n_padded_frames) & 
                                                                                     (frame_idxs_of_peaks_in_padded_signal < self.mean_intensity_over_time.shape[0] + n_padded_frames))]
        self.frame_idxs_of_peaks = frame_idxs_of_peaks_in_padded_signal - n_padded_frames
        self.peaks = {}
        for peak_frame_idx in self.frame_idxs_of_peaks:
            self.peaks[peak_frame_idx] = Peak(frame_idx = peak_frame_idx, intensity = self.mean_intensity_over_time[peak_frame_idx]) 
        self.peaks_count = self.frame_idxs_of_peaks.shape[0]


    def estimate_baseline(self, algorithm_acronym: str) -> None:
        baseline_estimation_method = BaselineEstimatorFactory().get_baseline_estimation_callable(algorithm_acronym)
        self.baseline = baseline_estimation_method(data = self.mean_intensity_over_time)[0]


    def compute_area_under_curve(self) -> None:
        self._get_unique_frame_idxs_of_intersections_between_signal_and_baseline()
        self._add_information_about_neighboring_intersections_to_peaks()
        area_under_curve_classification = {'peaks_with_auc': [], 'all_intersection_frame_idxs_pairs': []}
        for peak_frame_idx, peak in self.peaks.items():
            if peak.has_neighboring_intersections == True:
                idx_before_peak, idx_after_peak = peak.frame_idxs_of_neighboring_intersections
                peak.area_under_curve = np.trapz(self.mean_intensity_over_time[idx_before_peak:idx_after_peak + 1] - self.baseline[idx_before_peak:idx_after_peak + 1])
                area_under_curve_classification['peaks_with_auc'].append(peak)
                area_under_curve_classification['all_intersection_frame_idxs_pairs'].append(peak.frame_idxs_of_neighboring_intersections)
        self._classify_area_under_curve_types(area_under_curve_classification)
                                                                                    

    def _get_unique_frame_idxs_of_intersections_between_signal_and_baseline(self) -> None:
        quick_estimate_of_intersection_frame_idxs = np.argwhere(np.diff(np.sign(self.mean_intensity_over_time - self.baseline))).flatten()
        intersection_frame_idxs = np.asarray([self._improve_intersection_frame_idx_estimation_by_interpolation(idx) for idx in quick_estimate_of_intersection_frame_idxs])
        self.unique_intersection_frame_idxs = np.unique(intersection_frame_idxs)


    def _add_information_about_neighboring_intersections_to_peaks(self) -> None:
        for peak_frame_idx, peak in self.peaks.items():
            if (peak_frame_idx > self.unique_intersection_frame_idxs[0]) & (peak_frame_idx < self.unique_intersection_frame_idxs[-1]):
                peak.has_neighboring_intersections = True
                idx_pre_peak = self.unique_intersection_frame_idxs[self.unique_intersection_frame_idxs < peak_frame_idx][-1]
                idx_post_peak = self.unique_intersection_frame_idxs[self.unique_intersection_frame_idxs > peak_frame_idx][0]
                peak.frame_idxs_of_neighboring_intersections = (idx_pre_peak, idx_post_peak)
            else:
                peak.has_neighboring_intersections = False


    def _improve_intersection_frame_idx_estimation_by_interpolation(self, idx_frame_0: int) -> int:
        """
        Designed to resolve the bias of the quick estimation of intersection points, which will always 
        return the first index of two frames between which an intersection was determined. This is done 
        by interpolating the data (for both signal & baseline) to a sub-frame resolution between the 
        previously identified intersection frame index, and the following frame index - as the intersection 
        might actually happen closer to this following frame. If interpolation estimates the intersection 
        precisely in the middle between the two frames, the later frame is returned (0.5 is rounded up).
        """
        idx_frame_1 = idx_frame_0 + 1
        num_interpolated_steps = 7
        # interpolate signal & baseline to sub-frame resolution:
        interpolated_signal_intensities = np.linspace(self.mean_intensity_over_time[idx_frame_0], self.mean_intensity_over_time[idx_frame_1], num = num_interpolated_steps)
        interpolated_baseline = np.linspace(self.baseline[idx_frame_0], self.baseline[idx_frame_1], num = num_interpolated_steps)
        # identify whether frame_idx_0 or frame_idx_1 is closer to interpolated intersection point
        signed_differences = np.sign(interpolated_signal_intensities - interpolated_baseline)
        if 0 in signed_differences: #intersection exactly at one or multiple interpolated index
            intersection_idx_in_interpolation = np.argwhere(signed_differences == 0).flatten()[0]
        else: 
            results_for_intersection_idxs = np.argwhere(np.diff(signed_differences)).flatten()
            assert results_for_intersection_idxs.size != 0, ('get_improved_intersection_idx_estimation_by_interpolation() expected an intersection between frames '
                                                             f'{idx_frame_0} and {idx_frame_1} in the provided arrays, but none were found!')
            intersection_idx_in_interpolation = results_for_intersection_idxs[0]
        # Based on interpolation, select whether intersection is closer to frame_idx_0 or frame_idx_1
        if intersection_idx_in_interpolation < np.median(np.arange(0, num_interpolated_steps, 1)):
            interpolation_evaluated_intersection_frame_idx = idx_frame_0
        else:
            interpolation_evaluated_intersection_frame_idx = idx_frame_1
        return interpolation_evaluated_intersection_frame_idx
    

    def _classify_area_under_curve_types(self, data_for_auc_classification: Dict[str, List]) -> None:
        if len(data_for_auc_classification['all_intersection_frame_idxs_pairs']) != len(set(data_for_auc_classification['all_intersection_frame_idxs_pairs'])):
            counter = Counter(data_for_auc_classification['all_intersection_frame_idxs_pairs'])
            reoccuring_intersection_frame_idxs = [pair_of_intersection_frame_idxs for pair_of_intersection_frame_idxs, count in counter.items() if count > 1]
        else:
            reoccuring_intersection_frame_idxs = []
        for peak in self.peaks.values():
            if peak in data_for_auc_classification['peaks_with_auc']:
                if peak.frame_idxs_of_neighboring_intersections in reoccuring_intersection_frame_idxs:
                    peak.peak_type = 'clustered'
                else:
                    peak.peak_type = 'singular'
            else:
                peak.peak_type = 'isolated'

    
    def compute_amplitude_and_delta_f_over_f(self):
        for peak in self.peaks.values():
            peak.amplitude = self.mean_intensity_over_time[peak.frame_idx] - self.baseline[peak.frame_idx]
            peak.delta_f_over_f = peak.amplitude / self.baseline[peak.frame_idx]

    


def process_squares(square: Square, configs: Dict[str, Any]) -> Square:
    square.compute_mean_intensity_timeseries(configs['limit_analysis_to_frame_interval'], configs['start_frame_idx'], configs['end_frame_idx'])
    if np.mean(square.mean_intensity_over_time) >= configs['signal_average_threshold']:
        square.detect_peaks(configs['signal_to_noise_ratio'], configs['octaves_ridge_needs_to_spann'], configs['noise_window_size'])
        #if configs['compute_aucs'] == True:
        square.estimate_baseline(configs['baseline_estimation_method'])
        square.compute_area_under_curve()
        #if configs['compute_df_over_f'] == True:
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
            zstack_within_square = self.recording_zstack[:, square_row_coords_slice, square_col_coords_slice, :]
            squares.append(Square(grid_cell_label, (upper_left_row_pixel_idx, upper_left_col_pixel_idx), zstack_within_square))
        if self.roi_based == True:
            squares_filtered_by_roi = self._filter_squares_based_on_roi(squares)
            squares = squares_filtered_by_roi
        return squares


    def _filter_squares_based_on_roi(self, squares: List[Square]) -> List[Square]:
        filtered_squares = [square for square in squares if square.as_polygon.intersects(self.roi.as_polygon)]
        return filtered_squares

    
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
        self.overview_results = results.plot_activity_overview(filtered_squares = filtered_squares, 
                                                               preview_image = self.recording.preview, 
                                                               row_cropping_idx = self.row_cropping_idx, 
                                                               col_cropping_idx = self.col_cropping_idx, 
                                                               window_size = window_size, 
                                                               indicate_activity = True,
                                                               roi = self.roi)
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
                    overview_fig, ax = results.plot_activity_overview(filtered_squares = filtered_squares,
                                                                      preview_image = self.recording.preview, 
                                                                      row_cropping_idx = self.row_cropping_idx, 
                                                                      col_cropping_idx = self.col_cropping_idx, 
                                                                      window_size = window_size, 
                                                                      indicate_activity = indicate_activity,
                                                                      roi = self.roi)
                    pdf.savefig(overview_fig)
                    plt.close()
                for square in filtered_squares:
                    fig = results.plot_intensity_trace_with_identified_peaks_for_individual_square(square)
                    pdf.savefig(fig)
                    plt.close()