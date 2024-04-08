from pathlib import Path
import multiprocessing
import numpy as np

from .analysis import Square, process_squares
from .plotting import plot_activity_overview
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
                     signal_to_noise_ratio: float,
                     signal_average_threshold: float,
                     minimum_activity_counts: int,
                     baseline_estimation_method: str,
                     include_variance: bool,
                     variance: float,
                     limit_analysis_to_frame_interval: bool
                     # frame_interval_to_analyze: Tuple[int, int]
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
                                window_size: int
                               ) -> None:
        plot_activity_overview(self.processed_squares, self.recording_preview, window_size)
        # create_summary_text_files(whatever, might, be, needed, here, plus, additional, user, setttings)


    def create_detailed_results(self, user, settings, split, into, single, arguments, where, defaults, are, used, whenever, possible) -> None:
        #with multiprocessing.Pool(processes = self.num_processes) as pool:
            # check and align with multiprocessing of square processing above
            #pool.starmap(plot_intensity_trace_with_identified_peaks_for_individual_square, [(square, user, settings) for square in self.processed_squares])
        pass


    def _create_preview_with_superimposed_rois(self) -> None:
        #self.recording_preview_with_superimposed_rois = 
        pass

    
    def _create_squares(self, window_size: int) -> None:
        upper_left_coords_of_squares_in_grid = self._get_upper_left_coords_of_squares_in_grid(window_size)
        for upper_left_y, upper_left_x in upper_left_coords_of_squares_in_grid:
            square_y_coords_slice = slice(upper_left_y, upper_left_y + window_size)
            square_x_coords_slice = slice(upper_left_x, upper_left_x + window_size)
            zstack_within_square = self.recording_zstack[:, square_y_coords_slice, square_x_coords_slice, :]
            self.squares.append(Square((upper_left_y, upper_left_x), zstack_within_square))
            

    def _get_upper_left_coords_of_squares_in_grid(self, window_size: int) -> List[Tuple[int, int]]:
        grid_row_idxs = np.arange(start = 0, stop = self.recording_preview.shape[0], step = window_size)
        grid_col_idxs = np.arange(start = 0, stop = self.recording_preview.shape[1], step = window_size)
        upper_left_coords_of_squares_in_grid = []
        for row_idx in grid_row_idxs:
            for col_idx in grid_col_idxs:
                upper_left_coords_of_squares_in_grid.append((row_idx, col_idx))
        return upper_left_coords_of_squares_in_grid