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
    delta_f_over_f: Optional[float]=None
    has_neighboring_intersections: Optional[bool]=None
    frame_idxs_of_neighboring_intersections: Optional[Tuple[int, int]]=None
    area_under_curve: Optional[float]=None
    peak_type: Optional[str]=None



class Square:

    def __init__(self, idx: int, upper_left_corner_coords: Tuple[int, int], frames_zstack: np.ndarray) -> None:
        self.idx = idx
        self.upper_left_corner_coords = upper_left_corner_coords
        self.frames_zstack = frames_zstack
        self.center_coords = self._get_center_coords()
        self.peaks_count = 0


    def _get_center_coords(self) -> Tuple[int, int]:
        square_height = self.frames_zstack.shape[1]
        square_width = self.frames_zstack.shape[2]
        return (self.upper_left_corner_coords[0] + int(square_height/2), self.upper_left_corner_coords[1] + int(square_width/2))

        
    def compute_mean_intensity_timeseries(self, limit_analysis_to_frame_interval: bool, start_frame_idx: int, end_frame_idx: int) -> None:
        if limit_analysis_to_frame_interval == True:
            self.mean_intensity_over_time = np.mean(self.frames_zstack[start_frame_idx:end_frame_idx], axis = (1,2,3))
        else:
            self.mean_intensity_over_time = np.mean(self.frames_zstack, axis = (1,2,3))


    def detect_peaks(self, signal_to_noise_ratio: float, n_octaves_min: float=1.0, noise_window_size: int=200) -> None:
        # Add default exclusion of peaks within first / last 10-20 frames to avoid border effects? implement zero_padding / mirroring?
        widths = np.logspace(np.log10(1), np.log10(self.mean_intensity_over_time.shape[0]), 100)
        min_length = n_octaves_min / np.log2(widths[1] / widths[0])
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


    def compute_area_under_curve(self, improve_accuracy_via_interpolation: bool=True) -> None:
        self._get_unique_frame_idxs_of_intersections_between_signal_and_baseline(improve_accuracy_via_interpolation)
        self._add_information_about_neighboring_intersections_to_peaks()
        area_under_curve_classification = {'peaks_with_auc': [], 'all_intersection_frame_idxs_pairs': []}
        for peak_frame_idx, peak in self.peaks.items():
            if peak.has_neighboring_intersections == True:
                idx_before_peak, idx_after_peak = peak.frame_idxs_of_neighboring_intersections
                peak.area_under_curve = np.trapz(self.mean_intensity_over_time[idx_before_peak:idx_after_peak + 1] - self.baseline[idx_before_peak:idx_after_peak + 1])
                area_under_curve_classification['peaks_with_auc'].append(peak)
                area_under_curve_classification['all_intersection_frame_idxs_pairs'].append(peak.frame_idxs_of_neighboring_intersections)
        self._classify_area_under_curve_types(area_under_curve_classification)
                                                                                    

    def _get_unique_frame_idxs_of_intersections_between_signal_and_baseline(self, improve_accuracy_via_interpolation: bool) -> None:
        quick_estimate_of_intersection_frame_idxs = np.argwhere(np.diff(np.sign(self.mean_intensity_over_time - self.baseline))).flatten()
        if improve_accuracy_via_interpolation == True:
            intersection_frame_idxs = np.asarray([self._improve_intersection_frame_idx_estimation_by_interpolation(idx) for idx in quick_estimate_of_intersection_frame_idxs])
            unique_intersection_frame_idxs = np.unique(intersection_frame_idxs)
        else:
            unique_intersection_frame_idxs = np.unique(quick_estimate_of_intersection_frame_idxs)
        self.unique_intersection_frame_idxs = unique_intersection_frame_idxs


    def _add_information_about_neighboring_intersections_to_peaks(self) -> None:
        for peak_frame_idx, peak in self.peaks.items():
            #if peak_frame_idx in self.unique_intersection_frame_idxs:
                # log that this peak coincides with an intersection at the very same frame idx
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

    
    def compute_delta_f_over_f(self):
        for peak in self.peaks.values():
            peak.delta_f_over_f = (self.mean_intensity_over_time[peak.frame_idx] - self.baseline[peak.frame_idx]) / self.baseline[peak.frame_idx]

    


def process_squares(square: Square, configs: Dict[str, Any]) -> Square:
    square.compute_mean_intensity_timeseries(configs['limit_analysis_to_frame_interval'], configs['start_frame_idx'], configs['end_frame_idx'])
    if np.mean(square.mean_intensity_over_time) >= configs['signal_average_threshold']:
        square.detect_peaks(configs['signal_to_noise_ratio']) # add 'n_octaves_min' and 'noise_window_size' here!
        #if configs['compute_aucs'] == True:
        square.estimate_baseline(configs['baseline_estimation_method'])
        square.compute_area_under_curve() # add 'improve_accuracy_by_interpolation' here!
        #if configs['compute_df_over_f'] == True:
        square.compute_delta_f_over_f()
    return square