import numpy as np
from scipy import signal
from pybaselines import Baseline
from collections import Counter

from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass



@dataclass
class Peak:

    frame_idx: int
    intensity: float
    delta_f_over_f: Optional[float]=None
    has_neighboring_intersections: Optional[bool]=None
    frame_idxs_of_neighboring_intersections: Optional[Tuple[int, int]]=None
    area_under_curve: Optional[float]=None
    area_under_curve_type: Optional[str]=None



class Square:

    def __init__(self, upper_left_corner_coords: Tuple[int, int], frames_zstack: np.ndarray) -> None:
        self.upper_left_corner_coords = upper_left_corner_coords
        self.frames_zstack = frames_zstack
        self.center_coords = self._get_center_coords()
        

    def compute_mean_intensity_timeseries(self) -> None:
        self.mean_intensity_over_time = np.mean(self.frames_zstack, axis = (1,2,3))


    def detect_peaks(self, signal_to_noise_ratio: float) -> None:
        self.frame_idxs_of_peaks = signal.find_peaks_cwt(vector = self.mean_intensity_over_time, 
                                                         wavelet = signal.ricker, 
                                                         widths = np.arange(1, 81), 
                                                         min_length = 7, 
                                                         noise_perc = 10, 
                                                         min_snr = signal_to_noise_ratio)
        self.peaks = {}
        for peak_frame_idx in self.frame_idxs_of_peaks:
            self.peaks[peak_frame_idx] = Peak(frame_idx = peak_frame_idx, intensity = self.mean_intensity_over_time[peak_frame_idx]) 
        self.peaks_count = self.frame_idxs_of_peaks.shape[0]


    def estimate_baseline(self) -> None:
        baseline_estimator = Baseline()
        self.baseline = baseline_estimator.asls(data = self.mean_intensity_over_time)[0]


    def compute_area_under_curve(self) -> None:
        quick_estimate_of_intersection_frame_idxs = np.argwhere(np.diff(np.sign(self.mean_intensity_over_time - self.baseline))).flatten()
        self._add_information_about_neighboring_intersections_to_peaks(quick_estimate_of_intersection_frame_idxs)
        self._find_closest_neighboring_intersections_for_each_peak(quick_estimate_of_intersection_frame_idxs)
        area_under_curve_classification = {'peaks_with_auc': [], 'all_intersection_frame_idxs_pairs': []}
        for peak_frame_idx, peak in self.peaks.items():
            if peak.has_neighboring_intersections == True:
                idx_before_peak, idx_after_peak = peak.frame_idxs_of_neighboring_intersections
                peak.area_under_curve = np.trapz(self.mean_intensity_over_time[idx_before_peak:idx_after_peak + 1] - self.baseline[idx_before_peak:idx_after_peak + 1])
                area_under_curve_classification['peaks_with_auc'].append(peak)
                area_under_curve_classification['all_intersection_frame_idxs_pairs'].append(peak.frame_idxs_of_neighboring_intersections)
        self._classify_area_under_curve_types(area_under_curve_classification)
                                                                                    


    def _classify_area_under_curve_types(self, data_for_auc_classification: Dict[str, List]) -> None:
        if len(data_for_auc_classification['all_intersection_frame_idxs_pairs']) != len(set(data_for_auc_classification['all_intersection_frame_idxs_pairs'])):
            counter = Counter(data_for_auc_classification['all_intersection_frame_idxs_pairs'])
            reoccuring_intersection_frame_idxs = [pair_of_intersection_frame_idxs for pair_of_intersection_frame_idxs, count in counter.items() if count > 1]
        else:
            reoccuring_intersection_frame_idxs = []
        for peak in data_for_auc_classification['peaks_with_auc']:
            if peak.frame_idxs_of_neighboring_intersections in reoccuring_intersection_frame_idxs:
                peak.area_under_curve_type = 'event_train'
            else:
                peak.area_under_curve_type = 'individual_event'


    def _add_information_about_neighboring_intersections_to_peaks(self, intersection_frame_idxs: np.ndarray) -> None:
        frame_idxs_of_peaks_with_neighboring_intersections = self.frame_idxs_of_peaks[((self.frame_idxs_of_peaks > intersection_frame_idxs[0]) 
                                                                                       & (self.frame_idxs_of_peaks < intersection_frame_idxs[-1]))]
        for peak_frame_idx, peak in self.peaks.items():
            if peak_frame_idx in frame_idxs_of_peaks_with_neighboring_intersections:
                peak.has_neighboring_intersections = True
            else:
                peak.has_neighboring_intersections = False


    def _find_closest_neighboring_intersections_for_each_peak(self, quick_estimate_of_intersection_frame_idxs: np.ndarray, improve_accuracy_via_interpolation: bool=True) -> None:
        for peak_frame_idx, peak in self.peaks.items():
            if peak.has_neighboring_intersections == True:
                neighboring_intersection_frame_idxs = self._get_neighboring_intersection_frame_idxs(peak_frame_idx, quick_estimate_of_intersection_frame_idxs)
                if improve_accuracy_via_interpolation == True:
                    intersection_frame_idx_before_peak = self._improve_intersection_frame_idx_estimation_by_interpolation(neighboring_intersection_frame_idxs[0])
                    intersection_frame_idx_after_peak = self._improve_intersection_frame_idx_estimation_by_interpolation(neighboring_intersection_frame_idxs[1])
                else:
                    intersection_frame_idx_before_peak, intersection_frame_idx_after_peak = neighboring_intersection_frame_idxs
                peak.frame_idxs_of_neighboring_intersections = (intersection_frame_idx_before_peak, intersection_frame_idx_after_peak)


    def _get_neighboring_intersection_frame_idxs(self, peak_frame_idx: int, all_intersection_idxs: np.ndarray) -> Tuple[int, int]:
        rolling_diff_on_signed_distance = np.diff(np.sign(all_intersection_idxs - peak_frame_idx))
        assert 2 in rolling_diff_on_signed_distance, ('Failed to identify the two neighboring intersection indices between signal and baseline '
                                                      f'for identified peak at frame idx: {peak_frame_idx}')
        idx_before_peak_in_intersections = np.argmax(rolling_diff_on_signed_distance)
        intersection_frame_idx_before_peak = all_intersection_idxs[idx_before_peak_in_intersections]
        intersection_frame_idx_after_peak = all_intersection_idxs[idx_before_peak_in_intersections + 1]
        return intersection_frame_idx_before_peak, intersection_frame_idx_after_peak
    


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
    

    
    def compute_delta_f_over_f(self):
        for peak in self.peaks.values():
            peak.delta_f_over_f = (self.mean_intensity_over_time[peak.frame_idx] - self.baseline[peak.frame_idx]) / self.baseline[peak.frame_idx]

    

    def _get_center_coords(self) -> Tuple[int, int]:
        square_height = self.frames_zstack.shape[1]
        square_width = self.frames_zstack.shape[2]
        return (self.upper_left_corner_coords[0] + int(square_height/2), self.upper_left_corner_coords[1] + int(square_width/2))



def process_squares(square: Square, signal_to_noise_ratio: float) -> Square:
    square.compute_mean_intensity_timeseries()
    square.detect_peaks(signal_to_noise_ratio)
    #square.estimate_baseline()
    #square.compute_area_under_curve()
    #square.compute_delta_f_over_f()
    return square