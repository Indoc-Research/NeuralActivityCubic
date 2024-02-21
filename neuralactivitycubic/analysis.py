import numpy as np
from typing import Tuple
from scipy import signal

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
        self.peaks_count = self.frame_idxs_of_peaks.shape[0]


    def estimate_baseline(self):
        pass


    def compute_area_under_curve(self):
        pass


    def compute_delta_f_over_f(self):
        pass

    

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