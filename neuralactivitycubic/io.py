import imageio.v3 as iio
from pathlib import Path
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict


class RecordingLoader(ABC):


    @abstractmethod
    def _get_single_frame(self, frame_idx: int=0) -> np.ndarray:
        # To be implemented in individual subclasses
        # Should NOT keep the entire recording in memory!
        # Shape of returned numpy array: [frames, rows, cols, color_channels]
        pass


    @abstractmethod
    def _get_all_frames(self) -> np.ndarray: 
        # To be implemented in individual subclasses
        # Shape of returned numpy array: [frames, rows, cols, color_channels]
        pass
        

    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath


    def load_single_frame_as_preview(self, frame_idx: int=0) -> np.ndarray:
        preview_frame = self._get_single_frame(frame_idx)
        preview_frame = self._validate_shape_and_convert_to_grayscale_if_possible(preview_frame)
        return preview_frame


    def load_all_frames(self) -> np.ndarray: 
        all_frames = self._get_all_frames()
        all_frames = self._validate_shape_and_convert_to_grayscale_if_possible(all_frames)
        return all_frames


    def _validate_shape_and_convert_to_grayscale_if_possible(self, zstack: np.ndarray) -> np.ndarray:
        self._validate_correct_array_shape(zstack)
        if zstack.shape[3] > 1:
            if self._check_if_color_channels_are_redunant(zstack) == True:
                zstack = self._convert_to_grayscale(zstack)
        return zstack


    def _validate_correct_array_shape(self, zstack: np.ndarray) -> None:
        assert len(zstack.shape) == 4, ('The shape of the zstack numpy array is not correct. It should be a 4 dimensional array, like '
                                        f'[frames, rows, cols, color channels]. However, the current shape is: {zstack.shape}.')
        assert zstack.shape[3] in [1, 3], ('The color channels of the recording you attempted to load are incorrect. Currently, only single '
                                           f'channel or RGB (i.e. 1 or 3 color channels) are supported. However, your data has: {zstack.shape[3]}.')
        
    
    def _check_if_color_channels_are_redunant(self, zstack: np.ndarray) -> bool:
        reference_channel_idx = 0
        color_channels_are_equal = []
        for idx_of_channel_to_compare in range(1, zstack.shape[3]):
            if np.array_equal(zstack[:, :, :, reference_channel_idx], zstack[: , :, :, idx_of_channel_to_compare]) == True:
                color_channels_are_equal.append(True)
            else:
                color_channels_are_equal.append(False)
                break
        return all(color_channels_are_equal)


    def _convert_to_grayscale(self, zstack: np.ndarray) -> np.ndarray:
        return zstack[:, :, :, 0:1]
        
        
        


class AVILoader(RecordingLoader):

    def _get_single_frame(self, frame_idx: int=0) -> np.ndarray: 
        # To be implemented in individual subclasses
        # Should NOT keep the entire recording in memory!
        # Shape of returned numpy array: [frames, rows, cols, color_channels]
        pass


    def _get_all_frames(self) -> np.ndarray: 
        return iio.imread(self.filepath)
        



class RecordingLoaderFactory:

    @property
    def supported_extensions_per_recording_loader_subclass(self) -> Dict[RecordingLoader, List[str]]:
        supported_extensions_per_recording_loader = {AVILoader: ['.avi']}
        return supported_extensions_per_recording_loader

    def get_loader(self, filepath: Path) -> RecordingLoader:
        self._assert_validity_of_filepath(filepath)
        recording_loader = self._get_loader_for_file_extension(filepath)
        return recording_loader
        

    def _assert_validity_of_filepath(self, filepath: Path) -> None:
        assert isinstance(filepath, Path), f'filepath must be an instance of a pathlib.Path. However, you passed {filepath}, which is of type {type(filepath)}'
        assert filepath.exists(), f'The filepath you provided ({filepath}) does not seem to exist!'
        

    def _get_loader_for_file_extension(self, filepath: Path) -> RecordingLoader:
        matching_loader = None
        for loader_subclass, supported_extensions in self.supported_extensions_per_recording_loader_subclass.items():
            if filepath.suffix in supported_extensions:
                matching_loader = loader_subclass(filepath)
                break
        if matching_loader == None:
            raise NotImplementedError('It seems like there is no RecordingLoader implemented for the specific filetype you´re trying to load - sorry!')
        return matching_loader



