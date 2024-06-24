import imageio.v3 as iio
from pathlib import Path
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict


class RecordingLoader(ABC):


    @abstractmethod
    def _get_all_frames(self) -> np.ndarray: 
        # To be implemented in individual subclasses
        # Shape of returned numpy array: [frames, rows, cols, color_channels]
        pass
        

    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath


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

    def _get_all_frames(self) -> np.ndarray: 
        return iio.imread(self.filepath)
        



class RecordingLoaderFactory:

    @property
    def supported_extensions_per_recording_loader_subclass(self) -> Dict[RecordingLoader, List[str]]:
        supported_extensions_per_recording_loader = {AVILoader: ['.avi']}
        return supported_extensions_per_recording_loader

    @property
    def all_supported_extensions(self) -> List[str]:
        all_supported_extensions = []
        for value in self.supported_extensions_per_recording_loader.values():
            all_supported_extensions += value
        return all_supported_extensions

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




class ROILoader(ABC):

    @abstractmethod
    def _get_boundary_coords(self) -> np.ndarray: 
        # To be implemented in individual subclasses
        # Shape of returned numpy array: [(x or row coord, y or col coord)]
        pass
        

    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath

    
    def load_boundary_coords(self) -> np.ndarray: 
        boundary_coords = self._get_boundary_coords()
        polygon = self._convert_to_shapely_polygon(boundary_coords)
        self._assert_valid_shape(polygon)
        self.roi_as_polygon = polygon
        return boundary_coords


    def _convert_to_shapely_polygon(self, boundary_coords: np.ndarray) -> None: #Polygon:
        polygon = boundary_coords
        # conversion should happen here
        return polygon


    def _assert_valid_shape(self, polygon) -> None:
        # assertion that polygon has a valid shape should be performed here
        # IS THIS ACTUALLY NEEDED???
        pass



class ImageJROILoader(ROILoader):

    def _get_boundary_coords(self) -> np.ndarray:
        # read filepath and extract boundary coords of ROI
        # ensure that it´s only a single ROI
        placeholder_for_file_content = self.filepath
        return placeholder_for_file_content
        


class ROILoaderFactory:

    @property
    def supported_extensions_per_roi_loader_subclass(self) -> Dict[ROILoader, List[str]]:
        supported_extensions_per_roi_loader = {ImageJROILoader: ['.roi']}
        return supported_extensions_per_roi_loader

    @property
    def all_supported_extensions(self) -> List[str]:
        all_supported_extensions = []
        for value in self.supported_extensions_per_roi_loader.values():
            all_supported_extensions += value
        return all_supported_extensions

    def get_loader(self, filepath: Path) -> ROILoader:
        self._assert_validity_of_filepath(filepath)
        roi_loader = self._get_loader_for_file_extension(filepath)
        return roi_loader
        

    def _assert_validity_of_filepath(self, filepath: Path) -> None:
        assert isinstance(filepath, Path), f'filepath must be an instance of a pathlib.Path. However, you passed {filepath}, which is of type {type(filepath)}'
        assert filepath.exists(), f'The filepath you provided ({filepath}) does not seem to exist!'
        

    def _get_loader_for_file_extension(self, filepath: Path) -> ROILoader:
        matching_loader = None
        for loader_subclass, supported_extensions in self.supported_extensions_per_roi_loader.items():
            if filepath.suffix in supported_extensions:
                matching_loader = loader_subclass(filepath)
                break
        if matching_loader == None:
            raise NotImplementedError('It seems like there is no ROILoader implemented for the specific filetype you´re trying to load - sorry!')
        return matching_loader




class RecordingRoiCombiLoader:

    def __init__(self, dirpath: Path) -> None:
        self.dirpath = dirpath


    def get_a_recording_loader(self) -> None:
        recording_loader_factory = RecordingLoaderFactory()
        self.recording_filepath = self._get_filepath_of_supported_recording_in_dirpath(recording_loader_factory.all_supported_extensions, 1)[0]
        self.recording_loader = recording_loader_factory.get_loader(self.recording_filepath)


    def get_all_roi_loaders(self) -> None:
        roi_loader_factory = ROILoaderFactory()
        self.all_roi_filepaths = self._get_filepath_of_supported_recording_in_dirpath(roi_loader_factory.all_supported_extensions)
        self.all_roi_loaders = []
        for roi_filepath in self.all_roi_filepaths:
            self.all_roi_loaders.append(roi_loader_factory.get_loader(roi_filepath))
        

    def _get_filepaths_with_supported_extension_in_dirpath(self, all_supported_extensions: List[str], max_results: Optional[int]=None) -> List[Path]:
        all_filepaths_with_supported_extension = []
        for elem in self.dirpath.iterdir():
            if elem.is_file() == True:
                if elem.suffix in all_supported_extensions:
                    all_filepaths_with_supported_extension.append(elem)
        if type(max_results) == int:
            assert len(all_filepaths_with_supported_extension) <= , (
                f'There are more than {max_results} file(s) of supported type in {self.dirpath}, '
                f'but only a maximum of {max_results} are allowed. Please remove at least '
                f'{len(all_filepaths_with_supported_extension) - max_results} of the following' 
                f'files and try again: {all_filepaths_with_supported_extension}'
            )
        return all_filepaths_with_supported_extension

