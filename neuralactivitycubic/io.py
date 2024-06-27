import imageio.v3 as iio
from pathlib import Path
import numpy as np
from shapely import Polygon
import roifile
from abc import ABC, abstractmethod
from typing import List, Dict, Union, Optional



class InputData(ABC):

    @abstractmethod
    def inject_loaded_data(self, loaded_data: Any) -> None:
        pass

    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath


class Recording(InputData):

    def inject_loaded_data(self, loaded_data: np.ndarray) -> None:
        self.zstack = loaded_data
        self.estimated_bit_depth = self._estimate_bit_depth()
        self.preview = self._create_brightness_and_contrast_enhanced_preview()


    def _estimate_bit_depth(self) -> int:
        max_bit_value = self.zstack.max()
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
        raw_image = self.zstack[0, :, :, :].copy() # ensure that dimensions are the same as for ".get_single_frame_as_preview()"
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




class ROI(InputData):

    def inject_loaded_data(self, loaded_data: List[Tuple[int, int]]) -> None:
        self.boundary_coords = loaded_data
        self.as_polygon = self._convert_to_valid_polygon()


    def _convert_to_valid_polygon(self) -> Polygon:
        roi_as_polygon = Polygon(self.boundary_coords)
        assert roi_as_polygon.is_valid, f'Something went wrong when trying to create a Polygon out of your ROI: {self.filepath}.'
        return roi_as_polygon



class DataLoader(ABC):

    @abstractmethod
    def load_data(self) -> Union[Data, Tuple[Data, Data]]:
        # This method will be called when the data should be loaded for analysis
        pass

    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath



class RecordingLoader(DataLoader):


    @abstractmethod
    def _get_all_frames(self) -> np.ndarray: 
        # To be implemented in individual subclasses
        # Shape of returned numpy array: [frames, rows, cols, color_channels]
        pass


    def _load_all_frames(self) -> np.ndarray: 
        all_frames = self._get_all_frames()
        all_frames = self._validate_shape_and_convert_to_grayscale_if_possible(all_frames)
        return all_frames


    def load_data(self) -> Recording:
        all_frames = self._load_all_frames()
        recording = Recording(self.filepath)
        recording.inject_loaded_data(all_frames)
        return Recording


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




class ROILoader(DataLoader):

    @abstractmethod
    def _get_boundary_coords(self) -> List[Tuple[int, int]]: 
        # To be implemented in individual subclasses
        # Return a list of Tuples, where each tuple represents one boundary point: (row_coord, col_coord)
        pass

    
    def load_data(self) -> ROI: 
        boundary_coords = self._get_boundary_coords()
        boundary_coords = self._add_first_boundary_point_also_add_end_to_close_roi(boundary_coords)
        roi = ROI(self.filepath)
        roi.inject_loaded_data(boundary_coords)
        return roi


    def _add_first_boundary_point_also_add_end_to_close_roi(self, boundary_coords: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        first_boundary_point_coords = boundary_coords[0]
        boundary_coords.append(first_boundary_point_coords)
        return boundary_coords




class ImageJROILoader(ROILoader):

    def _get_boundary_coords(self) -> List[Tuple[int, int]]:
        roi_file_content = roifile.ImagejRoi.fromfile(self.filepath)
        assert type(roi_file_content) != list, (
            'NeuralActivityCubic can currently only handle loading ROI files that contain exactly one ROI per file. '
            f'However, the ROI file you selected ("{self.filepath}") contains more than a single ROI! Please adjust '
            'the file accordingly and retry.'
        )
        row_coords = roi_file_content.coordinates()[:, 1]
        col_coords = roi_file_content.coordinates()[:, 0]
        boundary_row_col_coords = list(zip(row_coords, col_coords))
        return boundary_row_col_coords


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




class RecordingRoiCombiLoader(DataLoader):


class RecLoaderROILoaderCombinator:

        
    def __init__(self, dir_path: Path) -> None:
        self.dir_path = dir_path

    
    def get_all_recording_and_roi_loader_combos(self) -> List[Tuple[RecordingLoader, ROILoader]]:
        recording_loader = self._get_the_recording_loader()
        all_roi_loaders = self._get_all_roi_loaders()
        rec_roi_loader_combos = [(recording_loader, roi_loader) for roi_loader in all_roi_loaders]
        return rec_roi_loader_combos
    

    def _get_the_recording_loader(self) -> RecordingLoader:
        recording_loader_factory = RecordingLoaderFactory()
        recording_filepath = self._get_filepaths_with_supported_extension_in_dirpath(recording_loader_factory.all_supported_extensions, 1)[0]
        recording_loader = recording_loader_factory.get_loader(recording_filepath)
        return recording_loader


    def _get_all_roi_loaders(self) -> List[ROILoader]:
        roi_loader_factory = ROILoaderFactory()
        all_roi_filepaths = self._get_filepaths_with_supported_extension_in_dirpath(roi_loader_factory.all_supported_extensions)
        all_roi_loaders = [roi_loader_factory.get_loader(filepath) for filepath in all_roi_filepaths]
        return all_roi_loaders



        

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

