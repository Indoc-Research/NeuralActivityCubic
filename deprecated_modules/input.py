import imageio.v3 as iio
from pathlib import Path
import numpy as np
from shapely import Polygon
import roifile
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Union, Optional, Any



class FocusAreaPathRestrictions:
    
    @property
    def supported_dir_names(self) -> List[str]:
        supported_dir_names =  ['focus_area', 'focus_areas', 'focus-area', 'focus-areas', 'focus area', 'focus areas',
                                'Focus_Area', 'Focus_Areas', 'Focus-Area', 'Focus-Areas', 'Focus Area', 'Focus Areas']
        return supported_dir_names
    

#################################
###### Source Data Handler ######
#################################

class Data(ABC):

    @abstractmethod
    def _parse_loaded_data(self, loaded_data: Any) -> None:
        pass

    def __init__(self, filepath: Path, loaded_data: Any) -> None:
        self.filepath = filepath
        self._parse_loaded_data(loaded_data)
        


class Recording(Data):

    def _parse_loaded_data(self, loaded_data: np.ndarray) -> None:
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



class ROI(Data):

    def _parse_loaded_data(self, loaded_data: List[Tuple[int, int]]) -> None:
        self.boundary_row_col_coords = loaded_data
        self.as_polygon = self._convert_to_valid_polygon()


    def _convert_to_valid_polygon(self) -> Polygon:
        roi_as_polygon = Polygon(self.boundary_row_col_coords)
        assert roi_as_polygon.is_valid, f'Something went wrong when trying to create a Polygon out of your ROI: {self.filepath}.'
        return roi_as_polygon


    def add_label_id(self, label_id: str) -> None:
        assert type(label_id) == str, f'"label_id" must be a string. However, you passed {label_id} which is of type {type(label_id)}.'
        setattr(self, 'label_id', label_id)



################################
###### Source Data Loader ######
################################


class DataLoader(ABC):

    @abstractmethod
    def load_and_parse_file_content(self) -> Union[Data, List[Data]]:
        # This method will be called when the data should be loaded for analysis
        pass

    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath



class GridWrapperROILoader(DataLoader):

    def set_configs_for_grid_creation(self, image_width: int, image_height: int, window_size: int) -> None:
        self.configs = {}
        self._add_to_configs_and_create_as_attribute('image_width', image_width)
        self._add_to_configs_and_create_as_attribute('image_height', image_height)
        self._add_to_configs_and_create_as_attribute('window_size', window_size)


    def _add_to_configs_and_create_as_attribute(self, attribute_name: str, value: Any) -> None:
        self.configs[attribute_name] = value
        setattr(self, attribute_name, self.configs[attribute_name])

    
    def load_and_parse_file_content(self) -> List[ROI]:
        row_cropping_idx, col_cropping_idx = self._get_cropping_indices_to_adjust_for_window_size()
        self._add_to_configs_and_create_as_attribute('row_cropping_idx', row_cropping_idx)
        self._add_to_configs_and_create_as_attribute('col_cropping_idx', col_cropping_idx)        
        grid_row_idxs, grid_col_idxs = self._get_row_col_idxs_of_grid()
        grid_row_labels, grid_col_labels = self._get_row_col_labels_for_rois_in_grid()
        self._add_to_configs_and_create_as_attribute('max_len_row_label_id', len(str(grid_row_labels[-1])))
        self._add_to_configs_and_create_as_attribute('max_len_col_label_id', len(str(grid_col_labels[-1])))
        all_rois = []
        for row_idx, row_label in zip(grid_row_idxs, grid_row_labels):
            for col_idx, col_label in zip(grid_col_idxs, grid_col_labels):
                square_corner_row_col_coords = self._get_boundary_row_col_coords_single_square(row_idx, col_idx)
                label_id = f'{row_label}/{col_label}'
                square_roi = ROI(self.filepath, square_corner_row_col_coords)
                square_roi.add_label_id(label_id)
                all_rois.append(square_roi)
        return all_rois     

    
    def _get_cropping_indices_to_adjust_for_window_size(self) -> Tuple[int, int]:
        row_cropping_index = (self.image_height // self.window_size) * self.window_size
        col_cropping_index = (self.image_width // self.window_size) * self.window_size
        return row_cropping_index, col_cropping_index
        

    def _get_row_col_idxs_of_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        grid_row_idxs = np.arange(start = 0, stop = self.row_cropping_idx, step = self.window_size)
        grid_col_idxs = np.arange(start = 0, stop = self.col_cropping_idx, step = self.window_size)
        return grid_row_idxs, grid_col_idxs


    def _get_row_col_labels_for_rois_in_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        grid_row_labels = np.arange(start = 1, stop = self.row_cropping_idx / self.window_size + 1, step = 1, dtype = 'int')
        grid_col_labels = np.arange(start = 1, stop = self.col_cropping_idx / self.window_size + 1, step = 1, dtype = 'int')
        return grid_row_labels, grid_col_labels
                
    
    def _get_boundary_row_col_coords_single_square(self, upper_left_corner_row_idx: int, upper_left_corner_col_idx: int) -> List[Tuple[(int, int)]]:
        upper_left_corner = (upper_left_corner_row_idx, upper_left_corner_col_idx)
        upper_right_corner = (upper_left_corner_row_idx, upper_left_corner_col_idx + self.window_size)
        lower_right_corner = (upper_left_corner_row_idx + self.window_size, upper_left_corner_col_idx + self.window_size)
        lower_left_corner = (upper_left_corner_row_idx + self.window_size, upper_left_corner_col_idx)
        return [upper_left_corner, lower_left_corner, lower_right_corner, upper_right_corner, upper_left_corner]



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


    def load_and_parse_file_content(self) -> Recording:
        all_frames = self._load_all_frames()
        recording = Recording(self.filepath, all_frames)
        return recording


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
        


class ROILoader(DataLoader):

    @abstractmethod
    def _get_boundary_row_col_coords_for_all_rois_in_source_data(self) -> List[List[Tuple[int, int]]]: 
        # To be implemented in individual subclasses
        # Return a list of Tuples, where each tuple represents one boundary point: (row_coord, col_coord)
        pass

    
    def load_and_parse_file_content(self) -> List[ROI]:
        boundary_row_col_coords_for_all_rois = self._get_boundary_row_col_coords_for_all_rois_in_source_data()
        all_rois = []
        for boundary_row_col_coords_single_roi in boundary_row_col_coords_for_all_rois:
            boundary_row_col_coords_single_roi = self._add_first_boundary_point_also_add_end_to_close_roi(boundary_row_col_coords_single_roi)
            roi = ROI(self.filepath, boundary_row_col_coords_single_roi)
            all_rois.append(roi)
        return all_rois


    def _add_first_boundary_point_also_add_end_to_close_roi(self, boundary_row_col_coords: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        first_boundary_point_coords = boundary_row_col_coords[0]
        boundary_row_col_coords.append(first_boundary_point_coords)
        return boundary_row_col_coords



class ImageJROILoader(ROILoader):

    
    def _get_boundary_row_col_coords_for_all_rois_in_source_data(self) -> List[List[Tuple[int, int]]]:
        roi_file_content = roifile.roiread(self.filepath)
        if type(roi_file_content) == list:
            all_rois = self._extract_boundary_row_col_coords_from_roi_set(roi_file_content)
        else:
            all_rois = [self._extract_boundary_row_col_coords_from_single_roi(roi_file_content)]
        return all_rois


    def _extract_boundary_row_col_coords_from_roi_set(self, all_imagej_rois: List[roifile.roifile.ImagejRoi]) -> List[List[Tuple[int, int]]]:
        boundary_coords_all_rois = []
        for imagej_roi in all_imagej_rois:
            boundary_row_col_coords_single_roi = self._extract_boundary_row_col_coords_from_single_roi(imagej_roi)
            boundary_coords_all_rois.append(boundary_row_col_coords_single_roi)
        return boundary_coords_all_rois
    
        
    def _extract_boundary_row_col_coords_from_single_roi(self, imagej_roi: roifile.roifile.ImagejRoi) -> List[Tuple[int, int]]:
        row_coords = imagej_roi.coordinates()[:, 1]
        col_coords = imagej_roi.coordinates()[:, 0]
        boundary_row_col_coords = list(zip(row_coords, col_coords))
        return boundary_row_col_coords
        


##############################
###### Loader Factories ######
##############################

class DataLoaderFactory(ABC):

    @property
    @abstractmethod
    def supported_extensions_per_data_loader(self) -> Dict[DataLoader, List[str]]:
        pass

    
    @property
    def all_supported_extensions(self) -> List[str]:
        all_supported_extensions = []
        for value in self.supported_extensions_per_data_loader.values():
            all_supported_extensions += value
        return all_supported_extensions

    
    def get_loader(self, filepath: Path) -> DataLoader:
        self._assert_validity_of_filepath(filepath)
        data_loader = self._get_loader_for_file_extension(filepath)
        return data_loader
        

    def _assert_validity_of_filepath(self, filepath: Path) -> None:
        assert isinstance(filepath, Path), f'filepath must be an instance of a pathlib.Path. However, you passed {filepath}, which is of type {type(filepath)}'
        assert filepath.exists(), f'The filepath you provided ({filepath}) does not seem to exist!'
        

    def _get_loader_for_file_extension(self, filepath: Path) -> DataLoader:
        matching_loader = None
        for loader_subclass, supported_extensions in self.supported_extensions_per_data_loader.items():
            if filepath.suffix in supported_extensions:
                matching_loader = loader_subclass(filepath)
                break
        if matching_loader == None:
            raise NotImplementedError('It seems like there is no DataLoader implemented for the specific filetype you´re trying to load - sorry!')
        return matching_loader



class RecordingLoaderFactory(DataLoaderFactory):

    @property
    def supported_extensions_per_data_loader(self) -> Dict[RecordingLoader, List[str]]:
        supported_extensions_per_data_loader = {AVILoader: ['.avi']}
        return supported_extensions_per_data_loader



class ROILoaderFactory(DataLoaderFactory):

    @property
    def supported_extensions_per_data_loader(self) -> Dict[ROILoader, List[str]]:
        supported_extensions_per_data_loader = {ImageJROILoader: ['.roi', '.zip']}
        return supported_extensions_per_data_loader



def get_filepaths_with_supported_extension_in_dirpath(dirpath: Path, all_supported_extensions: List[str], max_results: Optional[int]=None) -> List[Path]:
    all_filepaths_with_supported_extension = []
    for elem in dirpath.iterdir():
        if elem.is_file() == True:
            if elem.suffix in all_supported_extensions:
                all_filepaths_with_supported_extension.append(elem)
    if type(max_results) == int:
        assert len(all_filepaths_with_supported_extension) <= max_results, (
            f'There are more than {max_results} file(s) of supported type in {dirpath}, '
            f'but only a maximum of {max_results} are allowed. Please remove at least '
            f'{len(all_filepaths_with_supported_extension) - max_results} of the following' 
            f'files and try again: {all_filepaths_with_supported_extension}'
        )
    return all_filepaths_with_supported_extension


######################################
###### Handler for Combinations ######
######################################

class RecLoaderROILoaderCombinator:

        
    def __init__(self, dir_path: Path) -> None:
        self.dir_path = dir_path

    
    def get_all_recording_and_roi_loader_combos(self) -> List[Tuple[RecordingLoader, ROILoader]]:
        recording_loader = self._get_the_recording_loader()
        all_roi_loaders = self._get_all_roi_loaders()
        if len(all_roi_loaders) > 0:
            rec_roi_loader_combos = [(recording_loader, roi_loader) for roi_loader in all_roi_loaders]
        else:
            rec_roi_loader_combos = [(recording_loader, None)]
        return rec_roi_loader_combos
    

    def _get_the_recording_loader(self) -> RecordingLoader:
        recording_loader_factory = RecordingLoaderFactory()
        recording_filepath = get_filepaths_with_supported_extension_in_dirpath(self.dir_path, recording_loader_factory.all_supported_extensions, 1)[0]
        recording_loader = recording_loader_factory.get_loader(recording_filepath)
        return recording_loader


    def _get_all_roi_loaders(self) -> List[ROILoader]:
        roi_loader_factory = ROILoaderFactory()
        all_roi_filepaths = get_filepaths_with_supported_extension_in_dirpath(self.dir_path, roi_loader_factory.all_supported_extensions)
        all_roi_loaders = [roi_loader_factory.get_loader(filepath) for filepath in all_roi_filepaths]
        return all_roi_loaders
