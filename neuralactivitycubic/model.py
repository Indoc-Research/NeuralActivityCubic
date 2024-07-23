from pathlib import Path
import json
from datetime import datetime, timezone
import ipywidgets as w
import matplotlib.pyplot as plt
import inspect
import multiprocessing

from .processing import AnalysisJob
from .input import RecordingLoaderFactory, ROILoaderFactory, RecLoaderROILoaderCombinator, get_filepaths_with_supported_extension_in_dirpath

from typing import Tuple, Dict, Callable, Optional, Any
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes 



class Model:

    def __init__(self) -> None:
        self.num_processes = multiprocessing.cpu_count()
        self.analysis_job_queue = []
        self.logs = []
        self.gui_enabled = False


    def setup_connection_to_update_infos_in_view(self, update_infos: Callable) -> None:
        self.callback_view_update_infos = update_infos
        self._check_if_gui_setup_is_completed()


    def setup_connection_to_display_results(self, show_output_screen: Callable, output: w.Output, pixel_conversion: float) -> None:
        self.callback_view_show_output_screen = show_output_screen
        self.view_output = output
        self.pixel_conversion = pixel_conversion
        self._check_if_gui_setup_is_completed()


    def _check_if_gui_setup_is_completed(self) -> None:
        checklist_expected_attributes_with_type = {'callback_view_update_infos': Callable,
                                                   'callback_view_show_output_screen': Callable,
                                                   'view_output': w.Output,
                                                   'pixel_conversion': float}
        confirmed_attributes = []
        failed_attributes = []
        for attribute_name, expected_type in checklist_expected_attributes_with_type.items():
            if hasattr(self, attribute_name) == True:
                attribute = getattr(self, attribute_name)
                if type(attribute) == expected_type:
                    confirmed_attributes.append(True)
                elif isinstance(attribute, expected_type):
                    confirmed_attributes.append(True)
                else:
                    failed_attributes.append(attribute_name)
            else:
                failed_attributes.append(attribute_name)
        if len(confirmed_attributes) == len(checklist_expected_attributes_with_type.keys()):
            self.gui_enabled = True
            self.add_info_to_logs('NA3 GUI initialization completed. Start by selecting and loading source data.', True)
        else:
            self.gui_enabled = False
            for attribute_name in failed_attributes:
                self.add_info_to_logs(f'Setup of {attribute_name} missing before GUI connection can be enabled.')


    def add_info_to_logs(self, message: str, display_in_gui: bool=False, progress_in_percent: Optional[float]=None) -> None:
        time_prefix_in_utc = datetime.now(timezone.utc).strftime('%d-%m-%y %H:%M:%S.%f')
        self.logs.append(f'{time_prefix_in_utc} (UTC): {message}')
        if (display_in_gui == True) and (self.gui_enabled == True): 
            self.callback_view_update_infos(message, progress_in_percent)

    
    def create_analysis_jobs(self, configs: Dict[str, Any]) -> None:
        self._ensure_data_from_previous_jobs_was_removed()
        validated_configs = self._get_configs_required_for_specific_function(configs, self._assertion_for_create_analysis_jobs)
        self.add_info_to_logs('Basic configurations for data import validated. Starting creation of analysis job(s)...', True)
        if validated_configs['batch_mode'] == False:
            if validated_configs['focus_area'] == False:
                self._add_new_recording_without_rois_to_analysis_job_queue(validated_configs['data_source_path'], validated_configs['batch_mode'])
            else: #'focus_area' == True:
                self._add_new_recording_with_rois_to_analysis_job_queue(validated_configs['data_source_path'])
        else: #'batch_mode' == True:
            all_subdirs = [subdir for subdir in validated_configs['data_source_path'].iterdir() if subdir.is_dir() and (subdir.name.startswith('.') == False)]
            all_subdirs.sort()
            total_step_count = len(all_subdirs)
            progress_step_size = 100 / total_step_count
            if validated_configs['focus_area'] == False:
                for idx, subdir_path in enumerate(all_subdirs):
                    self._add_new_recording_without_rois_to_analysis_job_queue(subdir_path, validated_configs['batch_mode'])
                    self.add_info_to_logs(f'Job creation for {subdir_path} completed.', True, min((idx+1)*progress_step_size, 100.0))
            else: #'focus_area' == True:
                for idx, subdir_path in enumerate(all_subdirs):
                    self._add_new_recording_with_rois_to_analysis_job_queue(subdir_path)
                    self.add_info_to_logs(f'Job creation(s) for {subdir_path} completed.', True, min((idx+1)*progress_step_size, 100.0))
        self.add_info_to_logs('All job creation(s) completed.', True, 100.0)
        

    def _ensure_data_from_previous_jobs_was_removed(self) -> None:
        if len(self.analysis_job_queue) > 0:
            self.add_info_to_logs('Loading of new source data. All previously created jobs & logs will be deleted.', True)
            self.analysis_job_queue = []
            self.logs = []
            self._check_if_gui_setup_is_completed()

    
    def _assertion_for_create_analysis_jobs(self, roi_mode: bool, batch_mode: bool, focus_area: bool, data_source_path: Path) -> None:
        # just a convenience function to use the existing config validation and filtering methods
        pass
        

    def _add_new_recording_without_rois_to_analysis_job_queue(self, filepath: Path, batch_mode: bool) -> None:
        rec_loader_factory = RecordingLoaderFactory()
        job_idx = len(self.analysis_job_queue)
        if batch_mode == True:
            self.add_info_to_logs(f'Looking for a valid recording file in {filepath}...', True)
            valid_filepaths = get_filepaths_with_supported_extension_in_dirpath(filepath, rec_loader_factory.all_supported_extensions, 1)
            if len(valid_filepaths) == 0:
                self.add_info_to_logs(f'Could not find any recording files of supported type at {filepath}!', True)
            elif len(valid_filepaths) >  1:
                too_many_files_message = (f'Found more than a single recording file of supported type at {filepath}, i.e.: {valid_filepaths}. '
                                          f'However, only a single file was expected. NA3 continues with {valid_filepaths[0]} and will ignore the other files.')
                self.add_info_to_logs(too_many_files_message, True)
                recording_filepath = valid_filepaths[0]
            else:
                recording_filepath = valid_filepaths[0]
        else:
            recording_filepath = filepath
        self.add_info_to_logs(f'Starting to create a job for: {recording_filepath}', True)
        try:
            recording_loader = rec_loader_factory.get_loader(recording_filepath)
        except NotImplementedError:
            message = (f'Job creation failed! The data you selected ("{recording_filepath}") is not a supported recording file type! '
                       f'Currently supported file types for recordings are: {rec_loader_factory.all_supported_extensions}.')
            if batch_mode == True:
                message += 'For your convenience, this job is skipped and NA3 continues with the next one instead.'
            self.add_info_to_logs(message, True)
        except Exception as e:
            message = 'Job creation failed due to an unexpected error: '
            all_unexpected_exceptions = [exception_message for exception_message in e.args]
            for unexpected_exception in all_unexpected_exceptions:
                message += f'"{unexpected_exception}"; '
            self.add_info_to_logs(message, True)
            self.add_info_to_logs('Continuing with next job, if available.', True)
        else: # no errors in try
            self.analysis_job_queue.append(AnalysisJob(self.num_processes, recording_loader))
            self.add_info_to_logs(f'Successfully created job for: {recording_filepath} at index #{job_idx}.', True)


    def _add_new_recording_with_rois_to_analysis_job_queue(self, dir_path) -> None:
        self.add_info_to_logs(f'Looking for a recording file and ROI files in {dir_path} and trying to create corresponding jobs...', True)
        rec_roi_loader = RecLoaderROILoaderCombinator(dir_path)
        try:
            recording_and_roi_loader_combos = rec_roi_loader.get_all_recording_and_roi_loader_combos()
        except NotImplementedError:
            rec_loader_factory = RecordingLoaderFactory()
            message = (f'Job creation failed! Could not find a supported recording file in "{dir_path}"! '
                       f'Currently supported file types for recordings are: {rec_loader_factory.all_supported_extensions}. '
                        'For your convenience, this job is skipped and NA3 continues with the next one instead.')
            self.add_info_to_logs(message, True)
        except Exception as e:
            message = 'Job creation failed due to an unexpected error: '
            all_unexpected_exceptions = [exception_message for exception_message in e.args]
            for unexpected_exception in all_unexpected_exceptions:
                message += f'"{unexpected_exception}"; '
            self.add_info_to_logs(message, True)
            self.add_info_to_logs('Continuing with next job, if available.', True)
        else: # no errors in try
            for recording_loader, roi_loader in recording_and_roi_loader_combos:
                job_idx = len(self.analysis_job_queue)
                self.analysis_job_queue.append(AnalysisJob(self.num_processes, recording_loader, roi_loader))
                self.add_info_to_logs(f'Successfully created job for: {recording_loader.filepath} with {roi_loader.filepath} at index #{job_idx}.', True)
    

    def run_analysis(self, configs: Dict[str, Any]) -> None:
        sample_job_to_validate_configs = self.analysis_job_queue[0]
        validated_configs_for_analysis = self._get_configs_required_for_specific_function(configs, sample_job_to_validate_configs.run_analysis)
        validated_configs_for_result_creation = self._get_configs_required_for_specific_function(configs, sample_job_to_validate_configs.create_results)
        self.add_info_to_logs('Configurations for Analysis Settings and Result Creation validated successfully.', True)
        self.add_info_to_logs(f'Analysis Settings are:')
        for key, value in validated_configs_for_analysis.items():
            self.add_info_to_logs(f'{key}: {str(value)}')     
        self.add_info_to_logs('Starting analysis...', True)
        total_step_count = len(self.analysis_job_queue)
        progress_step_size = 100 / total_step_count
        for job_idx, analysis_job in enumerate(self.analysis_job_queue):
            self.add_info_to_logs(f'Starting to process analysis job with index #{job_idx}.')
            analysis_job.run_analysis(**validated_configs_for_analysis)
            self.add_info_to_logs(f'Analysis successfully completed. Continue with creation of results.. ')
            analysis_job.create_results(**validated_configs_for_result_creation)
            self.add_info_to_logs(f'Results successfully created at: {analysis_job.results_dir_path}')
            if self.gui_enabled == True:
                self.callback_view_show_output_screen()
                with self.view_output:
                    overview_results_fig, overview_results_ax = analysis_job.overview_results
                    overview_results_fig.set_figheight(400 * self.pixel_conversion)
                    overview_results_fig.tight_layout()
                    plt.show(overview_results_fig)
            self._save_user_settings_as_json(configs, analysis_job)
            self.add_info_to_logs(f'Successfully finished processing of analysis job with index #{job_idx}. {job_idx + 1} of {total_step_count} total job(s) completed.', 
                                  True,
                                  min((job_idx+1)*progress_step_size, 100.0))
            self._save_logs_at_current_moment(analysis_job.results_dir_path)
        self.add_info_to_logs('Updating all log files to contain all logs as final step. All valid logs files will end with this message.')
        for job_idx, analysis_job in enumerate(self.analysis_job_queue):
            self._save_logs_at_current_moment(analysis_job.results_dir_path)
            

    def _get_configs_required_for_specific_function(self, all_configs: Dict[str, Any], function_to_execute: Callable) -> Dict[str, Any]:
        filtered_and_validated_configs = {}
        for expected_parameter_name in inspect.signature(function_to_execute).parameters:
            self._validate_individual_config_value_against_function_type_hints(all_configs, function_to_execute, expected_parameter_name)
            filtered_and_validated_configs[expected_parameter_name] = all_configs[expected_parameter_name]
        return filtered_and_validated_configs


    def _validate_individual_config_value_against_function_type_hints(self, all_configs: Dict[str, Any], function_to_execute: Callable, expected_parameter_name: str) -> None:
        assert expected_parameter_name in all_configs.keys(), (
            f'{function_to_execute.__name__} requires the parameter "{expected_parameter_name}", '
            f'which is not included in the configs ({list(all_configs.keys())})'
        )
        value_in_configs = all_configs[expected_parameter_name]
        expected_parameter_type = inspect.signature(function_to_execute).parameters[expected_parameter_name].annotation
        if expected_parameter_type == Path:
            assert isinstance(value_in_configs, Path), (
                f'{function_to_execute.__name__} requires the parameter "{expected_parameter_name}" to be a '
                f'pathlib.Path object. However, {value_in_configs}, which is of type '
                f'{type(value_in_configs)}, was passed.'
            )
        else:
            assert expected_parameter_type == type(value_in_configs), (
                f'{function_to_execute.__name__} requires the parameter "{expected_parameter_name}" '
                f'to be of type {expected_parameter_type}. However, {value_in_configs}, which '
                f'is of type {type(value_in_configs)}, was passed.'
            )


    def _save_logs_at_current_moment(self, results_dir_path: Path) -> None:
        filepath = results_dir_path.joinpath('logs.txt')
        with open(filepath , 'w+') as logs_file:
            for log_message in self.logs:
                logs_file.write(f'{log_message}\n')


    def _save_user_settings_as_json(self, configs: Dict[str, Any], analysis_job: AnalysisJob) -> None:
        filepath = analysis_job.results_dir_path.joinpath('user_settings.json')
        configs['recording_filepath'] = analysis_job.recording.filepath
        if analysis_job.roi_based == True:
            configs['roi_filepath'] = analysis_job.roi.filepath
        else:
            configs['roi_filepath'] = None
        configs_preformatted_for_json = {}
        for key, value in configs.items():
            if isinstance(value, Path):
                configs_preformatted_for_json[key] = value.as_posix()
            else:
                configs_preformatted_for_json[key] = value
        with open(filepath, 'w+') as user_settings_json: 
            json.dump(configs_preformatted_for_json, user_settings_json)


    def preview_window_size(self, configs: Dict[str, Any]) -> Tuple[Figure, Axes]:
        job_for_preview = self.analysis_job_queue[0]
        validated_configs_for_preview = self._get_configs_required_for_specific_function(configs, job_for_preview.preview_window_size)
        preview_fig, preview_ax = job_for_preview.preview_window_size(**validated_configs_for_preview)
        return preview_fig, preview_ax

