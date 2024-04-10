from .model import Model
from .view import WidgetsInterface
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import inspect
from IPython.display import display

from typing import Any, Callable, Dict

class App:

    def __init__(self):
        self.model = Model()
        self.view = WidgetsInterface()
        self._bind_buttons_to_functions()

    
    def launch(self) -> None:
        display(self.view.widget)

    
    def _bind_buttons_to_functions(self) -> None:
        self.view.io_panel.run_analysis_button.on_click(self._run_button_clicked)
        self.view.io_panel.load_recording_button.on_click(self._load_recording_button_clicked)
        self.view.io_panel.load_roi_button.on_click(self._load_roi_button_clicked)


    def _run_button_clicked(self, change) -> None:
        self.view.enable_analysis(False)
        self.view.update_infos(logs_message = 'Validating user input for analysis', progress_in_percent = 5.0)
        validated_user_settings_analysis = self._get_validated_user_settings_required_for_model_function(self.model.run_analysis)
        self.view.update_infos(logs_message = 'Validation successful! Starting analysis ...', progress_in_percent = 10.0)
        self.model.run_analysis(**validated_user_settings_analysis)
        self.view.update_infos(logs_message = 'Analysis completed! Generating output results ...', progress_in_percent = 70.0)
        validated_user_settings_detailed_results = self._get_validated_user_settings_required_for_model_function(self.model.create_detailed_results)
        self.model.create_detailed_results(**validated_user_settings_detailed_results)
        self.view.update_infos(logs_message = 'Detailed results generated and saved! Generating overview ...', progress_in_percent = 95.0)
        validated_user_settings_overview_results = self._get_validated_user_settings_required_for_model_function(self.model.create_overview_results)
        self.view.main_screen.show_output_screen()
        with self.view.main_screen.output:
            self.model.create_overview_results(**validated_user_settings_overview_results)
        self.view.update_infos(logs_message = 'All result files have been created.')
        self.view.update_infos(logs_message = 'All jobs finished!', progress_in_percent = 100.0)
        self.view.enable_analysis(True)


    def _load_recording_button_clicked(self, change) -> None:
        validated_user_settings = self._get_validated_user_settings_required_for_model_function(self.model.load_recording)
        self.view.update_infos(logs_message = f'Loading data from specified input path: {validated_user_settings["recording_filepath"]}',
                               progress_in_percent = 10.0)
        self.model.load_recording(**validated_user_settings)
        self.view.adjust_widgets_to_loaded_data(total_frames = self.model.recording_zstack.shape[0])
        self.view.update_infos(logs_message = 'Recording data successfully loaded', progress_in_percent = 95.0)
        self.view.main_screen.show_output_screen()
        with self.view.main_screen.output:
            plt.imshow(self.model.recording_preview)
            plt.show()
        self.view.update_infos(progress_in_percent = 100.0)
        self.view.enable_analysis()


    def _load_roi_button_clicked(self, change) -> None:
        #validated_user_settings = self._get_validated_user_settings_required_for_model_function(self.model.load_roi)
        #self.model.load_roi(**validated_user_settings)
        #self.view.show_on_main_window(self.model.recording_preview_with_superimposed_rois)
        pass


    def _get_validated_user_settings_required_for_model_function(self, model_func: Callable) -> Dict[str, Any]:
        all_user_settings = self.view.export_user_settings()
        relevant_user_settings = {}
        for expected_parameter_name in inspect.signature(model_func).parameters:
            self._validate_user_settings_for_model_function(model_func, all_user_settings, expected_parameter_name)
            relevant_user_settings[expected_parameter_name] = all_user_settings[expected_parameter_name]
        return relevant_user_settings


    def _validate_user_settings_for_model_function(self, model_func: Callable, user_settings: Dict[str, Any], expected_parameter_name: str) -> None:
        assert expected_parameter_name in user_settings, (f'{model_func.__name__} requires the parameter "{expected_parameter_name}", '
                                                          f'which is not included in the user settings ({list(user_settings.keys())})')
        value_set_by_user = user_settings[expected_parameter_name]
        expected_parameter_type = inspect.signature(model_func).parameters[expected_parameter_name].annotation
        if expected_parameter_type == Path:
            assert isinstance(value_set_by_user, Path), (f'{model_func.__name__} requires the parameter "{expected_parameter_name}" to be a '
                                                         f'pathlib.Path object. However, {value_set_by_user}, which is of type '
                                                         f'{type(value_set_by_user)}, was passed.')
        else:
            assert expected_parameter_type == type(value_set_by_user), (f'{model_func.__name__} requires the parameter "{expected_parameter_name}" '
                                                                        f'to be of type {expected_parameter_type}. However, {value_set_by_user}, which '
                                                                        f'is of type {type(value_set_by_user)}, was passed.')