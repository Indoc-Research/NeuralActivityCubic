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
        self.view.run_analysis_button.on_click(self._run_button_clicked)
        self.view.load_recording_button.on_click(self._load_recording_button_clicked)
        self.view.load_roi_button.on_click(self._load_roi_button_clicked)


    def _run_button_clicked(self, change) -> None:
        validated_user_settings_analysis = self._get_validated_user_settings_required_for_model_function(self.model.run_analysis)
        self.model.run_analysis(**validated_user_settings_analysis)
        validated_user_settings_overview_results = self._get_validated_user_settings_required_for_model_function(self.model.create_overview_results)
        self.view.main_screen.clear_output()
        with self.view.main_screen:
            self.model.create_overview_results(**validated_user_settings_overview_results)


    def _load_recording_button_clicked(self, change) -> None:
        validated_user_settings = self._get_validated_user_settings_required_for_model_function(self.model.load_recording)
        self.model.load_recording(**validated_user_settings)
        with self.view.main_screen:
            plt.imshow(self.model.recording_preview)
            plt.show()


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

    # Was now moved to WidgetsInterface in view.py
    """
    def _export_current_user_settings(self) -> Dict[str, Any]:
        user_settings = {}
        for attribute_name, attribute_value in vars(self.view).items():
            if attribute_name.startswith('user_settings_'):
                value_set_by_user = attribute_value.value
                parameter_name = attribute_name.replace('user_settings_', '')
                if 'filepath' in parameter_name:
                    value_set_by_user = Path(value_set_by_user)
                user_settings[parameter_name] = value_set_by_user
        return user_settings
    """


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