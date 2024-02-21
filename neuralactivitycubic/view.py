import ipywidgets as w
import numpy as np

from typing import Dict

class WidgetsInterface:

    @property
    def layouts(self) -> Dict[str, w.Layout]:
        default_layout = w.Layout(justify_content = 'space-between', width = '100%')
        layouts = {'default': default_layout}
        return layouts

    @property
    def styles(self) -> Dict[str, w.Style]:
        default_style = {'description_width': 'initial'}
        styles = {'default': default_style}
        return styles

    def __init__(self) -> None:
        self.widget = self._build_widget()
        

    def _build_widget(self) -> w.VBox:
        welcome_banner = self._build_welcome_banner()
        directories_box = self._build_directories_box()
        display_elements = self._build_display_elements()
        settings_box = self._build_main_settings_box()
        widget = w.VBox([welcome_banner, directories_box, w.HBox([display_elements, settings_box])], layout = w.Layout(width = '100%', height = '880px'))
        return widget



    def _build_welcome_banner(self) -> w.HBox:
        welcome_html = w.HTML(value="<p style='font-size:32px; font-weight:bold; text-align:center;'>Welcome to NeuralActivityCubic</p>",
                              layout = w.Layout(width = '100%', height = '80px'))
        # include some branding logos
        return w.HBox([welcome_html], layout = w.Layout(width = '100%', height = '80px'))


    def _build_directories_box(self) -> w.VBox:
        self.user_settings_recording_filepath = w.Text(description = 'Filepath of Recording: ',
                                                       value = 'spiking_neuron.avi', 
                                                       placeholder = 'Provide filepath to ROI file',
                                                       style = {'description_width': 'initial'},
                                                       layout = w.Layout(width = '80%'))
        self.user_settings_roi_filepath = w.Text(description = 'Filepath of ROI file: ',
                                                 value = '',
                                                 placeholder = 'Provide filepath to ROI file',
                                                 style = {'description_width': 'initial'},
                                                 layout = w.Layout(width = '80%'))                         
        self.user_settings_results_directory = w.Text(description = 'Output directory for results: ',
                                                      value = '',
                                                      placeholder = 'Current working directory will be used if left empty',
                                                      style = {'description_width': 'initial'},
                                                      layout = w.Layout(width = '99%'))
        self.load_recording_button = w.Button(description = 'Load Recording', layout = w.Layout(width = '20%'))
        self.load_roi_button = w.Button(description = 'Load ROI', layout = w.Layout(width = '20%'))
        recordings_filepath_button_box = w.HBox([self.user_settings_recording_filepath, self.load_recording_button], layout = self.layouts['default'])
        roi_filepath_button_box = w.HBox([self.user_settings_roi_filepath, self.load_roi_button], layout = self.layouts['default'])
        directories_box = w.VBox([recordings_filepath_button_box, roi_filepath_button_box, self.user_settings_results_directory],
                                 layout = w.Layout(width = '100%', height = '128px', justify_content = 'space-around'))
        return directories_box

    

    def _build_display_elements(self) -> w.VBox:
        self.main_screen = w.Output(layout = w.Layout(width = '100%', height = '450px'))
        self.logs_screen = w.Output(layout = w.Layout(width = '100%', height = '100px'))
        display_elements = w.VBox([self.main_screen, self.logs_screen], layout = w.Layout(width = '60%', height = '5500 px'))
        return display_elements


    def _build_main_settings_box(self) -> w.VBox:
        self.user_settings_window_size = w.IntSlider(description='Window Size', 
                                                     value=8, 
                                                     min=1, 
                                                     max=999, 
                                                     step=1,
                                                     style = self.styles['default'],
                                                     layout = w.Layout(width = '99%'))
        self.user_settings_signal_to_noise_ratio = w.FloatSlider(description='Signal to Noise Ratio',
                                                                 value=3.0,
                                                                 min=0.05,
                                                                 max=30.0,
                                                                 step=0.05,
                                                                 style = self.styles['default'],
                                                                 layout = w.Layout(width = '99%'))

        self.user_settings_preview_only = w.Checkbox(description = 'Load preview frame only: ',
                                                     value = False,
                                                     style = self.styles['default'],
                                                     layout = w.Layout(width = '99%'))
        self.user_settings_frame_idx = w.BoundedIntText(description = 'Frame index to display for preview: ',
                                                        value = 0,
                                                        min = 0,
                                                        max = 999,
                                                        step = 1,
                                                        style = self.styles['default'],
                                                        layout = w.Layout(width = '99%'))                                                
        user_settings_box = w.VBox([self.user_settings_window_size,
                                    self.user_settings_signal_to_noise_ratio,
                                    self.user_settings_preview_only,
                                    self.user_settings_frame_idx], layout=w.Layout(width = '100%'))
        self.run_analysis_button = w.Button(description = 'Run analysis', icon = 'rocket', layout = w.Layout(width = '99%'))
        main_settings = w.VBox([user_settings_box, self.run_analysis_button], layout = w.Layout(justify_content = 'space-between', width = '40%', height = '550px'))
        return main_settings


    def show_on_main_window(self, image_to_show) -> None:
        # display the image passed 
        pass


    def add_to_logs(self, message: str) -> None:
        with self.logs_output:
            print(message)