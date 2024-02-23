import ipywidgets as w
from ipyfilechooser import FileChooser
from IPython.display import Image
import numpy as np
from datetime import datetime

from typing import Dict, Optional, Any
from pathlib import Path


class UserInfoPanel:

    def __init__(self) -> None:
        self.widget = self._build_widget()
        # Initialize basic parameters and configs:
        self.max_lines_that_can_be_displayed_in_output = int(self.detailed_logs_output.layout.max_height.replace('px', '')) / 15
        self.logs_message_count = 0
        self.progress_in_percent = 0.0


    def _build_widget(self) -> w.VBox:
        info = w.HTML(value="<p style='font-size:16px; font-weight:bold; text-align:center;'>Info</p>", layout = w.Layout(width = '40px'))
        self.latest_logs = w.Label(value = ' ... the latest logs message will be displayed here ... ',
                                   style = {'font_style': 'italic', 'text_color': 'gray', 'font_family': 'monospace', 'text_align': 'center'},
                                   layout = w.Layout(width = 'initial'))
        self.progress_bar = w.FloatProgress(description = 'Progress', style = {'description_width': 'initial'}, layout = w.Layout(width = '20%'))
        info_overview_box = w.HBox([info, self.latest_logs, self.progress_bar], layout = w.Layout(justify_content = 'space-between', align_items = 'center', width = '95%'))
        self.detailed_logs_output = w.Output(layout = w.Layout(max_height = '200px', y_overflow='scroll'))
        self.detailed_logs_accordion = w.Accordion(children = [self.detailed_logs_output], titles = ('Detailed logs', ), selected_index = None, layout = w.Layout(width = '95%'))
        vertical_spacer = w.HTML(value = '', layout = w.Layout(height = '15px'))
        return w.VBox([info_overview_box, self.detailed_logs_accordion, vertical_spacer], layout = w.Layout(align_items = 'center'))


    def add_new_logs(self, message: str) -> None:
        assert type(message) == str, f'UserInfoHandler.add_new_logs() expects a message of type string. However, you passed {message}, which is of type {type(message)}.'
        current_time = datetime.now(user_timezone)
        # Update latest logs:
        time_prefix_latest_logs = current_time.strftime('%H-%M-%S')
        if len(message) > 125:
            self.latest_logs.value = f'{time_prefix_latest_logs}: {message[:85]}...(find full message in detailed logs)'
        else:
            self.latest_logs.value = f'{time_prefix_latest_logs}: {message}'
        # Update detailed logs:
        time_prefix_detailed_logs = current_time.strftime('%d-%m-%y %H:%M:%S.%f')
        self.logs_message_count += 1
        # Check if title has already been changed:
        if self.detailed_logs_accordion.get_title(0) == 'Detailed logs':
            # If not, check if max lines have been reached with this new message and adjust title accordingly:
            if self.logs_message_count >= self.max_lines_that_can_be_displayed_in_output:
                self.detailed_logs_accordion.set_title(0, 'Detailed logs - please scroll down to see latest logs')
        with self.detailed_logs_output:
            print(f'{time_prefix_detailed_logs}: {message}')


    def update_progress_bar(self, progress_in_percent: float) -> None:
        assert type(progress_in_percent) == float, ('UserInfoHandler.update_progress_bar() expects the progress_in_percent of type float. '
                                                    f'However, you passed {progress_in_percent}, which is of type {type(progress_in_percent)}.')
        if progress_in_percent == 100.0:
            self.progress_bar.bar_style = 'success'
        self.progress_bar.value = progress_in_percent



class IOPanel:
    
    def __init__(self, user_info_panel: UserInfoPanel) -> None:
        self.user_info_panel = user_info_panel
        self.widget = self._build_widget()


    def _build_widget(self):
        self.io_recording_box = self._build_io_recording_box()
        self.io_roi_box = self._build_io_roi_box()
        self.io_results_box = self._build_io_results_box()
        return w.HBox([self.io_recording_box,
                       self.io_roi_box,
                       self.io_results_box],
                      layout = w.Layout(width = '100%'))



    def _build_io_results_box(self) -> w.VBox:
        io_results_info = w.HTML(value="<p style='font-size:16px; font-weight:bold; text-align:center;'>Results output</p>")
        self.user_settings_include_overview_results = w.Checkbox(description = 'Create overview results', value = True, style = {'description_width': 'initial'})
        self.user_settings_include_detailed_results = w.Checkbox(description = 'Create detailed results', value = True, style = {'description_width': 'initial'})
        self.user_settings_results_filepath = FileChooser(title = 'Please select directory in which the output files shall be saved:', show_only_dirs = True, layout = w.Layout(width = '90%'))
        self.user_settings_results_filepath.rows = 4
        self.run_analysis_button = w.Button(description = 'Run analysis',
                                            disabled = True,
                                            tooltip = 'You have to load some data first, before you can run the analysis!',
                                            button_style = 'danger',
                                            icon = 'rocket',
                                            layout = w.Layout(width = '90%'))
        io_results_box = w.VBox([io_results_info,
                                 self.user_settings_include_overview_results,
                                 self.user_settings_include_detailed_results,
                                 self.user_settings_results_filepath,
                                 self.run_analysis_button,
                                 self._get_spacer()],
                                layout = w.Layout(width = '33%', max_height = '400px', align_items='center'))
        return io_results_box
    


    def _build_io_roi_box(self) -> w.VBox:
        # Create and configure all elements:
        io_roi_info = w.HTML(value="<p style='font-size:16px; font-style:italic; text-align:center;'>ROIs to focus on (optional)</p>")
        self.user_settings_enable_rois = w.Checkbox(description = 'Enable ROI-based processing', value = False, style = {'description_width': 'initial'})
        self.batch_processing_info = w.Label(value = 'Batch processing is:', layout = w.Layout(visibility = 'hidden'))
        self.indicate_status_of_batch_processing = w.Button(description = 'disabled',
                                                            disabled = True,
                                                            tooltip = 'Please enable/disable batch processing in the "Recording data to analyze" section!',
                                                            layout = w.Layout(width = 'auto', visibility = 'hidden'))
        batch_processing_info_box = w.HBox([self.batch_processing_info, self.indicate_status_of_batch_processing], layout = w.Layout(width = '99%', justify_content = 'center'))
        self.user_settings_roi_filepath = FileChooser(title = 'Please select the ROI file:', layout = w.Layout(width = '90%'))
        self.user_settings_roi_filepath.rows = 4
        self.load_roi_button = w.Button(description = 'Load ROI data', disabled = True, tooltip = 'Please select which ROI data to load!', layout = w.Layout(width = '90%'))               
        # Enable event handling:
        self.user_settings_enable_rois.observe(self._change_roi_processing_config)
        self.user_settings_roi_filepath.register_callback(self._roi_filepath_chosen)
        # Arrange elements:
        io_roi_box = w.VBox([io_roi_info,
                             self.user_settings_enable_rois,
                             batch_processing_info_box,
                             self.user_settings_roi_filepath,
                             self.load_roi_button,
                             self._get_spacer()],
                            layout = w.Layout(width = '33%', max_height = '400px', align_items='center', border_right = '1px dashed'))
        return io_roi_box

    
    def _change_roi_processing_config(self, change) -> None:
        if change['name'] == 'value':
            if change['new'] == True:
                self._change_widget_state(self.batch_processing_info, visibility = 'visible')
                self._change_widget_state(self.indicate_status_of_batch_processing, visibility = 'visible')
                if self.user_settings_enable_batch_processing.value == True:
                    self._change_widget_state(self.indicate_status_of_batch_processing, description = 'enabled', button_style = 'success')
                    self.user_settings_roi_filepath.show_only_dirs = True
                    self.user_settings_roi_filepath.title = 'Please select the directory that contains all ROI files:'
                else:
                    self._change_widget_state(self.indicate_status_of_batch_processing, description = 'disabled', button_style = '')
                    self.user_settings_roi_filepath.show_only_dirs = False
                    self.user_settings_roi_filepath.title = 'Please select the ROI file:'  
            else:
                self._change_widget_state(self.batch_processing_info, visibility = 'hidden')
                self._change_widget_state(self.indicate_status_of_batch_processing, visibility = 'hidden')
                self._change_widget_state(self.load_roi_button, disabled = True, tooltip = 'Please select which ROI data to load!')
                self.user_settings_roi_filepath.reset()

            
    def _roi_filepath_chosen(self, file_chooser_obj) -> None:
        if file_chooser_obj.value != None:
            filepath = Path(file_chooser_obj.value)
            if self.user_settings_enable_batch_processing.value == True:
                if filepath.is_dir() == True:
                    self._change_widget_state(self.load_roi_button, disabled = False, tooltip = 'Click to load the selected data')
                else:
                    self.user_settings_roi_filepath.reset()
                    self._change_widget_state(self.load_roi_button, disabled = True, tooltip = 'Please select which data to load!')
                    message = 'You have to select a directory if batch processing is enabled!'
                    self.user_info_panel.add_new_logs(message)
            if self.user_settings_enable_batch_processing.value == False:
                if filepath.is_file() == True:
                    self._change_widget_state(self.load_roi_button, disabled = False, tooltip = 'Click to load the selected data')
                else:
                    self.user_settings_roi_filepath.reset()
                    self._change_widget_state(self.load_roi_button, disabled = True, tooltip = 'Please select which data to load!')
                    message = ('You have to select a file if batch processing is disabled! '
                               'If you want to analyze multiple recording files within a directory '
                               '- with the same settings - consider enabling batch mode.')
                    self.user_info_panel.add_new_logs(message)   

    

    def _build_io_recording_box(self) -> w.VBox:
        # Create and configure all elements:
        io_recording_info = w.HTML(value="<p style='font-size:16px; font-weight:bold; text-align:center;'>Recording data to analyze</p>")
        self.user_settings_enable_batch_processing = w.Checkbox(description = 'Enable batch processing', value = False, style = {'description_width': 'initial'})
        self.user_settings_recording_filepath = FileChooser(title = 'Please select the recording file:', layout = w.Layout(width = '90%'))
        self.user_settings_recording_filepath.rows = 4
        self.user_settings_idx_of_representative_recording_in_batch = w.BoundedIntText(description = 'Index of representative recording in batch to preview:', 
                                                                                       value = 0,
                                                                                       min = 0,
                                                                                       max = 10000,
                                                                                       step = 1,
                                                                                       style = {'description_width': 'initial'},
                                                                                       layout = w.Layout(visibility = 'hidden', width = '90%'))
        self.load_recording_button = w.Button(description = 'Load recording data', disabled = True, tooltip = 'Please select which recording data to load!', layout = w.Layout(width = '90%'))
        # Enable event handling:
        self.user_settings_enable_batch_processing.observe(self._change_batch_processing_config)
        self.user_settings_recording_filepath.register_callback(self._recording_filepath_chosen)
        # Arrange elements:
        io_recording_box = w.VBox([io_recording_info,
                                   self.user_settings_enable_batch_processing,
                                   self.user_settings_idx_of_representative_recording_in_batch,
                                   self.user_settings_recording_filepath,
                                   self.load_recording_button,
                                   self._get_spacer()],
                                  layout = w.Layout(width = '33%', max_height = '400px', align_items='center', border_right = '1px dashed', border_bottom = '1px dashed'))
        return io_recording_box

    
    def _change_batch_processing_config(self, change) -> None:
        if change['name'] == 'value':
            self._change_widget_state(self.load_recording_button, disabled = True, tooltip = 'Please select which data to load!')
            self.user_settings_recording_filepath.reset()
            # Apply changes also for relevant widgets in ROI IO Box:
            self._change_widget_state(self.load_roi_button, disabled = True, tooltip = 'Please select which data to load!')
            self.user_settings_roi_filepath.reset()
            if change['new'] == True:
                self._change_widget_state(self.user_settings_idx_of_representative_recording_in_batch, visibility = 'visible')
                self.user_settings_recording_filepath.show_only_dirs = True
                self.user_settings_recording_filepath.title = 'Please select the directory that contains all recordings:'
                # Apply changes also for relevant widgets in ROI IO Box:
                self._change_widget_state(self.indicate_status_of_batch_processing, description = 'enabled', button_style = 'success')
                self.user_settings_roi_filepath.show_only_dirs = True
                self.user_settings_roi_filepath.title = 'Please select the directory that contains all ROI files:'
            else:
                self._change_widget_state(self.user_settings_idx_of_representative_recording_in_batch, visibility = 'hidden')
                self.user_settings_recording_filepath.show_only_dirs = False
                self.user_settings_recording_filepath.title = 'Please select the recording file:'
                # Apply changes also for relevant widgets in ROI IO Box:
                self._change_widget_state(self.indicate_status_of_batch_processing, description = 'disabled', button_style = '')
                self.user_settings_roi_filepath.show_only_dirs = False
                self.user_settings_roi_filepath.title = 'Please select the ROI file:'                  

                    

    def _change_widget_state(self,
                             widget,
                             value: Optional[Any]=None,
                             description: Optional[str]=None,
                             disabled: Optional[bool]=None,
                             visibility: Optional[str]=None,
                             tooltip: Optional[str]=None,
                             button_style: Optional[str]=None
                            ) -> None:
        if value != None:
            widget.value = value
        if description != None:
            widget.description = description
        if disabled != None:
            widget.disabled = disabled
        if visibility != None:
            widget.layout.visibility = visibility
        if tooltip != None:
            widget.tooltip = tooltip
        if button_style != None:
            widget.button_style = button_style


    def _recording_filepath_chosen(self, file_chooser_obj) -> None:
        if file_chooser_obj.value != None:
            filepath = Path(file_chooser_obj.value)
            if self.user_settings_enable_batch_processing.value == True:
                if filepath.is_dir() == True:
                    self._change_widget_state(self.load_recording_button, disabled = False, tooltip = 'Click to load the selected data')
                else:
                    self.user_settings_recording_filepath.reset()
                    self._change_widget_state(self.load_recording_button, disabled = True, tooltip = 'Please select which data to load!')
                    message = 'You have to select a directory if batch processing is enabled!'
                    self.user_info_panel.add_new_logs(message)
            if self.user_settings_enable_batch_processing.value == False:
                if filepath.is_file() == True:
                    self._change_widget_state(self.load_recording_button, disabled = False, tooltip = 'Click to load the selected data')
                else:
                    self.user_settings_recording_filepath.reset()
                    self._change_widget_state(self.load_recording_button, disabled = True, tooltip = 'Please select which data to load!')
                    message = ('You have to select a file if batch processing is disabled! '
                               'If you want to analyze multiple recording files within a directory '
                               '- with the same settings - consider enabling batch mode.')
                    self.user_info_panel.add_new_logs(message)                
                
            
    def _get_spacer(self, height: str='10px', width: str='99%') -> w.HTML:
        # consider making a general function in view.py
        return w.HTML(layout = w.Layout(height = height, width = width))



class AnalysisSettingsPanel:

    def __init__(self) -> None:
        self.widget = self._build_default_widget()


    def _build_default_widget(self) -> None:
        # Create and configure all elements:
        analysis_settings_info = w.HTML(value="<p style='font-size:16px; font-weight:bold; text-align:center;'>Analysis Settings</p>", layout = w.Layout(width = '99%'))
        snr_label = w.Label(value = 'Signal to noise ratio:', style = {'text_align': 'left'}, layout = w.Layout(width = '90%'))
        self.user_settings_signal_to_noise_ratio = w.FloatSlider(value = 3.0, min = 0.0, max = 100.0, step = 0.05, disabled = True, layout = w.Layout(width = '90%'))
        sat_label = w.Label(value = 'Signal average threshold:', style = {'text_align': 'left'}, layout = w.Layout(width = '90%'))
        self.user_settings_signal_average_threshold = w.FloatSlider(value = 10.0, min = 0.0, max = 255.0, step = 0.5, disabled = True, layout = w.Layout(width = '90%'))
        mac_label = w.Label(value = 'Minimum activity counts:', style = {'text_align': 'left'}, layout = w.Layout(width = '90%'))
        self.user_settings_minimum_activity_counts = w.BoundedIntText(value = 2, min = 0, max = 100, step = 1, disabled = True, layout = w.Layout(width = '75%'))
        bem_label = w.Label(value = 'Baseline estimation method:', style = {'text_align': 'left'}, layout = w.Layout(width = '90%'))
        self.user_settings_baseline_estimation_method = w.Dropdown(value = "Asymmetric Least Squares", 
                                                                   options = ["Asymmetric Least Squares",
                                                                              "Fully Automatic Baseline Correction",
                                                                              "Peaked Signal's Asymmetric Least Squares Algorithm",
                                                                              "Standard Deviation Distribution"
                                                                             ],
                                                                   disabled = True,
                                                                   layout = w.Layout(width = '75%'))
        vertical_spacer = w.HTML(value = '', layout = w.Layout(height = '10px'))
        dashed_separator_line = w.HTML(value = "<hr style='border: none; border-bottom: 1px dashed;'>", layout = w.Layout(width = '95%'))
        optional_info = w.Label(value = 'Optional settings:', style = {'text_align': 'left', 'font_weight': 'bold'}, layout = w.Layout(width = '90%'))
        self.user_settings_include_variance = w.Checkbox(description = 'include Variance', value = False, disabled = True, layout = w.Layout(width = '90%'))
        self.user_settings_variance = w.BoundedFloatText(description = 'Variance:', disabled = True,
                                                         value = 3.0, min = 0.0, max = 30.0, step = 0.1,
                                                         style = {'description_width': 'initial'},
                                                         layout = w.Layout(width = '75%', visibility = 'hidden'))
        self.user_settings_limit_analysis_to_frame_interval = w.Checkbox(description = 'limit analysis to specific frame interval', 
                                                                         value = False, disabled = True, layout = w.Layout(width = '90%'))
        self.user_settings_frame_interval_to_analyze = w.IntRangeSlider(description = 'Frame interval:', disabled = True, 
                                                                        value = (0, 100), min = 0, max = 100, step = 1, 
                                                                        style = {'description_width': 'initial'}, layout = w.Layout(width = '90%', visibility = 'hidden'))
        # Enable event handling:
        self.user_settings_include_variance.observe(self._include_variance_config_changed)
        self.user_settings_limit_analysis_to_frame_interval.observe(self._limit_analysis_to_interval_changed)
        # Arrange elements:
        analysis_settings_box = w.VBox([analysis_settings_info,
                                        snr_label, self.user_settings_signal_to_noise_ratio,
                                        sat_label, self.user_settings_signal_average_threshold,
                                        mac_label, self.user_settings_minimum_activity_counts,
                                        bem_label, self.user_settings_baseline_estimation_method,
                                        vertical_spacer,
                                        dashed_separator_line, 
                                        optional_info,
                                        self.user_settings_include_variance, self.user_settings_variance,
                                        self.user_settings_limit_analysis_to_frame_interval, self.user_settings_frame_interval_to_analyze],
                                       layout = w.Layout(height = '512px', width = '33%', align_items = 'center', border_bottom = '1px dashed'))
        return analysis_settings_box
                                        
                                        
    def enable_analysis_settings(self, enable_all_widgets: bool=True) -> None:
        for attribute_name, attribute_obj in vars(self).items():
            if attribute_name.startswith('user_settings'):
                attribute_obj.disabled = not enable_all_widgets
    

    def _include_variance_config_changed(self, change) -> None:
        if change['name'] == 'value':
            if change['new'] == True:
                self.user_settings_variance.layout.visibility = 'visible'
            else:
                self.user_settings_variance.layout.visibility = 'hidden'


    def _limit_analysis_to_interval_changed(self, change) -> None:
        if change['name'] == 'value':
            if change['new'] == True:
                self.user_settings_frame_interval_to_analyze.layout.visibility = 'visible'
            else:
                self.user_settings_frame_interval_to_analyze.layout.visibility = 'hidden'





class MainScreen:

    def __init__(self) -> None:
        self.welcome_screen = self._build_welcome_screen()
        self.output_screen = self._build_output_screen()
        self.widget = w.VBox([], layout = w.Layout(height = '512px', width = '67%', justify_content = 'center', align_items = 'center', 
                                                                border_top = '1px solid', border_left = '1px solid', border_bottom = '1px solid'))
        self.show_welcome_screen()

    def _build_welcome_screen(self) -> w.VBox:
        welcome = w.HTML(value="<p style='font-size:32px; font-weight:bold; text-align:center;'>Welcome to</p>", layout = w.Layout(width = '99%'))
        logo = w.HTML(value='<img src="https://raw.githubusercontent.com/jpits30/NeuronActivityTool/master/Logo.png" width="256" height="256">')
        start_instructions = w.HTML(value="<p style='font-size:20px; font-weight:bold; text-align:center;'>Please start by selecting data of a recording to analyze</p>", layout = w.Layout(width = '99%'))
        welcome_screen = w.VBox([welcome, logo, start_instructions])
        return welcome_screen


    def _build_output_screen(self) -> w.VBox:
        self.output = w.Output()
        output_screen = w.VBox([self.output])
        return output_screen


    def show_welcome_screen(self) -> None:
        self.widget.children = self.welcome_screen.children
        self.current_screen = 'welcome'


    def show_output_screen(self, clear_output: bool=True) -> None:
        if clear_output == True:
            self.output.clear_output()
        self.widget.children = self.output_screen.children
        self.current_screen = 'output'   





class WidgetsInterface:

    def __init__(self) -> None:
        self.user_info_panel = UserInfoPanel()
        self.io_panel = IOPanel(user_info_panel = self.user_info_panel)
        self.analysis_settings_panel = AnalysisSettingsPanel()
        self.main_screen = MainScreen()
        self.widget = w.VBox([self.io_panel.widget, 
                              w.HBox([self.analysis_settings_panel.widget, self.main_screen.widget]),
                              self.user_info_panel.widget], layout = w.Layout(border = '1px solid'))


    def show_on_main_screen(self, image_to_show) -> None:
        # display the image passed 
        pass


    def add_to_logs(self, message: str) -> None:
        self.user_info_panel.add_new_logs(message)