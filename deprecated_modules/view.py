import ipywidgets as w
from ipyfilechooser import FileChooser
from IPython.display import Image
import numpy as np
from datetime import datetime

from typing import Dict, Optional, Any
from pathlib import Path


def change_widget_state(widget: w.Widget,
                        value: Optional[Any]=None,
                        description: Optional[str]=None,
                        disabled: Optional[bool]=None,
                        visibility: Optional[str]=None,
                        tooltip: Optional[str]=None,
                        button_style: Optional[str]=None
                       ) -> w.Widget:
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
    return widget




class UserInfoPanel:

    def __init__(self) -> None:
        self.widget = self._build_widget()
        # Initialize basic parameters and configs:
        self.user_timezone = datetime.now().astimezone().tzinfo
        self.max_lines_that_can_be_displayed_in_output = int(self.detailed_logs_output.layout.max_height.replace('px', '')) / 15
        self.logs_message_count = 0
        self.progress_in_percent = 0.0 # still required?


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
        current_time = datetime.now(self.user_timezone)
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
        assert type(progress_in_percent) == float, ('UserInfoPanel.update_progress_bar() expects the progress_in_percent of type float. '
                                                    f'However, you passed {progress_in_percent}, which is of type {type(progress_in_percent)}.')
        if progress_in_percent == 100.0:
            self.progress_bar.bar_style = 'success'
        else:
            self.progress_bar.bar_style = ''
        self.progress_bar.value = progress_in_percent



class SourceDataPanel:
    
    def __init__(self, user_info_panel: UserInfoPanel) -> None:
        self.user_info_panel = user_info_panel
        self.widget = self._build_widget()

    
    def _build_widget(self) -> w.HBox:
        # Create and configure all elements:
        io_source_data_info = w.HTML(value="<p style='font-size:16px; font-weight:bold; text-align:center;'>General Settings</p>")
        self.user_settings_batch_mode = w.Checkbox(description = 'Enable batch processing', value = False, indent = False)
        self.user_settings_focus_area_enabled = w.Checkbox(description = 'Limit analysis to focus area', value = False, indent = False)
        self.user_settings_roi_mode = w.Dropdown(options = [('Use adjustable grid to create ROIs (congruent squares)', 'grid'), ('Load predefined ROIs from source data', 'file')],
                                                 description = 'ROIs:',
                                                 value = 'grid',
                                                 style = {'description_width': 'initial'}, 
                                                 layout = {'width': 'initial'})
        processing_modes_box = w.VBox([self.user_settings_roi_mode, w.HBox([self.user_settings_batch_mode, self.user_settings_focus_area_enabled])], layout = {'width': '33%'})
        self.user_settings_data_source_path = FileChooser(title = 'Please select the recording file:', layout = w.Layout(width = '50%'))
        self.user_settings_data_source_path.rows = 8
        self.load_source_data_button = w.Button(description = 'Load Data', 
                                                disabled = True, 
                                                tooltip = 'Please first select which source data to load!', 
                                                layout = w.Layout(width = '12%', height = '55px'))
        vertical_spacer = w.HTML(value = '', layout = w.Layout(height = '5px'))
        # Enable event handling:
        self.user_settings_batch_mode.observe(self._change_batch_mode_config)
        self.user_settings_focus_area_enabled.observe(self._change_focus_area_config)
        self.user_settings_roi_mode.observe(self._change_roi_mode_config)
        self.user_settings_data_source_path.register_callback(self._data_source_path_chosen)
        # Arrange elements:
        source_data_settings_box = w.HBox([processing_modes_box,
                                           self.user_settings_data_source_path,
                                           self.load_source_data_button], 
                                          layout = w.Layout(width = '100%', justify_content = 'space-between'))
        general_settings_box = w.VBox([io_source_data_info, source_data_settings_box, vertical_spacer], layout = w.Layout(width = '95%', align_items = 'center'))
        return w.HBox([general_settings_box], layout = w.Layout(width = '100%', justify_content = 'center'))


    def _change_user_settings_data_source_path_configs(self, show_dirs_only: bool, title: str, reset: bool) -> None:
        if reset == True:
            self.user_settings_data_source_path.reset()
            self.load_source_data_button = change_widget_state(self.load_source_data_button, disabled = True, tooltip = 'Please select which source data to load!')
        self.user_settings_data_source_path.show_only_dirs = show_dirs_only
        self.user_settings_data_source_path.title = title
        
    
    def _change_roi_mode_config(self, change) -> None:
        if change['name'] == 'value':
            if change['new'] == 'file':
                show_only_dirs = True
                if self.user_settings_batch_mode.value == True:
                    title = 'Please select the parent directory that contains subdirectories with the individual source data:'
                elif self.user_settings_focus_area_enabled.value == True:
                    title = 'Please select the directory that contains the recording, all ROI files, and a directory with focus area ROI(s):'
                else:
                    title = 'Please select the directory that contains the recording and all ROI files:'
            else:
                if self.user_settings_batch_mode.value == True:
                    title = 'Please select the parent directory that contains subdirectories with the individual source data:'
                    show_only_dirs = True
                elif self.user_settings_focus_area_enabled.value == True:
                    title = 'Please select the directory that contains the recording and a directory with focus area ROI(s):'
                    show_only_dirs = True
                else:
                    title = 'Please select the recording file:'
                    show_only_dirs = False
            reset = self.user_settings_data_source_path.show_only_dirs != show_only_dirs
            self._change_user_settings_data_source_path_configs(show_only_dirs, title, reset)
           
        

    def _change_focus_area_config(self, change) -> None:
        if change['name'] == 'value':
            roi_mode = self.user_settings_roi_mode.value
            fake_change = {'name': 'value', 'new': roi_mode}
            self._change_roi_mode_config(fake_change)
           

    
    def _change_batch_mode_config(self, change) -> None:
        if change['name'] == 'value':
            roi_mode = self.user_settings_roi_mode.value
            fake_change = {'name': 'value', 'new': roi_mode}
            self._change_roi_mode_config(fake_change)                     


    def _data_source_path_chosen(self, file_chooser_obj) -> None:
        if file_chooser_obj.value != None:
            enable_loading = False
            source_data_path = Path(file_chooser_obj.value)
            if self.user_settings_roi_mode.value == 'file':
                if source_data_path.is_dir() == True:
                    enable_loading = True
            else:
                if (self.user_settings_batch_mode.value == True) or (self.user_settings_focus_area_enabled.value == True):
                    if source_data_path.is_dir() == True:
                        enable_loading = True
                else:
                    if source_data_path.is_file() == True:
                        enable_loading = True
            if enable_loading == True:
                self.load_source_data_button = change_widget_state(self.load_source_data_button, disabled = False, tooltip = 'Click to load the selected source data')
            else:
                self.load_source_data_button = change_widget_state(self.load_source_data_button, disabled = True, tooltip = 'Please select which source data to load!')


class AnalysisSettingsPanel:

    def __init__(self) -> None:
        self.widget = self._build_default_widget()


    def _build_default_widget(self) -> None:
        width_percentage_core_widgets = '95%'
        description_width = '35px'
        # Create and configure all elements:
        analysis_settings_info = w.HTML(value="<p style='font-size:16px; font-weight:bold; text-align:center;'>Analysis Settings</p>", layout = w.Layout(width = '99%'))

        self.user_settings_window_size = w.IntSlider(description = 'Grid size:', value = 10, min = 1, max = 128, step = 1, disabled = True, 
                                                          layout = w.Layout(width = '80%'), style = {'description_width': 'initial'})
        self.preview_window_size_button = w.Button(description = 'Preview', disabled = True, tooltip = 'Preview grid size. Does not start analysis', layout = w.Layout(width = '20%'))
        self.user_settings_signal_to_noise_ratio = w.BoundedFloatText(description = 'SNR: ', tooltip = 'Signal to noise ratio', value = 3.0, min = 0.1, max = 100.0, step = 0.05, disabled = True, 
                                                                      layout = w.Layout(width = width_percentage_core_widgets), style = {'description_width': description_width})
        self.user_settings_noise_window_size = w.BoundedIntText(description = 'NWS: ', tooltip = 'Noise window size', value = 200, min = 10, max = 1000, step = 1, disabled = True,
                                                                layout = w.Layout(width = width_percentage_core_widgets), style = {'description_width': description_width})
        self.user_settings_signal_average_threshold = w.BoundedFloatText(description = 'SAT: ', tooltip = 'Signal average threshold', value = 10.0, min = 0.0, max = 255.0, step = 0.5, 
                                                                         disabled = True, layout = w.Layout(width = width_percentage_core_widgets), 
                                                                         style = {'description_width': description_width})
        self.user_settings_minimum_activity_counts = w.BoundedIntText(description = 'MAC: ', tooltip = 'Minimum activity counts', value = 2, min = 0, max = 100, step = 1, disabled = True, 
                                                                      layout = w.Layout(width = width_percentage_core_widgets), style = {'description_width': description_width})
        self.user_settings_baseline_estimation_method = w.Dropdown(description = 'Baseline estimation method: ',
                                                                   value = 'asls', 
                                                                   options = [("Asymmetric Least Squares", "asls"),
                                                                              ("Fully Automatic Baseline Correction", "fabc"),
                                                                              ("Peaked Signal's Asymmetric Least Squares Algorithm", "psalsa"),
                                                                              ("Standard Deviation Distribution", "std_distribution")
                                                                             ],
                                                                   disabled = True,
                                                                   layout = w.Layout(width = '99%'), style = {'description_width': 'initial'})
        vertical_spacer = w.HTML(value = '', layout = w.Layout(height = '4px'))
        horizontal_spacer = w.HTML(value = '', layout = w.Layout(width = '5%'))
        vbox_core_settings_left = w.VBox([self.user_settings_signal_to_noise_ratio, vertical_spacer, self.user_settings_signal_average_threshold], 
                                         layout = w.Layout(width = '50%', align_items = 'flex-start'))
        vbox_core_settings_right = w.VBox([self.user_settings_noise_window_size, vertical_spacer, self.user_settings_minimum_activity_counts], 
                                           layout = w.Layout(width = '50%', align_items = 'flex-end'))      
        hbox_grid_window_size_settings = w.HBox([self.user_settings_window_size, self.preview_window_size_button], layout = w.Layout(width = '100%', justify_content = 'flex-start'))
        hbox_core_setting_vboxes = w.HBox([vbox_core_settings_left, vbox_core_settings_right], layout = w.Layout(width = '100%'))
        vbox_all_core_settings = w.VBox([hbox_grid_window_size_settings, 
                                         vertical_spacer,
                                         hbox_core_setting_vboxes,
                                         vertical_spacer,
                                         self.user_settings_baseline_estimation_method], 
                                        layout = w.Layout(width = '90%'))

        
        dashed_separator_line = w.HTML(value = "<hr style='border: none; border-bottom: 1px dashed;'>", layout = w.Layout(width = '95%'))
        optional_info = w.Label(value = 'Optional Settings:', style = {'text_align': 'left', 'font_weight': 'bold'}, layout = w.Layout(width = '90%'))
        self.user_settings_include_variance = w.Checkbox(description = 'include variance', value = False, indent = False, disabled = True, 
                                                         layout = w.Layout(width = '35%'), style = {'description_width': 'initial'})
        self.user_settings_variance = w.BoundedIntText(description = 'Variance:', disabled = True,
                                                         value = 15, min = 5, max = 200, step = 5,
                                                         style = {'description_width': 'initial'},
                                                         layout = w.Layout(width = '65%', visibility = 'hidden'))
        self.user_settings_limit_analysis_to_frame_interval = w.Checkbox(description = 'analyze interval', indent = False, 
                                                                         value = False, disabled = True, layout = w.Layout(width = '35%'), 
                                                                         style = {'description_width': 'initial'})
        self.user_settings_frame_interval_to_analyze = w.IntRangeSlider(description = 'Frames:', disabled = True, 
                                                                        value = (0, 500), min = 0, max = 500, step = 1, 
                                                                        style = {'description_width': 'initial'}, layout = w.Layout(width = '65%', visibility = 'hidden'))
        self.user_settings_configure_octaves = w.Checkbox(description = 'configure octaves', value = False, disabled = True, indent = False, 
                                                          layout = w.Layout(width = '35%'), style = {'description_width': 'initial'})
        self.user_settings_octaves_ridge_needs_to_spann = w.BoundedFloatText(description = 'Min. octaves:', tooltip = 'Minimum octaves a ridge needs to span', disabled = True,
                                                                             value = 1.0, min = 0.1, max = 30.0, step = 0.05,
                                                                             style = {'description_width': 'initial'},
                                                                             layout = w.Layout(width = '65%', visibility = 'hidden'))
        optional_variance_widgets = w.HBox([self.user_settings_include_variance, self.user_settings_variance], layout = w.Layout(width = '100%', align_items = 'flex-start'))
        optional_interval_widgets = w.HBox([self.user_settings_limit_analysis_to_frame_interval, self.user_settings_frame_interval_to_analyze], 
                                           layout = w.Layout(width = '100%', align_items = 'flex-start'))
        optional_octave_widgets = w.HBox([self.user_settings_configure_octaves, self.user_settings_octaves_ridge_needs_to_spann], 
                                         layout = w.Layout(width = '100%', align_items = 'flex-start'))
        
        optional_settings = w.VBox([optional_variance_widgets, vertical_spacer, optional_interval_widgets, vertical_spacer, optional_octave_widgets],
                                     layout = w.Layout(width = '90%', align_items = 'flex-start', align_content = 'flex-start', justify_content = 'flex-start'))

        results_info = w.Label(value = 'Results Settings:', style = {'text_align': 'left', 'font_weight': 'bold'}, layout = w.Layout(width = '90%'))
        self.user_settings_save_overview_png = w.Checkbox(description = 'Save overview plot', value = True, disabled = True, style = {'description_width': 'initial'})
        self.user_settings_save_detailed_results = w.Checkbox(description = 'Save detailed results', value = True, disabled = True, style = {'description_width': 'initial'})
        self.run_analysis_button = w.Button(description = 'Run Analysis',
                                            disabled = True,
                                            tooltip = 'You have to load some data first, before you can run the analysis!',
                                            button_style = '',
                                            icon = 'rocket',
                                            layout = w.Layout(width = '90%'))
        results_settings = w.VBox([w.HBox([self.user_settings_save_overview_png, self.user_settings_save_detailed_results],
                                          layout = w.Layout(width = '100%', align_items = 'flex-start'))],
                                   layout = w.Layout(width = '90%', align_items = 'flex-start', align_content = 'flex-start', justify_content = 'flex-start'))
        # Enable event handling:
        self.user_settings_include_variance.observe(self._include_variance_config_changed)
        self.user_settings_limit_analysis_to_frame_interval.observe(self._limit_analysis_to_interval_changed)
        self.user_settings_configure_octaves.observe(self._configure_octaves_changed)
        # Arrange elements:
        analysis_settings_box = w.VBox([analysis_settings_info,
                                        vbox_all_core_settings,
                                        vertical_spacer,
                                        dashed_separator_line, 
                                        optional_info,
                                        optional_settings,
                                        vertical_spacer,
                                        dashed_separator_line,
                                        results_info,
                                        results_settings,
                                        vertical_spacer,
                                        self.run_analysis_button,
                                        vertical_spacer],
                                       layout = w.Layout(height = '512px', width = '33%', align_items = 'center', border_top = '1px solid', border_bottom = '1px solid'))
        return analysis_settings_box

    
    def enable_analysis_settings(self, enable_all_widgets: bool, roi_mode: str) -> None:
        if enable_all_widgets == True:
            self.run_analysis_button = change_widget_state(self.run_analysis_button, 
                                                           disabled = False, 
                                                           button_style = 'success',
                                                           tooltip = 'Click here to start the analysis with all currently specified settings!')
        else:
            self.run_analysis_button = change_widget_state(self.run_analysis_button, 
                                                           disabled = True,
                                                           button_style = '')     
        for attribute_name, attribute_obj in vars(self).items():
            if attribute_name.startswith('user_settings'):
                attribute_obj.disabled = not enable_all_widgets
            elif attribute_name == 'preview_window_size_button':
                attribute_obj.disabled = not enable_all_widgets
        if roi_mode == 'file':
            self.preview_window_size_button.disabled = True
            self.user_settings_window_size.disabled = True
    

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


    def _configure_octaves_changed(self, change) -> None:
        if change['name'] == 'value':
            if change['new'] == True:
                self.user_settings_octaves_ridge_needs_to_spann.layout.visibility = 'visible'
            else:
                self.user_settings_octaves_ridge_needs_to_spann.layout.visibility = 'hidden'





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
        self.source_data_panel = SourceDataPanel(user_info_panel = self.user_info_panel)
        self.analysis_settings_panel = AnalysisSettingsPanel()
        self.main_screen = MainScreen()
        self.widget = w.VBox([self.source_data_panel.widget, 
                              w.HBox([self.analysis_settings_panel.widget, self.main_screen.widget]),
                              self.user_info_panel.widget], layout = w.Layout(align_items = 'stretch', border = '1px solid'))
        self._setup_observer_for_roi_mode_config_change()


    def _setup_observer_for_roi_mode_config_change(self) -> None:
        self.source_data_panel.user_settings_roi_mode.observe(self._enable_window_size_widgets)


    def _enable_window_size_widgets(self, change) -> None:
        if change['name'] == 'value':
            if change['new'] == 'file':
                change_widget_state(self.analysis_settings_panel.user_settings_window_size, disabled = True)
                change_widget_state(self.analysis_settings_panel.preview_window_size_button, disabled = True)
            else:
                if self.analysis_settings_panel.user_settings_signal_to_noise_ratio.disabled == True:
                    pass
                else:
                    change_widget_state(self.analysis_settings_panel.user_settings_window_size, disabled = False)
                    change_widget_state(self.analysis_settings_panel.preview_window_size_button, disabled = False)                


    def update_infos(self, logs_message: Optional[str]=None, progress_in_percent: Optional[float]=None) -> None:
        if logs_message != None:
            self.user_info_panel.add_new_logs(logs_message)
        if progress_in_percent != None:
            self.user_info_panel.update_progress_bar(progress_in_percent)


    def adjust_widgets_to_loaded_data(self, total_frames: int) -> None:
        self._adjust_frame_interval_selection_widget(total_frames)


    def _adjust_frame_interval_selection_widget(self, total_frames: int) -> None:
        self.analysis_settings_panel.user_settings_frame_interval_to_analyze.max = total_frames + 1
        self.analysis_settings_panel.user_settings_frame_interval_to_analyze.value = (1, total_frames + 1)


    def export_user_settings(self) -> Dict[str, Any]:
        user_settings = {}
        for panel_name_with_user_settings in ['analysis_settings_panel', 'source_data_panel']:
            panel = getattr(self, panel_name_with_user_settings)
            for attribute_name, attribute_value in vars(panel).items():
                if attribute_name == 'user_settings_frame_interval_to_analyze':
                    user_settings['start_frame_idx'] = attribute_value.value[0]
                    user_settings['end_frame_idx'] = attribute_value.value[1]
                elif attribute_name.startswith('user_settings_'):
                    value_set_by_user = attribute_value.value
                    if value_set_by_user != None:
                        parameter_name = attribute_name.replace('user_settings_', '')
                        if 'path' in parameter_name:
                            value_set_by_user = Path(value_set_by_user)
                        user_settings[parameter_name] = value_set_by_user               
        return user_settings


    def enable_analysis(self, enable: bool=True) -> None:
        roi_mode = self.source_data_panel.user_settings_roi_mode.value
        self.analysis_settings_panel.enable_analysis_settings(enable, roi_mode)     