import ipywidgets as w
from ipyfilechooser import FileChooser
import numpy as np

from typing import Dict, Optional, Any
from pathlib import Path


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





class IOPanel:
    
    def __init__(self) -> None:
        self.widget = self._build_widget()


    def _build_widget(self) -> w.VBox:
        self.io_selection_box = self._build_io_selection_box()
        self.logs = w.Output(description = 'Logs:')
        logs_box = w.HBox([self.logs], layout=w.Layout(height='250px', overflow_y='scroll'))
        return w.VBox([self.io_selection_box])


    def _build_io_selection_box(self):
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
                                layout = w.Layout(width = '33%', max_height = '400px', align_items='center', border = '1px solid'))
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
                            layout = w.Layout(width = '33%', max_height = '400px', align_items='center', border = '1px solid'))
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
                    self.add_entry_to_logs(message)
            if self.user_settings_enable_batch_processing.value == False:
                if filepath.is_file() == True:
                    self._change_widget_state(self.load_roi_button, disabled = False, tooltip = 'Click to load the selected data')
                else:
                    self.user_settings_roi_filepath.reset()
                    self._change_widget_state(self.load_roi_button, disabled = True, tooltip = 'Please select which data to load!')
                    message = ('You have to select a file if batch processing is disabled! '
                               'If you want to analyze multiple recording files within a directory '
                               '- with the same settings - consider enabling batch mode.')
                    self.add_entry_to_logs(message)   

    

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
                                  layout = w.Layout(width = '33%', max_height = '400px', align_items='center', border = '1px solid'))
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
                    self.add_entry_to_logs(message)
            if self.user_settings_enable_batch_processing.value == False:
                if filepath.is_file() == True:
                    self._change_widget_state(self.load_recording_button, disabled = False, tooltip = 'Click to load the selected data')
                else:
                    self.user_settings_recording_filepath.reset()
                    self._change_widget_state(self.load_recording_button, disabled = True, tooltip = 'Please select which data to load!')
                    message = ('You have to select a file if batch processing is disabled! '
                               'If you want to analyze multiple recording files within a directory '
                               '- with the same settings - consider enabling batch mode.')
                    self.add_entry_to_logs(message)                
                
            
    def _get_spacer(self, height: str='10px', width: str='99%') -> w.HTML:
        return w.HTML(layout = w.Layout(height = height, width = width))



    def add_entry_to_logs(self, message: str) -> None:
        # Add timestamp
        # Add new entry to displayed logs
        pass