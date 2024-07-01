from .model import Model
from .view import WidgetsInterface
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from IPython.display import display

from typing import Any, Callable, Dict

class App:

    def __init__(self):
        self.model = Model()
        self.view = WidgetsInterface()
        self.pixel_conversion = 1/plt.rcParams['figure.dpi']
        self._setup_interaction_between_model_and_view()


    def _setup_interaction_between_model_and_view(self) -> None:
        self._bind_buttons_of_view_to_functions_of_model()
        self.model.setup_connection_to_update_infos_in_view(self.view.update_infos)
        self.model.setup_connection_to_display_results(self.view.main_screen.show_output_screen, self.view.main_screen.output, self.pixel_conversion)


    def _bind_buttons_of_view_to_functions_of_model(self) -> None:
        self.view.source_data_panel.load_source_data_button.on_click(self._load_data_button_clicked)
        self.view.analysis_settings_panel.run_analysis_button.on_click(self._run_button_clicked)
        self.view.analysis_settings_panel.preview_window_size_button.on_click(self._preview_window_size_button_clicked)

    
    def launch(self) -> None:
        display(self.view.widget)


    def _load_data_button_clicked(self, change) -> None:
        user_settings = self.view.export_user_settings()
        self.model.create_analysis_jobs(user_settings)
        if len(self.model.analysis_job_queue) < 1:
            self.model.add_info_to_logs('Failed to create any analysis job(s). Please inspect logs for more details!', True)
            self.view.user_info_panel.progress_bar.bar_style = 'danger'
        else:
            representative_job = self.model.analysis_job_queue[0]
            representative_job.load_data_into_memory()
            self.view.adjust_widgets_to_loaded_data(total_frames = representative_job.recording.zstack.shape[0])
            self.view.main_screen.show_output_screen()
            with self.view.main_screen.output:
                fig = plt.figure(figsize = (600*self.pixel_conversion, 400*self.pixel_conversion))
                if representative_job.roi_based == True:
                    roi_boundary_coords = np.asarray(representative_job.roi.boundary_coords)
                    plt.plot(roi_boundary_coords[:, 1], roi_boundary_coords[:, 0], c = 'cyan', linewidth = 2)
                    plt.imshow(representative_job.recording.preview, cmap = 'gray')
                else:
                    plt.imshow(representative_job.recording.preview, cmap = 'gray')
                plt.tight_layout()
                plt.show()
            self.model.add_info_to_logs(f'Data import completed! {len(self.model.analysis_job_queue)} job(s) in queue.', True, 100.0)
            self.view.enable_analysis()


    def _run_button_clicked(self, change) -> None:
        self.view.enable_analysis(False)
        user_settings = self.view.export_user_settings()
        self.model.run_analysis(user_settings)
        self.model.add_info_to_logs(f'Processing of all jobs completed! Feel free to load more data & continue analyzing!', True, 100.0)
        self.view.enable_analysis(True)


    def _preview_window_size_button_clicked(self, change) -> None:
        self.view.main_screen.show_output_screen()
        with self.view.main_screen.output:
            user_settings = self.view.export_user_settings()
            preview_fig, preview_ax = self.model.preview_window_size(user_settings)
            preview_fig.set_figheight(400 * self.pixel_conversion)
            preview_fig.tight_layout()
            plt.show(preview_fig)
