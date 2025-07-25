"""Defines the interface between View & Model."""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/00_controller.ipynb.

# %% auto 0
__all__ = ['App', 'open_gui']

# %% ../nbs/00_controller.ipynb 3
import matplotlib.pyplot as plt
from IPython.display import display

# %% ../nbs/00_controller.ipynb 4
from .model import Model
from .view import WidgetsInterface
from . import results

# %% ../nbs/00_controller.ipynb 5
class App:

    def __init__(self):
        self.view = WidgetsInterface()
        self._bind_buttons_of_view_to_functions_of_model()
        self.pixel_conversion = 1/plt.rcParams['figure.dpi']


    def _setup_interaction_between_model_and_view(self) -> None:
        self.model.setup_connection_to_update_infos_in_view(self.view.update_infos)
        self.model.setup_connection_to_display_results(self.view.main_screen.show_output_screen, self.view.main_screen.output, self.pixel_conversion)


    def _bind_buttons_of_view_to_functions_of_model(self) -> None:
        self.view.source_data_panel.load_source_data_button.on_click(self._load_data_button_clicked)
        self.view.analysis_settings_panel.run_analysis_button.on_click(self._run_button_clicked)
        self.view.analysis_settings_panel.preview_window_size_button.on_click(self._preview_window_size_button_clicked)


    def launch(self) -> None:
        display(self.view.widget)


    def _load_data_button_clicked(self, change) -> None:
        self.model = Model(self.view.export_user_settings())
        self._setup_interaction_between_model_and_view()
        self.model.create_analysis_jobs()
        if len(self.model.analysis_job_queue) < 1:
            self.model.add_info_to_logs('Failed to create any analysis job(s). Please inspect logs for more details!', True)
            self.view.user_info_panel.progress_bar.bar_style = 'danger'
        else:
            self._display_preview_of_representative_job(self.view.grid_size)
            self.model.add_info_to_logs(f'Data import completed! {len(self.model.analysis_job_queue)} job(s) in queue.', True, 100.0)
            self.view.enable_analysis(True)


    def _display_preview_of_representative_job(self, grid_size: int) -> None:
        representative_job = self.model.analysis_job_queue[0]
        representative_job.load_data_into_memory(grid_size)
        self.view.adjust_widgets_to_loaded_data(total_frames = representative_job.recording.zstack.shape[0])
        self.view.main_screen.show_output_screen()
        with self.view.main_screen.output:
            fig = plt.figure(figsize = (600*self.pixel_conversion, 400*self.pixel_conversion))
            if representative_job.focus_area_enabled:
                results.plot_roi_boundaries(representative_job.focus_area, 'cyan', 'solid', 2)
            if representative_job.rois_source == 'file':
                for roi in representative_job.all_rois:
                    results.plot_roi_boundaries(roi, 'magenta', 'solid', 1)
            plt.imshow(representative_job.recording.preview, cmap = 'gray')
            plt.tight_layout()
            plt.show()


    def _run_button_clicked(self, change) -> None:
        self.view.enable_analysis(False)
        self.model.config = self.view.export_user_settings()
        if self.view.analysis_settings_panel.settings.export_to_nwb:
            self.model.nwb_metadata = self.view.export_nwb_metadata()
        self.model.run_analysis()
        self.model.add_info_to_logs(f'Processing of all jobs completed! Feel free to load more data & continue analyzing!', True, 100.0)
        self.view.enable_analysis(True)


    def _preview_window_size_button_clicked(self, change) -> None:
        self.view.main_screen.show_output_screen()
        with self.view.main_screen.output:
            preview_fig, preview_ax = self.model.preview_window_size(self.view.grid_size)
            preview_fig.set_figheight(400 * self.pixel_conversion)
            preview_fig.tight_layout()
            plt.show(preview_fig)

# %% ../nbs/00_controller.ipynb 6
def open_gui():
    """Start the interactive widgets interface for NeuralActivityCubic"""
    na3 = App()
    return na3.launch()
