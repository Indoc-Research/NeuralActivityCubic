import ipywidgets as w
from IPython.display import display
from .core import run_analysis

STYLE = {'description_width': 'initial'}


class GUI:

    def __init__(self) -> None:
        self.widget = self._build_widget()
        self._bind_button_functions()


    def _build_widget(self) -> w.HBox:
        self.video_filepath = w.Text(value = 'spiking_neuron.avi')
        self.window_size_text = w.BoundedIntText(value=12, min=0, max=100, step=1, description='Window Size:', style = STYLE)
        self.process_button = w.Button(description='Process Image Stack', tooltip='Process Image Stack', style = STYLE)
        self.snr_slider = w.IntSlider(value=3, min=0, max=10, step=1, description='Signal to Noise Ratio:', style = STYLE)
        self.sat_slider = w.IntSlider(value=15, min=0, max=30, step=1, description='Signal Average Threshold:', disabled = True, style = STYLE)
        self.variance_slider = w.IntSlider(value=30, min=0,max=100, step=1, description='Include Variance:', disabled = True, style = STYLE)
        self.activity_checkbox = w.Checkbox(value=False, description='General Activity Tendency', disabled = True, style = STYLE)
        self.min_activity_text = w.BoundedIntText(value=1, min=0, max=100, step=1, description='Minimum Activity Counts:', disabled = True, style = STYLE)
        self.detect_button = w.Button( description='Detect Activity', tooltip='Detect Activity', disabled = True)
        self.reset_button = w.Button(description='Reset', tooltip='Reset', disabled = True)
        self.output = w.Output()
        interface = w.VBox([self.video_filepath,
                            self.window_size_text,
                            self.process_button,
                            self.snr_slider,
                            self.sat_slider,
                            self.activity_checkbox,
                            self.variance_slider,
                            self.min_activity_text,
                            w.HBox([self.detect_button, self.reset_button])],
                          layout = {'width': '40%'})
        return w.HBox([interface, self.output])


    def _bind_button_functions(self) -> None:
        self.process_button.on_click(self._on_process_button_clicked)


    def _on_process_button_clicked(self, b):
        with self.output:
            self.output.clear_output()
            self.process_button.description = 'Analysis running ...'
            self.process_button.disabled = True
            run_analysis(video_filepath = self.video_filepath.value, 
                         window_size = self.window_size_text.value,
                         signal_to_noise_ratio = self.snr_slider.value)
            self.process_button.description = 'Process Image Stack'
            self.process_button.disabled = False


def show():
    gui = GUI()
    display(gui.widget)
