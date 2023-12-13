import ipywidgets as w
from IPython.display import display
from .core import run_analysis

# Button to load a movie file from the user local computer
video_filepath = w.Text(value = 'spiking_neuron.avi')

# Numeric text field for window size
window_size_text = w.BoundedIntText(
    value=12,
    min=0,
    max=100,
    step=1,
    description='Window Size:',
)

# Button to process image stack
process_button = w.Button(
    description='Process Image Stack',
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Process Image Stack',
)

# Sliders for Signal to Noise Ratio, Signal Average Threshold, and Include Variance
snr_slider = w.IntSlider(
    value=3,
    min=0,
    max=10,
    step=1,
    description='Signal to Noise Ratio:',
)

sat_slider = w.IntSlider(
    value=15,
    min=0,
    max=30,
    step=1,
    description='Signal Average Threshold:',
    disabled = True
)

variance_slider = w.IntSlider(
    value=30,
    min=0,
    max=100,
    step=1,
    description='Include Variance:',
    disabled = True
)

# Checkbox for General Activity Tendency
activity_checkbox = w.Checkbox(
    value=False,
    description='General Activity Tendency',
    disabled = True
)

# IntText for Minimum Activity Counts
min_activity_text = w.BoundedIntText(
    value=1,
    min=0,
    max=100,
    step=1,
    description='Minimum Activity Counts:',
    disabled = True
)

# Buttons for Detect Activity and Reset
detect_button = w.Button(
    description='Detect Activity',
    button_style='info', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Detect Activity',
    disabled = True
)

reset_button = w.Button(
    description='Reset',
    button_style='danger', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Reset',
    disabled = True
)

# Layout the w
widget = w.VBox([video_filepath,
                window_size_text,
                process_button,
                snr_slider,
                sat_slider,
                activity_checkbox,
                variance_slider,
                min_activity_text,
                w.HBox([detect_button, reset_button])])

def show():
    display(widget)

def on_process_button_clicked(b):
    process_button.description = 'Analysis running ...'
    process_button.disabled = True
    run_analysis(video_filepath = video_filepath.value, 
                 window_size = window_size_text.value,
                 signal_to_noise_ratio = snr_slider.value)
    process_button.description = 'Process Image Stack'
    process_button.disabled = False
    

# Attach the callback function to the load button
process_button.on_click(on_process_button_clicked)