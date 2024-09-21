# NeuralActivityCubic

NeuralActivityCubic (NA³) is an open-source tool for analyzing calcium imaging data, designed to identify subtle spatiotemporal calcium signals in neuronal networks, including low signal-to-noise ratio data. It was published by [Prada et al. 2018 (Plos Comp. Biol.)](https://doi.org/10.1371/journal.pcbi.1006054), but the original [implementation](https://github.com/jpits30/NeuronActivityTool) could no longer be maintained. This is the revamped version, which includes significant performance enhancements and modernized architecture to facilitate high-throughput pharmacological screenings.


## Table of Contents
1. [Features](#features)
2. [Installation](#installation)
3. [Documentation](#documentation)
4. [Usage](#usage)
5. [Contributing](#contributing)
6. [License](#license)


## Features

- Optimized for low signal-to-noise ratio calcium imaging data
- Multi-threading support for faster data processing
- Batch processing for high-throughput experiments


## Installation

To install the latest version of NA³, use the following:

### Using pip

```
pip install neuralactivitycubic
```

### From source by cloning our GitHub repository

```
git clone https://github.com/your-username/neuralactivitycubic.git
cd neuralactivitycubic
pip install .
``` 


## Documentation

We´re currently still working actively on the new implementation of NeuralActivityCubic and will create the documentation as soon as we reached a stable implementation. For now, please feel free to test NeuralActivityCubic for free and without any local installation, simply click [>>here<<](https://mybinder.org/v2/gh/Indoc-Research/NeuralActivityCubic/HEAD?urlpath=lab/tree/NeuralActivityCubic_demo.ipynb) to launch the Jupyter Notebook directly in your Webbrowser. This will run a container hosted completely for free on [MyBinder](https://mybinder.org/).


## Usage

You need to run this tool in a Jupyter Notebook. To launch the GUI, run: 

```
import neuralactivitycubic as na3

na3.open_gui()
```

## Contributing

We welcome contributions from the community! Please read our [CONTRIBUTING.md](https://github.com/Indoc-Research/NeuralActivityCubic/blob/main/CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests.


## License

NeuralActivityCubic is licensed under the AGPL-3.0 License. See the [LICENSE](https://github.com/Indoc-Research/NeuralActivityCubic/blob/main/LICENSE) file for more details.

## Citation

If you use NeuralActivityCubic in your research, please cite:

Prada J, Sasi M, Martin C, Jablonka S, Dandekar T, Blum R (2018) An open source tool for automatic spatiotemporal assessment of calcium transients and local 'signal-close-to-noise' activity in calcium imaging data. PLoS computational biology 14:e1006054

DOI: 10.1371/journal.pcbi.1006054