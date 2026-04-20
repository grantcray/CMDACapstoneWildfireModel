# iGuide_SpatialAI_2024

## Overview

**iGuide_SpatialAI_2024** is a project focused on spatial artificial intelligence. The project comprises three main Jupyter notebooks: [`main.ipynb`](main.ipynb), [`utils.ipynb`](utils.ipynb), and [`unet.ipynb`](unet.ipynb). The primary workflow is executed from the [`main.ipynb`](main.ipynb) notebook, which leverages utility functions and a U-Net model for spatial data processing.

## Repository Structure

```
iGuide_SpatialAI_2024/
|-- main.ipynb
|-- utils.ipynb
|-- unet.ipynb
|-- requirements.txt
|-- README.md
|-- LICENSE.txt
|-- Datasets/
|-- models/
|-- resources/

```

- **main.ipynb**: The entry point for the project. This notebook coordinates the overall workflow, calling functions and models defined in the other notebooks.
- **utils.ipynb**: Contains utility functions that support data preprocessing, visualization, and other auxiliary tasks.
- **unet.ipynb**: Implements the U-Net model, a convolutional neural network designed for image segmentation tasks, and functions associated with the U-Net.
- **requirements.txt**: Lists the Python packages required to run the project.
- **README.md**: Provides an overview of the project, installation instructions, and usage information.
- **LICENSE.txt**: Contains the MIT License for the project.
- **Datasets/**: Directory to store datasets used in the project.
- **models/**: Directory to save trained models, mask predictions, and related files.
- **resources/**: Directory for additional resources such as images, documentation, or other files.

## Getting Started

### Prerequisites

To run the notebooks, you need to have the following software and libraries installed:
#### Development Environment
- [**Python 3.11.x**](https://www.python.org/downloads/release/python-3110/)
- [**Jupyter Notebook**](https://jupyter.org/)

#### Standard library imports
- [**os**](https://docs.python.org/3/library/os.html) \*: Provides a way of using operating system-dependent functionality.
- [**sys**](https://docs.python.org/3/library/sys.html) \*: Provides access to some variables used or maintained by the Python interpreter.
- [**pathlib**](https://docs.python.org/3/library/pathlib.html) \* : Object-oriented filesystem paths.
- [**warnings**](https://docs.python.org/3/library/warnings.html) \*: Provides a way to issue warning messages.
- [**import_ipynb**](https://pypi.org/project/import-ipynb/): Allows importing Jupyter Notebooks as modules.

#### Third-party libraries
- [**NumPy 2.1.x**](https://numpy.org/): Numerical computing with arrays and matrices.
- [**TensorFlow 2.19.x**](https://www.tensorflow.org/): Machine learning and AI platform.
- [**Matplotlib 3.10.x**](https://matplotlib.org/): Plotting and visualization library.
- [**Scikit-Learn 1.6.x**](https://scikit-learn.org/): Machine learning tools for data analysis.
- [**PIL (Pillow) 11.1.x**](https://pillow.readthedocs.io/en/stable/index.html): Image processing library.

#### Python 3.x built-in libraries \*
- [**shutil**](https://docs.python.org/3/library/shutil.html): High-level file operations.

- [**re (Regular Expressions)**](https://docs.python.org/3/library/re.html): Provides support for regular expressions.
- [**typing**](https://docs.python.org/3/library/typing.html): Support for type hints.
- [**random**](https://docs.python.org/3/library/random.html): Generates pseudo-random numbers.

**Note:** You can install these libraries using the [`requirements.txt`](requirements.txt) in this repository.

\* These are standard built-in Python (3.4 or higher) libraries that **do not** need to be installed separately. 

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/cna-iguide/iGuide_SpatialAI_2024.git
    cd iGuide_SpatialAI_2024
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Project

1. Open the Jupyter Notebook server:
    ```bash
    jupyter notebook
    ```

2. Navigate to the [`main.ipynb`](main.ipynb) notebook and open it.

3. Execute the cells in [`main.ipynb`](main.ipynb) to run the project. This notebook will call functions from [`utils.ipynb`](utils.ipynb) and use the U-Net model from [`unet.ipynb`](unet.ipynb) to process spatial data.

## Usage

### [`main.ipynb`](main.ipynb)

This notebook is the main entry point for the project. It orchestrates the data loading, preprocessing, model training, and evaluation steps.

### [`utils.ipynb`](utils.ipynb)

Contains utility functions for data preprocessing, visualization, and dataset management. Includes both internal and public functions with detailed docstrings and comments for clarity.

### [`unet.ipynb`](unet.ipynb)

Implements the U-Net model and its associated functions for image segmentation. Includes both internal and public functions with detailed docstrings and comments for clarity.

## License

The code and notebooks associated with this project are licensed under the MIT License. See the [LICENSE](LICENSE.txt) for more details.

## Acknowledgements

- The authors would like to thank the contributors and the open-source community for their valuable tools and libraries.