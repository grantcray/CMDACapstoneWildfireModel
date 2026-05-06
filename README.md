# CMDA Capstone Wildfire Model

## Overview

This project was worked on by the Smokey the Databears team of the Spring 2026 CMDA Capstone Course at Virginia Tech. Most of this code is borrowed with permission from the **Wildfire_Infrastructure_Threat_Unet** Git Project created by CNA. With their assistance, our team developed the following Wildfire U-Net Model which builds off of their original work.

## Repository Structure

```
CMDACapstoneWildfireModel/
|-- main.ipynb
|-- utils.ipynb
|-- unet.ipynb
|-- dataInspection.ipynb
|-- multiOutput.py
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
- **dataInspection.ipynb**: Contains data auditing and inspection utilities used to validate image-mask consistency, analyze class distributions, detect dataset irregularities, and inspect segmentation labels before model training.
- **multiOutput.py**: Contains the multi output approach used to rebalance the wildfire risk classes in this model.
- **requirements.txt**: Lists the Python packages required to run the project.
- **README.md**: Provides an overview of the project, installation instructions, and usage information.
- **Datasets/**: Directory to store datasets used in the project.
- **models/**: Directory to save trained models, mask predictions, and related files.
- **resources/**: Directory for additional resources such as images, documentation, or other files.

## License

The code and notebooks associated with this project are licensed under the MIT License. See the [LICENSE](LICENSE.txt) for more details.

## Acknowledgements

- Our team would like to thank CNA, the CMDA Capstone Teaching Team, and Virginia Tech for allowing us to create this project.
