# CMDA Capstone Wildfire Model

## Overview

This project was worked on by the Smokey the Databears team of the Spring 2026 CMDA Capstone Course at Virginia Tech. Most of this code is borrowed with permission from the **Wildfire_Infrastructure_Threat_Unet** Git Project created by CNA. With their assistance, our team developed the following Wildfire U-Net Model which builds off of their original work.

## Repository Structure

```
CMDACapstoneWildfireModel/
|-- main.ipynb
|-- main_with_woolsey.ipynb
|-- main_dual_datasets.ipynb
|-- utils.ipynb
|-- unet.ipynb
|-- requirements.txt
|-- README.md
|-- LICENSE.txt
|-- Datasets/
|-- models/
|-- resources/

```

- **main.ipynb**: The entry point for the project. This notebook coordinates the overall workflow, calling functions and models defined in the other notebooks. Runs on the Pala Mesa Dataset by default.
- **main_with_woolsey.ipynb**: The main notebook for this project, but ran on the Woolsey dataset. Used to illustrate the differences between the Pala Mesa and Woolsey datasets.
- **main_dual_datasets.ipynb**: The main notebook for this project, but using the dual dataset training strategy. Combines the Pala Mesa and Woolsey datasets by default and runs the U-Net model on the merged dataset.
- **utils.ipynb**: Contains utility functions that support data preprocessing, visualization, and other auxiliary tasks.
- **unet.ipynb**: Implements the U-Net model, a convolutional neural network designed for image segmentation tasks, and functions associated with the U-Net.
- **requirements.txt**: Lists the Python packages required to run the project.
- **README.md**: Provides an overview of the project, installation instructions, and usage information.
- **WeightedLoss/**: Directory that contains the Weighted Loss function and versions of the model that integrate it.
- **DataInspection/**: Directory that contains the dataInspection code and versions of the model that integrate it.
- **Multi-Output/**: Directory that contains the Multi-Output prediction framework.
- **Visualizations/**: Directory to store graphs used to visualize the results of our implementations.
- **Datasets/**: Directory to store datasets used in the project.
- **models/**: Directory to save trained models, mask predictions, and related files.
- **resources/**: Directory for additional resources such as images, documentation, or other files.

## License

The code and notebooks associated with this project are licensed under the MIT License. See the [LICENSE](LICENSE.txt) for more details.

## Acknowledgements

- Our team would like to thank CNA, the CMDA Capstone Teaching Team, and Virginia Tech for allowing us to work on this project.
