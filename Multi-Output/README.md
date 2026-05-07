# Multi-Output

## Overview

This project was worked on by the Smokey the Databears team of the Spring 2026 CMDA Capstone Course at Virginia Tech. Most of this code is borrowed with permission from the **Wildfire_Infrastructure_Threat_Unet** Git Project created by CNA. With their assistance, our team developed the following Wildfire U-Net Model which builds off of their original work.

## Repository Structure

```
|-- main_multi_output.ipynb
|-- README.md
```

- **main.ipynb**: The entry point for the project. This notebook coordinates the overall workflow, calling functions and models defined in the other notebooks. Runs on the Pala Mesa Dataset by default.
- **main_with_woolsey.ipynb**: The main notebook for this project, but ran on the Woolsey dataset. Used to illustrate the differences between the Pala Mesa and Woolsey datasets.
- **main_dual_datasets.ipynb**: The main notebook for this project, but using the dual dataset training strategy. Combines the Pala Mesa and Woolsey datasets by default and runs the U-Net model on the merged dataset.
- **utils.ipynb**: Contains utility functions that support data preprocessing, visualization, and other auxiliary tasks.
- **unet.ipynb**: Implements the U-Net model, a convolutional neural network designed for image segmentation tasks, and functions associated with the U-Net.
