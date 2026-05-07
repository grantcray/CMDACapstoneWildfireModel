# DataInspection

## Overview

This file contains the data auditing and label validation pipeline used to inspect the preprocessed data before model training. These visualizations were used to verify label consistency, inspect spatial relationships between wildfire risk classes and transportation infrastructure, and identify potential labeling irregularities across datasets.

## Repository Structure

```
DataInspection/
|-- main_dual_datasets_output.ipynb
|-- unet_output.ipynb
|-- dataInspection.ipynb
|-- README.md
```

- **main_dual_datasets_output.ipynb**: This file runs the Dual Dataset training approach on the U-Net model, but displays additional information related to the Pala Mesa and Woolsey merged dataset. This notebook doesn't differ much from the regular **main_dual_datasets.ipynb** file.
- **unet_output.ipynb**: This file is a modified version of the **unet.ipynb** file that displays additional information related to the image data trained by the U-Net model.
- **dataInspection.ipynb**: This file contains the main data auditing process used to anaylze the image masks, labels, and wildfire risk class distributions of our U-Net data. By default, this file anaylzes the Pala Mesa dataset.
- **README.md**: This file, explains the structure of the DataInspection directory.

