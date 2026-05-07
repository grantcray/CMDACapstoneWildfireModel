# WeightedLoss

## Overview

This file contains the weighted categorical cross-entropy loss function for training the U-Net model. Due to the substantial class imbalance observed across the datasets, this class weighting strategy was incorporated during model training to improve sensitivity to underrepresented wildfire risk classes. Initial weight selection was informed by normalized inverse frequency distributions derived from the dataset audit. These weights were then adjusted experimentally to improve training stability and reduce overcorrection toward minority classes. The use of class weighting was directly motivated by the audit results, which showed that background regions dominated the datasets while several minority classes occupied only a small percentage of total pixels and rarely appeared as dominant classes within image masks.


## Repository Structure

```
WeightedLoss/
|-- main_dual_datasets_weighted_loss.ipynb
|-- main_weighted_loss.ipynb
|-- unet_weighted_loss.ipynb
|-- README.md
```

- **main_dual_datasets_weighted_loss.ipynb**: This file is a modified version of the **main_dual_datasets.ipynb** file uses updated functions tailored to the weighted loss design.
- **main_weighted_loss.ipynb**: This file is a modified version of the **main.ipynb** file that uses updated functions tailored to the weighted loss design.
- **unet_weighted_loss.ipynb**: This file is a modified version of the **unet.ipynb** file that uses updated functions tailored to the weighted loss design.
- **README.md**: This file, explains the structure of the WeightedLoss directory.


