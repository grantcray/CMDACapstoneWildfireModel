# Multi-Output

## Overview

This file contains the Multi-Output prediction framework used on the U-Net model. This model simultaneously predicts wildfire presence, vegetation classification, and road risk using satellite imagery. The model was trained on 1,000 RGB satellite images (256 × 256 pixels) and corresponding segmentation masks. Additional environmental features were also generated, including NDVI-based vegetation maps and road risk maps calculated using distance from roads. 

The model produced three outputs:
1. Burn prediction (fire vs. non-fire)
2. Vegetation classification
3. Road risk classification

## Repository Structure

```
Multi-Output/
|-- main_multi_output.ipynb
|-- README.md
```

- **main_multi_output.ipynb**: This file contains the Multi-Output prediction framework used on the Pala Mesa trained U-Net model. This notebook doesn't differ much from the regular **main.ipynb** file besides from using the multi-output functions.
- **README.md**: This file, explains the structure of the Multi-Output prediction framework.
