# Data for Neural Network

This folder contains various datasets for the neural network. Each dataset is organized into subfolders, and each subfolder contains images and corresponding labels (masks).

## Folder Structure

The folder structure is organized as follows:
```
Datasets/
|-- Pala_Mesa_Roads_Rails/
|  |-- images/
|  |-- labels/
```

## Subfolder Details

This set of subfolders gives the general idea of the organization of the datasets, and their file formats

### Images

- **Location**: `Pala_Mesa_Roads_Rails/images/`
- **Formats**: `.tif`, `.png`
- **Description**: This folder contains the input images for the neural network.

### Labels

- **Location**: `Pala_Mesa_Roads_Rails/labels/`
- **Formats**: `.tif`, `.png`
- **Description**: This folder contains the corresponding masks (or labels) for the images, used for training the neural network.

## Notes

- Ensure that each image in the `images` folder has a corresponding label in the `labels` folder with the same filename, where possible.
- The datasets may contain images and labels in `.tif` or `.png` format.

## Usage

To use this data for training a neural network, you can load the images and labels from the respective subfolders, using the built in functions in `utils.ipynb`
