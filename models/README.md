# U-Net Models and Data

This folder contains trained U-Net models that were saved, as well as a folder containing the predicted masks and associated files saved after U-Net training in `main.ipynb`.

## Folder Structure

The folder structure is organized as follows:
```
models/
|-- predicted_masks/
|-- trained_UNet.keras
```

## Subfolder Details

The predicted_masks subfolder contains the predicted masks and any associated files (such as `.tfw` files that contain relevant geospatial data.)

### predicted_masks

- **Location**: `Pala_Mesa_Roads_Rails/images/`
- **Formats**: `.tif`, `.png`, `.tfw`
- **Description**: This folder contains the predicted masks and any associated files.


## Notes

- The datasets may contain masks in `.tif` or `.png` format, and any other file format for additional data.

## Usage

