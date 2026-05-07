# Multi-Output

## Overview

This file contains the Multi-Output prediction framework used on the U-Net model. This model simultaneously predicts wildfire presence, vegetation classification, and road risk using satellite imagery. The model was trained on 1,000 RGB satellite images (256 × 256 pixels) and corresponding segmentation masks. Additional environmental features were also generated, including NDVI-based vegetation maps and road risk maps calculated using distance from roads. 

This model produces three outputs:
1. Burn prediction (fire vs. non-fire)
2. Vegetation classification
3. Road risk classification

## Motivation 
The original CNA wildfire model predicts wildfire risk using multiple segmentation classes (background, low, moderate, high, extreme). However, several limitations were identified during analysis:
- The dataset is highly imbalanced, with most pixels belonging to the background class. 
- Multi-class wildfire prediction can become unstable due to overlapping or ambiguous labels
- The original model lacks environmental and infrastructure context
- Minority wildfire classes are difficult for the model to learn consistently

To address these issues, this project:
- Converted wildfire prediction into a binary classification problem (fire vs. no fire)
- Added vegetation information using NDVI-derived features
- Added road proximity information to estimate infrastructure-related wildfire risk

The goal of this approach was to explore whether combining multiple related prediction tasks into a single framework could improve wildfire analysis and provide more contextual outputs.

## Model Design
### Architecture
The model is based on a U-Net architecture with:
- A shared encoder-decoder backbone
- Multiple output heads for separate prediction tasks
- The shared backbone allows the model to learn common spatial features from satellite imagery while each output specializes in a different task.

### Input
- Sentinel-2 satellite imagery
- Image size: (256 x 256 x 3)

### Outputs
1. Burn Output (Binary Classification)
Shape: (256 x 256 x 1)
Activation: Sigmoid
Loss Function: Binary Cross-Entropy

This output predicts whether each pixel represents:
- Fire
- No fire

The original CNA multi-class wildfire labels were converted into a binary representation for this task.

2. Vegetation Output
Shape: (256 x 256 x 3)
Activation: Softmax
Loss Function: Sparse Categorical Cross-Entropy

This output predicts vegetation density levels:
- Low vegetation
- Medium vegetation
- High vegetation

Vegetation classes are generated using the Normalized Difference Vegetation Index (NDVI).

3. Road Risk Output
Shape: (256 x 256 x 4)
Activation: Softmax
Loss Function: Sparse Categorical Cross-Entropy

This output predicts infrastructure-related wildfire risk based on distance to roads:
- Low risk
- Medium risk
- High risk
- Very high risk

Road risk maps are generated using a distance transform from road pixels.

## How It Works
The model uses a multi-task learning approach. A shared feature extractor learns spatial patterns from the input image while separate output heads specialize in different prediction tasks.
All outputs are trained simultaneously, allowing the model to:
- Learn shared features across tasks
- Improve efficiency by using one model instead of multiple independent models
- Generate multiple predictions from a single satellite image
- Incorporate environmental and infrastructure context into wildfire analysis


## Results
Current Model Performance
OutputAccuracyBurn Detection~79%Vegetation Classification~98%Road Risk Classification~47%

## Interpretation of Results
### Vegetation Classification
Vegetation classification achieved the highest performance because NDVI provides strong spectral signals that are relatively easy to distinguish within satellite imagery.
Burn Detection
Burn detection achieved moderate performance but continues to be affected by:
- Class imbalance
- Limited wildfire pixel representation
- Dataset variability


### Road Risk Classification
Road risk classification was the most difficult prediction task because:
- The labels are indirectly derived from road distance
- Infrastructure-related wildfire risk is more abstract than direct image segmentation
- Spatial relationships are more difficult for the model to learn consistently

## Advantages
- Produces multiple outputs from a single model
- Incorporates environmental and infrastructure context
- Reduces the need for multiple independent models
- Expands wildfire analysis beyond simple fire segmentation
- Demonstrates the potential for more comprehensive wildfire risk assessment systems


## Important Notes and Limitations
### Experimental Nature of the Model
This multi-output framework is still highly experimental and should not be considered production-ready or fully validated.
The current implementation:
- Requires substantially more training and refinement
- Needs improved validation and testing
- Should not be relied upon for operational wildfire prediction or emergency response decisions



## Relationship to CNA's Original Model
CNA had previously discussed the concept of a multi-output wildfire framework but had not implemented it. This project serves as an initial exploratory step toward expanding the original single-output multi-class segmentation model into a more comprehensive system.

## Separation From Main Team Pipeline
Due to instability and ongoing implementation issues, this multi-output framework was kept separate from the team’s primary project components:
- Dual-dataset training
- Weighted loss implementation
- Data auditing and label validation pipeline

Those components were more thoroughly tested and validated than the multi-output framework.

## Ethical Considerations
Because the model remains experimental and partially unvalidated, there are ethical concerns associated with over-interpreting or over-relying on its predictions.
Potential concerns include:
- Incorrect wildfire risk predictions
- Misleading infrastructure risk assessments
- Poor generalization to new geographic regions
- Overconfidence in unstable outputs

For these reasons, the system should only be interpreted as:
- A research prototype
- A proof-of-concept framework
- A decision-support exploration tool

It should not be treated as a fully reliable predictive system.

## Current Limitations
- Severe class imbalance still affects wildfire prediction
- Road risk labels are approximated rather than true ground truth labels
- Training is computationally intensive
- GPU limitations restricted experimentation and tuning
- Some outputs are inherently more difficult to predict
- Validation across multiple geographic regions remains limited
- Additional hyperparameter tuning is needed
- The model was not fully optimized for all outputs



## Future Work
Potential future improvements include:
- Implementing improved class-weighted loss strategies
- Incorporating more accurate GIS road datasets
- Adding environmental variables such as:
    - Weather
    - Elevation
    - Humidity
    - Wind conditions
- Expanding training data across additional wildfire-prone regions
- Performing more extensive hyperparameter tuning
- Improving validation procedures
- Exploring the reintroduction of multi-class wildfire prediction
- Testing alternative segmentation architectures



## Summary
This multi-output framework expands a traditional wildfire segmentation model into a more comprehensive system capable of simultaneously predicting wildfire presence, vegetation characteristics, and infrastructure-related risk.
Although the framework remains experimental and requires substantial additional development, it demonstrates the potential for integrating multiple environmental prediction tasks into a single deep learning pipeline. This project provides an initial foundation for future exploration of multi-output wildfire risk assessment systems.

## Repository Structure

```
Multi-Output/
|-- main_multi_output.ipynb
|-- README.md
```

- **main_multi_output.ipynb**: This file contains the Multi-Output prediction framework used on the Pala Mesa trained U-Net model. This notebook doesn't differ much from the regular **main.ipynb** file besides from using the multi-output functions.
- **README.md**: This file, explains the structure of the Multi-Output prediction framework.
