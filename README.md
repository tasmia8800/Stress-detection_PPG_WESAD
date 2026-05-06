# Stress Detection via Multi-Modal Fusion and LOSO Validation

This repository contains the implementation of a robust multi-modal stress detection framework that intelligently synergizes Photoplethysmography (PPG), Electrodermal Activity (EDA), Accelerometer (ACC), and Skin Temperature (TEMP) signals.

## Project Overview
The core objective of this project is to enhance the accuracy of wearable stress detection by addressing motion artifacts and physiological ambiguities through:
- **OMDP Framework**: Orchestrating Multiple Denoising and Peak-detecting for robust PPG signal recovery.
- **Multi-Modal Fusion**: Combining cardiac, electrodermal, kinetic, and thermal data.
- **LOSO Validation**: Leave-One-Subject-Out cross-validation to ensure subject-independent generalization.

## Dataset Information ⚠️
The **WESAD (Wearable Stress and Affect Detection)** dataset is used for this research. 

> [!IMPORTANT]
> Due to the large size (~17 GB), the raw dataset is **not included** in this repository. 

### How to obtain the dataset:
1. Download the dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/WESAD+(Wearable+Stress+and+Affect+Detection)).
2. Extract the contents.
3. Place the `WESAD` folder inside the `OMDP/` directory so the path looks like: `OMDP/WESAD/`.

## Installation
Ensure you have Python 3.8+ installed. Install the required dependencies:
```bash
pip install -r Codes/requirements.txt
```

## Usage
To run the stress classification pipeline:
```bash
python Codes/run_pipeline.py
```

## License
This project is for academic and research purposes.
