# Diploma Thesis Project - Source Code and Notebooks

This README serves as a guide to navigate the submitted files and source code for the diploma thesis titled: **Analysis of Multiple Myelomas in Dual-Energy CT Based Vertebral Data**. 

The repository consists of Python scripts for radiomics feature extraction and feature selection, alongside several Jupyter Notebooks containing the results of various statistical and data analyses.

---

## Directory Structure and File Description

### Feature Extraction & Selection (Scripts)

*   **`radiomics_features_extractor.py`**
    Contains the configuration and setup of the Pyradiomics extractor used to compute radiomic features from medical images.
*   **`extract_features_for_patients.py`**
    Extracts radiomic features from the available patient images and segmentation masks. It expects each patient folder to contain the respective images and matching segmentation masks.
*   **`extract_features_for_healthy_vertebrae.py`**
    A specialized script designed to extract radiomic features exclusively from the vertebrae masks. It assumes that each patient folder contains the images and a correctly labeled vertebrae segmentation mask.
*   **`feature_selection_methods.py`**
    Implements all feature selection methods and their combinations. This script assumes that each patient already has pre-generated CSV files containing the extracted radiomic features.
*   **`visualization_manager.py`**
    A helper utility script that manages and generates visualizations produced by the feature selection methods.

### Data Analysis (Jupyter Notebooks)

*   **`clinical_parameters_analysis.ipynb`**
    Contains the results and visualization of the clinical data analysis.
*   **`radiomic-clinical_relationships.ipynb`**
    Investigates and visualizes the relationships and correlations between clinical parameters and radiomic features.
*   **`shape_features_analysis.ipynb`**
    Contains the results and insights from the shape-based feature analysis.
*   **`vertebrae_analysis.ipynb`**
    Dedicated notebook containing the results and evaluation of the vertebrae analysis.
*   **`clustering_analysis.ipynb`**
    Contains the workflow and results of the clustering analysis.
*   **`follow-up_analysis.ipynb`**
    Focuses on longitudinal data, containing the results of the patient follow-up analysis.
*   **`myeloma_vs_healthy_analysis.ipynb`**
    Contains the comparative analysis and results highlighting differences between healthy subjects and myeloma patients.
*   **`HU_distribution_experiment.ipynb`**
    Contains the results and data behavior from the Hounsfield Unit (HU) distribution experiment.

---

## Technical Prerequisites and Assumptions

To successfully run the scripts and replicate the analyses, please ensure the following conditions are met:

1.  **Feature Extraction Phase:**
    The feature extraction scripts (`extract_features_for_patients.py` and `extract_features_for_healthy_vertebrae.py`) rely heavily on a specific directory structure. The main patient directory must contain individual folders for each patient, and each folder **must include properly named medical images and corresponding segmentation masks**.
    
2.  **Analysis and Feature Selection Phase:**
    All analysis notebooks and the `feature_selection_methods.py` script **depend entirely on pre-extracted features saved in CSV format** within the patient folders. **Without these generated CSV files, no analysis or feature selection methods can be executed.** You must run the extraction scripts first to populate the data before running any notebook.

---

## GitHub Repository

Additional files, including preliminary test analyses, experimental notebooks, and various utility helper scripts, can be found in the remote repository on GitHub:  
👉 **https://github.com/lukas1110/Analysis-of-multiple-myelomas-from-dual-CT**