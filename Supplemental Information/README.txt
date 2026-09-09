================================================================================
SPECTRAL ANALYSIS AND MACHINE LEARNING FOR CONTAMINATED PLASTIC IDENTIFICATION
================================================================================

PROJECT OVERVIEW
----------------
This project processes Infrared (IR) and Raman spectroscopic data of contaminated 
plastics and uses machine learning to perform multi-output classification for 
identifying both plastic types and contaminants from spectral signatures.
Currently, no Raman data is used.

PROBLEM STATEMENT
-----------------
Multi-output classification problem with two target outputs:
  1. Plastic Type Classification: PP, LDPE, LLDPE, HDPE
  2. Contaminant Identification: BSA (protein), Oil, Starch, CMC, Guar, Casein

PLASTIC TYPES
-------------
  - PP    : Polypropylene
  - LDPE  : Low-Density Polyethylene
  - LLDPE : Linear Low-Density Polyethylene
  - HDPE  : High-Density Polyethylene

CONTAMINANTS
------------
  - BSA (Bovine Serum Albumin) - Protein contaminant
  - Oil (Olive Oil)            - Lipid/fat contaminant
  - Starch                     - Carbohydrate contaminant
  - CMC (Carboxymethyl cellulose) - Polysaccharide contaminant
  - Guar (Guar gum)            - Polysaccharide contaminant
  - Casein                     - Protein contaminant

SPECTROSCOPIC TECHNIQUES
------------------------
  - FTIR (Fourier Transform Infrared Spectroscopy)
  - Raman Spectroscopy

================================================================================
DIRECTORY STRUCTURE
================================================================================

project/
│
├── data/
│   ├── flattened_IR_data/
│   │   └── raw/               # Raw IR spectral CSV files
│   │
│   └── flattened_raman_data/  # Raman spectral data
│
├── src/
│   ├── preprocessing/         # Data preprocessing scripts
│   │                          # - Baseline correction
│   │                          # - Normalization
│   │                          # - Feature extraction
│   │                          # - Label encoding
│   │
│   ├── models/                # Machine learning model implementations
│   │   ├── CNN/               # Convolutional Neural Network
│   │   ├── KNN/               # K-Nearest Neighbors
│   │   ├── logistic_regression/  # Logistic Regression
│   │   └── random_forest/     # Random Forest
│   │
│   ├── evaluation/            # Model evaluation scripts
│   │                          # - Metrics calculation
│   │                          # - Cross-validation
│   │                          # - Confusion matrices
│   │
│   ├── calculated_evaluation_metrics/  # Stored evaluation results
│   │
│   └── scripts/               # Utility and automation scripts
│
├── reports/
│   ├── figures/               # Generated plots and visualizations
│   └── text_reports/          # Analysis reports and summaries
│
└── README.txt                 # This file

================================================================================
DATA FORMAT
================================================================================

INPUT DATA
----------
CSV files containing spectral data with the following naming convention:
  [Sample ID] [Plastic Type] in [Contaminant List]-[Replicate Number].CSV

Example filenames:
  - "1203-54-2 pp-1.CSV"                         # Pure PP sample
  - "1203-55-1 PP in 10% Oil + 1% Starch + 0.5% Guar-1.CSV"  # Contaminated PP
  - "HDPE contaminated with BSA and Oil.CSV"     # HDPE with multiple contaminants

SPECTRAL DATA STRUCTURE
-----------------------
Each CSV file contains:
  - Column 1: Wavenumber (cm^-1) - typically 400-4000 cm^-1 for IR
  - Column 2: Absorbance/Intensity values

================================================================================
MACHINE LEARNING WORKFLOW
================================================================================

1. DATA PREPROCESSING
   - Load raw spectral CSV files
   - Parse filenames to extract labels (plastic type, contaminants)
   - Apply baseline correction (e.g., rubberband, polynomial)
   - Normalize spectra (e.g., SNV, min-max, area normalization)
   - Optional: Smoothing (Savitzky-Golay filter)
   - Split into training/validation/test sets

2. FEATURE ENGINEERING (Optional)
   - Peak detection and characterization
   - Principal Component Analysis (PCA)
   - Spectral region selection

3. MODEL TRAINING
   - Multi-output classifiers:
     * Random Forest with MultiOutputClassifier
     * CNN with multiple output heads
     * KNN with multi-label support
     * Logistic Regression with OneVsRest

4. MODEL EVALUATION
   - Per-output metrics:
     * Accuracy, Precision, Recall, F1-Score
     * Confusion matrices
   - Overall multi-output metrics:
     * Hamming loss
     * Subset accuracy (exact match ratio)

5. RESULTS AND REPORTING
   - Generate classification reports
   - Plot confusion matrices
   - Visualize feature importance
   - Save trained models

================================================================================
REQUIREMENTS
================================================================================

Python 3.8+

Required packages:
  - numpy
  - pandas
  - scipy
  - scikit-learn
  - matplotlib
  - seaborn
  - tensorflow / keras (for CNN)

Optional packages:
  - rampy (Raman spectroscopy utilities)
  - baselinewrapper (baseline correction)

================================================================================
USAGE
================================================================================

1. Place raw spectral CSV files in:
   - data/flattened_IR_data/raw/    (for IR data)
   - data/flattened_raman_data/     (for Raman data)

2. Run preprocessing scripts to prepare data

3. Train models using scripts in src/models/

4. Evaluate models using scripts in src/evaluation/

5. View results in reports/

================================================================================
NOTES
================================================================================

- Ensure consistent wavenumber ranges across all spectral files
- Multiple replicates per sample improve model robustness
- Consider class imbalance when training models
- Contaminant detection is a multi-label problem (samples may have multiple contaminants)
- Plastic type detection is a multi-class problem (one plastic type per sample)

================================================================================
AUTHOR
================================================================================

Michael Behrendt

================================================================================
