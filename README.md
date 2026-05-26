# Heart Rate Modeling with Neural Networks

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20388876.svg)](https://doi.org/10.5281/zenodo.20388876)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21%2B-orange)](https://www.tensorflow.org/)

## Overview
A complete deep learning pipeline for heart health modeling comprising 
two parts: heart rate prediction using regression and arrhythmia 
detection using multi-class classification. Built with TensorFlow 
and Keras.

## Part A — Heart Rate Prediction (Regression)
Predicts resting heart rate in BPM from patient health and activity data.

### Input Features
Age, Weight, Height, BMI, Activity Level, Daily Steps, Sleep Hours,
Exercise Duration, Stress Level, Caffeine Intake

### Model Performance
| Metric | Score |
|---|---|
| MAE | 3.80 BPM |
| RMSE | 4.81 BPM |
| R2 Score | 0.8682 |

### Sample Prediction

## Part B — Arrhythmia Detection (Classification)
Detects cardiac arrhythmia type from ECG-derived features.

### Classes Detected
Normal, Atrial Fibrillation, Bradycardia, Tachycardia, PVC

### Input Features
RR Interval, Heart Rate, PR Interval, QRS Duration, QT Interval,
SDNN, RMSSD, pNN50, LF Power, HF Power

### Model Performance
| Metric | Score |
|---|---|
| Accuracy | 99.5% |
| ROC-AUC | 0.9999 |

### Sample Prediction

## Neural Network Architecture

### Part A — Regression

### Part B — Classification

## Output Figures
| Figure | Description |
|---|---|
| figA1_distributions.png | Feature distributions — heart rate dataset |
| figA2_correlation.png | Feature correlation heatmap |
| figA3_scatter.png | Heart rate vs key features scatter plots |
| figA4_regression_evaluation.png | Full regression evaluation dashboard |
| figB1_class_distribution.png | Arrhythmia class distribution and proportions |
| figB2_ecg_boxplots.png | ECG features by arrhythmia type |
| figB3_classification_evaluation.png | Full classification evaluation dashboard |
| figB4_architecture.png | Neural network architecture diagrams |

## How to Run
Install dependencies:
```bash
pip install -r requirements.txt
```

Run the pipeline:
```bash
python heart_rate_model.py
```

## Key Findings
- BMI and stress level are strong predictors of elevated resting heart rate
- Neural network achieves 99.5% accuracy in arrhythmia classification
- Atrial Fibrillation shows highest RR interval variability (SDNN)
- Tachycardia patients exhibit lowest RMSSD and pNN50 values

## Author
Khan Gulrez Shagufa Fazal Ahmed
Independent Researcher, Maharashtra, India

## Citation
If you use this code in your research please cite this repository
using the DOI badge above (available after Zenodo registration).
