**PsyBio-Stack**
PsyBio-Stack is a machine learning framework for analyzing gut microbiome data to predict psychiatric disorders.

This repository contains the processed datasets and analysis scripts used in the study:

"PsyBio-Stack: A Machine Learning Framework Linking Gut Microbiome and Psychiatric Disorders"

**Repository Structure**

data

•	metadata.tsv — sample metadata containing clinical information.

•	feature-table.tsv — microbial abundance feature table used as input for the machine learning model.
scripts

•	main_script.py — main machine learning pipeline for training and evaluating the predictive model.

•	stability_analysis.py — performs microbial feature stability analysis across cross-validation folds.

•	gen_ai_pipeline.py — interpretability and generative AI–based analysis of microbial features.

**Requirements**

Python packages required:

•	numpy

•	pandas

•	scikit-learn

•	matplotlib

•	seaborn

•	joblib

**Example Usage**
Run the main machine learning pipeline:
python scripts/main_script.py \
--feature data/feature-table.tsv \
--meta data/metadata.tsv \
--target depression_bipolar_schizophrenia

**Data Availability**
The processed datasets and machine learning scripts used in this study are publicly available in this repository.
