# AI Authenticity Detector

## Overview
This project detects whether input text or images are AI-generated or human-created.

## Problem
With the rise of generative AI, detecting synthetic essays and images has become important for academic integrity and digital media authenticity.

## Modules
- Text Detection Module
- Image Detection Module

## Datasets
- AI vs Human Text dataset
- CIFAKE dataset

## Models
### Text
- TF-IDF + Logistic Regression
- TF-IDF + MultinomialNB
- TF-IDF + XGBoost
- TF-IDF + LightGBM
- RoBERTa

### Image
- CNN / ResNet / EfficientNet on CIFAKE

## How to Run
### Install
pip install -r requirements.txt

### Run Streamlit App
streamlit run src/app/streamlit_app.py

## Results
Add key metrics here.

## Repository Structure
Brief explanation of folders.