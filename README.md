# AI Authenticity Detector

## Overview
This project detects whether input text or images are AI-generated or human-created. It consists of two independent modules deployed together in a Streamlit web application.

## Problem
With the rise of generative AI tools such as ChatGPT and Midjourney, detecting synthetic content has become critical for academic integrity and digital media authenticity.

## Models & Results

### Text Detection
- **Model:** Multinomial Naive Bayes + TF-IDF (10,000 features, 1–3 n-grams)
- **Dataset:** AI_Human — 487,235 samples (62.8% human, 37.2% AI-generated)
- **Accuracy:** 96% on 97,447 held-out test samples
- **F1 Score:** 0.96 weighted average

### Image Detection
- **Model:** ResNet18 fine-tuned on CIFAKE (ImageNet pretrained)
- **Dataset:** CIFAKE — real CIFAR-10 photos vs Stable Diffusion generated images
- **Accuracy:** 98% on held-out CIFAKE test set

## How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the Streamlit app
```bash
python -m streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

## Repository Structure
```
app.py               — Streamlit front-end (tab-based UI)
text_detector.py     — TF-IDF preprocessing and Naive Bayes inference
image_detector.py    — ResNet18 loading, Grad-CAM explainability, inference
models/              — Serialised model artefacts
  nb_model.pkl           Naive Bayes classifier
  tfidf_vectorizer.pkl   TF-IDF vectorizer
  label_encoder.pkl      Label encoder
  resnet18_cifake.pth    Fine-tuned ResNet18 weights
notebooks/           — Training experiments (NB, XGBoost, LightGBM)
requirements.txt     — Python dependencies
```

## Features
- **Text tab:** Paste any text to get a Human/AI verdict with probability breakdown and color-coded word-level explanation (green = human-leaning words, red = AI-leaning words)
- **Image tab:** Upload any image to get a Real/AI verdict with Grad-CAM heatmap showing which regions influenced the prediction
