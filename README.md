# Grammar-Scoring-Engine-for-Voice-Samples-SHL-Hiring-Assessment

An end to end Audio Grammar Scoring System that predicts spoken English grammar proficiency scores from audio recordings using:-

 Speech to Text (Whisper)

 BERT Text Embeddings

 Audio Signal Features (MFCC, Chroma, Spectral Contrast, etc.)

 Linguistic Handcrafted Features

 Ridge Regression with 5-Fold Cross Validation

This project was built for an audio scoring challenge and combines NLP + Speech Processing + Machine Learning into a single pipeline.

The system takes raw speech audio as input and predicts a grammar proficiency score (0–5 range).

- Pipeline Architecture:- 
Audio File
   ↓
Speech-to-Text (Whisper)
   ↓
Feature Extraction:
   • BERT Embeddings (Text Semantics)
   • Audio Features (MFCC, Chroma, Spectral)
   • Linguistic Features (Grammar Signals)
   ↓
Feature Concatenation
   ↓
StandardScaler
   ↓
Ridge Regression (5-Fold CV)
   ↓
Predicted Grammar Score

- Tech Stack:- 

Python

PyTorch

HuggingFace Transformers

Faster Whisper

Librosa

Scikit-learn

Pandas / NumPy
