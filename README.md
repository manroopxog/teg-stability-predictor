Bipolar Organic Semiconductor Screening Engine ⚛️
Overview
This repository contains a deployed, continuous-learning machine learning pipeline for the high-throughput screening of organic semiconductor stability. The engine utilizes a Graph Attention Network (GAT) to predict the Lowest Unoccupied Molecular Orbital (LUMO) energy levels directly from SMILES strings, enabling rapid classification of n-type, p-type, and ambipolar materials.
Core Architecture
The predictive engine is built on PyTorch Geometric and processes molecules as 3D spatial graphs.
Node Features: 11 distinct unified atom features (hybridization, aromaticity, formal charge, etc.) extracted via RDKit.
Edge Features: 3D spatial coordinate distances calculated post-UFF (Universal Force Field) optimization.
Network Design: A 4-layer GAT architecture with coordinate-aware message passing, followed by global mean pooling and linear regression layers.
Key Features
Deterministic Inference: The deployed app.py script enforces strict evaluation modes (model.eval()) and locked random seeds across both PyTorch and RDKit to ensure 100% precise, repeatable point estimations for academic reliability.
Transfer Learning Calibration: The base weights were actively calibrated using a targeted bipolar dataset (combining deep electron acceptors like F4-TCNQ with shallow donors like Pentacene) to eliminate out-of-distribution p-type bias and mathematically define the full redox potential spectrum.
High-Throughput Batch Screening: A dedicated batch-processing pipeline allows researchers to upload CSV datasets of SMILES strings for automated, bulk LUMO calculation and rapid candidate filtering.
Application Deployment
The engine is currently deployed as a live web application via Streamlit.
Capabilities within the UI:
Single Molecule Prediction: Instant LUMO calculation and categorical classification (Air-Stable N-Type, Borderline, Unstable P-Type).
Live Fine-Tuning: A built-in training module allowing researchers to upload proprietary CSV datasets to fine-tune the GAT weights via browser-based PyTorch training loops.
Batch Screening: Automated processing for large-scale virtual screening.
Tech Stack
Deep Learning: PyTorch, PyTorch Geometric
Computational Chemistry: RDKit
Data Processing: Pandas, Scikit-learn (Joblib for standard scaling)
Deployment: Streamlit
