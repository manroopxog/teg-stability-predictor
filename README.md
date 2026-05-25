<h1>Bipolar Organic Semiconductor Screening Engine ⚛️</h1>

# Model A: High-Throughput LUMO Screener
🚀 **[Click Here to Test the Live Web App](https://teg-stability-predictor-ajupdywh7jmcxqcfaoiecy.streamlit.app/)**

This repository contains the Graph Attention Network (GAT) architecture used to predict the air-stability (LUMO levels) of n-type organic semiconductors...

<h2>Overview</h2>
<p>This repository contains a deployed, continuous-learning machine learning pipeline for the high-throughput screening of organic semiconductor stability. The engine utilizes a <strong>Graph Attention Network (GAT)</strong> to predict the Lowest Unoccupied Molecular Orbital (LUMO) energy levels directly from SMILES strings, enabling rapid classification of n-type, p-type, and ambipolar materials.</p>

<h2>Core Architecture</h2>
<p>The predictive engine is built on <strong>PyTorch Geometric</strong> and processes molecules as 3D spatial graphs.</p>
<ul>
    <li><strong>Node Features:</strong> 11 distinct unified atom features (hybridization, aromaticity, formal charge, etc.) extracted via <strong>RDKit</strong>.</li>
    <li><strong>Edge Features:</strong> 3D spatial coordinate distances calculated post-<strong>UFF</strong> (Universal Force Field) optimization.</li>
    <li><strong>Network Design:</strong> A 4-layer GAT architecture with coordinate-aware message passing, followed by global mean pooling and linear regression layers.</li>
</ul>

<h2>Key Features</h2>
<ul>
    <li><strong>Deterministic Inference:</strong> The deployed <code>app.py</code> script enforces strict evaluation modes (<code>model.eval()</code>) and locked random seeds across both PyTorch and RDKit to ensure 100% precise, repeatable point estimations for academic reliability.</li>
    <li><strong>Transfer Learning Calibration:</strong> The base weights were actively calibrated using a targeted bipolar dataset (combining deep electron acceptors like F4-TCNQ with shallow donors like Pentacene) to eliminate out-of-distribution p-type bias and mathematically define the full redox potential spectrum.</li>
    <li><strong>High-Throughput Batch Screening:</strong> A dedicated batch-processing pipeline allows researchers to upload CSV datasets of SMILES strings for automated, bulk LUMO calculation and rapid candidate filtering.</li>
</ul>

<h2>Application Deployment</h2>
<p>The engine is currently deployed as a live web application via <strong>Streamlit</strong>.</p>
<p><strong>Capabilities within the UI:</strong></p>
<ul>
    <li><strong>Single Molecule Prediction:</strong> Instant LUMO calculation and categorical classification (Air-Stable N-Type, Borderline, Unstable P-Type).</li>
    <li><strong>Live Fine-Tuning:</strong> A built-in training module allowing researchers to upload proprietary CSV datasets to fine-tune the GAT weights via browser-based PyTorch training loops.</li>
    <li><strong>Batch Screening:</strong> Automated processing for large-scale virtual screening.</li>
</ul>

<h2>Tech Stack</h2>
<ul>
    <li><strong>Deep Learning:</strong> PyTorch, PyTorch Geometric</li>
    <li><strong>Computational Chemistry:</strong> RDKit</li>
    <li><strong>Data Processing:</strong> Pandas, Scikit-learn (Joblib for standard scaling)</li>
    <li><strong>Deployment:</strong> Streamlit</li>
</ul>
