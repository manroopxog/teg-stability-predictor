<div align="center">
  <h1>⚛️ Bipolar Organic Semiconductor Screening Engine</h1>
  <p><i>High-Throughput Computational Screening Pipeline for Organic Semiconductors</i></p>

  <p>
    <img src="https://img.shields.io/badge/Python-3.10-blue?style=flat&logo=python" alt="Python">
    <img src="https://img.shields.io/badge/PyTorch-Geometric-red?style=flat&logo=pytorch" alt="PyTorch">
    <img src="https://img.shields.io/badge/Framework-Streamlit-orange?style=flat&logo=streamlit" alt="Streamlit">
    <img src="https://img.shields.io/badge/Chemistry-RDKit-green?style=flat" alt="RDKit">
  </p>
</div>

<hr>

<h2>Overview</h2>
<p>This project implements a continuous-learning machine learning pipeline for the high-throughput screening of organic semiconductor stability. Using a <strong>Graph Attention Network (GAT)</strong>, the engine predicts the <strong>LUMO energy levels</strong> directly from SMILES strings, facilitating rapid classification of n-type, p-type, and ambipolar electronic materials.</p>

<h2>Core Architecture</h2>
<p>Built on <strong>PyTorch Geometric</strong>, the engine processes molecules as 3D spatial graphs:</p>
<ul>
  <li><strong>Node Features:</strong> 11 unified atomic descriptors (hybridization, aromaticity, formal charge, etc.) extracted via <strong>RDKit</strong>.</li>
  <li><strong>Edge Features:</strong> Coordinate-aware spatial distances calculated post-<strong>UFF</strong> (Universal Force Field) optimization.</li>
  <li><strong>Network Design:</strong> A 4-layer GAT architecture with global mean pooling and linear regression head.</li>
</ul>

<h2>Key Features</h2>
<ul>
  <li><strong>Deterministic Inference:</strong> Enforces strict <code>model.eval()</code> modes and locked random seeds to ensure 100% precise, repeatable point estimations.</li>
  <li><strong>Transfer Learning Calibration:</strong> Weights calibrated against a targeted bipolar dataset to define the full redox spectrum and eliminate p-type bias.</li>
  <li><strong>Batch Screening Pipeline:</strong> Automated processing for large-scale virtual screening, allowing for bulk CSV analysis and report generation.</li>
</ul>

<h2>Application Deployment</h2>
<p>The engine is deployed as a live web application, structured into three functional modules:</p>
<table>
  <tr>
    <th>Module</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>🧪 <b>Predict Stability</b></td>
    <td>Single-molecule LUMO prediction with air-stability classification.</td>
  </tr>
  <tr>
    <td>⚙️ <b>Live Fine-Tuning</b></td>
    <td>Upload proprietary datasets to refine weights via browser-based PyTorch training.</td>
  </tr>
  <tr>
    <td>📁 <b>Batch Screening</b></td>
    <td>Automated screening of large molecular libraries.</td>
  </tr>
</table>

<h2>Tech Stack</h2>
<ul>
  <li><strong>Deep Learning:</strong> PyTorch, PyTorch Geometric</li>
  <li><strong>Computational Chemistry:</strong> RDKit</li>
  <li><strong>Data Processing:</strong> Pandas, Scikit-learn (Joblib)</li>
  <li><strong>Deployment:</strong> Streamlit Cloud</li>
</ul>

<hr>
<p align="center"><i>Developed for high-throughput materials discovery.</i></p>
