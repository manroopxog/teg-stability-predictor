# ⚡ Model A: Solvent-Aware Deep LUMO Suite

<a href="YOUR_STREAMLIT_APP_LINK_HERE" target="_blank">
  <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Open in Streamlit">
</a>

---

## 🔬 Overview
Model A is a high-throughput, AI-driven computational chemistry suite designed to evaluate the ambient air stability of n-type organic thermoelectric (OTE) polymers. Moving beyond traditional gas-phase vacuum predictions, this tool integrates a **multi-modal Graph Attention Network (GAT)** that dynamically calculates Lowest Unoccupied Molecular Orbital (LUMO) energy shifts based on real-world fluid dynamics and solvent environments.

## 🚀 Core Architecture & Features

* **Solvent-Aware Physics Engine:** Predicts LUMO energy levels across multiple chemical environments (e.g., Toluene, Dichloromethane, Acetonitrile). Calculates exact electrostatic stabilization shifts using dielectric constants and dipole moments for custom solvent interpolation.
* **Deep Graph Attention Network:** Employs a 4-layer GAT processing 11 unique quantum node features (hybridization, aromaticity, formal charge) and bond-specific edge featurizations, trained on experimental cyclic voltammetry datasets.
* **Real-Time 3D Structural Minimization:** Integrates the Universal Force Field (UFF) to optimize 3D geometry instantly and renders interactive Van der Waals (VDW) electron density envelopes to visualize steric bulk.
* **Toxicity & Bioaccumulation Screening:** Routes candidates through PAINS (Pan Assay Interference Compounds) filters to flag reactive sub-structures and connects directly to the NIH PubChem API to fetch compound CIDs, IUPAC nomenclature, and XLogP values.
* **High-Throughput Batch Processing:** Supports massive `.csv` library uploads for instant batch evaluation, generating a downloadable matrix of solvated LUMO predictions alongside a real-time interactive structural inspector.

---

## 🛠️ Local Installation

If you wish to run the model architecture locally:

    # Clone the repository
    git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
    cd YOUR_REPO_NAME

    # Install required dependencies
    pip install torch torch-geometric rdkit streamlit pandas numpy joblib py3Dmol stmol requests

    # Launch the Streamlit Dashboard
    streamlit run app.py

## 🧠 Model Weights
The repository contains the pre-trained weights (`solvent_aware_model.pth`) and their respective scalers (`lumo_scaler.pkl` & `solvent_scaler.pkl`). These files must remain in the root directory for the app to initialize the inference pipeline.

---
