# ⚡ Indigenous μ-TEG Discovery Suite

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://teg-stability-predictor-ajupdywh7jmcxqcfaoiecy.streamlit.app/)

**A Physics-Informed, Dual-Engine High-Throughput Virtual Screening (HTVS) Pipeline for n-Type Organic Thermoelectrics**

Organic Micro-Thermoelectric Generators (μ-TEGs) require semiconductors with deep Lowest Unoccupied Molecular Orbitals (LUMO) for ambient stability, and low Reorganization Energies (λ) for high charge carrier mobility. 

Traditional Density Functional Theory (DFT) calculations are computationally expensive, making large-scale library screening bottlenecked. This repository introduces a dual-engine Graph Attention Network (GAT) pipeline designed to act as a pre-DFT ranker, processing molecular topologies in seconds while respecting core quantum mechanical principles.

---

## 🧠 Core Architecture: The Dual Engines

### 1. Model A: The Thermodynamic Shield (Solvated LUMO)
* **Objective:** Predict the LUMO energy levels in varied dielectric environments.
* **Architecture:** A Solvent-Aware Graph Attention Network that dynamically adjusts predictions based on the user-defined solvent's Dielectric Constant and Dipole Moment.
* **Utility:** Rapidly filters out shallow-LUMO candidates that would rapidly oxidize in ambient air.

### 2. Model B: The Kinetic Engine (Reorganization Energy)
* **Objective:** Estimate the structural rigidity and electron mobility of the π-conjugated backbone.
* **Physics Engine:** Utilizes Heteroatom-Parameterized Hückel Theory to mathematically account for the aggressive electron-withdrawing nature of halogens and cyano groups (e.g., F, N, O).
* **Anchored Training:** Utilizes an Oversampled Anchoring technique. High-performance n-type literature standards (like TCNQ and F4-TCNQ) are heavily weighted during training to explicitly teach the AI the physics of electron acceptors, overcoming p-type donor bias in open-source datasets.

---

## 🛡️ Built-In Security & Validation Filters

* **Tanimoto Domain of Applicability:** Calculates the Morgan Fingerprint of every input and compares it to the training data. Flags out-of-domain molecules (e.g., pharmaceutical drugs), making the AI aware of its own mathematical blind spots.
* **PAINS Toxicity Filter:** Integrates RDKit's Pan Assay Interference Compounds catalog to automatically flag structurally unstable functional groups.
* **PubChem Cross-Referencing:** Automatically fetches Systematic IUPAC names, CIDs, and calculated XLogP values via the NIH REST API to assist with solvent processability planning.

---

## 🔬 Scientific Disclaimer

This pipeline is an HTVS pre-ranker, not a replacement for high-level DFT methods. By reducing 3D conformational analysis to 2D topological graph embeddings, the suite trades absolute numerical precision for massive computational scalability. The intended workflow is to isolate the top 1% of highly rigid candidates for supercomputer DFT optimizations.

---

## 💻 Local Verification & Reproducibility 

To run this pipeline locally, verify the PyTorch weights, or bypass the web interface for batch screening:

`git clone https://github.com/your-username/muteg-discovery-suite.git`  
`cd muteg-discovery-suite`  
`pip install -r requirements.txt`  
`streamlit run app.py`

*(Note: Ensure all `.pth` model weights and `.pkl` scalers remain in the root directory)*

---

**Developer:** Manroop Manota    
**Focus:** Computational Materials Chemistry & AI Data Workflow Integration
