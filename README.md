# ⚡ Indigenous μ-TEG Discovery Suite

**Live Application:** [https://teg-stability-predictor-ajupdywh7jmcxqcfaoiecy.streamlit.app/]

**A Physics-Informed, Dual-Engine High-Throughput Virtual Screening (HTVS) Pipeline for n-Type Organic Thermoelectrics**

Organic Micro-Thermoelectric Generators (μ-TEGs) require semiconductors with deep Lowest Unoccupied Molecular Orbitals (LUMO) for ambient stability, and low Reorganization Energies (λ) for high charge carrier mobility. 

Traditional Density Functional Theory (DFT) calculations are computationally expensive, making large-scale library screening bottlenecked. This web application introduces a dual-engine Graph Attention Network (GAT) pipeline designed to act as a pre-DFT ranker, processing molecular topologies in seconds while respecting core quantum mechanical principles.

---

## 🧠 Core Architecture: The Dual Engines

This suite decouples the thermodynamic and kinetic properties of molecular discovery into two specialized neural networks, preventing feature confusion and allowing for physics-specific data extraction.

### 1. Model A: The Thermodynamic Shield (Solvated LUMO)
* Objective: Predict the LUMO energy levels in varied dielectric environments.
* Architecture: A Solvent-Aware Graph Attention Network that dynamically adjusts predictions based on the user-defined solvent's Dielectric Constant and Dipole Moment.
* Utility: Rapidly filters out shallow-LUMO candidates that would rapidly oxidize in ambient air, isolating robust n-type materials.

### 2. Model B: The Kinetic Engine (Reorganization Energy)
* Objective: Estimate the structural rigidity and electron mobility of the π-conjugated backbone.
* Physics Engine: Utilizes Heteroatom-Parameterized Hückel Theory. Standard topological matrices are blind to electronegativity. This engine injects empirical Coulomb Integrals and Resonance Integrals into the adjacency matrix to mathematically account for the aggressive electron-withdrawing nature of halogens and cyano groups (e.g., F, N, O).
* Anchored Training: To overcome the inherent p-type donor bias present in massive open-source material datasets, the training pipeline utilizes an Oversampled Anchoring technique. High-performance n-type literature standards (like TCNQ and F4-TCNQ) are heavily weighted during training to explicitly teach the neural network the physics of electron acceptors.

---

## 🛡️ Built-In Security & Validation Filters

Machine learning models are probabilistic estimators, and relying on them blindly in physical chemistry is dangerous. This application includes automated guardrails:

* Tanimoto Domain of Applicability: The suite calculates the Morgan Fingerprint (Radius=2, 2048-bit) of every input and runs a Tanimoto Similarity index against the original training data. If a user inputs an out-of-domain molecule (e.g., a pharmaceutical drug), the system flags it, effectively making the AI "self-aware" of its own mathematical blind spots.
* PAINS Toxicity Filter: Integrates RDKit's Pan Assay Interference Compounds (PAINS) catalog to automatically flag structurally unstable or highly reactive functional groups.
* PubChem Cross-Referencing: Automatically fetches Systematic IUPAC names, CIDs, and calculated XLogP values (hydrophobicity) via the NIH PubChem REST API to assist with solvent processability planning.

---

## 🔬 Scientific Disclaimer

This pipeline is an HTVS pre-ranker, not a replacement for high-level ab initio or DFT methods (e.g., B3LYP/6-31G*). By reducing 3D conformational analysis to 2D topological graph embeddings and 1D Hückel eigenvalues, the suite trades absolute numerical precision for massive computational scalability. 

The intended workflow is to use this suite to screen libraries of 100,000+ molecules, isolate the top 1% of deep-LUMO, highly rigid candidates, and pass those isolated structures onto supercomputer DFT optimizations.

---

## 💻 For Developers: Local Verification & Reproducibility 

If you wish to run this pipeline locally, verify the PyTorch weights, or bypass the web interface for massive batch screening, you can clone this repository. 

Prerequisites: Python 3.9+

    git clone https://github.com/your-username/muteg-discovery-suite.git
    cd muteg-discovery-suite
    pip install -r requirements.txt
    streamlit run app.py

Make sure all `.pth` model weights and `.pkl` scalers remain in the root directory alongside `app.py`.

---

**Developer / Lead Researcher:** Manroop Manota
**Institution:** Swami Shraddhanand College, University of Delhi
**Focus:** Computational Materials Chemistry & AI Data Workflow Integration
