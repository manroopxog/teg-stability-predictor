# ⚡ Model A: Solvent-Aware Deep LUMO Suite

<div align="center">
  <a href="https://teg-stability-predictor-ajupdywh7jmcxqcfaoiecy.streamlit.app/" target="_blank">
    <img src="https://img.shields.io/badge/Launch_Interactive_Dashboard-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Launch Dashboard">
  </a>
</div>

---

## 🔬 Scientific Overview
Model A represents a paradigm shift in high-throughput computational screening for n-type organic thermoelectric (OTE) polymers. Historically, machine learning predictors have relied on gas-phase vacuum calculations (DFT), which fail to account for the crucial solvation energies present during real-world Cyclic Voltammetry (CV) testing and device fabrication. 

Model A bridges this gap. By deploying a multi-modal Graph Attention Network (GAT), this suite dynamically calculates the Lowest Unoccupied Molecular Orbital (LUMO) energy shifts based on the exact physical fluid dynamics of the processing solvent. By mapping how polar environments electrostatically stabilize radical anions, Model A allows researchers to accurately predict ambient air stability against oxygen and moisture degradation before any physical synthesis occurs.

---

## 🚀 Core Architecture & Features

### 1. Dynamic Solvent-Aware Physics Engine
* **Environmental Interpolation:** Predicts absolute LUMO energy levels across distinct chemical environments (e.g., Toluene, Dichloromethane, Chloroform, Acetonitrile).
* **Electrostatic Scaling:** Calculates precise energetic stabilization shifts using continuous solvent descriptors (**Dielectric Constants** and **Dipole Moments**). This allows the neural network to mathematically interpolate and predict behavior in custom, user-defined solvents not explicitly seen in the training data.

### 2. Multi-Modal Graph Attention Network (GAT)
* **Topological Mapping:** The AI brain is built on a 4-layer GAT architecture that processes 11 unique quantum node features (including atomic hybridization states, aromaticity, and formal charge) alongside bond-order edge featurizations.
* **Feature Concatenation:** Molecular graph embeddings are globally pooled into a 64-dimensional vector and concatenated with the physical solvent tensors. This allows the dense linear layers to weigh the structural topology of the semiconductor against the electrostatic pressure of the solvent.

### 3. Real-Time 3D Structural Minimization
* **Force Field Geometry:** Integrates the Universal Force Field (UFF) to instantly optimize the 3D conformation of target molecules and calculate steric strain energies.
* **Electronic Visualization:** Automatically renders interactive Van der Waals (VDW) electron density envelopes, allowing researchers to visually assess steric bulk, backbone planarization, and potential intermolecular packing disruptions.

### 4. Toxicity & Bioaccumulation Screening
* **Reactive Flagging:** Routes all molecular candidates through the PAINS (Pan Assay Interference Compounds) structural catalog to automatically flag highly reactive, unstable, or assay-interfering sub-structures.
* **Database Integration:** Connects directly to the NIH PubChem REST API to fetch verified compound CIDs, systematic IUPAC nomenclature, and calculated `XLogP` values to assess environmental bioaccumulation risks.

### 5. High-Throughput Batch Pipeline
* **Library Screening:** Supports the upload of massive `.csv` libraries for instant, solvent-specific batch evaluation. 
* **Data Matrix Generation:** The system automatically compiles and generates a downloadable matrix of solvated LUMO predictions and toxicity flags, paired with a real-time, interactive 3D structural inspector for deep-diving into individual batch candidates.

---
