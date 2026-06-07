<h1 align="left">⚡ &mu;-TEG Discovery Suite</h1>

<p align="center">
  <a href="https://teg-stability-predictor-ajupdywh7jmcxqcfaoiecy.streamlit.app/" target="_blank">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Live Application">
  </a>
</p>

<p><strong>A Physics-Informed, Dual-Engine High-Throughput Virtual Screening (HTVS) Pipeline for n-Type Organic Thermoelectrics</strong></p>

<p>Organic Micro-Thermoelectric Generators (&mu;-TEGs) require semiconductors with deep Lowest Unoccupied Molecular Orbitals (LUMO) for ambient stability, and low Reorganization Energies (&lambda;) for high charge carrier mobility.</p>

<p>Traditional Density Functional Theory (DFT) calculations are computationally expensive, making large-scale library screening bottlenecked. This repository introduces a dual-engine Graph Attention Network (GAT) pipeline designed to act as a pre-DFT ranker, processing molecular topologies in seconds while respecting core quantum mechanical principles.</p>

<hr>

<h2>🧠 Core Architecture: The Dual Engines</h2>

<h3>1. Model A: The Thermodynamic Shield (Solvated LUMO)</h3>
<ul>
  <li><strong>Objective:</strong> Predict the LUMO energy levels in varied dielectric environments.</li>
  <li><strong>Architecture:</strong> A Solvent-Aware Graph Attention Network that dynamically adjusts predictions based on the user-defined solvent's Dielectric Constant and Dipole Moment.</li>
  <li><strong>Utility:</strong> Rapidly filters out shallow-LUMO candidates that would rapidly oxidize in ambient air.</li>
</ul>

<h3>2. Model B: The Kinetic Engine (Reorganization Energy)</h3>
<ul>
  <li><strong>Objective:</strong> Estimate the structural rigidity and electron mobility of the &pi;-conjugated backbone.</li>
  <li><strong>Physics Engine:</strong> Utilizes Heteroatom-Parameterized Hückel Theory to mathematically account for the aggressive electron-withdrawing nature of halogens and cyano groups (e.g., F, N, O).</li>
  <li><strong>Anchored Training:</strong> Utilizes an Oversampled Anchoring technique. High-performance n-type literature standards (like TCNQ and F4-TCNQ) are heavily weighted during training to explicitly teach the AI the physics of electron acceptors, overcoming p-type donor bias in open-source datasets.</li>
</ul>

<hr>

<h2>🛡️ Built-In Security & Validation Filters</h2>
<ul>
  <li><strong>Tanimoto Domain of Applicability:</strong> Calculates the Morgan Fingerprint of every input and compares it to the training data. Flags out-of-domain molecules (e.g., pharmaceutical drugs), making the AI aware of its own mathematical blind spots.</li>
  <li><strong>PAINS Toxicity Filter:</strong> Integrates RDKit's Pan Assay Interference Compounds catalog to automatically flag structurally unstable functional groups.</li>
  <li><strong>PubChem Cross-Referencing:</strong> Automatically fetches Systematic IUPAC names, CIDs, and calculated XLogP values via the NIH REST API to assist with solvent processability planning.</li>
</ul>

<hr>

<h2>🔬 Scientific Disclaimer</h2>
<p>This pipeline is an HTVS pre-ranker, not a replacement for high-level DFT methods. By reducing 3D conformational analysis to 2D topological graph embeddings, the suite trades absolute numerical precision for massive computational scalability. The intended workflow is to isolate the top 1% of highly rigid candidates for supercomputer DFT optimizations.</p>

<hr>

<h2>💻 Local Verification & Reproducibility</h2>
<p>To run this pipeline locally, verify the PyTorch weights, or bypass the web interface for batch screening:</p>

<pre><code>git clone https://github.com/your-username/muteg-discovery-suite.git
cd muteg-discovery-suite
pip install -r requirements.txt
streamlit run app.py</code></pre>

<p><em>(Note: Ensure all <code>.pth</code> model weights and <code>.pkl</code> scalers remain in the root directory)</em></p>

<hr>

<p>
  <strong>Developer:</strong> Manroop Manota<br>
  <strong>Focus:</strong> Computational Materials Chemistry & Machine Data Workflow Integration
</p>
