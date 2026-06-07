<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; max-width: 850px; margin: 0 auto; line-height: 1.6; color: #24292e;">

    <h1 style="border-bottom: 1px solid #eaecef; padding-bottom: 10px; color: #0366d6;">⚡ Indigenous &mu;-TEG Discovery Suite</h1>

    <p style="font-size: 1.1em;">
        <strong>Live Application:</strong> <a href="https://teg-stability-predictor-ajupdywh7jmcxqcfaoiecy.streamlit.app/" style="color: #0366d6; text-decoration: none;">Click for Live Application</a>
    </p>

    <p><strong>A Physics-Informed, Dual-Engine High-Throughput Virtual Screening (HTVS) Pipeline for n-Type Organic Thermoelectrics</strong></p>

    <p>Organic Micro-Thermoelectric Generators (&mu;-TEGs) require semiconductors with deep Lowest Unoccupied Molecular Orbitals (LUMO) for ambient stability, and low Reorganization Energies (&lambda;) for high charge carrier mobility.</p>

    <p>Traditional Density Functional Theory (DFT) calculations are computationally expensive, making large-scale library screening bottlenecked. This web application introduces a dual-engine Graph Attention Network (GAT) pipeline designed to act as a pre-DFT ranker, processing molecular topologies in seconds while respecting core quantum mechanical principles.</p>

    <hr style="height: 1px; border: none; background-color: #eaecef; margin: 24px 0;">

    <h2 style="color: #24292e;">🧠 Core Architecture: The Dual Engines</h2>
    <p>This suite decouples the thermodynamic and kinetic properties of molecular discovery into two specialized neural networks, preventing feature confusion and allowing for physics-specific data extraction.</p>

    <h3 style="color: #24292e;">1. Model A: The Thermodynamic Shield (Solvated LUMO)</h3>
    <ul>
        <li><strong>Objective:</strong> Predict the LUMO energy levels in varied dielectric environments.</li>
        <li><strong>Architecture:</strong> A Solvent-Aware Graph Attention Network that dynamically adjusts predictions based on the user-defined solvent's Dielectric Constant and Dipole Moment.</li>
        <li><strong>Utility:</strong> Rapidly filters out shallow-LUMO candidates that would rapidly oxidize in ambient air, isolating robust n-type materials.</li>
    </ul>

    <h3 style="color: #24292e;">2. Model B: The Kinetic Engine (Reorganization Energy)</h3>
    <ul>
        <li><strong>Objective:</strong> Estimate the structural rigidity and electron mobility of the &pi;-conjugated backbone.</li>
        <li><strong>Physics Engine:</strong> Utilizes Heteroatom-Parameterized Hückel Theory. Standard topological matrices are blind to electronegativity. This engine injects empirical Coulomb Integrals and Resonance Integrals into the adjacency matrix to mathematically account for the aggressive electron-withdrawing nature of halogens and cyano groups (e.g., F, N, O).</li>
        <li><strong>Anchored Training:</strong> To overcome the inherent p-type donor bias present in massive open-source material datasets, the training pipeline utilizes an Oversampled Anchoring technique. High-performance n-type literature standards (like TCNQ and F4-TCNQ) are heavily weighted during training to explicitly teach the neural network the physics of electron acceptors.</li>
    </ul>

    <hr style="height: 1px; border: none; background-color: #eaecef; margin: 24px 0;">

    <h2 style="color: #24292e;">🛡️ Built-In Security & Validation Filters</h2>
    <p>Machine learning models are probabilistic estimators, and relying on them blindly in physical chemistry is dangerous. This application includes automated guardrails:</p>
    <ul>
        <li><strong>Tanimoto Domain of Applicability:</strong> The suite calculates the Morgan Fingerprint (Radius=2, 2048-bit) of every input and runs a Tanimoto Similarity index against the original training data. If a user inputs an out-of-domain molecule (e.g., a pharmaceutical drug), the system flags it, effectively making the AI "self-aware" of its own mathematical blind spots.</li>
        <li><strong>PAINS Toxicity Filter:</strong> Integrates RDKit's Pan Assay Interference Compounds (PAINS) catalog to automatically flag structurally unstable or highly reactive functional groups.</li>
        <li><strong>PubChem Cross-Referencing:</strong> Automatically fetches Systematic IUPAC names, CIDs, and calculated XLogP values (hydrophobicity) via the NIH PubChem REST API to assist with solvent processability planning.</li>
    </ul>

    <hr style="height: 1px; border: none; background-color: #eaecef; margin: 24px 0;">

    <h2 style="color: #24292e;">🔬 Scientific Disclaimer</h2>
    <p>This pipeline is an HTVS pre-ranker, not a replacement for high-level <em>ab initio</em> or DFT methods (e.g., B3LYP/6-31G*). By reducing 3D conformational analysis to 2D topological graph embeddings and 1D Hückel eigenvalues, the suite trades absolute numerical precision for massive computational scalability.</p>
    <p>The intended workflow is to use this suite to screen libraries of 100,000+ molecules, isolate the top 1% of deep-LUMO, highly rigid candidates, and pass those isolated structures onto expensive supercomputer DFT optimizations.</p>

    <hr style="height: 1px; border: none; background-color: #eaecef; margin: 24px 0;">

    <h2 style="color: #24292e;">💻 For Developers: Local Verification & Reproducibility</h2>
    <p>If you wish to run this pipeline locally, verify the PyTorch weights, or bypass the web interface for massive batch screening, you can clone this repository.</p>
    
    <p><strong>Prerequisites:</strong> Python 3.9+</p>

    <pre style="background-color: #f6f8fa; padding: 16px; border-radius: 6px; overflow: auto; font-family: ui-monospace, SFMono-Regular, Consolas, 'Liberation Mono', Menlo, monospace; font-size: 0.9em;"><code>git clone https://github.com/your-username/muteg-discovery-suite.git
cd muteg-discovery-suite
pip install -r requirements.txt
streamlit run app.py</code></pre>

    <p>Make sure all <code>.pth</code> model weights and <code>.pkl</code> scalers remain in the root directory alongside <code>app.py</code>.</p>

    <hr style="height: 1px; border: none; background-color: #eaecef; margin: 24px 0;">

    <div style="background-color: #f1f8ff; padding: 15px; border-radius: 6px; border-left: 4px solid #0366d6;">
        <p style="margin: 0;"><strong>Developer / Lead Researcher:</strong> Manroop Manota</p>
        <p style="margin: 5px 0 0 0;"><strong>Institution:</strong> Swami Shraddhanand College, University of Delhi</p>
        <p style="margin: 5px 0 0 0;"><strong>Focus:</strong> Computational Materials Chemistry & AI Data Workflow Integration</p>
    </div>

</div>
