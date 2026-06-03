import streamlit as st
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem, FilterCatalog
import joblib
import numpy as np
import pandas as pd
import urllib.parse
import requests
import py3Dmol
import io
from stmol import showmol

# ==========================================
# 1. CORE ARCHITECTURE (PERFECTLY MATCHED)
# ==========================================
class GATModel(torch.nn.Module):
    def __init__(self, num_node_features=11, edge_dim=1):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(num_node_features, 64, heads=2, concat=True, edge_dim=edge_dim)
        self.conv2 = GATConv(128, 64, heads=2, concat=True, edge_dim=edge_dim)
        self.conv3 = GATConv(128, 64, heads=2, concat=True, edge_dim=edge_dim)
        self.conv4 = GATConv(128, 64, heads=1, concat=False, edge_dim=edge_dim)
        
        self.linear1 = torch.nn.Linear(64, 32)
        self.linear2 = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index, batch, edge_attr=None):
        x = F.elu(self.conv1(x, edge_index, edge_attr=edge_attr))
        x = F.elu(self.conv2(x, edge_index, edge_attr=edge_attr))
        x = F.elu(self.conv3(x, edge_index, edge_attr=edge_attr))
        x = F.elu(self.conv4(x, edge_index, edge_attr=edge_attr))
        x = global_mean_pool(x, batch)
        x = F.elu(self.linear1(x))
        x = self.linear2(x)
        return x

def get_node_features(atom):
    features = [float(atom.GetAtomicNum() == i) for i in [6, 7, 8, 16, 9]]
    hybridization = atom.GetHybridization()
    features += [
        float(hybridization == Chem.rdchem.HybridizationType.SP),
        float(hybridization == Chem.rdchem.HybridizationType.SP2),
        float(hybridization == Chem.rdchem.HybridizationType.SP3)
    ]
    features.append(float(atom.GetIsAromatic()))
    features.append(float(atom.GetFormalCharge()))
    features.append(float(atom.GetTotalNumHs()))
    return features

def get_edge_features(bond):
    bt = bond.GetBondType()
    val = 1.0
    if bt == Chem.rdchem.BondType.DOUBLE: val = 2.0
    elif bt == Chem.rdchem.BondType.TRIPLE: val = 3.0
    elif bt == Chem.rdchem.BondType.AROMATIC: val = 1.5
    return [val]

def smiles_to_graph(smiles, target_val=None):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None: return None
    x = torch.tensor([get_node_features(atom) for atom in mol.GetAtoms()], dtype=torch.float)
    edges, edge_attrs = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.extend([[i, j], [j, i]])
        e_feat = get_edge_features(bond)
        edge_attrs.extend([e_feat, e_feat])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float) if edges else torch.empty((0, 1), dtype=torch.float)
    
    if target_val is not None:
        y = torch.tensor([[target_val]], dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# ==========================================
# 2. TOXICITY & PUBCHEM UTILS
# ==========================================
params = FilterCatalog.FilterCatalogParams()
params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
toxicity_catalog = FilterCatalog.FilterCatalog(params)

def check_toxicity(mol):
    if mol and toxicity_catalog.HasMatch(mol):
        return f"Fail: {toxicity_catalog.GetFirstMatch(mol).GetDescription()}"
    return "Pass"

def get_pubchem_data(smiles):
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{urllib.parse.quote(smiles)}/property/IUPACName,MolecularWeight,XLogP/JSON"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            props = response.json()['PropertyTable']['Properties'][0]
            return {"CID": props.get('CID', 'N/A'), "Name": props.get('IUPACName', 'N/A'), "XLogP": props.get('XLogP', 'N/A')}
    except:
        pass
    return {"CID": "N/A", "Name": "N/A", "XLogP": "N/A"}

# ==========================================
# 3. STREAMLIT UI SETUP & STYLING
# ==========================================
st.set_page_config(page_title="Model A | OTE Deep LUMO Dashboard", layout="wide")

@st.cache_resource
def load_assets():
    model = GATModel(num_node_features=11, edge_dim=1)
    model.load_state_dict(torch.load('upgraded_n_type_expert (4) (2).pth (1)', map_location=torch.device('cpu')))
    scaler = joblib.load('polymer_lumo_scaler (1).pkl')
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"⚠️ Initialization Error: Ensure 'upgraded_n_type_expert (4) (2).pth (1)' and 'polymer_lumo_scaler (1).pkl' are in your GitHub repo. Details: {e}")
    st.stop()

# --- HEADER DISPLAY ---
st.title("🔬 Model A: Deep LUMO Analytical Suite")
st.markdown("---")

tab1, tab2, tab3 = st.tabs([
    "🎯 Single Molecule Studio", 
    "📑 High-Throughput Batch Pipeline", 
    "🔄 Active Fine-Tuning Engine"
])

# ==========================================
# --- TAB 1: SINGLE MOLECULE STUDIO ---
# ==========================================
with tab1:
    col_input, col_viz = st.columns([1, 1.2])
    
    with col_input:
        with st.container(border=True):
            st.subheader("🛠️ Molecular Functionalization")
            if "lumo_smiles" not in st.session_state:
                st.session_state.lumo_smiles = "N#CC(C#N)=C1C=CC(=C(C#N)C#N)C=C1"

            def mutate(rxn_smarts):
                try:
                    mol = Chem.MolFromSmiles(st.session_state.lumo_smiles)
                    products = AllChem.ReactionFromSmarts(rxn_smarts).RunReactants((mol,))
                    if products:
                        Chem.SanitizeMol(products[0][0])
                        st.session_state.lumo_smiles = Chem.MolToSmiles(products[0][0])
                except: pass

            c1, c2 = st.columns(2)
            if c1.button("🧬 Halogenate: Add Fluorine (-F)", use_container_width=True): mutate('[cH:1]>>[c:1](F)')
            if c2.button("🧬 Acceptor: Add Cyano (-C#N)", use_container_width=True): mutate('[cH:1]>>[c:1](C#N)')
            
            smiles = st.text_input("Target Chemical Structure (SMILES Notation):", key="lumo_smiles").strip()
            mol = Chem.MolFromSmiles(smiles) if smiles else None

        if mol:
            with st.container(border=True):
                st.subheader("🤖 Neural Network Evaluation")
                with st.spinner("Executing Graph Attention Network (GAT)..."):
                    model.eval()
                    graph = smiles_to_graph(smiles)
                    if graph is not None:
                        batch = torch.zeros(graph.x.shape[0], dtype=torch.long)
                        with torch.no_grad():
                            scaled_pred = model(graph.x, graph.edge_index, batch, edge_attr=graph.edge_attr).numpy()
                            pred_ev = scaler.inverse_transform(scaled_pred)[0][0]
                        
                        st.metric(label="Predicted LUMO Energy Level", value=f"{pred_ev:.3f} eV")
                        if pred_ev <= -4.0: 
                            st.success("✅ Deep LUMO: High expected ambient air stability.")
                        elif pred_ev <= -3.5: 
                            st.warning("⚠️ Intermediate LUMO: Marginal environmental stability.")
                        else: 
                            st.error("❌ Shallow LUMO: Severe risk of air oxidation.")
                    
            with st.container(border=True):
                st.subheader("🌐 Cross-Reference: PubChem Registry")
                pc = get_pubchem_data(smiles)
                st.markdown(f"**Compound Identification (CID):** `{pc['CID']}`")
                st.markdown(f"**IUPAC Structure Name:** `{pc['Name']}`")
                st.markdown(f"**Bioaccumulation Proxy (XLogP):** `{pc['XLogP']}`")

    with col_viz:
        if mol:
            with st.container(border=True):
                st.subheader("📐 Structural Energy & Physics Optimization")
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, randomSeed=42)
                
                try:
                    ff = AllChem.UFFGetMoleculeForceField(mol)
                    pre_e = ff.CalcEnergy() if ff else 0
                    AllChem.UFFOptimizeMolecule(mol)
                    ff2 = AllChem.UFFGetMoleculeForceField(mol)
                    post_e = ff2.CalcEnergy() if ff2 else 0
                except:
                    pre_e, post_e = 0, 0
                    
                tox = check_toxicity(mol)
                
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Unoptimized Strain", f"{pre_e:.1f} kcal/mol")
                mc2.metric("Relaxed Geometry", f"{post_e:.1f} kcal/mol")
                if tox == "Pass": mc3.success("🛡️ PAINS Filter: PASS")
                else: mc3.error(f"🚨 ALERT: {tox}")

                st.markdown("---")
                st.caption("Conformal Van der Waals (VDW) Electron Density Envelope Approximation")
                viewer = py3Dmol.view(width=500, height=400)
                viewer.addModel(Chem.MolToMolBlock(mol), "mol")
                viewer.setStyle({'stick': {}, 'sphere': {'radius': 0.3}})
                viewer.addSurface(py3Dmol.VDW, {'opacity': 0.45, 'colorscheme': 'cyanCarbon'})
                viewer.zoomTo()
                showmol(viewer, height=400, width=500)

# ==========================================
# --- TAB 2: BATCH SCREENING PIPELINE ---
# ==========================================
with tab2:
    st.subheader("📊 High-Throughput In-Silico Screening")
    
    # Structural File Rules Box
    with st.expander("ℹ️ CRITICAL FILE ENTRY SYSTEM REQUIREMENTS - READ BEFORE UPLOADING", expanded=True):
        st.info("""
        **Your uploaded `.csv` file must perfectly match the constraints below:**
        1. **File Format:** Standard comma-separated values (`.csv`).
        2. **Required Columns:** At least one column containing clean molecular **SMILES strings** (e.g., `CCO`, `c1ccccc1`).
        3. **Data Uniformity:** Ensure there are no empty rows or corrupt/partial chemical annotations.
        *The system will automatically generate new columns detailing the exact AI-predicted LUMO configurations, structural toxicity assessments, and real-time interactive evaluation tools.*
        """)

    uploaded_file = st.file_uploader("Upload Target Screening Candidates File (.csv)", type=["csv"], key="batch_screen_uploader")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        with st.container(border=True):
            st.subheader("Configure Dataset Mapping")
            smiles_col = st.selectbox("Identify your dataset's SMILES Column:", df.columns)
            run_btn = st.button("Initialize Computational Screening Protocol", type="primary")
        
        if run_btn:
            bar = st.progress(0)
            preds, toxs = [], []
            model.eval()
            
            for i, s in enumerate(df[smiles_col]):
                try:
                    m = Chem.MolFromSmiles(str(s).strip())
                    toxs.append(check_toxicity(m) if m else "Corrupt SMILES")
                    
                    g = smiles_to_graph(str(s).strip())
                    if g:
                        b = torch.zeros(g.x.shape[0], dtype=torch.long)
                        with torch.no_grad():
                            p = model(g.x, g.edge_index, b, edge_attr=g.edge_attr).numpy()
                        preds.append(round(scaler.inverse_transform(p)[0][0], 3))
                    else: preds.append("Inference Failed")
                except:
                    preds.append("Error"); toxs.append("Error")
                bar.progress((i+1)/len(df))
                
            df['Predicted_LUMO_eV'] = preds
            df['Toxicity_PAINS_Status'] = toxs
            st.session_state.batch_df = df
            st.success("🔬 Virtual screening matrix successfully compiled!")
            
        if "batch_df" in st.session_state:
            res_df = st.session_state.batch_df
            st.markdown("### Processed Compound Library Overview")
            st.dataframe(res_df, use_container_width=True)
            
            st.download_button(
                label="📥 Export Computed Candidate Data (.csv)", 
                data=res_df.to_csv(index=False).encode('utf-8'), 
                file_name="screened_ote_library.csv",
                mime="text/csv"
            )
            
            # Interactive Dropdown Row Inspector Feature
            st.markdown("---")
            with st.container(border=True):
                st.subheader("🔍 Real-Time Individual Compound Inspector")
                st.write("Click any molecule SMILES string from your batch run to generate its structural 3D envelope and verify live database parameters:")
                valid_smiles = [s for s in res_df[smiles_col] if type(s) == str and Chem.MolFromSmiles(s.strip()) is not None]
                
                if valid_smiles:
                    inspect_smiles = st.selectbox("Select Target Compound from Screened Batch:", valid_smiles)
                    
                    b_col1, b_col2 = st.columns(2)
                    with b_col1:
                        i_mol = Chem.MolFromSmiles(inspect_smiles.strip())
                        i_mol = Chem.AddHs(i_mol)
                        AllChem.EmbedMolecule(i_mol, randomSeed=42)
                        try:
                            AllChem.UFFOptimizeMolecule(i_mol)
                        except: pass
                            
                        i_mblock = Chem.MolToMolBlock(i_mol)
                        viewer2 = py3Dmol.view(width=450, height=350)
                        viewer2.addModel(i_mblock, "mol")
                        viewer2.setStyle({'stick': {}})
                        viewer2.addSurface(py3Dmol.VDW, {'opacity': 0.5, 'colorscheme': 'cyanCarbon'})
                        viewer2.zoomTo()
                        showmol(viewer2, height=350, width=450)
                    
                    with b_col2:
                        pc_info = get_pubchem_data(inspect_smiles.strip())
                        st.markdown(f"**Structural Format:** `{inspect_smiles}`")
                        st.markdown(f"**PubChem CID Link:** `{pc_info['CID']}`")
                        st.markdown(f"**Systematic Title Name:** {pc_info['Name']}")
                        st.markdown(f"**Calculated XLogP Parameter:** `{pc_info['XLogP']}`")

# ==========================================
# --- TAB 3: ACTIVE FINE-TUNING ENGINE ---
# ==========================================
with tab3:
    st.subheader("🔄 In-Situ Weight Calibration & Fine-Tuning")
    
    with st.expander("ℹ️ MODEL CALIBRATION FILE PROTOCOLS - READ BEFORE RETRAINING", expanded=True):
        st.warning("""
        **To re-train and update the underlying GAT node weights, your spreadsheet must adhere to the following setup:**
        1. **Two-Column Requirement:** The CSV file must contain precisely two functional rows of parameters.
        2. **Column A (Structure):** Clear **SMILES text entries** (e.g. `C1=CC=CC=C1`).
        3. **Column B (Experimental Target):** The corresponding **Empirical LUMO Energy value** calculated in electron-volts (`eV`). This column must contain numerical floating point variables (e.g. `-4.21`, `-3.85`).
        *Retraining on custom experimental or high-level DFT datasets recalibrates structural parameters instantly.*
        """)

    train_file = st.file_uploader("Upload Experimental Calibration Dataset (.csv)", type=["csv"], key="engine_retrain_uploader")
    
    if train_file is not None:
        train_df = pd.read_csv(train_file)
        st.markdown("### Retraining Dataset Stream Preview")
        st.dataframe(train_df.head(4), use_container_width=True)
        
        with st.container(border=True):
            st.subheader("Map Calibration Array Injections")
            t_col1, t_col2 = st.columns(2)
            train_smiles_col = t_col1.selectbox("Map Target Structure Column (SMILES):", train_df.columns, index=0)
            train_target_col = t_col2.selectbox("Map Target Vector Column (LUMO eV):", train_df.columns, index=1 if len(train_df.columns) > 1 else 0)
            
            c3, c4, c5 = st.columns(3)
            epochs = c3.number_input("Optimization Iterations (Epochs)", min_value=1, max_value=500, value=30, step=10)
            lr = c4.number_input("Gradient Optimization Step Size (Learning Rate)", min_value=0.0001, max_value=0.01, value=0.0005, format="%.4f")
            batch_size = c5.number_input("Stochastic Batch Slicing Size", min_value=4, max_value=64, value=16, step=4)
            
            trigger_train = st.button("Execute Model Weight Modification Run", type="primary", use_container_width=True)
        
        if trigger_train:
            train_df[train_target_col] = pd.to_numeric(train_df[train_target_col], errors='coerce')
            train_df = train_df.dropna(subset=[train_smiles_col, train_target_col])
            
            with st.spinner("Transforming molecular data structures into structural graphs..."):
                train_graphs, raw_targets = [], []
                for _, row in train_df.iterrows():
                    graph = smiles_to_graph(str(row[train_smiles_col]).strip(), target_val=row[train_target_col])
                    if graph is not None:
                        train_graphs.append(graph)
                        raw_targets.append(row[train_target_col])
                
                if train_graphs:
                    scaled_targets = scaler.transform(np.array(raw_targets).reshape(-1, 1))
                    for i, graph in enumerate(train_graphs):
                        graph.y = torch.tensor([scaled_targets[i]], dtype=torch.float)
            
            if len(train_graphs) == 0:
                st.error("❌ Fatal Configuration Error: No valid molecular array definitions could be extracted from your setup.")
            else:
                loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                criterion = torch.nn.MSELoss()
                
                prog_text = st.empty()
                p_bar = st.progress(0)
                
                model.train()
                for epoch in range(1, epochs + 1):
                    total_loss = 0
                    for data in loader:
                        optimizer.zero_grad()
                        out = model(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr)
                        loss = criterion(out, data.y)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        total_loss += loss.item()
                    
                    avg_loss = total_loss / len(loader)
                    p_bar.progress(epoch / epochs)
                    prog_text.markdown(f"**Optimization Execution Status:** Epoch `{epoch}/{epochs}` | Evaluated Loss Mean Error: `{avg_loss:.5f}`")
                
                st.success("🎉 Fine-Tuning optimization completed without matrix failure!")
                
                # Setup model download payload
                buffer = io.BytesIO()
                torch.save(model.state_dict(), buffer)
                buffer.seek(0)
                
                st.download_button(
                    label="💾 Download Adjusted GAT Architecture Weights (.pth)", 
                    data=buffer, 
                    file_name="upgraded_n_type_expert (4).pth (1)", 
                    mime="application/octet-stream",
                    use_container_width=True
            )
