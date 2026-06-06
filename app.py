import streamlit as st
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem, FilterCatalog, DataStructs, rdFingerprintGenerator
import joblib
import numpy as np
import pandas as pd
import urllib.parse
import requests
import py3Dmol
from stmol import showmol
from scipy.linalg import eigh

# ==========================================
# 1. THE DUAL ARCHITECTURES
# ==========================================
class SolventAwareGAT(torch.nn.Module):
    def __init__(self, num_node_features=11, edge_dim=1, num_solvent_features=2):
        super(SolventAwareGAT, self).__init__()
        self.conv1 = GATConv(num_node_features, 64, heads=2, concat=True, edge_dim=edge_dim)
        self.conv2 = GATConv(128, 64, heads=2, concat=True, edge_dim=edge_dim)
        self.conv3 = GATConv(128, 64, heads=2, concat=True, edge_dim=edge_dim)
        self.conv4 = GATConv(128, 64, heads=1, concat=False, edge_dim=edge_dim)
        self.linear1 = torch.nn.Linear(64 + num_solvent_features, 32)
        self.linear2 = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index, batch, solvent_features, edge_attr=None):
        x = F.elu(self.conv1(x, edge_index, edge_attr=edge_attr))
        x = F.elu(self.conv2(x, edge_index, edge_attr=edge_attr))
        x = F.elu(self.conv3(x, edge_index, edge_attr=edge_attr))
        x = F.elu(self.conv4(x, edge_index, edge_attr=edge_attr))
        mol_embedding = global_mean_pool(x, batch)
        combined = torch.cat([mol_embedding, solvent_features], dim=1)
        out = F.elu(self.linear1(combined))
        return self.linear2(out)

class SpectralSpatialGAT(torch.nn.Module):
    def __init__(self, num_node_features=6, num_spectral_features=3):
        super(SpectralSpatialGAT, self).__init__()
        self.conv1 = GATConv(num_node_features, 64, heads=2, concat=True)
        self.conv2 = GATConv(128, 64, heads=2, concat=True)
        self.conv3 = GATConv(128, 64, heads=1, concat=False)
        self.linear1 = torch.nn.Linear(64 + num_spectral_features, 32)
        self.linear2 = torch.nn.Linear(32, 1)

    def forward(self, data):
        x, edge_index, batch, spectral = data.x, data.edge_index, data.batch, data.spectral
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = F.elu(self.conv3(x, edge_index))
        mol_embedding = global_mean_pool(x, batch)
        combined = torch.cat([mol_embedding, spectral], dim=1)
        out = F.elu(self.linear1(combined))
        return self.linear2(out)

# ==========================================
# 2. MATH ENGINES & UTILS
# ==========================================
SOLVENTS = {
    "Vacuum (Gas Phase)": [1.0, 0.0],
    "Toluene": [2.38, 0.36],
    "Chloroform": [4.81, 1.04],
    "Dichloromethane": [8.93, 1.60],
    "Acetonitrile": [37.5, 3.92]
}

params = FilterCatalog.FilterCatalogParams()
params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
toxicity_catalog = FilterCatalog.FilterCatalog(params)

def check_toxicity(mol):
    if mol and toxicity_catalog.HasMatch(mol): return f"Fail: {toxicity_catalog.GetFirstMatch(mol).GetDescription()}"
    return "Pass"

def get_pubchem_data(smiles):
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{urllib.parse.quote(smiles)}/property/IUPACName,MolecularWeight,XLogP/JSON"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            props = response.json()['PropertyTable']['Properties'][0]
            return {"CID": props.get('CID', 'N/A'), "Name": props.get('IUPACName', 'N/A'), "XLogP": props.get('XLogP', 'N/A')}
    except: pass
    return {"CID": "N/A", "Name": "N/A", "XLogP": "N/A"}

def extract_spectral_signatures(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if not mol: return [0.0, 0.0, 0.0]
    num_atoms = mol.GetNumAtoms()
    if num_atoms < 2: return [0.0, 0.0, 0.0]
    
    A = np.zeros((num_atoms, num_atoms))
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        A[i, j] = A[j, i] = bond.GetBondTypeAsDouble()
        
    try:
        w, v = eigh(A)
        idx = w.argsort()
        w, v = w[idx], v[:, idx]
        spectral_radius = np.max(np.abs(w))
        mid = len(w) // 2
        spectral_gap = w[mid] - w[mid - 1] if len(w) > 1 else 0
        wave_localization = np.var(v[:, mid]) if len(w) > 1 else 0
        return [spectral_radius / 5.0, spectral_gap / 5.0, wave_localization * 10.0]
    except: return [0.0, 0.0, 0.0]

def get_model_a_node_features(atom):
    features = [float(atom.GetAtomicNum() == i) for i in [6, 7, 8, 16, 9]]
    hyb = atom.GetHybridization()
    features += [float(hyb == Chem.rdchem.HybridizationType.SP), float(hyb == Chem.rdchem.HybridizationType.SP2), float(hyb == Chem.rdchem.HybridizationType.SP3)]
    features.extend([float(atom.GetIsAromatic()), float(atom.GetFormalCharge()), float(atom.GetTotalNumHs())])
    return features

def get_model_b_node_features(atom):
    return [float(atom.GetAtomicNum() == i) for i in [6, 7, 8, 16, 9]] + [float(atom.GetIsAromatic())]

def get_edge_features(bond):
    bt = bond.GetBondType()
    val = 2.0 if bt == Chem.rdchem.BondType.DOUBLE else 3.0 if bt == Chem.rdchem.BondType.TRIPLE else 1.5 if bt == Chem.rdchem.BondType.AROMATIC else 1.0
    return [val]

def smiles_to_graphs(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None: return None, None
    
    # Model A Graph
    x_a = torch.tensor([get_model_a_node_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    edges, edge_attrs = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.extend([[i, j], [j, i]])
        e_feat = get_edge_features(bond)
        edge_attrs.extend([e_feat, e_feat])
    edge_index_a = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
    edge_attr_a = torch.tensor(edge_attrs, dtype=torch.float) if edges else torch.empty((0, 1), dtype=torch.float)
    graph_a = Data(x=x_a, edge_index=edge_index_a, edge_attr=edge_attr_a)

    # Model B Graph
    x_b = torch.tensor([get_model_b_node_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    spectral_feats = torch.tensor([extract_spectral_signatures(smiles)], dtype=torch.float)
    graph_b = Data(x=x_b, edge_index=edge_index_a, spectral=spectral_feats) if torch.sum(spectral_feats) != 0 else None

    return graph_a, graph_b

@st.cache_data
def load_training_fingerprints():
    try:
        df_ref = pd.read_csv('training_smiles.csv')
        fps = []
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        for s in df_ref['smiles']:
            m = Chem.MolFromSmiles(str(s))
            if m: fps.append(mfpgen.GetFingerprint(m))
        return fps
    except: return []

def calculate_tanimoto_domain(target_smiles, train_fps):
    if not train_fps: return 1.0 
    mol = Chem.MolFromSmiles(target_smiles)
    if not mol: return 0.0
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    target_fp = mfpgen.GetFingerprint(mol)
    sims = DataStructs.BulkTanimotoSimilarity(target_fp, train_fps)
    return max(sims)

# ==========================================
# 3. STREAMLIT UI & ASSET LOADING
# ==========================================
st.set_page_config(page_title="Indigenous μ-TEG Screening Suite", layout="wide")

@st.cache_resource
def load_assets():
    # Load Model A (Thermodynamics)
    m_a = SolventAwareGAT()
    m_a.load_state_dict(torch.load('solvent_aware_model.pth', map_location='cpu'))
    l_scaler = joblib.load('lumo_scaler.pkl')
    s_scaler = joblib.load('solvent_scaler.pkl')
    
    # Load Model B (Kinetics / Mobility)
    m_b = SpectralSpatialGAT()
    m_b.load_state_dict(torch.load('model_b_reorg_engine.pth', map_location='cpu'))
    r_scaler = joblib.load('reorg_scaler.pkl')
    
    return m_a, l_scaler, s_scaler, m_b, r_scaler

try:
    model_A, lumo_scaler, solvent_scaler, model_B, reorg_scaler = load_assets()
    training_fps = load_training_fingerprints()
except Exception as e:
    st.error(f"⚠️ Initialization Error. Are all 6 model/scaler files in GitHub? Details: {e}")
    st.stop()

st.title("⚡ Indigenous μ-TEG Discovery Suite")
st.markdown("Dual-engine AI pipeline screening n-type organic semiconductors for ambient stability (LUMO) and charge carrier mobility (Reorganization Energy).")

tab1, tab2 = st.tabs(["🎯 Single Molecule Studio", "📑 High-Throughput Batch Pipeline"])

# --- TAB 1: STUDIO ---
with tab1:
    col_input, col_viz = st.columns([1, 1.2])
    
    with col_input:
        with st.container(border=True):
            st.subheader("🧪 Doping Environment")
            sol_choice = st.selectbox("Select Target Solvent/Environment:", list(SOLVENTS.keys()) + ["Custom Solvent"])
            if sol_choice == "Custom Solvent":
                diel = st.number_input("Dielectric Constant", value=10.0)
                dip = st.number_input("Dipole Moment", value=2.0)
            else:
                diel, dip = SOLVENTS[sol_choice]

        with st.container(border=True):
            st.subheader("🛠️ μ-TEG Core Structure")
            smiles = st.text_input("Target Chemical Structure (SMILES):", value="N#CC(C#N)=Cc1ccsc1").strip()
            mol = Chem.MolFromSmiles(smiles) if smiles else None

        if mol:
            max_sim = calculate_tanimoto_domain(smiles, training_fps)
            if max_sim < 0.45:
                st.warning(f"⚠️ **Domain of Applicability Alert:** Tanimoto Similarity is {max_sim:.2f}. Molecule is out-of-distribution.")

            with st.container(border=True):
                st.subheader("🤖 Dual-Engine Prediction")
                with st.spinner("Processing thermodynamic and kinetic profiles..."):
                    model_A.eval()
                    model_B.eval()
                    graph_a, graph_b = smiles_to_graphs(smiles)
                    
                    if graph_a is not None and graph_b is not None:
                        batch = torch.zeros(graph_a.x.shape[0], dtype=torch.long)
                        
                        # Predict Model A (LUMO)
                        scaled_solvent = solvent_scaler.transform(np.array([[diel, dip]]))
                        sol_tensor = torch.tensor(scaled_solvent, dtype=torch.float)
                        with torch.no_grad():
                            scaled_lumo = model_A(graph_a.x, graph_a.edge_index, batch, sol_tensor, edge_attr=graph_a.edge_attr).numpy()
                            pred_lumo = lumo_scaler.inverse_transform(scaled_lumo)[0][0]
                            
                            # Predict Model B (Reorganization Energy)
                            batch_b = torch.zeros(graph_b.x.shape[0], dtype=torch.long)
                            data_b = Data(x=graph_b.x, edge_index=graph_b.edge_index, batch=batch_b, spectral=graph_b.spectral)
                            scaled_reorg = model_B(data_b).numpy()
                            pred_reorg = reorg_scaler.inverse_transform(scaled_reorg)[0][0]
                        
                        # Display Results
                        subcol1, subcol2 = st.columns(2)
                        subcol1.metric(label=f"LUMO in {sol_choice}", value=f"{pred_lumo:.3f} eV")
                        subcol2.metric(label="Reorganization Energy (λ)", value=f"{pred_reorg:.3f} eV")
                        
                        if pred_lumo <= -3.8: st.success("✅ Deep LUMO: Resists Ambient Oxidation.")
                        else: st.error("❌ Shallow LUMO: Oxidation Risk in air.")
                        
                        if pred_reorg <= 0.200: st.success("⚡ High Mobility: Rigid structural backbone.")
                        else: st.warning("🧱 Lower Mobility: High structural distortion upon charging.")

            with st.container(border=True):
                st.subheader("🌐 PubChem Database Cross-Reference")
                pc = get_pubchem_data(smiles)
                st.markdown(f"**Compound CID:** `{pc['CID']}` | **IUPAC Name:** `{pc['Name']}` | **XLogP:** `{pc['XLogP']}`")


    with col_viz:
        if mol:
            with st.container(border=True):
                st.subheader("📐 3D Spatial Inspector")
                mol_3d = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol_3d, randomSeed=42)
                try: AllChem.UFFOptimizeMolecule(mol_3d)
                except: pass
                
                tox = check_toxicity(mol)
                if tox == "Pass": st.success("🛡️ PAINS Toxicity Filter: PASS")
                else: st.error(f"🚨 ALERT: {tox}")

                viewer = py3Dmol.view(width=500, height=400)
                viewer.addModel(Chem.MolToMolBlock(mol_3d), "mol")
                viewer.setStyle({'stick': {}, 'sphere': {'radius': 0.3}})
                viewer.addSurface(py3Dmol.VDW, {'opacity': 0.45, 'colorscheme': 'cyanCarbon'})
                viewer.zoomTo()
                showmol(viewer, height=400, width=500)

# --- TAB 2: BATCH SCREENING ---
with tab2:
    st.subheader("📊 High-Throughput μ-TEG Screening")
    col_bsolv, col_bupl = st.columns(2)
    batch_sol_choice = col_bsolv.selectbox("Batch Processing Solvent:", list(SOLVENTS.keys()))
    b_diel, b_dip = SOLVENTS[batch_sol_choice]
    
    uploaded_file = col_bupl.file_uploader("Upload Candidates (.csv)", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        smiles_col = st.selectbox("Identify SMILES Column:", df.columns)
        
        if st.button("Initialize Dual-Engine Screening", type="primary"):
            bar = st.progress(0)
            lumos, reorgs, toxs, sims = [], [], [], []
            model_A.eval()
            model_B.eval()
            
            sol_tensor = torch.tensor(solvent_scaler.transform(np.array([[b_diel, b_dip]])), dtype=torch.float)
            
            for i, s in enumerate(df[smiles_col]):
                try:
                    s_str = str(s).strip()
                    m = Chem.MolFromSmiles(s_str)
                    toxs.append(check_toxicity(m) if m else "Corrupt")
                    sims.append(calculate_tanimoto_domain(s_str, training_fps))
                    
                    ga, gb = smiles_to_graphs(s_str)
                    if ga and gb:
                        with torch.no_grad():
                            ba = torch.zeros(ga.x.shape[0], dtype=torch.long)
                            p_lumo = model_A(ga.x, ga.edge_index, ba, sol_tensor, edge_attr=ga.edge_attr).numpy()
                            lumos.append(round(lumo_scaler.inverse_transform(p_lumo)[0][0], 3))
                            
                            bb = torch.zeros(gb.x.shape[0], dtype=torch.long)
                            db = Data(x=gb.x, edge_index=gb.edge_index, batch=bb, spectral=gb.spectral)
                            p_reorg = model_B(db).numpy()
                            reorgs.append(round(reorg_scaler.inverse_transform(p_reorg)[0][0], 3))
                    else:
                        lumos.append("Failed")
                        reorgs.append("Failed")
                except:
                    lumos.append("Error"); reorgs.append("Error"); toxs.append("Error"); sims.append(0.0)
                bar.progress((i+1)/len(df))
                
            df[f'LUMO_eV_in_{batch_sol_choice}'] = lumos
            df['Reorganization_Energy_eV'] = reorgs
            df['Toxicity_Status'] = toxs
            df['Tanimoto_Similarity'] = sims
            st.session_state.batch_df = df
            st.success("🔬 Dual-engine virtual screening complete!")
            
        if "batch_df" in st.session_state:
            res_df = st.session_state.batch_df
            st.dataframe(res_df, use_container_width=True)
            st.download_button("📥 Export Screened Data", res_df.to_csv(index=False).encode('utf-8'), "screened_muteg.csv", "text/csv")
            st.markdown("---")
            with st.container(border=True):
                st.subheader("🔍 Real-Time Compound Inspector")
                valid_smiles = [s for s in res_df[smiles_col] if type(s) == str and Chem.MolFromSmiles(s.strip()) is not None]
                
                if valid_smiles:
                    inspect_smiles = st.selectbox("Select Target Compound from Screened Batch:", valid_smiles)
                    
                    b_col1, b_col2 = st.columns(2)
                    with b_col1:
                        i_mol = Chem.MolFromSmiles(inspect_smiles.strip())
                        i_mol = Chem.AddHs(i_mol)
                        AllChem.EmbedMolecule(i_mol, randomSeed=42)
                        try: AllChem.UFFOptimizeMolecule(i_mol)
                        except: pass
                            
                        viewer2 = py3Dmol.view(width=450, height=350)
                        viewer2.addModel(Chem.MolToMolBlock(i_mol), "mol")
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
                        
    
