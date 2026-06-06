import streamlit as st
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
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
# 1. SOLVENT-AWARE ARCHITECTURE
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

def get_node_features(atom):
    features = [float(atom.GetAtomicNum() == i) for i in [6, 7, 8, 16, 9]]
    hyb = atom.GetHybridization()
    features += [float(hyb == Chem.rdchem.HybridizationType.SP), float(hyb == Chem.rdchem.HybridizationType.SP2), float(hyb == Chem.rdchem.HybridizationType.SP3)]
    features.extend([float(atom.GetIsAromatic()), float(atom.GetFormalCharge()), float(atom.GetTotalNumHs())])
    return features

def get_edge_features(bond):
    bt = bond.GetBondType()
    val = 2.0 if bt == Chem.rdchem.BondType.DOUBLE else 3.0 if bt == Chem.rdchem.BondType.TRIPLE else 1.5 if bt == Chem.rdchem.BondType.AROMATIC else 1.0
    return [val]

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None: return None
    x = torch.tensor([get_node_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    edges, edge_attrs = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.extend([[i, j], [j, i]])
        e_feat = get_edge_features(bond)
        edge_attrs.extend([e_feat, e_feat])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float) if edges else torch.empty((0, 1), dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# ==========================================
# 2. CONSTANTS & UTILS
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

# ==========================================
# 3. STREAMLIT UI SETUP
# ==========================================
st.set_page_config(page_title="Solvent-Aware LUMO Screener", layout="wide")

@st.cache_resource
def load_assets():
    m = SolventAwareGAT()
    # Expecting the exact base names you download from Drive
    m.load_state_dict(torch.load('solvent_aware_model.pth', map_location='cpu'))
    l_scaler = joblib.load('lumo_scaler.pkl')
    s_scaler = joblib.load('solvent_scaler.pkl')
    return m, l_scaler, s_scaler

try:
    model, lumo_scaler, solvent_scaler = load_assets()
except Exception as e:
    st.error(f"⚠️ Initialization Error: Please ensure 'solvent_aware_model.pth', 'lumo_scaler.pkl', and 'solvent_scaler.pkl' are uploaded. Details: {e}")
    st.stop()

st.title("🌊 Model A: Solvent-Aware Deep LUMO Suite")
st.markdown("Predicts highly accurate ambient air stability for n-type organic semiconductors by dynamically calculating solvation energy shifts.")

tab1, tab2 = st.tabs(["🎯 Single Molecule Studio", "📑 High-Throughput Batch Pipeline"])

# --- TAB 1: STUDIO ---
with tab1:
    col_input, col_viz = st.columns([1, 1.2])
    
    with col_input:
        with st.container(border=True):
            st.subheader("🧪 Chemical Environment")
            sol_choice = st.selectbox("Select Target Solvent Environment:", list(SOLVENTS.keys()) + ["Custom Solvent"])
            
            if sol_choice == "Custom Solvent":
                diel = st.number_input("Dielectric Constant", value=10.0)
                dip = st.number_input("Dipole Moment", value=2.0)
            else:
                diel, dip = SOLVENTS[sol_choice]
            
            st.info(f"**Current Fluid Dynamics:** Dielectric: {diel} | Dipole: {dip}")

        with st.container(border=True):
            st.subheader("🛠️ Molecular Structure")
            smiles = st.text_input("Target Chemical Structure (SMILES):", value="N#CC(C#N)=Cc1ccsc1").strip()
            mol = Chem.MolFromSmiles(smiles) if smiles else None

        if mol:
            with st.container(border=True):
                st.subheader("🤖 Solvent-Aware Prediction")
                with st.spinner("Calculating physical solvation shifts..."):
                    model.eval()
                    graph = smiles_to_graph(smiles)
                    if graph is not None:
                        batch = torch.zeros(graph.x.shape[0], dtype=torch.long)
                        
                        # Apply Solvent Scaler
                        scaled_solvent = solvent_scaler.transform(np.array([[diel, dip]]))
                        sol_tensor = torch.tensor(scaled_solvent, dtype=torch.float)

                        with torch.no_grad():
                            scaled_pred = model(graph.x, graph.edge_index, batch, sol_tensor, edge_attr=graph.edge_attr).numpy()
                            pred_ev = lumo_scaler.inverse_transform(scaled_pred)[0][0]
                        
                        st.metric(label=f"Predicted LUMO in {sol_choice}", value=f"{pred_ev:.3f} eV")
                        if pred_ev <= -4.0: st.success("✅ Deep LUMO: Highly Air-Stable in this environment.")
                        elif pred_ev <= -3.5: st.warning("⚠️ Intermediate Stability.")
                        else: st.error("❌ Shallow LUMO: Oxidation Risk.")

    with col_viz:
        if mol:
            with st.container(border=True):
                st.subheader("📐 Structural Energy Optimization")
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, randomSeed=42)
                try: AllChem.UFFOptimizeMolecule(mol)
                except: pass
                
                tox = check_toxicity(mol)
                if tox == "Pass": st.success("🛡️ PAINS Toxicity Filter: PASS")
                else: st.error(f"🚨 ALERT: {tox}")

                viewer = py3Dmol.view(width=500, height=400)
                viewer.addModel(Chem.MolToMolBlock(mol), "mol")
                viewer.setStyle({'stick': {}, 'sphere': {'radius': 0.3}})
                viewer.addSurface(py3Dmol.VDW, {'opacity': 0.45, 'colorscheme': 'cyanCarbon'})
                viewer.zoomTo()
                showmol(viewer, height=400, width=500)

# --- TAB 2: BATCH SCREENING ---
with tab2:
    st.subheader("📊 Batch Screening in Liquid Environments")
    st.info("Upload a CSV containing your SMILES strings. Select a processing solvent, and the AI will predict the solvated LUMO for every molecule in the batch instantly.")
    
    col_bsolv, col_bupl = st.columns(2)
    batch_sol_choice = col_bsolv.selectbox("Batch Processing Solvent:", list(SOLVENTS.keys()))
    b_diel, b_dip = SOLVENTS[batch_sol_choice]
    
    uploaded_file = col_bupl.file_uploader("Upload Screening Candidates (.csv)", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        smiles_col = st.selectbox("Identify SMILES Column:", df.columns)
        
        if st.button("Initialize Environment Screening", type="primary"):
            bar = st.progress(0)
            preds, toxs = [], []
            model.eval()
            
            # Prepare constant solvent tensor for the whole batch
            scaled_solvent = solvent_scaler.transform(np.array([[b_diel, b_dip]]))
            sol_tensor = torch.tensor(scaled_solvent, dtype=torch.float)
            
            for i, s in enumerate(df[smiles_col]):
                try:
                    m = Chem.MolFromSmiles(str(s).strip())
                    toxs.append(check_toxicity(m) if m else "Corrupt SMILES")
                    
                    g = smiles_to_graph(str(s).strip())
                    if g:
                        b = torch.zeros(g.x.shape[0], dtype=torch.long)
                        with torch.no_grad():
                            p = model(g.x, g.edge_index, b, sol_tensor, edge_attr=g.edge_attr).numpy()
                        preds.append(round(lumo_scaler.inverse_transform(p)[0][0], 3))
                    else: preds.append("Inference Failed")
                except:
                    preds.append("Error"); toxs.append("Error")
                bar.progress((i+1)/len(df))
                
            df[f'Predicted_LUMO_eV_in_{batch_sol_choice}'] = preds
            df['Toxicity_Status'] = toxs
            st.session_state.batch_df = df
            st.success("🔬 Solvated virtual screening complete!")
            
        if "batch_df" in st.session_state:
            st.dataframe(st.session_state.batch_df, use_container_width=True)
            st.download_button(
                label="📥 Export Screened Data (.csv)", 
                data=st.session_state.batch_df.to_csv(index=False).encode('utf-8'), 
                file_name=f"screened_ote_{batch_sol_choice}.csv",
                mime="text/csv"
        )
            
