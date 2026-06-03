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
# 3. STREAMLIT UI SETUP
# ==========================================
st.set_page_config(page_title="Model A | OTE Deep LUMO", layout="wide")

@st.cache_resource
def load_assets():
    model = GATModel(num_node_features=11, edge_dim=1)
    
    # HARDWIRED FOR YOUR EXACT MOBILE DOWNLOAD NAMES
    model.load_state_dict(torch.load('upgraded_n_type_expert (4).pth (1)', map_location=torch.device('cpu')))
    scaler = joblib.load('polymer_lumo_scaler (1).pkl')
    
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"Waiting for files... Please ensure 'upgraded_n_type_expert (4).pth (1)' and 'polymer_lumo_scaler (1).pkl' are in your GitHub repo. System error: {e}")
    st.stop()

# --- HEADER ---
st.title("⚡ Model A: Deep LUMO Architecture")
st.markdown("High-throughput ambient air stability screener for n-type organic thermoelectric (OTE) polymers. Integrates size-invariant GAT predictions with structural minimization and bio-accumulation filtering.")

tab1, tab2 = st.tabs(["🔬 Single Molecule Studio", "📑 Batch Screening Pipeline"])

# --- TAB 1: STUDIO ---
with tab1:
    col_input, col_viz = st.columns([1, 1.2])
    
    with col_input:
        st.subheader("Molecular Editor")
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
        if c1.button("🧬 Add Fluorine (-F)"): mutate('[cH:1]>>[c:1](F)')
        if c2.button("🧬 Add Cyano (-C#N)"): mutate('[cH:1]>>[c:1](C#N)')
        
        smiles = st.text_input("SMILES String:", key="lumo_smiles").strip()
        mol = Chem.MolFromSmiles(smiles) if smiles else None

        if mol:
            st.markdown("### 🤖 Quantum Prediction")
            with st.spinner("Processing Graph Attention..."):
                model.eval()
                graph = smiles_to_graph(smiles)
                if graph is not None:
                    batch = torch.zeros(graph.x.shape[0], dtype=torch.long)
                    with torch.no_grad():
                        scaled_pred = model(graph.x, graph.edge_index, batch, edge_attr=graph.edge_attr).numpy()
                        pred_ev = scaler.inverse_transform(scaled_pred)[0][0]
                    
                    st.metric(label="Predicted LUMO Energy", value=f"{pred_ev:.3f} eV")
                    if pred_ev <= -4.0: st.success("✅ Deep LUMO: High Air-Stability")
                    elif pred_ev <= -3.5: st.warning("⚠️ Moderate LUMO")
                    else: st.error("❌ Shallow LUMO: Oxidation Risk")
                
            with st.expander("🌐 External Database (PubChem)", expanded=True):
                pc = get_pubchem_data(smiles)
                st.write(f"**CID:** {pc['CID']} | **XLogP:** {pc['XLogP']}")
                st.write(f"**IUPAC:** {pc['Name']}")

    with col_viz:
        if mol:
            st.subheader("3D Structure & Physics")
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
            mc1.metric("Pre-Min Strain", f"{pre_e:.1f} kcal/mol")
            mc2.metric("Post-Min Strain", f"{post_e:.1f} kcal/mol")
            if tox == "Pass": mc3.success("✅ PAINS: Pass")
            else: mc3.error("⚠️ PAINS: Fail")

            st.caption("Rendering Electrostatic VDW Surface (LUMO Approximation)")
            viewer = py3Dmol.view(width=500, height=400)
            viewer.addModel(Chem.MolToMolBlock(mol), "mol")
            viewer.setStyle({'stick': {}, 'sphere': {'radius': 0.3}})
            viewer.addSurface(py3Dmol.VDW, {'opacity': 0.5, 'colorscheme': 'cyanCarbon'})
            viewer.zoomTo()
            showmol(viewer, height=400, width=500)

# --- TAB 2: BATCH SCREENING ---
with tab2:
    st.subheader("High-Throughput Batch Processing")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        smiles_col = st.selectbox("SMILES Column:", df.columns)
        
        if st.button("Start AI Screening", type="primary"):
            bar = st.progress(0)
            preds, toxs = [], []
            model.eval()
            
            for i, s in enumerate(df[smiles_col]):
                try:
                    m = Chem.MolFromSmiles(str(s).strip())
                    toxs.append(check_toxicity(m) if m else "Error")
                    
                    g = smiles_to_graph(str(s).strip())
                    if g:
                        b = torch.zeros(g.x.shape[0], dtype=torch.long)
                        with torch.no_grad():
                            p = model(g.x, g.edge_index, b, edge_attr=g.edge_attr).numpy()
                        preds.append(round(scaler.inverse_transform(p)[0][0], 3))
                    else: preds.append("Invalid")
                except:
                    preds.append("Error"); toxs.append("Error")
                bar.progress((i+1)/len(df))
                
            df['Predicted_LUMO_eV'] = preds
            df['Toxicity_Filter'] = toxs
            st.session_state.batch_df = df
            st.success("Screening Complete!")
            
        if "batch_df" in st.session_state:
            res_df = st.session_state.batch_df
            st.dataframe(res_df, height=250)
            st.download_button("📥 Export Screened Candidates", data=res_df.to_csv(index=False).encode('utf-8'), file_name="screened_ote.csv")
            
            st.markdown("---")
            st.subheader("🔍 Interactive Molecule Inspector")
            valid_smiles = [s for s in res_df[smiles_col] if type(s) == str and Chem.MolFromSmiles(s.strip()) is not None]
            
            if valid_smiles:
                inspect_smiles = st.selectbox("Select a molecule from your batch to inspect:", valid_smiles)
                
                col_b1, col_b2 = st.columns(2)
                with col_b1:
                    i_mol = Chem.MolFromSmiles(inspect_smiles.strip())
                    i_mol = Chem.AddHs(i_mol)
                    AllChem.EmbedMolecule(i_mol, randomSeed=42)
                    
                    try:
                        AllChem.UFFOptimizeMolecule(i_mol)
                    except:
                        pass
                        
                    i_mblock = Chem.MolToMolBlock(i_mol)
                    viewer2 = py3Dmol.view(width=350, height=350)
                    viewer2.addModel(i_mblock, "mol")
                    viewer2.setStyle({'stick': {}})
                    viewer2.addSurface(py3Dmol.VDW, {'opacity': 0.5, 'colorscheme': 'cyanCarbon'})
                    viewer2.zoomTo()
                    showmol(viewer2, height=350, width=350)
                
                with col_b2:
                    pc_info = get_pubchem_data(inspect_smiles.strip())
                    st.write(f"**SMILES:** `{inspect_smiles}`")
                    st.write(f"**PubChem CID:** {pc_info['CID']}")
                    st.write(f"**Name:** {pc_info['Name']}")
                    st.write(f"**XLogP:** {pc_info['XLogP']}")
                                        
