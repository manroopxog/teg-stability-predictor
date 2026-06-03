import streamlit as st
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import FilterCatalog
import joblib
import numpy as np
import pandas as pd
import io
import urllib.parse
import requests
import py3Dmol
from stmol import showmol

# ==========================================
# 1. DEFINE THE GAT MODEL ARCHITECTURE (PERFECTLY MATCHED TO MODEL A)
# ==========================================
class GATModel(torch.nn.Module):
    def __init__(self, num_node_features=11, edge_dim=1):
        super(GATModel, self).__init__()
        
        # Layer 1: 11 in -> 64 out, 2 heads, concat=True (Output = 128)
        self.conv1 = GATConv(num_node_features, 64, heads=2, concat=True, edge_dim=edge_dim)
        
        # Layer 2: 128 in -> 64 out, 2 heads, concat=True (Output = 128)
        self.conv2 = GATConv(128, 64, heads=2, concat=True, edge_dim=edge_dim)
        
        # Layer 3: 128 in -> 64 out, 2 heads, concat=True (Output = 128)
        self.conv3 = GATConv(128, 64, heads=2, concat=True, edge_dim=edge_dim)
        
        # Layer 4: 128 in -> 64 out, 1 head, concat=False (Output = 64)
        self.conv4 = GATConv(128, 64, heads=1, concat=False, edge_dim=edge_dim)
        
        # MLPs matching checkpoint funnels (64 -> 32 -> 1)
        self.linear1 = torch.nn.Linear(64, 32)
        self.linear2 = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index, batch, edge_attr=None):
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = self.conv3(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = self.conv4(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        
        x = global_mean_pool(x, batch)
        
        x = self.linear1(x)
        x = F.elu(x)
        x = self.linear2(x)
        return x

# ==========================================
# 2. FEATURIZER & TOXICITY FILTERS (UPDATED FOR 1D EDGE)
# ==========================================
def get_node_features(atom):
    features = []
    atomic_num = atom.GetAtomicNum()
    # 5 features
    features += [float(atomic_num == i) for i in [6, 7, 8, 16, 9]]
    
    # 3 features
    hybridization = atom.GetHybridization()
    features += [
        float(hybridization == Chem.rdchem.HybridizationType.SP),
        float(hybridization == Chem.rdchem.HybridizationType.SP2),
        float(hybridization == Chem.rdchem.HybridizationType.SP3)
    ]
    
    # 3 features (Aromaticity, Charge, Hydrogen Count)
    features.append(float(atom.GetIsAromatic()))
    features.append(float(atom.GetFormalCharge()))
    features.append(float(atom.GetTotalNumHs()))
    
    return features

def get_edge_features(bond):
    # Condensing bond type into a 1D float as required by the checkpoint
    bt = bond.GetBondType()
    if bt == Chem.rdchem.BondType.SINGLE:
        return [1.0]
    elif bt == Chem.rdchem.BondType.DOUBLE:
        return [2.0]
    elif bt == Chem.rdchem.BondType.TRIPLE:
        return [3.0]
    elif bt == Chem.rdchem.BondType.AROMATIC:
        return [1.5]
    return [1.0]

def smiles_to_graph(smiles, target_val=None):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None: return None
    
    # Node features
    node_features = [get_node_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(node_features, dtype=torch.float)
    
    # Edge features
    edges = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.extend([[i, j], [j, i]])
        
        e_feat = get_edge_features(bond)
        edge_attrs.extend([e_feat, e_feat])
        
    if not edges:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    
    if target_val is not None:
        y = torch.tensor([[target_val]], dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# Setup PAINS Toxicity Filter
params = FilterCatalog.FilterCatalogParams()
params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
toxicity_catalog = FilterCatalog.FilterCatalog(params)

def check_toxicity(mol):
    if mol and toxicity_catalog.HasMatch(mol):
        entry = toxicity_catalog.GetFirstMatch(mol)
        return f"Fail: {entry.GetDescription()}"
    return "Pass"

def get_pubchem_data(smiles):
    try:
        safe_smiles = urllib.parse.quote(smiles)
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{safe_smiles}/property/IUPACName,MolecularWeight,MolecularFormula,XLogP/JSON"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            props = response.json()['PropertyTable']['Properties'][0]
            return {
                "CID": props.get('CID', 'Unknown'),
                "Name": props.get('IUPACName', 'Unknown'),
                "Mass": props.get('MolecularWeight', 'Unknown'),
                "XLogP": props.get('XLogP', 'N/A')
            }
        return {"CID": "Novel/Not Found", "Name": "N/A", "Mass": "N/A", "XLogP": "N/A"}
    except:
        return {"CID": "API Error", "Name": "N/A", "Mass": "N/A", "XLogP": "N/A"}

# ==========================================
# 3. LOAD ASSETS 
# ==========================================
st.set_page_config(page_title="Model A | LUMO Screener", layout="wide")

@st.cache_resource
def load_assets():
    model = GATModel(num_node_features=11, edge_dim=1)
    model.load_state_dict(torch.load('upgraded_n_type_expert (4).pth', map_location=torch.device('cpu')))
    scaler = joblib.load('polymer_lumo_scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"Failed to load model or scaler. Ensure files exist. Error: {e}")
    st.stop()

# ==========================================
# 4. STREAMLIT UI & TABS
# ==========================================
st.title("🔬 Model A: Deep LUMO Screener")
st.markdown("Predicting Lowest Unoccupied Molecular Orbital (LUMO) levels to ensure ambient air stability in n-type organic thermoelectrics.")

tab1, tab2, tab3 = st.tabs(["Single Molecule", "Batch Screening", "Active Fine-Tuning"])

# --- TAB 1: SINGLE MOLECULE STUDIO ---
with tab1:
    if "lumo_smiles_input" not in st.session_state:
        st.session_state.lumo_smiles_input = "N#CC(C#N)=C1C=CC(=C(C#N)C#N)C=C1"

    def run_mutation(rxn_smarts):
        try:
            mol = Chem.MolFromSmiles(st.session_state.lumo_smiles_input.strip())
            rxn = AllChem.ReactionFromSmarts(rxn_smarts)
            products = rxn.RunReactants((mol,))
            if products:
                new_mol = products[0][0] 
                Chem.SanitizeMol(new_mol)
                st.session_state.lumo_smiles_input = Chem.MolToSmiles(new_mol)
        except:
            pass 

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧬 Mutate: Add Fluorine (-F)"):
            run_mutation('[cH:1]>>[c:1](F)')
    with col2:
        if st.button("🧬 Mutate: Add Cyano (-C#N)"):
            run_mutation('[cH:1]>>[c:1](C#N)')

    user_smiles = st.text_input("Current Molecule SMILES:", key="lumo_smiles_input")
    clean_smiles = user_smiles.strip() if user_smiles else ""

    if clean_smiles:
        mol = Chem.MolFromSmiles(clean_smiles)
        if mol is not None:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            
            try:
                ff_pre = AllChem.UFFGetMoleculeForceField(mol)
                e_pre = ff_pre.CalcEnergy() if ff_pre else 0.0
                
                AllChem.UFFOptimizeMolecule(mol)
                
                ff_post = AllChem.UFFGetMoleculeForceField(mol)
                e_post = ff_post.CalcEnergy() if ff_post else 0.0
            except:
                e_pre, e_post = 0.0, 0.0
            
            col_viz, col_metrics = st.columns([1, 1])
            
            with col_viz:
                st.subheader("3D Electron Surface (Blobs)")
                mblock = Chem.MolToMolBlock(mol)
                viewer = py3Dmol.view(width=400, height=400)
                viewer.addModel(mblock, "mol")
                viewer.setStyle({'stick': {}, 'sphere': {'radius': 0.3}})
                viewer.addSurface(py3Dmol.VDW, {'opacity': 0.5, 'colorscheme': 'cyanCarbon'})
                viewer.zoomTo()
                showmol(viewer, height=400, width=400)
            
            with col_metrics:
                st.subheader("Physical Properties")
                st.metric("Initial Strain Energy", f"{e_pre:.2f} kcal/mol")
                st.metric("Minimized Strain Energy", f"{e_post:.2f} kcal/mol")
                
                tox_status = check_toxicity(mol)
                if tox_status == "Pass":
                    st.success("✅ Toxicity: PASS (No PAINS alerts)")
                else:
                    st.error(f"⚠️ Toxicity Alert: {tox_status}")
        else:
            st.error("Invalid SMILES string.")

    if st.button("Predict LUMO Level & Search PubChem", type="primary") and clean_smiles and mol is not None:
        model.eval() 
        with st.spinner("Calculating quantum features & pinging global databases..."):
            graph = smiles_to_graph(clean_smiles)
            batch = torch.zeros(graph.x.shape[0], dtype=torch.long)
            
            with torch.no_grad():
                scaled_pred = model(graph.x, graph.edge_index, batch, edge_attr=graph.edge_attr).numpy()
                predicted_ev = scaler.inverse_transform(scaled_pred)[0][0]
            
            st.subheader("🤖 AI Prediction Result")
            st.metric(label="Predicted LUMO Energy", value=f"{predicted_ev:.3f} eV")
            
            if predicted_ev <= -4.0:
                st.success("✅ DEEP LUMO: Excellent expected air stability.")
            elif -4.0 < predicted_ev <= -3.5:
                st.warning("⚠️ MODERATE LUMO: Marginal air stability.")
            else:
                st.error("❌ SHALLOW LUMO: Highly prone to oxidation in air.")

            pc_data = get_pubchem_data(clean_smiles)
            st.subheader("🌐 PubChem Reality Check")
            st.info(f"**CID:** {pc_data['CID']}\n\n**Name:** {pc_data['Name']}\n\n**Mass:** {pc_data['Mass']} g/mol\n\n**XLogP (Bioaccumulation proxy):** {pc_data['XLogP']}")

# --- TAB 2: BATCH SCREENING ---
with tab2:
    st.subheader("CSV Batch Screening with Toxicity Filters")
    uploaded_file = st.file_uploader("Upload candidate CSV", type=["csv"], key="batch_uploader")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        smiles_col = st.selectbox("Select SMILES column:", df.columns, key="batch_smiles")
        
        if st.button("Run Batch Screening", key="batch_run"):
            progress_text = "Processing molecules..."
            my_bar = st.progress(0, text=progress_text)
            
            preds, tox_alerts = [], []
            total = len(df)
            model.eval()
            
            for i, smiles in enumerate(df[smiles_col]):
                try:
                    mol = Chem.MolFromSmiles(str(smiles).strip())
                    tox_alerts.append(check_toxicity(mol) if mol else "Invalid")
                    
                    graph = smiles_to_graph(str(smiles).strip())
                    if graph is not None:
                        batch = torch.zeros(graph.x.shape[0], dtype=torch.long)
                        with torch.no_grad():
                            scaled_pred = model(graph.x, graph.edge_index, batch, edge_attr=graph.edge_attr).numpy()
                        real_val = scaler.inverse_transform(scaled_pred)[0][0]
                        preds.append(round(real_val, 4))
                    else:
                        preds.append("Invalid")
                except:
                    preds.append("Error")
                    tox_alerts.append("Error")
                my_bar.progress((i + 1) / total, text=f"Processed {i+1}/{total} molecules")
            
            df['Predicted_LUMO_eV'] = preds
            df['Toxicity_PAINS'] = tox_alerts
            st.session_state.batch_results = df
            st.success("Batch screening complete!")
        
        if "batch_results" in st.session_state:
            res_df = st.session_state.batch_results
            st.dataframe(res_df)
            csv = res_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results", data=csv, file_name="screened_LUMO_candidates.csv", mime="text/csv")
            
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

# --- TAB 3: ACTIVE FINE-TUNING ---
with tab3:
    st.subheader("Retrain & Fine-Tune Model Weights")
    st.info("Requires 2 columns: SMILES and Target LUMO (eV).")
    train_file = st.file_uploader("Upload Training Dataset (CSV)", type=["csv"], key="train_uploader")
    
    if train_file is not None:
        train_df = pd.read_csv(train_file)
        st.write("Dataset Preview:", train_df.head(3))
        
        col1, col2 = st.columns(2)
        train_smiles_col = col1.selectbox("SMILES Column:", train_df.columns, index=0)
        train_target_col = col2.selectbox("LUMO Column:", train_df.columns, index=1 if len(train_df.columns)>1 else 0)
        
        c3, c4, c5 = st.columns(3)
        epochs = c3.number_input("Epochs", min_value=1, max_value=500, value=50, step=10)
        lr = c4.number_input("Learning Rate", min_value=0.0001, max_value=0.01, value=0.0005, format="%.4f")
        batch_size = c5.number_input("Batch Size", min_value=4, max_value=64, value=16, step=4)
        
        if st.button("Start Fine-Tuning", type="primary"):
            train_df[train_target_col] = pd.to_numeric(train_df[train_target_col], errors='coerce')
            train_df = train_df.dropna(subset=[train_smiles_col, train_target_col])
            
            with st.spinner("Converting SMILES to Graph representations..."):
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
                st.error("No valid molecules found.")
            else:
                loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                criterion = torch.nn.MSELoss()
                progress_bar = st.progress(0, text="Training...")
                loss_text = st.empty()
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
                    progress_bar.progress(epoch / epochs, text=f"Epoch {epoch}/{epochs}")
                    loss_text.text(f"Current MSE Loss: {avg_loss:.4f}")
                
                st.success("Fine-tuning complete!")
                buffer = io.BytesIO()
                torch.save(model.state_dict(), buffer)
                buffer.seek(0)
                st.download_button("💾 Download Updated Model Weights (.pth)", data=buffer, file_name="upgraded_n_type_expert (4).pth", mime="application/octet-stream")
        
