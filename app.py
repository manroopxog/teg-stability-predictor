import streamlit as st
import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from rdkit import Chem
from rdkit.Chem import AllChem
import joblib
import pandas as pd
import io
import py3Dmol
from stmol import showmol


# --- PAGE SETUP ---
st.set_page_config(page_title="N-Type Semiconductor AI", page_icon="⚛️", layout="centered")

# --- ARCHITECTURE ---
class LumoPredictorGAT(torch.nn.Module):
    def __init__(self, num_node_features):
        super(LumoPredictorGAT, self).__init__()
        self.conv1 = GATConv(num_node_features, 64, heads=2, edge_dim=1)
        self.conv2 = GATConv(128, 64, heads=2, edge_dim=1)
        self.conv3 = GATConv(128, 64, heads=2, edge_dim=1)
        self.conv4 = GATConv(128, 64, heads=1, edge_dim=1)
        self.dropout = Dropout(p=0.2) 
        self.linear1 = Linear(64, 32)
        self.linear2 = Linear(32, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        edge_attr = edge_attr.view(-1, 1)
        x = F.relu(self.conv1(x, edge_index, edge_attr=edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr=edge_attr))
        x = F.relu(self.conv3(x, edge_index, edge_attr=edge_attr))
        x = F.relu(self.conv4(x, edge_index, edge_attr=edge_attr))
        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

def get_unified_features(atom):
    return [
        float(atom.GetAtomicNum()), float(atom.GetChiralTag()), float(atom.GetDegree()), 
        float(atom.GetFormalCharge()), float(atom.GetHybridization()), float(atom.GetIsAromatic()), 
        float(atom.GetMass()), float(atom.GetNumImplicitHs()), float(atom.GetNumRadicalElectrons()), 
        float(atom.GetTotalValence()), float(atom.GetTotalNumHs())
    ]

@st.cache_resource
def load_ai_engine():
    device = torch.device('cpu') 
    model = LumoPredictorGAT(num_node_features=11).to(device)
    try:
        model.load_state_dict(torch.load('upgraded_n_type_expert (4).pth', map_location=device))
    except:
        pass # dont crash if file is missing
    scaler = joblib.load('polymer_lumo_scaler.pkl')
    return model, scaler, device

model, scaler, device = load_ai_engine()

st.title("🧪 TEG Discovery & Training Studio")

# made 3 tabs now instead of 2
tab1, tab2, tab3 = st.tabs(["🧪 Predict Stability", "⚙️ Train the AI", "📁 Batch Screening"])


# ==========================================
# TAB 1: PREDICTION & DISCOVERY STUDIO
# ==========================================
with tab1:
    import urllib.parse
    import requests
    from rdkit.Chem import AllChem

    st.markdown("Enter a SMILES string, or use the Mutator buttons to alter the chemistry.")
    
    # --- UI STATE MANAGEMENT ---
    # This allows the buttons to physically change the text inside the input box
    if "smiles_input" not in st.session_state:
        st.session_state.smiles_input = "N#CC(C#N)=C1C=CC(=C(C#N)C#N)C=C1"

    # --- THE MOLECULE MUTATOR ENGINE ---
    def run_mutation(rxn_smarts):
        try:
            mol = Chem.MolFromSmiles(st.session_state.smiles_input)
            rxn = AllChem.ReactionFromSmarts(rxn_smarts)
            # Run the reaction on the current molecule
            products = rxn.RunReactants((mol,))
            if products:
                new_mol = products[0][0] # Grab the first generated variant
                Chem.SanitizeMol(new_mol)
                st.session_state.smiles_input = Chem.MolToSmiles(new_mol)
        except:
            pass # If the reaction fails (e.g. no valid attachment point), do nothing

    col1, col2 = st.columns(2)
    with col1:
        # Reaction: Finds an aromatic Carbon-Hydrogen bond and replaces it with Carbon-Fluorine
        if st.button("🧬 Mutate: Add Fluorine (-F)"):
            run_mutation('[cH:1]>>[c:1](F)')
    with col2:
        # Reaction: Finds an aromatic Carbon-Hydrogen bond and replaces it with a Cyano group
        if st.button("🧬 Mutate: Add Cyano (-C#N)"):
            run_mutation('[cH:1]>>[c:1](C#N)')

    # --- THE VISUALIZER ---
    user_smiles = st.text_input("Current Molecule SMILES:", st.session_state.smiles_input)
    
    # Update memory if you type manually
    if user_smiles != st.session_state.smiles_input:
        st.session_state.smiles_input = user_smiles

    if user_smiles:
        mol = Chem.MolFromSmiles(user_smiles)
        
        if mol is not None:
            st.subheader("Interactive 3D Geometry:")
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
            mblock = Chem.MolToMolBlock(mol)
            viewer = py3Dmol.view(width=400, height=400)
            viewer.addModel(mblock, "mol")
            viewer.setStyle({'stick': {}, 'sphere': {'radius': 0.4}})
            viewer.zoomTo()
            showmol(viewer, height=400, width=400)
        else:
            st.error("Invalid SMILES string. Please check your input.")

    # --- THE AI & API PREDICTION ---
    if st.button("Predict Stability & Search PubChem", type="primary"):
        torch.manual_seed(42)
        model.eval() 
        
        with st.spinner("Calculating quantum features & pinging global databases..."):
            mol = Chem.MolFromSmiles(user_smiles)
            if not mol:
                st.error("Invalid SMILES string.")
            else:
                # 1. AI Prediction Logic
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
                AllChem.UFFOptimizeMolecule(mol, maxIters=1000)
                    
                x = torch.tensor([get_unified_features(a) for a in mol.GetAtoms()], dtype=torch.float).to(device)
                edges, dists = [], []
                conf = mol.GetConformer()
                for bond in mol.GetBonds():
                    i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    d = conf.GetAtomPosition(i).Distance(conf.GetAtomPosition(j))
                    edges.extend([[i, j], [j, i]])
                    dists.extend([d, d])
                        
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
                edge_attr = torch.tensor(dists, dtype=torch.float).to(device)
                batch = torch.zeros(mol.GetNumAtoms(), dtype=torch.long).to(device)

                with torch.no_grad():
                    scaled_pred = model(x, edge_index, edge_attr, batch).cpu().numpy()
                    predicted_ev = scaler.inverse_transform(scaled_pred)[0][0]
                
                st.subheader("🤖 AI Prediction Result")
                st.metric(label="Predicted LUMO", value=f"{predicted_ev:.2f} eV")
                
                if predicted_ev <= -3.85:
                    st.success("✅ AIR-STABLE N-TYPE SEMICONDUCTOR.")
                elif predicted_ev <= -3.50:
                    st.warning("⚠️ BORDERLINE (Weak Acceptor).")
                else:
                    st.error("❌ UNSTABLE P-TYPE/DONOR.")

                # 2. PubChem API Logic
                st.subheader("🌐 PubChem Reality Check")
                try:
                    safe_smiles = urllib.parse.quote(user_smiles)
                    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{safe_smiles}/property/Title,MolecularWeight/JSON"
                    response = requests.get(url)
                    
                    if response.status_code == 200:
                        data = response.json()
                        props = data['PropertyTable']['Properties'][0]
                        name = props.get('Title', 'Unnamed Compound')
                        weight = props.get('MolecularWeight', 'Unknown')
                        st.info(f"**Molecule Recognized!**\n\n**Common Name:** {name}\n\n**Mass:** {weight} g/mol")
                    else:
                        st.success("🌟 **Novel Molecule!** No matches found in the PubChem database. You just engineered this.")
                except:
                    st.warning("Could not connect to PubChem API.")
            
# ==========================================
# TAB 2: THE TRAINING DASHBOARD
# ==========================================
with tab2:
    st.markdown("### Fine-Tune on Custom Data")
    st.info("Upload a CSV file with two columns: `smiles` and `LUMO`. The AI will adapt its weights to your dataset.")
    
    uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=['csv'])
    epochs = st.number_input("Set Number of Epochs", min_value=1, max_value=1000, value=100, step=10)
    
    if uploaded_file and st.button("Start Fine-Tuning 🚀"):
        df_custom = pd.read_csv(uploaded_file)
        
        if 'smiles' not in df_custom.columns or 'LUMO' not in df_custom.columns:
            st.error("CSV must contain 'smiles' and 'LUMO' columns.")
        else:
            st.write(f"Loaded {len(df_custom)} molecules. Processing 3D geometries...")
            graphs = []
            df_custom['Scaled_LUMO'] = scaler.transform(df_custom[['LUMO']])
            
            progress_bar = st.progress(0)
            for idx, row in df_custom.iterrows():
                mol = Chem.MolFromSmiles(row['smiles'])
                if not mol: continue
                mol = Chem.AddHs(mol)
                try:
                    AllChem.EmbedMolecule(mol, randomSeed=42)
                    AllChem.UFFOptimizeMolecule(mol)
                except: continue
                
                x = torch.tensor([get_unified_features(a) for a in mol.GetAtoms()], dtype=torch.float)
                edges, dists = [], []
                conf = mol.GetConformer()
                for bond in mol.GetBonds():
                    i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    d = conf.GetAtomPosition(i).Distance(conf.GetAtomPosition(j))
                    edges.extend([[i, j], [j, i]])
                    dists.extend([d, d])
                    
                if not edges: continue
                data = Data(x=x, edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(), 
                            edge_attr=torch.tensor(dists, dtype=torch.float), 
                            y=torch.tensor([[row['Scaled_LUMO']]], dtype=torch.float))
                graphs.append(data)
                progress_bar.progress(min((idx + 1) / len(df_custom), 1.0))
            
            st.success("Geometries generated! Starting PyTorch Training...")
            loader = DataLoader(graphs, batch_size=16, shuffle=True)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            loss_fn = torch.nn.MSELoss()
            
            model.train()
            train_progress = st.progress(0)
            status_text = st.empty()
            
            for epoch in range(1, epochs + 1):
                total_loss = 0
                for data in loader:
                    data = data.to(device)
                    optimizer.zero_grad()
                    out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                    loss = loss_fn(out, data.y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                if epoch % 5 == 0 or epoch == epochs:
                    train_progress.progress(epoch / epochs)
                    status_text.text(f"Epoch {epoch}/{epochs} | Loss: {total_loss/len(loader):.4f}")
            
            st.success("🎉 Fine-Tuning Complete!")
            
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            buffer.seek(0)
            
            st.download_button(
                label="📥 Download Upgraded AI Weights (.pth)",
                data=buffer,
                file_name="upgraded_n_type_expert.pth",
                mime="application/octet-stream"
            )

# ==========================================
# TAB 3: BATCH SCREENING
# ==========================================
with tab3:
    st.markdown("### High-Throughput Batch Screening")
    st.info("Upload a CSV with a column named `smiles`. The engine will predict LUMO for every molecule and generate a report.")
    
    batch_file = st.file_uploader("Upload Batch CSV", type=['csv'], key="batch_uploader")
    
    if batch_file and st.button("Run Batch Screen ⚡", type="primary"):
        df_batch = pd.read_csv(batch_file)
        
        if 'smiles' not in df_batch.columns:
            st.error("Error: The CSV must have a column named 'smiles'.")
        else:
            st.write(f"Found {len(df_batch)} molecules. Starting quantum predictions...")
            results = []
            
            # lock the ai brain for the whole batch
            torch.manual_seed(42)
            model.eval()
            
            progress_bar = st.progress(0)
            
            for idx, row in df_batch.iterrows():
                smiles_input = row['smiles']
                mol = Chem.MolFromSmiles(str(smiles_input))
                
                if not mol:
                    results.append("Invalid SMILES")
                    continue
                    
                mol = Chem.AddHs(mol)
                try:
                    # lock the geometry generation
                    AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
                    AllChem.UFFOptimizeMolecule(mol, maxIters=1000)
                    
                    x = torch.tensor([get_unified_features(a) for a in mol.GetAtoms()], dtype=torch.float).to(device)
                    edges, dists = [], []
                    conf = mol.GetConformer()
                    for bond in mol.GetBonds():
                        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                        d = conf.GetAtomPosition(i).Distance(conf.GetAtomPosition(j))
                        edges.extend([[i, j], [j, i]])
                        dists.extend([d, d])
                    
                    if not edges:
                        results.append("Geometry Error")
                        continue
                        
                    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
                    edge_attr = torch.tensor(dists, dtype=torch.float).to(device)
                    batch_tensor = torch.zeros(mol.GetNumAtoms(), dtype=torch.long).to(device)

                    with torch.no_grad():
                        scaled_pred = model(x, edge_index, edge_attr, batch_tensor).cpu().numpy()
                        predicted_ev = scaler.inverse_transform(scaled_pred)[0][0]
                        results.append(round(predicted_ev, 3))
                except Exception as e:
                    results.append("Error")
                
                progress_bar.progress(min((idx + 1) / len(df_batch), 1.0))
                
            df_batch['Predicted_LUMO'] = results
            st.success("Batch screening finished!")
            
            csv_buffer = df_batch.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results CSV",
                data=csv_buffer,
                file_name="screening_results.csv",
                mime="text/csv",
            )
            
            
