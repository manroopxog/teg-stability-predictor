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
        model.load_state_dict(torch.load('model_A_polymer_expert.pth', map_location=device))
    except:
        pass # Will handle missing file gracefully in the UI
    scaler = joblib.load('polymer_lumo_scaler.pkl')
    return model, scaler, device

model, scaler, device = load_ai_engine()

st.title("🧪 TEG Discovery & Training Studio")

# Create Two Tabs for the App
tab1, tab2 = st.tabs(["🧪 Predict Stability", "⚙️ Train the AI"])

# ==========================================
# TAB 1: PREDICTION (Your existing code)
# ==========================================
with tab1:
    st.markdown("Enter the SMILES string of a hypothetical organic semiconductor.")
    user_smiles = st.text_input("Enter Molecule SMILES:", "N#CC(C#N)=C1C=CC(=C(C#N)C#N)C=C1")

    if st.button("Predict Stability", type="primary"):
     # 1. LOCK THE AI BRAIN
     torch.manual_seed(42)
     model.eval() 
    
    with st.spinner("Calculating quantum spatial features..."):
        mol = Chem.MolFromSmiles(user_smiles)
        if not mol:
            st.error("Invalid SMILES string.")
        else:
            mol = Chem.AddHs(mol)
            # 2. LOCK THE 3D GEOMETRY
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
                
                st.subheader("AI Prediction Result")
                st.metric(label="Predicted LUMO", value=f"{predicted_ev:.2f} eV")
                
                if predicted_ev <= -3.85:
                    st.success("✅ AIR-STABLE N-TYPE SEMICONDUCTOR.")
                elif predicted_ev <= -3.50:
                    st.warning("⚠️ BORDERLINE (Weak Acceptor).")
                else:
                    st.error("❌ UNSTABLE P-TYPE/DONOR.")

# ==========================================
# TAB 2: THE TRAINING DASHBOARD (New!)
# ==========================================
with tab2:
    st.markdown("### Fine-Tune on Custom Data")
    st.info("Upload a CSV file with two columns: `smiles` and `LUMO`. The AI will adapt its weights to your dataset.")
    
    # 1. File Uploader
    uploaded_file = st.file_uploader("Upload N-Type Dataset (CSV)", type=['csv'])
    
    # 2. Epoch Editor
    epochs = st.number_input("Set Number of Epochs", min_value=1, max_value=1000, value=100, step=10)
    
    if uploaded_file and st.button("Start Fine-Tuning 🚀"):
        df_custom = pd.read_csv(uploaded_file)
        
        if 'smiles' not in df_custom.columns or 'LUMO' not in df_custom.columns:
            st.error("CSV must contain 'smiles' and 'LUMO' columns.")
        else:
            st.write(f"Loaded {len(df_custom)} molecules. Processing 3D geometries...")
            
            # Prepare Data
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
                
                # Update progress bar
                progress_bar.progress(min((idx + 1) / len(df_custom), 1.0))
            
            st.success("Geometries generated! Starting PyTorch Training...")
            
            # Training Loop
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
                
                # Update UI every 5 epochs
                if epoch % 5 == 0 or epoch == epochs:
                    train_progress.progress(epoch / epochs)
                    status_text.text(f"Epoch {epoch}/{epochs} | Loss: {total_loss/len(loader):.4f}")
            
            st.success("🎉 Fine-Tuning Complete!")
            
            # Save the newly trained weights into memory so user can download them
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            buffer.seek(0)
            
            st.download_button(
                label="📥 Download Upgraded AI Weights (.pth)",
                data=buffer,
                file_name="upgraded_n_type_expert.pth",
                mime="application/octet-stream"
            )
