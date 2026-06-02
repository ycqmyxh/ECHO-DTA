# data_utils.py — 数据加载、图构建、缓存处理
import os
import sys
import json
import random
import numpy as np
import torch
import networkx as nx
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig

from config import (
    DATA_PATH, PLM_FEAT_PATH, CACHE_DIR,
    SMILES_MAX_LEN, PROT_MAX_LEN,
    DRUG_FEAT_DIM_RAW, PROT_FEAT_DIM_RAW,
)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ---------------------------
# 分子特征提取
# ---------------------------
def one_hot_embedding(value, options):
    embedding = [0] * (len(options) + 1)
    index = options.index(value) if value in options else -1
    if index >= 0:
        embedding[index] = 1
    return embedding


def get_atom_features(atom):
    possible_atoms = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'H']
    atom_type = one_hot_embedding(atom.GetSymbol(), possible_atoms)
    possible_degrees = list(range(11))
    degree = one_hot_embedding(atom.GetTotalDegree(), possible_degrees)
    possible_implicit_valence = list(range(7))
    implicit_valence = one_hot_embedding(atom.GetImplicitValence(), possible_implicit_valence)
    aromatic = [1] if atom.GetIsAromatic() else [0]
    feat = np.array(atom_type + degree + implicit_valence + aromatic, dtype=np.float32)
    pad_len = DRUG_FEAT_DIM_RAW - feat.shape[0]
    if pad_len > 0:
        feat = np.concatenate([feat, np.zeros(pad_len, dtype=np.float32)])
    else:
        feat = feat[:DRUG_FEAT_DIM_RAW]
    return feat


def get_residue_features(residue_char):
    aa_list = list("ACDEFGHIKLMNPQRSTVWY")
    aa_map = {aa: i for i, aa in enumerate(aa_list)}
    idx = aa_map.get(residue_char.upper(), -1)
    feat = np.zeros(PROT_FEAT_DIM_RAW, dtype=np.float32)
    if 0 <= idx < PROT_FEAT_DIM_RAW:
        feat[idx] = 1.0
    return feat


def get_bond_features(bond):
    bt = bond.GetBondType()
    feat = [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
    ]
    return np.array(feat, dtype=np.float32)


# ---------------------------
# 超图构建
# ---------------------------
def get_molecule_hyperedges(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fdef_file = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdef_file)
    feats = factory.GetFeaturesForMol(mol)

    node_indices = []
    edge_indices = []
    edge_counter = 0

    for f in feats:
        atom_ids = f.GetAtomIds()
        for a_id in atom_ids:
            node_indices.append(a_id)
            edge_indices.append(edge_counter)
        edge_counter += 1

    rings = mol.GetRingInfo().AtomRings()
    for ring in rings:
        for atom_idx in ring:
            node_indices.append(atom_idx)
            edge_indices.append(edge_counter)
        edge_counter += 1

    if not node_indices:
        return torch.tensor(
            [[i for i in range(mol.GetNumAtoms())], [0] * mol.GetNumAtoms()],
            dtype=torch.long
        )

    return torch.tensor([node_indices, edge_indices], dtype=torch.long)


# ---------------------------
# 图构建
# ---------------------------
def build_drug_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return nx.Graph()
    g = nx.Graph()
    for atom in mol.GetAtoms():
        feat = get_atom_features(atom)
        g.add_node(atom.GetIdx(), feat=feat)
    for bond in mol.GetBonds():
        feat = get_bond_features(bond)
        g.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), feat=feat)
    return g


def build_prot_graph(seq):
    g = nx.Graph()
    seq = seq[:PROT_MAX_LEN]
    for i, aa in enumerate(seq):
        feat = get_residue_features(aa)
        g.add_node(i, feat=feat)
    for i in range(len(seq) - 1):
        g.add_edge(i, i + 1)
        if i < len(seq) - 2:
            g.add_edge(i, i + 2)
    return g


def process_graph(G, smiles, is_drug=True):
    N = G.number_of_nodes()
    if N == 0:
        return None
    x = np.array([G.nodes[n]['feat'] for n in sorted(G.nodes())], dtype=np.float32)
    A = np.asarray(nx.adjacency_matrix(G).todense(), dtype=np.int8)

    if is_drug:
        E = np.zeros((N, N, 4), dtype=np.float32)
        for u, v, d in G.edges(data=True):
            E[u, v] = d['feat']
            E[v, u] = d['feat']
        H = get_molecule_hyperedges(smiles)
        return x, A, E, H
    return x, A


def prepad_graphs(graphs, feat_dim, is_drug=True):
    max_nodes = 0
    valid = []
    for item in graphs:
        if is_drug:
            x, A, E = item
            if x is None:
                x, A, E = np.zeros((1, feat_dim)), np.eye(1), np.zeros((1, 1, 4))
        else:
            x, A = item
            if x is None:
                x, A = np.zeros((1, feat_dim)), np.eye(1)

        max_nodes = max(max_nodes, x.shape[0])
        valid.append((x, A, E) if is_drug else (x, A))

    num = len(valid)
    X_t = torch.zeros((num, max_nodes, feat_dim))
    A_norm_t = torch.zeros((num, max_nodes, max_nodes))
    M_t = torch.zeros((num, max_nodes))
    A_bool_t = torch.zeros((num, max_nodes, max_nodes), dtype=torch.bool)
    E_t = torch.zeros((num, max_nodes, max_nodes, 4)) if is_drug else None

    for i, item in enumerate(valid):
        x = item[0]
        A = item[1]
        N = x.shape[0]
        X_t[i, :N, :] = torch.from_numpy(x)
        M_t[i, :N] = 1.0
        A_bool_t[i, :N, :N] = torch.from_numpy(A) > 0
        if is_drug:
            E_t[i, :N, :N, :] = torch.from_numpy(item[2])

        if N > 0:
            A_mat = torch.from_numpy(A.astype(np.float32)) + torch.eye(N)
            deg_inv_sqrt = torch.pow(A_mat.sum(1), -0.5)
            deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
            D = torch.diag(deg_inv_sqrt)
            A_norm_t[i, :N, :N] = D.mm(A_mat).mm(D)

    return (X_t, A_bool_t, M_t, A_norm_t, E_t) if is_drug else (X_t, A_bool_t, M_t, A_norm_t)


# ---------------------------
# 主数据加载
# ---------------------------
def load_data_and_plm():
    files_needed = {
        "Drugs": os.path.join(DATA_PATH, "drug_smiles.json"),
        "Proteins": os.path.join(DATA_PATH, "protein_seq.json"),
        "Affinity": os.path.join(DATA_PATH, "Y.txt")
    }
    for name, path in files_needed.items():
        if not os.path.exists(path):
            print(f"[FATAL] missing {name} at {path}")
            sys.exit(1)

    with open(files_needed["Drugs"]) as f:
        drug_data = json.load(f)
        drug_ids = sorted(drug_data.keys())
        drugs = [drug_data[k] for k in drug_ids]
    with open(files_needed["Proteins"]) as f:
        prot_data = json.load(f)
        prot_ids = sorted(prot_data.keys())
        prots = [prot_data[k] for k in prot_ids]

    Y = np.loadtxt(files_needed["Affinity"])
    n_drugs = len(drug_ids)
    n_prots = len(prot_ids)
    if Y.shape == (n_prots, n_drugs):
        Y = Y.T

    drug_indices, prot_indices, aff_values = [], [], []
    for i in range(n_drugs):
        for j in range(n_prots):
            if not np.isnan(Y[i, j]):
                drug_indices.append(i)
                prot_indices.append(j)
                aff_values.append(Y[i, j])

    aff_values = torch.tensor(aff_values, dtype=torch.float32)
    aff_mean = aff_values.mean()
    aff_std = aff_values.std()
    aff_norm = (aff_values - aff_mean) / aff_std

    print(f"[INFO] Target Standardization Applied:")
    print(f"       Mean: {aff_mean.item():.4f} | Std: {aff_std.item():.4f}")

    if not os.path.exists(PLM_FEAT_PATH):
        sys.exit(f"[FATAL] PLM features not found: {PLM_FEAT_PATH}")

    plm_data = torch.load(PLM_FEAT_PATH, map_location='cpu')
    d_id2idx = plm_data['drug_id2idx']
    p_id2idx = plm_data['prot_id2idx']

    d_gather_idx = torch.tensor([d_id2idx[k] for k in drug_ids], dtype=torch.long)
    p_gather_idx = torch.tensor([p_id2idx[k] for k in prot_ids], dtype=torch.long)

    raw_d_feat = plm_data['drug_feats'].float()
    raw_p_feat = plm_data['prot_feats'].float()
    raw_d_mask = plm_data['drug_masks'].float()
    raw_p_mask = plm_data['prot_masks'].float()

    d_feat = raw_d_feat[d_gather_idx]
    d_mask = raw_d_mask[d_gather_idx]
    p_feat = raw_p_feat[p_gather_idx]
    p_mask = raw_p_mask[p_gather_idx]

    if d_feat.shape[1] > SMILES_MAX_LEN:
        d_feat = d_feat[:, :SMILES_MAX_LEN]
        d_mask = d_mask[:, :SMILES_MAX_LEN]
    if p_feat.shape[1] > PROT_MAX_LEN:
        p_feat = p_feat[:, :PROT_MAX_LEN]
        p_mask = p_mask[:, :PROT_MAX_LEN]

    return (
        drugs, prots,
        torch.nan_to_num(d_feat), d_mask,
        torch.nan_to_num(p_feat), p_mask,
        torch.tensor(drug_indices, dtype=torch.long),
        torch.tensor(prot_indices, dtype=torch.long),
        aff_values, aff_mean, aff_std
    )


def prepare_graph_cache(data):
    cache_file = os.path.join(CACHE_DIR, "graphs_processed_maxspeed.pt")

    if os.path.exists(cache_file):
        print("[INFO] Graph cache found.")
        return

    print("[INFO] Building graph cache (Relational + Hypergraph)...")
    drugs, prots, *_ = data

    drug_graphs_raw = []
    drug_hyper_list = []

    for s in tqdm(drugs, desc="Drug Graph"):
        res = process_graph(build_drug_graph(s), s, is_drug=True)
        if res:
            drug_graphs_raw.append((res[0], res[1], res[2]))
            drug_hyper_list.append(res[3])

    prot_graphs_raw = []
    for s in tqdm(prots, desc="Protein Graph"):
        res = process_graph(build_prot_graph(s), None, is_drug=False)
        if res:
            prot_graphs_raw.append(res)

    print("[INFO] Padding drug graphs...")
    dX, dA_bool, dM, dA_norm, dE = prepad_graphs(drug_graphs_raw, DRUG_FEAT_DIM_RAW, is_drug=True)

    print("[INFO] Padding protein graphs...")
    pX, pA_bool, pM, pA_norm = prepad_graphs(prot_graphs_raw, PROT_FEAT_DIM_RAW, is_drug=False)

    g_data = {
        "drug": (dX, dA_bool, dM, dA_norm, dE, drug_hyper_list),
        "prot": (pX, pA_bool, pM, pA_norm)
    }
    torch.save(g_data, cache_file)
    print(f"[INFO] Graph cache saved to {cache_file}")
