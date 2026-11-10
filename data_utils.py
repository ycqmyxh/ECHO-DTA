# -*- coding: utf-8 -*-

import os
import json
import hashlib
import random
import numpy as np
import networkx as nx
import torch
from tqdm import tqdm
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures
import config as cfg


def get_molecule_hyperedges(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None

    # 加载 RDKit 默认的药效团定义
    fdef_file = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdef_file)
    feats = factory.GetFeaturesForMol(mol)

    node_indices = []
    edge_indices = []
    edge_counter = 0

    # 1. 药效团超边 (语义维度)
    for f in feats:
        atom_ids = f.GetAtomIds()
        for a_id in atom_ids:
            node_indices.append(a_id)
            edge_indices.append(edge_counter)
        edge_counter += 1

    # 2. 环结构超边 (几何维度 - 保留作为补充)
    rings = mol.GetRingInfo().AtomRings()
    for ring in rings:
        for atom_idx in ring:
            node_indices.append(atom_idx)
            edge_indices.append(edge_counter)
        edge_counter += 1

    if not node_indices: # 兜底逻辑：如果没有提取到特征，使用全原子超边
        return torch.tensor([[i for i in range(mol.GetNumAtoms())], [0]*mol.GetNumAtoms()], dtype=torch.long)

    return torch.tensor([node_indices, edge_indices], dtype=torch.long)


def get_bond_features(bond):
    """提取 4 维边特征：单键、双键、三键、芳香键"""
    bt = bond.GetBondType()
    feat = [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
    ]
    return np.array(feat, dtype=np.float32)


def read_affinity_pairs(load_plm=True):

    files = {"Drugs": os.path.join(cfg.DATA_PATH, "drug_smiles.json"),
             "Proteins": os.path.join(cfg.DATA_PATH, "protein_seq.json"),
             "Affinity": os.path.join(cfg.DATA_PATH, "Y.txt")}
    for name, path in files.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing {name}: {path}")
    with open(files["Drugs"], encoding="utf-8") as f:
        drug_data = json.load(f)
    with open(files["Proteins"], encoding="utf-8") as f:
        prot_data = json.load(f)


    drug_ids = sorted(drug_data.keys())
    prot_ids = sorted(prot_data.keys())
    if list(drug_data) != drug_ids or list(prot_data) != prot_ids:
        print('[WARN] JSON insertion order differs from sorted IDs; original ECHO main'
              ' assumes Y.txt rows/columns follow sorted IDs. Verify the matrix indexing!')
    drugs = [drug_data[k] for k in drug_ids]
    prots = [prot_data[k] for k in prot_ids]

    Y = np.loadtxt(files["Affinity"], dtype=np.float64)
    expected = (len(drug_ids), len(prot_ids))
    if Y.shape == expected[::-1] and Y.shape != expected:
        Y = Y.T
    if Y.shape != expected:
        raise ValueError(f"Y.txt shape {Y.shape} does not match expected drug×protein {expected}")
    if np.isinf(Y).any():
        raise ValueError('Y.txt contains infinity')
    di, pi = np.where(~np.isnan(Y))
    if len(di) == 0:
        raise ValueError('No observed affinity pairs in Y.txt')
    labels = Y[di, pi].copy()
    if cfg.DATASET == 'davis':
        unit = cfg.LABEL_UNIT
        if unit == 'auto':
            unit = 'kd_nm' if float(np.max(labels)) > 100 else 'pkd'
        if unit == 'kd_nm':
            if (labels <= 0).any():
                raise ValueError('Davis Kd_nM labels must be strictly positive')
            labels = 9.0 - np.log10(labels)
        elif unit != 'pkd':
            raise ValueError('Davis supports --label-unit auto/pkd/kd_nm')
        label_name = 'pKd'
        print(f'[DATA] Davis unit input={unit}; reporting unit=pKd')
    else:
        if cfg.LABEL_UNIT not in ('auto', 'raw'):
            raise ValueError('KIBA labels are native scores; use --label-unit auto/raw')
        label_name = 'KIBA score'

    aff = torch.from_numpy(labels.astype(np.float32))
    if not torch.isfinite(aff).all():
        raise ValueError('Non-finite converted affinities')
    print(f'[DATA] {cfg.DATASET.upper()}: {len(drugs)} drugs, {len(prots)} proteins, '
          f'{len(aff)} observed interactions; {label_name} range '
          f'[{float(aff.min()):.4f}, {float(aff.max()):.4f}]')

    base = (drugs, prots, torch.from_numpy(di.astype(np.int64)),
            torch.from_numpy(pi.astype(np.int64)), aff)
    if not load_plm:
        return base
    if not os.path.isfile(cfg.PLM_FEAT_PATH):
        raise FileNotFoundError(f'Missing pretrained PLM features: {cfg.PLM_FEAT_PATH}')
    plm_data = torch.load(cfg.PLM_FEAT_PATH, map_location='cpu', weights_only=False)
    d_id2idx = plm_data['drug_id2idx']
    p_id2idx = plm_data['prot_id2idx']
    d_gather_idx = torch.tensor([d_id2idx[k] for k in drug_ids], dtype=torch.long)
    p_gather_idx = torch.tensor([p_id2idx[k] for k in prot_ids], dtype=torch.long)
    d_feat = plm_data['drug_feats'].float()[d_gather_idx]
    d_mask = plm_data['drug_masks'].float()[d_gather_idx]
    p_feat = plm_data['prot_feats'].float()[p_gather_idx]
    p_mask = plm_data['prot_masks'].float()[p_gather_idx]
    d_feat, d_mask = d_feat[:, :cfg.SMILES_MAX_LEN], d_mask[:, :cfg.SMILES_MAX_LEN]
    p_feat, p_mask = p_feat[:, :cfg.PROT_MAX_LEN], p_mask[:, :cfg.PROT_MAX_LEN]
    if d_feat.shape[-1] != cfg.PLM_DRUG_DIM or p_feat.shape[-1] != cfg.PLM_PROT_DIM:
        raise ValueError(f'Unexpected PLM feature dimensions: {d_feat.shape} and {p_feat.shape}')
    if d_mask.shape != d_feat.shape[:2] or p_mask.shape != p_feat.shape[:2]:
        raise ValueError('PLM embedding/mask shape mismatch')
    if not d_mask.bool().any(dim=1).all() or not p_mask.bool().any(dim=1).all():
        raise ValueError('PLM mask has an entity with zero valid tokens')
    return (drugs, prots, torch.nan_to_num(d_feat), d_mask,
            torch.nan_to_num(p_feat), p_mask, base[2], base[3], aff)


def load_data_and_plm():
    return read_affinity_pairs(load_plm=True)


def random_interaction_split(n, seed):

    if n < 20:
        raise ValueError('At least 20 observed interactions are required for 80/10/10')

    rng = np.random.RandomState(seed)
    indices = np.arange(n, dtype=np.int64)
    rng.shuffle(indices)
    n10 = int(.1 * n)
    te, val, tr = indices[:n10], indices[n10:2*n10], indices[2*n10:]
    assert len(np.unique(np.concatenate([tr, val, te]))) == n
    return tr, val, te


def split_digest(tr, val, te, di, pi, labels):
    h = hashlib.sha256()
    for arr in (tr, val, te, di.numpy(), pi.numpy(), labels.numpy()):
        h.update(np.asarray(arr).tobytes())
    return h.hexdigest()


def source_graph_fingerprint(drugs, prots):

    h = hashlib.sha256()
    h.update(b'ECHO_DTA_GRAPH_CACHE_V3_atom78_protein21_hyperfeatures')
    h.update(json.dumps({'drugs':list(drugs), 'proteins':list(prots)},
                        ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
    return h.hexdigest()


def write_seed_split(seed, tr, val, te, di, pi, aff):
    os.makedirs(cfg.CACHE_DIR, exist_ok=True)
    digest = split_digest(tr, val, te, di, pi, aff)
    path = os.path.join(cfg.CACHE_DIR, f'random_split_seed_{seed}.npz')
    if os.path.exists(path):
        with np.load(path) as old:
            for k, values in [('train_idx',tr), ('val_idx',val), ('test_idx',te),
                              ('pair_drug_idx',di.numpy()), ('pair_protein_idx',pi.numpy()),
                              ('labels_native',aff.numpy())]:
                if k not in old.files or not np.array_equal(old[k], values):
                    raise RuntimeError(f'Split content changed at {path} ({k}). '
                                       'Use another --output-dir; never silently reuse a split.')
            if str(old['split_sha256'].item()) != digest:
                raise RuntimeError(f'Split digest mismatch: {path}')
    else:
        np.savez_compressed(path, train_idx=tr, val_idx=val, test_idx=te,
                            pair_drug_idx=di.numpy(), pair_protein_idx=pi.numpy(),
                            labels_native=aff.numpy(), split_sha256=digest)
    return digest


def train_only_standardize(aff, tr_idx):
    mean = aff[tr_idx].double().mean().item()
    std = aff[tr_idx].double().std(unbiased=False).item()
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 1e-12:
        raise ValueError(f'Invalid training label stats: mean={mean}, std={std}')
    normalized = (aff.float() - mean) / std
    assert abs(float(normalized[tr_idx].double().mean())) < 1e-5
    return normalized, float(mean), float(std)


def one_hot_embedding(value, options):
    embedding = [0] * (len(options) + 1)
    index = options.index(value) if value in options else -1
    if index >= 0: embedding[index] = 1
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
    pad_len = cfg.DRUG_FEAT_DIM_RAW - feat.shape[0]
    if pad_len > 0:
        feat = np.concatenate([feat, np.zeros(pad_len, dtype=np.float32)])
    else:
        feat = feat[:cfg.DRUG_FEAT_DIM_RAW]
    return feat


def get_residue_features(residue_char):
    aa_list = list("ACDEFGHIKLMNPQRSTVWY")
    aa_map = {aa: i for i, aa in enumerate(aa_list)}
    idx = aa_map.get(residue_char.upper(), -1)
    feat = np.zeros(cfg.PROT_FEAT_DIM_RAW, dtype=np.float32)
    if 0 <= idx < cfg.PROT_FEAT_DIM_RAW: feat[idx] = 1.0
    return feat


def build_drug_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return nx.Graph()
    g = nx.Graph()
    for atom in mol.GetAtoms():
        feat = get_atom_features(atom)
        g.add_node(atom.GetIdx(), feat=feat)
    for bond in mol.GetBonds():
        # [修改] 为边添加化学键特征
        feat = get_bond_features(bond)
        g.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), feat=feat)
    return g


def build_prot_graph(seq):
    g = nx.Graph()
    seq = seq[:cfg.PROT_MAX_LEN]
    for i, aa in enumerate(seq):
        feat = get_residue_features(aa)
        g.add_node(i, feat=feat)
    for i in range(len(seq) - 1):
        g.add_edge(i, i + 1)
        if i < len(seq) - 2:
            g.add_edge(i, i + 2)
    return g


def process_graph(G, smiles, is_drug=True): # 增加 smiles 参数
    N = G.number_of_nodes()
    if N == 0: return None
    x = np.array([G.nodes[n]['feat'] for n in sorted(G.nodes())], dtype=np.float32)
    A = np.asarray(nx.adjacency_matrix(G).todense(), dtype=np.int8)

    if is_drug:
        E = np.zeros((N, N, 4), dtype=np.float32)
        for u, v, d in G.edges(data=True):
            E[u, v] = d['feat']
            E[v, u] = d['feat']
        # [新增] 提取超图索引
        H = get_molecule_hyperedges(smiles)
        return x, A, E, H
    return x, A


def prepad_graphs(graphs, feat_dim, is_drug=True):
    max_nodes = 0
    valid = []
    for item in graphs:
        # 药物分支解包 3 个参数，蛋白质分支解包 2 个
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
        if is_drug: E_t[i, :N, :N, :] = torch.from_numpy(item[2])

        if N > 0:
            A_mat = torch.from_numpy(A.astype(np.float32)) + torch.eye(N)
            deg_inv_sqrt = torch.pow(A_mat.sum(1), -0.5)
            deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
            D = torch.diag(deg_inv_sqrt)
            A_norm_t[i, :N, :N] = D.mm(A_mat).mm(D)

    return (X_t, A_bool_t, M_t, A_norm_t, E_t) if is_drug else (X_t, A_bool_t, M_t, A_norm_t)


def prepare_graph_cache(data):
    cache_file = os.path.join(cfg.CACHE_DIR, "graphs_processed_maxspeed.pt")
    drugs, prots, *_ = data
    fingerprint = source_graph_fingerprint(drugs, prots)
    if os.path.exists(cache_file):
        g = torch.load(cache_file, map_location='cpu', weights_only=False)
        if g.get('graph_fingerprint') != fingerprint:
            raise RuntimeError('Graph cache is stale or has no fingerprint. '
                               f'Delete {cache_file} or use a new --output-dir.')
        if len(g['drug'][5]) != len(drugs) or g['prot'][0].size(0) != len(prots):
            raise RuntimeError('Graph cache has incorrect entity counts')
        print('[INFO] Verified graph cache fingerprint:', fingerprint[:16])
        return

    print("[INFO] Building graph cache (Relational + Hypergraph)...")

    drug_graphs_raw = []
    drug_hyper_list = []

    for idx, smi in enumerate(tqdm(drugs, desc="Drug Graph")):
        res = process_graph(build_drug_graph(smi), smi, is_drug=True)
        if res is None:
            raise ValueError(f'Invalid/empty SMILES at drug index {idx}: {smi!r}')
        drug_graphs_raw.append((res[0], res[1], res[2]))
        drug_hyper_list.append(res[3])

    prot_graphs_raw = []
    for idx, seq in enumerate(tqdm(prots, desc="Protein Graph")):
        res = process_graph(build_prot_graph(seq), None, is_drug=False)
        if res is None:
            raise ValueError(f'Empty protein sequence at protein index {idx}')
        prot_graphs_raw.append(res)

    print("[INFO] Padding drug graphs...")
    dX, dA_bool, dM, dA_norm, dE = prepad_graphs(drug_graphs_raw, cfg.DRUG_FEAT_DIM_RAW, is_drug=True)

    print("[INFO] Padding protein graphs...")
    pX, pA_bool, pM, pA_norm = prepad_graphs(prot_graphs_raw, cfg.PROT_FEAT_DIM_RAW, is_drug=False)

    # 包含超图列表的最终数据字典
    g_data = {
        "graph_fingerprint": fingerprint,
        "drug": (dX, dA_bool, dM, dA_norm, dE, drug_hyper_list),
        "prot": (pX, pA_bool, pM, pA_norm)
    }
    torch.save(g_data, cache_file)
    print(f"[INFO] Graph cache saved to {cache_file}")
