import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoConfig

# ==========================================
# 配置区域
# ==========================================
MODEL_CHOICE = 'molformer'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "Data", "KIBA")

DRUG_JSON = os.path.join(DATA_PATH, "drug_smiles.json")
PROT_JSON = os.path.join(DATA_PATH, "protein_seq.json")

OUTPUT_DIR = os.path.join(DATA_PATH, "plm_features_SOTA")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "plm_embeddings_sota_final.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 24
MAX_LEN_DRUG = 512
MAX_LEN_PROT = 1024


# ==========================================
# 模型加载
# ==========================================
def get_drug_model():
    if MODEL_CHOICE == 'molformer':
        print("[INFO] Loading MoLFormer...")
        model_name = "ibm/MoLFormer-XL-both-10pct"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    else:
        model_name = "DeepChem/ChemBERTa-77M-MTR"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    return tokenizer, model.to(DEVICE)


def get_prot_model():
    print("[INFO] Loading ESM-2 3B...")
    model_name = "facebook/esm2_t36_3B_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16).to(DEVICE)
    return tokenizer, model


# ==========================================
# 特征提取函数
# ==========================================
def extract_embeddings_and_masks(sequence_list, tokenizer, model, is_protein=False):
    embeddings = []
    masks = []
    model.eval()

    max_len_limit = MAX_LEN_PROT if is_protein else MAX_LEN_DRUG

    with torch.no_grad():
        for i in tqdm(range(0, len(sequence_list), BATCH_SIZE),
                      desc=f"Extracting ({'Prot' if is_protein else 'Drug'})"):
            batch_seqs = sequence_list[i: i + BATCH_SIZE]

            inputs = tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=max_len_limit,
                return_attention_mask=True
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            masks.append(inputs["attention_mask"].cpu())

            if is_protein:
                outputs = model(**inputs, output_hidden_states=True)
                emb = outputs.hidden_states[-1]
            else:
                outputs = model(**inputs)
                emb = outputs.last_hidden_state

            embeddings.append(emb.float().cpu())

    # 统一 Fixed Max Len
    print(f"[INFO] Aligning to fixed length = {max_len_limit}")

    feat_dim = embeddings[0].shape[2]
    total_samples = sum([t.shape[0] for t in embeddings])

    final_feats = torch.zeros((total_samples, max_len_limit, feat_dim), dtype=torch.float32)
    final_masks = torch.zeros((total_samples, max_len_limit), dtype=torch.float32)

    idx = 0
    for batch_feat, batch_mask in zip(embeddings, masks):
        b_size, b_len, _ = batch_feat.shape

        final_feats[idx:idx + b_size, :b_len, :] = batch_feat
        final_masks[idx:idx + b_size, :b_len] = batch_mask
        idx += b_size

    return final_feats, final_masks


# ==========================================
# 主程序
# ==========================================
def main():
    print(f"[INFO] Loading data from {DATA_PATH}")

    # 1. Load JSONs (并对齐排序)
    with open(DRUG_JSON) as f:
        d_data = json.load(f)

    # **关键修正：必须排序！确保与 Y.txt 对齐**
    d_keys = sorted(d_data.keys())
    drugs = [d_data[k] for k in d_keys]

    with open(PROT_JSON) as f:
        p_data = json.load(f)

    # **关键修正：必须排序！确保与 Y.txt 对齐**
    p_keys = sorted(p_data.keys())
    prots = [p_data[k] for k in p_keys]

    # 构建 idx lookup（与你 phc.py 完全一致）
    drug_id2idx = {k: i for i, k in enumerate(d_keys)}
    prot_id2idx = {k: i for i, k in enumerate(p_keys)}

    # 2. Drug Emb
    d_tok, d_model = get_drug_model()
    print(f"[INFO] Extracting drug embeddings: N={len(drugs)}")
    d_emb, d_mask = extract_embeddings_and_masks(drugs, d_tok, d_model, is_protein=False)

    del d_model
    del d_tok
    torch.cuda.empty_cache()

    # 3. Protein Emb
    p_tok, p_model = get_prot_model()
    print(f"[INFO] Extracting protein embeddings: N={len(prots)}")
    p_emb, p_mask = extract_embeddings_and_masks(prots, p_tok, p_model, is_protein=True)

    # 4. Save
    print(f"[INFO] Saving to {OUTPUT_FILE}")
    torch.save({
        "drug_feats": d_emb,
        "drug_masks": d_mask,
        "prot_feats": p_emb,
        "prot_masks": p_mask,
        "drug_ids": d_keys,
        "prot_ids": p_keys,
        "drug_id2idx": drug_id2idx,
        "prot_id2idx": prot_id2idx,
        "model_info": {
            "drug_model": MODEL_CHOICE,
            "prot_model": "esm2_3b_fp16",
            "padding_strategy": "fixed_max_len"
        }
    }, OUTPUT_FILE)

    print("\n[FINISHED] Feature generation complete.")
    print(f"Drug Feats: {d_emb.shape}")
    print(f"Prot Feats: {p_emb.shape}")
    print("=" * 50)


if __name__ == "__main__":
    main()
