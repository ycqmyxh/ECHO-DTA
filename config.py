# config.py — 环境配置、常量定义与路径声明
import os
import torch

# ---------------------------
# 1. 极致性能环境配置
# ---------------------------
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# ---------------------------
# 2. 实验配置
# ---------------------------
DATA_PATH = "Data/KIBA/"
PLM_FEAT_PATH = os.path.join(DATA_PATH, "plm_features_SOTA", "plm_embeddings_sota_final.pt")
CACHE_DIR = os.path.join(DATA_PATH, "cache_kiba_final_no_Pocket_and_Broadcast")
os.makedirs(CACHE_DIR, exist_ok=True)

# 维度配置
PLM_DRUG_DIM = 768
PLM_PROT_DIM = 2560
SMILES_MAX_LEN = 512
PROT_MAX_LEN = 1024
DRUG_FEAT_DIM_RAW = 78
PROT_FEAT_DIM_RAW = 21
GNN_HID = 512
GNN_OUT = 512
DROPOUT = 0.25

BATCH_SIZE = 128
EPOCHS = 120
PATIENCE = 12
MAX_LR = 8e-5
WEIGHT_DECAY = 1e-2

ALPHA_MSE = 1.0
ALPHA_RANK = 0.6
ALPHA_CL = 0.05

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BINARY_THRESHOLD = 12.1

# [OPTIM] 混合精度自动选择
try:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        AMP_TYPE = torch.bfloat16
        USE_SCALER = False
        print("[INFO] Acceleration: bfloat16 enabled (No Scaler).")
    else:
        AMP_TYPE = torch.float16
        USE_SCALER = True
        print("[INFO] Acceleration: float16 enabled (With Scaler).")
except:
    AMP_TYPE = torch.float16
    USE_SCALER = True

_SOFTMAX_NEG_INF = -1e4