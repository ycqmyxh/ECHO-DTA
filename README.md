# ECHO-DTA: Drug-Target Affinity Prediction

A deep learning model for drug-target affinity prediction, leveraging relational graph neural networks, interaction-aware hypergraph encoding, and multi-scale sequence modeling.

## Project Structure

```
MLP/
├── config.py          # Environment configuration, hyperparameters, path constants
├── data_utils.py      # Data loading, molecular/protein graph construction, graph caching
├── model.py           # Model components (GIN/GAT encoders, hypergraph interaction, DCI-DTA)
├── train.py           # Loss functions, evaluation metrics, training loop, main entry
├── requirements.txt   # Python dependencies

```

## Quick Start

**Note:** Before running the model, you must first execute `Data/extract_Kiba.py` and `Data/extract_Davis.py` to generate the required `.pt` feature files.

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare data: ensure the following files exist under `Data/KIBA/`:
   - `drug_smiles.json`
   - `protein_seq.json`
   - `Y.txt`
   - `plm_features_SOTA/plm_embeddings_sota_final.pt`

3. Run training:
```bash
python train.py
```

## Module Overview

| Module | Description |
|--------|-------------|
| `config.py` | Global configuration (mixed precision, dimensions, hyperparameters, paths) |
| `data_utils.py` | SMILES parsing, graph construction (drug/protein), hypergraph feature extraction, PLM feature loading and graph caching |
| `model.py` | EMA, RobustProjector, GINLayer, GATLayer, RelationalGINLayer, InteractionDynamicHyperEncoder, MultiScaleDilatedCNN, TokenCrossAttention, DCI_DTA_Model |
| `train.py` | Pairwise ranking loss, SupCon loss, CI / rm2 metrics, checkpoint resumption, EMA validation, main training loop |
