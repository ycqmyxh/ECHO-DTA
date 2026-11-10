# -*- coding: utf-8 -*-

import os
import gc
import json
import random
import math
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim.lr_scheduler import OneCycleLR
from torch.amp import autocast, GradScaler
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
import config as cfg
from data_utils import (
    read_affinity_pairs, load_data_and_plm, random_interaction_split,
    write_seed_split, train_only_standardize, source_graph_fingerprint,
    prepare_graph_cache,
)
from model import EMA, DCI_DTA_Model, HypergraphConv


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def pairwise_ranking_loss(y_pred, y_true, num_samples=2048):
    B = y_true.shape[0]
    idx1 = torch.randint(0, B, (num_samples,), device=y_true.device)
    idx2 = torch.randint(0, B, (num_samples,), device=y_true.device)

    diff = y_true[idx1] - y_true[idx2]
    pred_diff = y_pred[idx1] - y_pred[idx2]

    # 带通滤波：忽略 <0.05 的纯噪声，忽略 >0.5 的送分题
    # 让算力 100% 聚焦于“微观困难对”
    abs_diff = diff.abs()
    mask = ((abs_diff > 0.05) & (abs_diff < 0.5)).float()

    valid_count = mask.sum()
    if valid_count == 0:
        return torch.tensor(0.0, device=y_true.device)

    #Softplus 连续平滑排序
    loss = F.softplus(-torch.sign(diff) * pred_diff)

    return (loss * mask).sum() / (valid_count + 1e-6)


def supcon_loss(features, labels, temperature=0.07, num_bins=10):
    device = features.device
    features = F.normalize(features, dim=1)

    try:
        quantiles = torch.linspace(0.0, 1.0, steps=num_bins + 1, device=device)
        bins = torch.quantile(labels, quantiles)
        labels_q = torch.bucketize(labels, torch.quantile(labels, torch.linspace(0,1,6)))
    except:
        return torch.tensor(0.0, device=device)

    mask = torch.eq(labels_q.unsqueeze(1), labels_q.unsqueeze(0)).float()
    logits = torch.matmul(features, features.T) / temperature

    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    logits_mask = 1.0 - torch.eye(features.shape[0], device=device)
    mask = mask * logits_mask

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
    valid = (mask.sum(1) > 0).float()

    if valid.sum() == 0: return torch.tensor(0.0, device=device)
    loss = -(valid * mean_log_prob_pos).sum() / (valid.sum() + 1e-6)
    return loss


def fast_concordance_index_vectorized(y_true, y_pred, max_pairs=None):

    true = np.asarray(y_true, dtype=np.float64).ravel()
    pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if len(true) != len(pred) or len(true) < 2:
        return float('nan')
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError('Non-finite labels/predictions in CI')
    order = np.argsort(true, kind='stable')
    t, p = true[order], pred[order]
    uniq_pred = np.unique(p)
    rank = np.searchsorted(uniq_pred, p).astype(np.int64)
    tree = np.zeros(len(uniq_pred) + 1, dtype=np.int64)

    def prefix(n):
        count = 0
        while n > 0:
            count += int(tree[n])
            n -= n & -n
        return count

    def add(pos):
        pos += 1
        while pos < len(tree):
            tree[pos] += 1
            pos += pos & -pos

    concord = 0.0
    comparable = 0
    n_earlier = 0
    i = 0
    while i < len(t):
        j = i + 1
        while j < len(t) and t[j] == t[i]:
            j += 1
  
        for k in range(i, j):
            less = prefix(int(rank[k]))
            leq = prefix(int(rank[k]) + 1)
            concord += less + .5 * (leq - less)
        comparable += n_earlier * (j - i)
        for k in range(i, j):
            add(int(rank[k]))
        n_earlier += j - i
        i = j
    return concord / comparable if comparable else float('nan')


def calculate_rm2(y_true, y_pred):

    y = np.asarray(y_true, dtype=np.float64).ravel()
    p = np.asarray(y_pred, dtype=np.float64).ravel()
    if y.size < 2 or np.std(y) == 0 or np.std(p) == 0:
        return float('nan')
    r_sq = float(np.corrcoef(y, p)[0, 1] ** 2)
    denom = float(np.dot(p, p))
    sst = float(np.square(y-y.mean()).sum())
    if denom <= 0 or sst <= 0:
        return float('nan')
    k = float(np.dot(y, p) / denom)
    r0_sq = 1.0 - float(np.square(y - k*p).sum() / sst)
    return r_sq * (1.0 - np.sqrt(abs(r_sq - r0_sq)))


def native_predictions(y_true_normalized, y_pred_normalized, mean, std):
    def npflat(x):
        return x.detach().float().cpu().numpy().reshape(-1) if isinstance(x, torch.Tensor) else np.asarray(x).reshape(-1)
    y_t = npflat(y_true_normalized).astype(np.float64) * std + mean
    y_p = npflat(y_pred_normalized).astype(np.float64) * std + mean
    if len(y_t) != len(y_p) or not np.isfinite(y_t).all() or not np.isfinite(y_p).all():
        raise ValueError('Invalid predicted/true affinities')
    return y_t, y_p


def compute_metrics(y_true_normalized, y_pred_normalized, mean, std):
    """Only ONE inverse transform, on both normalized true and predicted labels."""
    y_t, y_p = native_predictions(y_true_normalized, y_pred_normalized, mean, std)
    if len(y_t) < 2:
        raise ValueError('At least 2 validation/testing interactions required')
    mse_val = float(np.mean(np.square(y_t - y_p)))
    r2 = float(r2_score(y_t, y_p))
    ci = float(fast_concordance_index_vectorized(y_t, y_p))
    if not np.isfinite(ci):
        raise ValueError('Undefined CI: no comparable labels; inspect split')
    pear = float(pearsonr(y_t, y_p)[0]) if np.std(y_t) > 0 and np.std(y_p) > 0 else float('nan')
    spear = float(spearmanr(y_t, y_p)[0]) if np.std(y_t) > 0 and np.std(y_p) > 0 else float('nan')
    return {'MSE':mse_val,'CI':ci,'R2':r2,'rm2':float(calculate_rm2(y_t,y_p)),
            'Pearson':pear,'Spearman':spear}


def clean_model(model):
    return model._orig_mod if hasattr(model, '_orig_mod') else model


def get_clean_state_dict(model):
    return {k: v.detach().cpu() for k, v in clean_model(model).state_dict().items()}


def make_hyperedge_batch(drug_ids, hyper_list, real_d):
   
    h_list = [hyper_list[int(i)] for i in drug_ids.detach().cpu().tolist()]
    valid = [(idx, h) for idx, h in enumerate(h_list) if h is not None and h.numel() > 0]
    if not valid:
        return None
    new_parts = []
    total_edges = 0
    for pos, h in valid:
        h = h.to(device=cfg.DEVICE, dtype=torch.long).clone()
        h[0] += pos * real_d
        h[1] += total_edges
        total_edges += int(h[1].max().item()) - total_edges + 1
        new_parts.append(h)
    return torch.cat(new_parts, dim=1)


def run_cycle(seed, run_idx, data):
    drugs, prots, d_plm, d_mask, p_plm, p_mask, d_idx, p_idx, aff = data
    seed_everything(seed)
    tr_idx, val_idx, te_idx = random_interaction_split(len(aff), seed)
    digest = write_seed_split(seed, tr_idx, val_idx, te_idx, d_idx, p_idx, aff)
    normalized_aff, mean, std = train_only_standardize(aff, tr_idx)
    with open(os.path.join(cfg.CACHE_DIR, f'target_stats_seed_{seed}.json'), 'w') as f:
        json.dump({'seed': seed, 'dataset': cfg.DATASET, 'split_type': 'interaction_random_80_10_10',
                   'target_mean': mean, 'target_std_population': std,
                   'unit': 'pKd' if cfg.DATASET == 'davis' else 'KIBA native score',
                   'split_sha256': digest,
                   'train_n':len(tr_idx), 'val_n':len(val_idx), 'test_n':len(te_idx)}, f, indent=2)
    print(f'[SPLIT seed={seed}] train={len(tr_idx)} val={len(val_idx)} test={len(te_idx)} '
          f'train-only mean={mean:.6f} std={std:.6f} SHA={digest[:12]}')

    metric_file = os.path.join(cfg.CACHE_DIR, f'test_metrics_seed_{seed}.json')
    best_model_path = os.path.join(cfg.CACHE_DIR, f'best_model_seed_{seed}.pth')
    last_ckpt_path = os.path.join(cfg.CACHE_DIR, f'last_checkpoint_seed_{seed}.pth')
    if os.path.isfile(metric_file) and not cfg.OVERWRITE:
        with open(metric_file) as f:
            old_result = json.load(f)
        if (old_result.get('split_sha256') != digest
            or old_result.get('dataset') != cfg.DATASET
            or abs(float(old_result.get('target_mean',float('nan'))) - mean) > 1e-6
            or abs(float(old_result.get('target_std',float('nan'))) - std) > 1e-6):
            raise RuntimeError('Existing test result does not match current split or label stats. '
                               'Use --overwrite or another --output-dir.')
        print(f'[SKIP] Seed {seed} already completed: {metric_file}')
        return old_result['metrics']
    if cfg.OVERWRITE:
        for p in [metric_file, best_model_path, last_ckpt_path]:
            if os.path.isfile(p):
                os.remove(p)

    model_raw = DCI_DTA_Model().to(cfg.DEVICE)
    ema = EMA(model_raw, decay=0.995)
    model = model_raw
    if cfg.ENABLE_COMPILE:
        try:
            model = torch.compile(model_raw, mode='default', dynamic=True, fullgraph=False)
            print('[INFO] torch.compile(dynamic=True) enabled')
        except Exception as e:
            model = model_raw
            print('[WARN] compile unavailable, eager fallback:', e)
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.MAX_LR,
                                       weight_decay=cfg.WEIGHT_DECAY, fused=(cfg.DEVICE.type == 'cuda'))
    except TypeError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.MAX_LR, weight_decay=cfg.WEIGHT_DECAY)
    steps_per_epoch = math.ceil(len(tr_idx)/cfg.BATCH_SIZE)
    scheduler = OneCycleLR(optimizer, max_lr=cfg.MAX_LR, epochs=cfg.EPOCHS,
                           steps_per_epoch=steps_per_epoch, pct_start=0.1, div_factor=20)
    scaler = GradScaler(enabled=cfg.USE_SCALER)

    G_d_plm, G_d_mask = d_plm.to(cfg.DEVICE, dtype=cfg.AMP_TYPE), d_mask.to(cfg.DEVICE).bool()
    G_p_plm, G_p_mask = p_plm.to(cfg.DEVICE, dtype=cfg.AMP_TYPE), p_mask.to(cfg.DEVICE).bool()
    G_idx_d, G_idx_p = d_idx.to(cfg.DEVICE), p_idx.to(cfg.DEVICE)
    # CRITICAL: G_aff is STANDARDIZED, not overwritten with raw aff.
    G_aff = normalized_aff.to(cfg.DEVICE)
    if not np.isclose(float(G_aff[torch.as_tensor(tr_idx,device=cfg.DEVICE)].float().mean()), 0, atol=2e-5):
        raise AssertionError('Training labels are not standardized')

    g_data = torch.load(os.path.join(cfg.CACHE_DIR, 'graphs_processed_maxspeed.pt'),
                        map_location='cpu', weights_only=False)
    if (g_data.get('graph_fingerprint') != source_graph_fingerprint(drugs, prots)
        or g_data['drug'][0].shape[0] != len(drugs)
        or g_data['prot'][0].shape[0] != len(prots)):
        raise RuntimeError('Graph cache incompatible with current entity order or contents')
    G_dE = g_data['drug'][4].to(cfg.DEVICE, dtype=torch.float32)
    G_dX = g_data['drug'][0].to(cfg.DEVICE, dtype=torch.float32)
    G_dA_norm = g_data['drug'][3].to(cfg.DEVICE, dtype=cfg.AMP_TYPE)
    G_dA_bool = g_data['drug'][1].to(cfg.DEVICE).bool()
    G_dM = g_data['drug'][2].to(cfg.DEVICE).bool()
    G_pX = g_data['prot'][0].to(cfg.DEVICE, dtype=torch.float32)
    G_pA_norm = g_data['prot'][3].to(cfg.DEVICE, dtype=cfg.AMP_TYPE)
    G_pA_bool = g_data['prot'][1].to(cfg.DEVICE).bool()
    G_pM = g_data['prot'][2].to(cfg.DEVICE).bool()
    G_d_hyper = g_data['drug'][5]

    def inference(target_indices):
        model.eval()
        preds, trues = [], []
        with torch.inference_mode(), autocast('cuda', dtype=cfg.AMP_TYPE):
            for i in range(0, len(target_indices), cfg.BATCH_SIZE * 2):
                b_idx = target_indices[i:i+cfg.BATCH_SIZE*2]
                bd, bp = G_idx_d[b_idx], G_idx_p[b_idx]
                real_d = max(1, int(G_dM[bd].sum(1).max().item()))
                real_p = max(1, int(G_pM[bp].sum(1).max().item()))
                cur_hyper = make_hyperedge_batch(bd, G_d_hyper, real_d)
                p, _, _, _ = model(
                    G_d_plm[bd], G_d_mask[bd], G_p_plm[bp], G_p_mask[bp],
                    G_dX[bd][:,:real_d], G_dA_norm[bd][:,:real_d,:real_d],
                    G_dA_bool[bd][:,:real_d,:real_d], G_dE[bd][:,:real_d,:real_d],
                    G_dM[bd][:,:real_d],
                    G_pX[bp][:,:real_p], G_pA_norm[bp][:,:real_p,:real_p],
                    G_pA_bool[bp][:,:real_p,:real_p], G_pM[bp][:,:real_p], cur_hyper)
                preds.append(p.detach().float().cpu())
                trues.append(G_aff[b_idx].detach().float().cpu())
        return torch.cat(trues), torch.cat(preds)

    tr_gpu = torch.as_tensor(tr_idx, dtype=torch.long, device=cfg.DEVICE)
    val_gpu = torch.as_tensor(val_idx, dtype=torch.long, device=cfg.DEVICE)
    te_gpu = torch.as_tensor(te_idx, dtype=torch.long, device=cfg.DEVICE)
    start_epoch, global_step, early_stop = 0, 0, 0
    best_score, best_ci, best_mse = -float('inf'), float('nan'), float('inf')
    if os.path.isfile(last_ckpt_path):
        ckpt = torch.load(last_ckpt_path, map_location=cfg.DEVICE, weights_only=False)
        if (ckpt.get('format_version') != 2 or ckpt.get('split_sha256') != digest
           or abs(float(ckpt.get('target_mean',float('nan'))) - mean) > 1e-6
           or abs(float(ckpt.get('target_std',float('nan'))) - std) > 1e-6
           or ckpt.get('epochs') != cfg.EPOCHS or ckpt.get('max_lr') != cfg.MAX_LR
           or ckpt.get('batch_size') != cfg.BATCH_SIZE):
            raise RuntimeError('Checkpoint incompatible with new labels/split/configuration. '
                               'Use a new --output-dir, or --overwrite.')
        model_raw.load_state_dict(ckpt['model_state_dict'])
        ema.load_state_dict(ckpt['ema_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if ckpt.get('scaler_state_dict'):
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = int(ckpt['epoch'])+1
        global_step = int(ckpt['global_step'])
        early_stop = int(ckpt['early_stop'])
        best_score = float(ckpt['best_score'])
        best_ci = float(ckpt['best_ci'])
        best_mse = float(ckpt['best_mse'])
        if ckpt.get('torch_rng_state') is not None:
            torch.set_rng_state(ckpt['torch_rng_state'].cpu())
        if ckpt.get('cuda_rng_state') is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ckpt['cuda_rng_state'])
        if ckpt.get('numpy_rng_state') is not None:
            np.random.set_state(ckpt['numpy_rng_state'])
        if ckpt.get('python_rng_state') is not None:
            random.setstate(ckpt['python_rng_state'])
        print(f'[RESUME] Seed {seed}, epoch {start_epoch+1}, patience {early_stop}/{cfg.PATIENCE}')

    if start_epoch < cfg.EPOCHS and early_stop < cfg.PATIENCE:
        for epoch in range(start_epoch, cfg.EPOCHS):
            model.train()
            perm = torch.randperm(len(tr_gpu), device=cfg.DEVICE)
            tr_shuffled = tr_gpu[perm]
            running_loss = 0.
            nsteps = 0
            for i in tqdm(range(0, len(tr_gpu), cfg.BATCH_SIZE), leave=False, desc=f'Seed {seed} Ep {epoch+1}'):
                b_idx = tr_shuffled[i:i+cfg.BATCH_SIZE]
                bd, bp = G_idx_d[b_idx], G_idx_p[b_idx]
                y = G_aff[b_idx]
                real_d = max(1, int(G_dM[bd].sum(1).max().item()))
                real_p = max(1, int(G_pM[bp].sum(1).max().item()))
                cur_hyper = make_hyperedge_batch(bd, G_d_hyper, real_d)
                optimizer.zero_grad(set_to_none=True)
                with autocast('cuda', dtype=cfg.AMP_TYPE):
                    pred, vec_d, vec_p, div_l = model(
                        G_d_plm[bd], G_d_mask[bd], G_p_plm[bp], G_p_mask[bp],
                        G_dX[bd][:,:real_d], G_dA_norm[bd][:,:real_d,:real_d],
                        G_dA_bool[bd][:,:real_d,:real_d], G_dE[bd][:,:real_d,:real_d],
                        G_dM[bd][:,:real_d],
                        G_pX[bp][:,:real_p], G_pA_norm[bp][:,:real_p,:real_p],
                        G_pA_bool[bp][:,:real_p,:real_p], G_pM[bp][:,:real_p],cur_hyper)
                    current_std = y.std(unbiased=False).clamp(min=1e-4)
                    mse_l = F.smooth_l1_loss(pred, y, beta=(0.1*current_std).item())
                    rank_l = pairwise_ranking_loss(pred, y)
                    cl_l = supcon_loss(vec_d, y)+supcon_loss(vec_p, y) if cfg.ALPHA_CL>0 else 0.
                    loss = cfg.ALPHA_MSE*mse_l + cfg.ALPHA_RANK*rank_l + cfg.ALPHA_CL*cl_l + .01*div_l
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Non-finite loss at seed={seed} epoch={epoch+1}')
                scaler.scale(loss).backward()
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)  # CRITICAL: unscale before clipping in FP16
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scale_before = scaler.get_scale() if scaler.is_enabled() else None
                scaler.step(optimizer)
                scaler.update()
                skipped_step = scaler.is_enabled() and scaler.get_scale() < scale_before
                # OneCycleLR and EMA advance only when optimizer step actually happens.
                if not skipped_step:
                    global_step += 1
                    if global_step > 100:
                        ema.update()
                    scheduler.step()
                running_loss += float(loss.detach())
                nsteps += 1

            ema.apply_shadow()
            try:
                val_true, val_pred = inference(val_gpu)
                val_met = compute_metrics(val_true, val_pred, mean, std)
                cur_score = val_met['CI'] - 0.1*val_met['MSE']
                if not np.isfinite(cur_score):
                    raise ValueError('Undefined validation model selection score')
                if cur_score > best_score:
                    best_score = cur_score
                    best_ci, best_mse = val_met['CI'], val_met['MSE']
                    early_stop = 0
                    torch.save(get_clean_state_dict(model), best_model_path)
                    with open(os.path.join(cfg.CACHE_DIR,f'best_model_seed_{seed}_meta.json'),'w') as f:
                        json.dump({'seed':seed, 'best_epoch':epoch+1,
                                   'split_sha256':digest,'target_mean':mean,'target_std':std,
                                   'validation_metrics':val_met},f,indent=2)
                    print(f'[*] New best EMA, seed={seed}: CI={best_ci:.4f} MSE={best_mse:.4f}')
                else:
                    early_stop += 1
            finally:
                ema.restore()
            print(f'Seed {seed} Ep {epoch+1} | train_loss={running_loss/max(nsteps,1):.4f}'
                  f' | val MSE={val_met["MSE"]:.4f} CI={val_met["CI"]:.4f}'
                  f' | score={cur_score:.4f} best={best_score:.4f} patience={early_stop}/{cfg.PATIENCE}')
            ckpt = {'format_version':2, 'epoch':epoch, 'global_step':global_step,
                    'best_score':best_score, 'best_ci':best_ci, 'best_mse':best_mse,
                    'early_stop':early_stop, 'split_sha256':digest,
                    'target_mean':mean,'target_std':std,
                    'epochs':cfg.EPOCHS,'max_lr':cfg.MAX_LR,'batch_size':cfg.BATCH_SIZE,
                    'model_state_dict':get_clean_state_dict(model),
                    'ema_state_dict':{k:v.cpu() for k,v in ema.state_dict().items()},
                    'optimizer_state_dict':optimizer.state_dict(),
                    'scheduler_state_dict':scheduler.state_dict(),
                    'scaler_state_dict':scaler.state_dict(),
                    'torch_rng_state':torch.get_rng_state(),
                    'cuda_rng_state':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    'numpy_rng_state':np.random.get_state(),
                    'python_rng_state':random.getstate()}
            torch.save(ckpt, last_ckpt_path)
            if early_stop >= cfg.PATIENCE:
                print(f'[INFO] Early stopping at epoch {epoch+1}')
                break
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not os.path.isfile(best_model_path):
        raise RuntimeError('Best model checkpoint missing. Cannot evaluate test')
    model_raw.load_state_dict(torch.load(best_model_path,map_location=cfg.DEVICE,weights_only=False))
    test_true, test_pred = inference(te_gpu)
    result = compute_metrics(test_true, test_pred, mean, std)
    raw_true, raw_pred = native_predictions(test_true,test_pred,mean,std)
    pd.DataFrame({'interaction_index':te_idx, 'drug_index':d_idx[te_idx].numpy(),
                  'protein_index':p_idx[te_idx].numpy(), 'y_true_native':raw_true,
                  'y_pred_native':raw_pred}).to_csv(
        os.path.join(cfg.CACHE_DIR,f'test_predictions_seed_{seed}.csv'),index=False)
    with open(metric_file,'w') as f:
        json.dump({'seed':seed,'dataset':cfg.DATASET,'split_type':'interaction_random_80_10_10',
                   'split_sha256':digest,'target_mean':mean,'target_std':std,
                   'metrics':result},f,indent=2)
    print(f'>> TEST seed={seed} MSE={result["MSE"]:.4f} CI={result["CI"]:.4f} '
          f'rm2={result["rm2"]:.4f}')
    return result


def self_test():

    for seed in (666,1008,1004,1,42):
        n=347
        legacy=np.arange(n)
        np.random.seed(seed)
        np.random.shuffle(legacy)
        tr,val,te=random_interaction_split(n,seed)
        k=int(n*.1)
        assert np.array_equal(te,legacy[:k])
        assert np.array_equal(val,legacy[k:2*k])
        assert np.array_equal(tr,legacy[2*k:])
        x=torch.tensor([1.,2.,3.,5.,9.,11.])
        norm,m,s=train_only_standardize(x,np.array([0,1,2,3]))
        y,z=native_predictions(norm,norm+.15,m,s)
        assert np.allclose(y,x.numpy(),atol=1e-6)
        assert np.isclose(np.mean((y-z)**2),(.15*s)**2,atol=1e-6)
    for _ in range(16):
        r=np.random.default_rng(_)
        yt=r.integers(0,5,size=35).astype(float)
        yp=r.integers(0,7,size=35).astype(float)
        brute_den=brute_num=0
        for i in range(len(yt)):
            for j in range(i):
                if yt[i] == yt[j]:
                    continue
                brute_den+=1
                d=(yt[i]-yt[j])*(yp[i]-yp[j])
                brute_num+=1 if d>0 else (.5 if d==0 else 0)
        assert np.isclose(fast_concordance_index_vectorized(yt,yp),brute_num/brute_den)
    q=torch.ones(1,1,2,2)
    kv=torch.ones_like(q)
    
    val=torch.tensor([[[[1.],[100.]]]])
    out=F.scaled_dot_product_attention(q,kv,val,attn_mask=torch.tensor([[[[True,False],[True,False]]]]))
    assert torch.allclose(out,torch.ones_like(out))
    print('[SELF-TEST PASS] original split order, train-only labels, inverse scale, exact CI, attention mask')


def main():
    parser=argparse.ArgumentParser(description='Corrected ECHO-DTA main experiment: random interaction 8:1:1')
    parser.add_argument('--dataset',choices=['kiba','davis'],default='kiba')
    parser.add_argument('--data-path',default=None)
    parser.add_argument('--plm-path',default=None)
    parser.add_argument('--output-dir',default=None)
    parser.add_argument('--seeds',nargs='+',type=int,default=[666,1008,1004])
    parser.add_argument('--label-unit',choices=['auto','raw','pkd','kd_nm'],default='auto')
    parser.add_argument('--epochs',type=int,default=120)
    parser.add_argument('--patience',type=int,default=12)
    parser.add_argument('--max-lr',type=float,default=8e-5)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--device',default=None)
    parser.add_argument('--no-compile',action='store_true')
    parser.add_argument('--split-only',action='store_true',help='No PLM/GPU/PyG needed; print and save split + label statistics')
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--overwrite',action='store_true',help='Restart selected seeds; delete their old checkpoints')
    args=parser.parse_args()
    if args.self_test:
        self_test()
        return
    cfg.DATASET=args.dataset
    cfg.LABEL_UNIT=args.label_unit
    cfg.DATA_PATH=args.data_path or os.path.join('Data',cfg.DATASET.upper() if cfg.DATASET=='kiba' else 'Davis')
    cfg.PLM_FEAT_PATH=args.plm_path or os.path.join(cfg.DATA_PATH,'plm_features_SOTA','plm_embeddings_sota_final.pt')
    cfg.CACHE_DIR=args.output_dir or os.path.join(cfg.DATA_PATH,'cache_echo_main_correct_targets_v3')
    os.makedirs(cfg.CACHE_DIR,exist_ok=True)
    cfg.EPOCHS, cfg.PATIENCE, cfg.MAX_LR, cfg.BATCH_SIZE=args.epochs,args.patience,args.max_lr,args.batch_size
    cfg.OVERWRITE=args.overwrite
    cfg.ENABLE_COMPILE=not args.no_compile
    if min(cfg.EPOCHS,cfg.PATIENCE,cfg.BATCH_SIZE) <= 0 or cfg.MAX_LR <= 0:
        raise ValueError('Epochs/patience/batch-size/max-lr must be positive')
    if args.split_only:
        drugs,prots,di,pi,aff=read_affinity_pairs(load_plm=False)
        for seed in args.seeds:
            tr,val,te=random_interaction_split(len(aff),seed)
            digest=write_seed_split(seed,tr,val,te,di,pi,aff)
            _,m,s=train_only_standardize(aff,tr)
            with open(os.path.join(cfg.CACHE_DIR, f'target_stats_seed_{seed}.json'),'w') as f:
                json.dump({'seed':seed, 'dataset':cfg.DATASET, 'split_sha256':digest,
                           'target_mean':m,'target_std_population':s,
                           'train_n':len(tr),'val_n':len(val),'test_n':len(te)}, f,indent=2)
            print(f'[SPLIT seed={seed}] N={len(tr)}/{len(val)}/{len(te)} mean={m:.6f} std={s:.6f} SHA={digest[:16]}')
        return
    if HypergraphConv is None:
        raise ImportError('torch_geometric is required for full ECHO-DTA training. Use --split-only to audit.')
    cfg.DEVICE=torch.device(args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    if cfg.DEVICE.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('Full main experiment requires CUDA GPU; use --split-only or --self-test on CPU.')
    if torch.cuda.is_bf16_supported():
        cfg.AMP_TYPE=torch.bfloat16
        cfg.USE_SCALER=False
    else:
        cfg.AMP_TYPE=torch.float16
        cfg.USE_SCALER=True
    print(f'[CONFIG] dataset={cfg.DATASET}, split=random 80/10/10, AMP={cfg.AMP_TYPE}, '
          f'max_lr={cfg.MAX_LR}, epochs={cfg.EPOCHS}, patience={cfg.PATIENCE}, compile={cfg.ENABLE_COMPILE}')
    data=load_data_and_plm()
    prepare_graph_cache(data)
    results=[]
    for run_idx,seed in enumerate(args.seeds):
        res=run_cycle(seed,run_idx,data)
        results.append({'seed':seed,**res})
        # Aggregate ALL completed seeds in this output directory, even when
        # --seeds contains only the newly requested subset.
        completed=[]
        for metric_path in Path(cfg.CACHE_DIR).glob('test_metrics_seed_*.json'):
            with open(metric_path) as f:
                record=json.load(f)
            if record.get('dataset') == cfg.DATASET and record.get('split_type') == 'interaction_random_80_10_10':
                completed.append({'seed':int(record['seed']),**record['metrics']})
        df=pd.DataFrame(completed).sort_values('seed')
        df.to_csv(os.path.join(cfg.CACHE_DIR,'metrics_per_seed.csv'),index=False)
        numerical=df.drop(columns='seed')
        summary=pd.DataFrame([dict(metric='mean',**numerical.mean(numeric_only=True).to_dict()),
                              dict(metric='std',**numerical.std(numeric_only=True,ddof=1).to_dict())])
        summary.to_csv(os.path.join(cfg.CACHE_DIR,'metrics_mean_std.csv'),index=False)
        print(f'\n[PROGRESS] {run_idx+1}/{len(args.seeds)} requested seeds completed '
              f'({len(df)} total in output directory)\n{df.to_string(index=False)}')
    print('[DONE] Output directory:',cfg.CACHE_DIR)

if __name__ == '__main__':
    main()
