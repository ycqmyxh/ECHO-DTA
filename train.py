# train.py — 训练循环、损失函数、评估指标与主入口
import os
import gc
import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim.lr_scheduler import OneCycleLR
from torch.amp import autocast, GradScaler
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr

from config import (
    DEVICE, AMP_TYPE, USE_SCALER,
    BATCH_SIZE, EPOCHS, PATIENCE,
    MAX_LR, WEIGHT_DECAY, ALPHA_CL,
    CACHE_DIR,
)
from data_utils import seed_everything, load_data_and_plm, prepare_graph_cache
from model import DCI_DTA_Model, EMA


# ---------------------------
# 损失函数
# ---------------------------
def pairwise_ranking_loss(y_pred, y_true, num_samples=2048):
    B = y_true.shape[0]
    idx1 = torch.randint(0, B, (num_samples,), device=y_true.device)
    idx2 = torch.randint(0, B, (num_samples,), device=y_true.device)

    diff = y_true[idx1] - y_true[idx2]
    pred_diff = y_pred[idx1] - y_pred[idx2]
    abs_diff = diff.abs()
    mask = ((abs_diff > 0.05) & (abs_diff < 0.5)).float()

    valid_count = mask.sum()
    if valid_count == 0:
        return torch.tensor(0.0, device=y_true.device)

    loss = F.softplus(-torch.sign(diff) * pred_diff)
    return (loss * mask).sum() / (valid_count + 1e-6)


def supcon_loss(features, labels, temperature=0.07, num_bins=10):
    device = features.device
    features = F.normalize(features, dim=1)

    try:
        quantiles = torch.linspace(0.0, 1.0, steps=num_bins + 1, device=device)
        bins = torch.quantile(labels, quantiles)
        labels_q = torch.bucketize(labels, torch.quantile(labels, torch.linspace(0, 1, 6)))
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

    if valid.sum() == 0:
        return torch.tensor(0.0, device=device)
    loss = -(valid * mean_log_prob_pos).sum() / (valid.sum() + 1e-6)
    return loss


def fast_concordance_index_vectorized(y_true, y_pred, max_pairs=20000):
    try:
        t = np.asarray(y_true)
        p = np.asarray(y_pred)
        n = len(t)
        if n > max_pairs:
            idx = np.random.choice(n, max_pairs, replace=False)
            t = t[idx]
            p = p[idx]
        perm = np.argsort(t)
        t = t[perm]
        p = p[perm]
        dt = t[:, None] - t[None, :]
        dp = p[:, None] - p[None, :]
        mask = dt > 0
        concordant = (dp > 0) & mask
        ties = (dp == 0) & mask
        n_pairs = mask.sum()
        if n_pairs == 0:
            return 0.0
        return (concordant.sum() + 0.5 * ties.sum()) / n_pairs
    except:
        return 0.0


def calculate_rm2(y_true, y_pred):
    try:
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        r2 = r2_score(y_true, y_pred)

        numerator = np.sum(y_true * y_pred)
        denominator = np.sum(y_pred ** 2)
        if denominator == 0:
            return 0.0

        k = numerator / denominator
        y_pred_0 = k * y_pred
        sse_0 = np.sum((y_true - y_pred_0) ** 2)
        sst = np.sum((y_true - np.mean(y_true)) ** 2)

        if sst == 0:
            return 0.0
        r02 = 1 - (sse_0 / sst)
        rm2 = r2 * (1 - np.sqrt(np.abs(r2 - r02)))
        return rm2
    except Exception as e:
        return 0.0


def compute_metrics(y_true, y_pred, mean, std):
    y_t_norm = y_true.detach().float().cpu().numpy().flatten()
    y_p_norm = y_pred.detach().float().cpu().numpy().flatten()
    y_t = (y_t_norm * std) + mean
    y_p = (y_p_norm * std) + mean

    mse = ((y_t - y_p) ** 2).mean()
    r2 = r2_score(y_t, y_p)
    rm2 = calculate_rm2(y_t, y_p)
    ci = fast_concordance_index_vectorized(y_t, y_p)

    try:
        pearson = pearsonr(y_t, y_p)[0]
        spearman = spearmanr(y_t, y_p)[0]
    except:
        pearson = 0.0
        spearman = 0.0

    return {
        "MSE": mse,
        "CI": ci,
        "R2": r2,
        "rm2": rm2,
        "Pearson": pearson,
        "Spearman": spearman
    }


# ---------------------------
# 训练循环
# ---------------------------
def run_cycle(seed, run_idx, data):
    def get_clean_state_dict(model):
        state_dict = model.state_dict()
        return {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    def run_inference(eval_model, target_indices, label="Evaluation"):
        eval_model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for i in range(0, len(target_indices), BATCH_SIZE * 2):
                b_idx = target_indices[i: i + BATCH_SIZE * 2]
                bd, bp = G_idx_d[b_idx], G_idx_p[b_idx]

                real_d = max(1, int(G_dM[bd].sum(1).max().item()))
                real_p = max(1, int(G_pM[bp].sum(1).max().item()))

                batch_h_raw = [G_d_hyper[idx.item()] for idx in bd]
                valid_mask = [h is not None and h.numel() > 0 for h in batch_h_raw]
                valid_h = [h for h, m in zip(batch_h_raw, valid_mask) if m]

                if valid_h:
                    cur_hyper = torch.cat(valid_h, dim=1).to(DEVICE)
                    valid_batch_indices = torch.tensor(
                        [idx for idx, m in enumerate(valid_mask) if m], device=DEVICE
                    )
                    e_counts = torch.tensor([h.size(1) for h in valid_h], device=DEVICE)

                    node_offsets = valid_batch_indices * real_d
                    cur_hyper[0] += torch.repeat_interleave(node_offsets, e_counts)

                    h_edge_counts = torch.tensor(
                        [h[1].max().item() + 1 for h in valid_h], device=DEVICE
                    )
                    edge_offsets = torch.cumsum(h_edge_counts, dim=0) - h_edge_counts
                    cur_hyper[1] += torch.repeat_interleave(edge_offsets, e_counts)
                else:
                    cur_hyper = None

                cur_e = G_dE_tensor[bd][:, :real_d, :real_d]
                with autocast('cuda', dtype=AMP_TYPE):
                    p, _, _, _ = eval_model(
                        G_d_plm[bd], G_d_mask[bd], G_p_plm[bp], G_p_mask[bp],
                        G_dX[bd][:, :real_d], G_dA_norm[bd][:, :real_d, :real_d],
                        G_dA_bool[bd][:, :real_d, :real_d], cur_e, G_dM[bd][:, :real_d],
                        G_pX[bp][:, :real_p], G_pA_norm[bp][:, :real_p, :real_p],
                        G_pA_bool[bp][:, :real_p, :real_p], G_pM[bp][:, :real_p],
                        cur_hyper
                    )
                preds.append(p.detach().clone())
                trues.append(G_aff[b_idx].detach())
        return torch.cat(trues), torch.cat(preds)

    drugs, prots, d_plm, d_mask, p_plm, p_mask, d_idx, p_idx, aff, aff_mean, aff_std = data
    m, s = aff_mean.item(), aff_std.item()

    seed_everything(seed)
    total_len = len(aff)
    indices = np.arange(total_len)
    np.random.shuffle(indices)

    split_1, split_2 = int(0.1 * total_len), int(0.2 * total_len)
    te_idx, val_idx, tr_idx = indices[:split_1], indices[split_1:split_2], indices[split_2:]

    train_aff_values = aff[tr_idx]
    tr_mean = train_aff_values.mean()
    tr_std = train_aff_values.std()
    m, s = tr_mean.item(), tr_std.item()
    G_aff = (aff.to(DEVICE) - tr_mean.to(DEVICE)) / tr_std.to(DEVICE)

    tr_idx_gpu = torch.tensor(tr_idx, device=DEVICE)
    val_idx_gpu = torch.tensor(val_idx, device=DEVICE)
    te_idx_gpu = torch.tensor(te_idx, device=DEVICE)

    model = DCI_DTA_Model().to(DEVICE)
    ema = EMA(model, decay=0.995)

    if hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode="default", dynamic=True, fullgraph=False)
            print("[INFO] torch.compile enabled with dynamic shapes.")
        except Exception as e:
            print(f"[WARN] torch.compile failed: {e}. Falling back to eager mode.")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY, fused=True
    )
    steps_per_epoch = (len(tr_idx) + BATCH_SIZE - 1) // BATCH_SIZE
    scheduler = OneCycleLR(
        optimizer, max_lr=MAX_LR, epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch, pct_start=0.1, div_factor=20
    )
    scaler = GradScaler(enabled=USE_SCALER)

    G_d_plm, G_d_mask = d_plm.to(DEVICE, dtype=AMP_TYPE), d_mask.to(DEVICE).bool()
    G_p_plm, G_p_mask = p_plm.to(DEVICE, dtype=AMP_TYPE), p_mask.to(DEVICE).bool()
    G_idx_d, G_idx_p, G_aff = d_idx.to(DEVICE), p_idx.to(DEVICE), aff.to(DEVICE)

    g_data = torch.load(
        os.path.join(CACHE_DIR, "graphs_processed_maxspeed.pt"), map_location='cpu'
    )
    G_dE_tensor = g_data['drug'][4].to(DEVICE, dtype=torch.float32)
    G_dX, G_dA_norm, G_dA_bool, G_dM = (
        g_data['drug'][0].to(DEVICE, dtype=torch.float32),
        g_data['drug'][3].to(DEVICE, dtype=AMP_TYPE),
        (g_data['drug'][1].to(DEVICE) > 0).to(torch.bool),
        g_data['drug'][2].to(DEVICE).bool()
    )
    G_pX, G_pA_norm, G_pA_bool, G_pM = (
        g_data['prot'][0].to(DEVICE, dtype=torch.float32),
        g_data['prot'][3].to(DEVICE, dtype=AMP_TYPE),
        (g_data['prot'][1].to(DEVICE) > 0).to(torch.bool),
        g_data['prot'][2].to(DEVICE).bool()
    )
    G_d_hyper = g_data['drug'][5]

    best_model_path = os.path.join(CACHE_DIR, f"best_model_seed_{seed}.pth")
    last_ckpt_path = os.path.join(CACHE_DIR, f"last_checkpoint_seed_{seed}.pth")
    best_score = -999.0
    best_mse = 999.0
    best_ci = 0.0
    start_epoch = 0
    early_stop = 0
    global_step = 0

    if os.path.exists(last_ckpt_path):
        print(f"[INFO] Resuming from checkpoint: {last_ckpt_path}")
        ckpt = torch.load(last_ckpt_path, map_location=DEVICE, weights_only=False)

        saved_state_dict = ckpt['model_state_dict']
        current_model_dict = model.state_dict()
        has_prefix = any(k.startswith('_orig_mod.') for k in current_model_dict.keys())

        new_state_dict = {}
        for k, v in saved_state_dict.items():
            if has_prefix and not k.startswith('_orig_mod.'):
                new_state_dict[f'_orig_mod.{k}'] = v
            elif not has_prefix and k.startswith('_orig_mod.'):
                new_state_dict[k.replace('_orig_mod.', '')] = v
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])

        if 'ema_state_dict' in ckpt:
            ema.load_state_dict(ckpt['ema_state_dict'])
            print("[INFO] EMA shadow weights resumed.")

        start_epoch = ckpt['epoch'] + 1
        global_step = ckpt.get('global_step', 0)
        best_score = ckpt.get('best_score', -999.0)
        best_mse = ckpt['best_mse']
        best_ci = ckpt.get('best_ci', 0.0)
        early_stop = ckpt['early_stop']
        print(f"[INFO] Resumed! Starting from Epoch {start_epoch + 1}, Current Best Val MSE: {best_mse:.4f}")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        perm = torch.randperm(len(tr_idx), device=DEVICE)
        tr_idx_perm = tr_idx_gpu[perm]
        loss_accum, steps = 0, 0
        pbar = tqdm(range(0, len(tr_idx), BATCH_SIZE), leave=False, desc=f"Ep {epoch + 1}")

        for i in pbar:
            global_step += 1
            b_idx = tr_idx_perm[i: i + BATCH_SIZE]
            bd, bp = G_idx_d[b_idx], G_idx_p[b_idx]
            y = G_aff[b_idx]

            bd_aug = bd
            real_d = max(1, int(G_dM[bd_aug].sum(1).max().item()))
            real_p = max(1, int(G_pM[bp].sum(1).max().item()))
            cur_dE = G_dE_tensor[bd][:, :real_d, :real_d]

            batch_h_raw = [G_d_hyper[idx.item()] for idx in bd]
            valid_mask = [h is not None and h.numel() > 0 for h in batch_h_raw]
            valid_h = [h for h, m in zip(batch_h_raw, valid_mask) if m]

            if valid_h:
                cur_hyper = torch.cat(valid_h, dim=1).to(DEVICE)
                valid_batch_indices = torch.tensor(
                    [idx for idx, m in enumerate(valid_mask) if m], device=DEVICE
                )
                e_counts = torch.tensor([h.size(1) for h in valid_h], device=DEVICE)

                node_offsets = valid_batch_indices * real_d
                cur_hyper[0] += torch.repeat_interleave(node_offsets, e_counts)

                h_edge_counts = torch.tensor(
                    [h[1].max().item() + 1 for h in valid_h], device=DEVICE
                )
                edge_offsets = torch.cumsum(h_edge_counts, dim=0) - h_edge_counts
                cur_hyper[1] += torch.repeat_interleave(edge_offsets, e_counts)
            else:
                cur_hyper = None

            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda', dtype=AMP_TYPE):
                pred, vec_d, vec_p, div_l = model(
                    G_d_plm[bd], G_d_mask[bd], G_p_plm[bp], G_p_mask[bp],
                    G_dX[bd_aug][:, :real_d], G_dA_norm[bd_aug][:, :real_d, :real_d],
                    G_dA_bool[bd_aug][:, :real_d, :real_d], cur_dE, G_dM[bd_aug][:, :real_d],
                    G_pX[bp][:, :real_p], G_pA_norm[bp][:, :real_p, :real_p],
                    G_pA_bool[bp][:, :real_p, :real_p], G_pM[bp][:, :real_p],
                    cur_hyper
                )

                current_std = y.std().clamp(min=1e-4)
                dynamic_beta = 0.1 * current_std
                mse_l = F.smooth_l1_loss(pred, y, beta=dynamic_beta.item())
                rank_l = pairwise_ranking_loss(pred, y)
                cl_l = (supcon_loss(vec_d, y) + supcon_loss(vec_p, y)) if ALPHA_CL > 0 else 0
                loss = 1.0 * mse_l + 0.8 * rank_l + ALPHA_CL * cl_l + (0.01 * div_l)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            if global_step > 100:
                ema.update()
            scheduler.step()

            loss_accum += loss.item()
            steps += 1

        ema.apply_shadow()
        val_trues_ema, val_preds_ema = run_inference(model, val_idx_gpu)
        metrics_ema = compute_metrics(val_trues_ema, val_preds_ema, m, s)
        cur_ci = metrics_ema["CI"]
        cur_mse = metrics_ema["MSE"]
        cur_score = cur_ci - 0.1 * cur_mse

        if cur_score > best_score:
            best_score, best_ci, best_mse = cur_score, cur_ci, cur_mse
            early_stop = 0
            torch.save(get_clean_state_dict(model), best_model_path)
            print(f"[*] New Best EMA Model! Score: {best_score:.4f} (CI: {cur_ci:.4f})")
        else:
            early_stop += 1
        ema.restore()

        print(
            f"Ep {epoch + 1} | EMA MSE: {cur_mse:.4f} | EMA CI: {cur_ci:.4f} | "
            f"Best CI: {best_ci:.4f} | Stop: {early_stop}/{PATIENCE}"
        )

        del val_trues_ema, val_preds_ema
        torch.cuda.empty_cache()

        ckpt_to_save = {
            'epoch': epoch,
            'best_score': best_score,
            'best_ci': best_ci,
            'best_mse': best_mse,
            'early_stop': early_stop,
            'model_state_dict': get_clean_state_dict(model),
            'ema_state_dict': ema.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        torch.save(ckpt_to_save, last_ckpt_path)

        if early_stop >= PATIENCE:
            print("[INFO] Early Stopping triggered.")
            break

    print("\n" + "-" * 30 + "\n[Final Testing] Comparing Best vs. Last Models\n" + "-" * 30)

    best_state = torch.load(best_model_path, map_location=DEVICE, weights_only=False)
    current_model_dict = model.state_dict()
    has_prefix = any(k.startswith('_orig_mod.') for k in current_model_dict.keys())

    adjusted_state = {}
    for k, v in best_state.items():
        if has_prefix and not k.startswith('_orig_mod.'):
            adjusted_state[f'_orig_mod.{k}'] = v
        elif not has_prefix and k.startswith('_orig_mod.'):
            adjusted_state[k.replace('_orig_mod.', '')] = v
        else:
            adjusted_state[k] = v

    model.load_state_dict(adjusted_state)
    te_trues_b, te_preds_b = run_inference(model, te_idx_gpu)
    res_best = compute_metrics(te_trues_b, te_preds_b, m, s)
    print(f">> [Best Model Result] MSE: {res_best['MSE']:.4f} | CI: {res_best['CI']:.4f} | rm2: {res_best['rm2']:.4f}")

    return res_best


# ---------------------------
# 主函数
# ---------------------------
def main():
    gc.collect()
    torch.cuda.empty_cache()
    data = load_data_and_plm()
    prepare_graph_cache(data)

    seeds = [666, 1008, 1004]
    results = []

    for i, s in enumerate(seeds):
        res = run_cycle(s, i, data)
        results.append(res)
        print(f"\n--- Progress: {i + 1}/{len(seeds)} Seeds Completed ---")
        print(pd.DataFrame(results).tail(1))

    print("\n" + "=" * 30)
    print("DCI-DTA FINAL AVERAGE RESULTS")
    print("=" * 30)
    df_res = pd.DataFrame(results)
    print(df_res)
    print("\nMean Performance:")
    print(df_res.mean())


if __name__ == "__main__":
    main()