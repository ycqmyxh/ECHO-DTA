# model.py — DCI-DTA 完整模型组件定义
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv

from config import (
    DEVICE, DROPOUT,
    PLM_DRUG_DIM, PLM_PROT_DIM,
    DRUG_FEAT_DIM_RAW, PROT_FEAT_DIM_RAW,
    GNN_HID, GNN_OUT,
    _SOFTMAX_NEG_INF,
)


# ---------------------------
# EMA (指数移动平均)
# ---------------------------
class EMA:
    def __init__(self, model, decay=0.995):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(DEVICE)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        for name, param in state_dict.items():
            if name in self.shadow:
                device = next(self.model.parameters()).device
                self.shadow[name].copy_(param.to(device))

    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name].copy_(new_average)

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


# ---------------------------
# 投影器
# ---------------------------
class RobustProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        bottleneck = 256
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, bottleneck),
            nn.GELU(),
            nn.LayerNorm(bottleneck),
            nn.Dropout(DROPOUT),
            nn.Linear(bottleneck, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        return self.proj(x)


# ---------------------------
# GIN / GAT 编码器
# ---------------------------
class GINLayer(nn.Module):
    def __init__(self, ind, outd):
        super().__init__()
        self.ind = ind
        self.outd = outd
        self.mlp = nn.Sequential(
            nn.Linear(ind if ind == outd else ind, outd),
            nn.LayerNorm(outd),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(outd, outd),
            nn.LayerNorm(outd),
            nn.GELU()
        )
        self.eps = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, adj_norm, node_mask):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        mask = node_mask.unsqueeze(-1)
        x = x * mask
        neigh = torch.matmul(adj_norm, x)
        out = neigh + self.eps * x
        B, N, F = out.shape
        out = self.mlp(out.view(-1, F)).view(B, N, -1)
        if self.ind == self.outd:
            out = out + x
        return out * mask


class GATLayer(nn.Module):
    def __init__(self, ind, outd, dropout=DROPOUT, alpha=0.2):
        super().__init__()
        self.fc = nn.Linear(ind, outd, bias=False)
        self.attn_l = nn.Linear(outd, 1, bias=False)
        self.attn_r = nn.Linear(outd, 1, bias=False)
        self.leaky = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(outd)
        self.use_residual = (ind == outd)

    def forward(self, x, adj_bool, node_mask):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        mask = node_mask.unsqueeze(-1)
        x = x * mask
        h = self.fc(x)
        e_l = self.attn_l(h)
        e_r = self.attn_r(h).permute(0, 2, 1)
        e = e_l + e_r
        e = self.leaky(e)

        node_pair_mask = (node_mask.unsqueeze(1) * node_mask.unsqueeze(2))
        allowed = (adj_bool > 0) & (node_pair_mask.bool())
        e_masked = e.masked_fill(~allowed, _SOFTMAX_NEG_INF)

        attention = F.softmax(e_masked, dim=-1)
        attention = torch.nan_to_num(attention, nan=0.0)
        out = torch.matmul(self.dropout(attention), h)
        if self.use_residual:
            out = out + x
        out = self.norm(out * mask)
        return out


class GIN_GAT_Encoder(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.gin1 = GINLayer(in_dim, hid_dim)
        self.gin2 = GINLayer(hid_dim, hid_dim)
        self.gat = GATLayer(hid_dim, out_dim)

    def forward(self, x, adj_norm, adj_bool, node_mask):
        x = self.gin1(x, adj_norm, node_mask)
        x = self.gin2(x, adj_norm, node_mask)
        x = self.gat(x, adj_bool, node_mask)
        return F.elu(x)


# ---------------------------
# 关系型 GIN（化学键感知）
# ---------------------------
class RelationalGINLayer(nn.Module):
    def __init__(self, ind, outd, num_relations=4):
        super().__init__()
        self.rel_weight = nn.Parameter(torch.Tensor(num_relations, ind, ind))
        nn.init.xavier_uniform_(self.rel_weight)
        self.mlp = nn.Sequential(
            nn.Linear(ind, outd),
            nn.LayerNorm(outd),
            nn.GELU()
        )
        self.eps = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, adj_bool, edge_feat, node_mask):
        B, N, D = x.shape
        x_rel = torch.einsum('bnd,rdk->brnk', x, self.rel_weight)
        adj_rel = edge_feat.permute(0, 3, 1, 2) * adj_bool.unsqueeze(1).float()
        neighbor_agg = torch.matmul(adj_rel, x_rel).sum(dim=1)
        out = self.mlp((1 + self.eps) * x + neighbor_agg)
        return out * node_mask.unsqueeze(-1)


# ---------------------------
# 交互式动态超图编码器
# ---------------------------
class InteractionDynamicHyperEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.hid_dim = hid_dim
        self.out_dim = out_dim

        self.rel_gin = RelationalGINLayer(in_dim, hid_dim)
        self.interaction_gate = nn.Sequential(
            nn.Linear(hid_dim + out_dim, hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, hid_dim),
            nn.Sigmoid()
        )
        self.static_hyper_conv = HypergraphConv(hid_dim, out_dim)
        self.dynamic_hyper_conv = HypergraphConv(hid_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        self.res_proj = nn.Linear(hid_dim, out_dim)
        self.adaptive_fusion_net = nn.Sequential(
            nn.Linear(hid_dim, 32),
            nn.GELU(),
            nn.Linear(32, 3)
        )

    def forward(self, x, adj_norm, adj_bool, edge_feat, node_mask, static_hyper_idx, p_vec):
        B, N, _ = x.shape

        # A. Relational GIN
        x_base = self.rel_gin(x, adj_bool, edge_feat, node_mask)

        # B. Interaction Gating
        p_ctx = p_vec.unsqueeze(1).expand(-1, N, -1)
        gate = self.interaction_gate(torch.cat([x_base, p_ctx], dim=-1))
        x_aware = x_base * gate
        x_flat = x_aware.reshape(B * N, self.hid_dim)

        # C. Static hypergraph
        h_static = torch.zeros((B * N, self.out_dim), device=x.device)
        if static_hyper_idx is not None and static_hyper_idx.numel() > 0:
            h_static = self.static_hyper_conv(x_flat, static_hyper_idx)

        # D. Dynamic hypergraph
        scores = (x_aware * p_ctx).sum(-1)
        scores = scores / math.sqrt(x_aware.size(-1))
        scores = scores.masked_fill(node_mask == 0, -1e4)

        num_edges_per_mol = 4
        node_counts = node_mask.sum(dim=1)
        safe_k = torch.clamp(node_counts // 2, min=num_edges_per_mol * 2)
        safe_k = torch.maximum(safe_k, torch.tensor(num_edges_per_mol, device=x.device))
        safe_k = safe_k.int()
        k = torch.minimum(safe_k, torch.tensor(N, device=x.device))
        k = int(torch.median(k).item())

        _, topk_indices = torch.topk(scores, k, dim=1)
        k_per_edge = k // num_edges_per_mol
        topk_indices = topk_indices[:, :num_edges_per_mol * k_per_edge]
        topk_indices = topk_indices.view(B, num_edges_per_mol, k_per_edge)

        batch_offset = torch.arange(B, device=x.device).view(B, 1, 1) * N
        dynamic_node_idx = (topk_indices + batch_offset).flatten()
        edge_base = torch.arange(num_edges_per_mol, device=x.device).view(1, num_edges_per_mol, 1)
        edge_offset = (torch.arange(B, device=x.device) * num_edges_per_mol).view(B, 1, 1)
        dynamic_edge_idx = (edge_base + edge_offset).expand(-1, -1, k_per_edge).flatten()

        dynamic_hyper_idx = torch.stack([dynamic_node_idx, dynamic_edge_idx], dim=0)
        valid_connection_mask = node_mask.reshape(-1)[dynamic_node_idx].bool()
        dynamic_hyper_idx = dynamic_hyper_idx[:, valid_connection_mask]

        if dynamic_hyper_idx.size(1) > 0:
            h_dynamic = self.dynamic_hyper_conv(x_flat, dynamic_hyper_idx)
        else:
            h_dynamic = torch.zeros_like(x_flat)

        # E. Adaptive fusion
        gate_logits = self.adaptive_fusion_net(x_flat)
        w = torch.softmax(gate_logits, dim=-1)
        self.last_gate = w.mean(dim=0).detach()

        if not self.training and getattr(self, '_print_flag', True):
            w_mean_tensor = w.mean(dim=0).detach()
            print("\n[Adaptive Gate Avg] 基础RGNN / 静态超图 / 动态超图:", w_mean_tensor)
            self._print_flag = False
        elif self.training:
            self._print_flag = True

        out = w[:, 0:1] * self.res_proj(x_flat) + \
              w[:, 1:2] * F.gelu(h_static) + \
              w[:, 2:3] * F.gelu(h_dynamic)

        return out.view(B, N, -1) * node_mask.unsqueeze(-1), h_static, h_dynamic, scores


# ---------------------------
# 注意力池化
# ---------------------------
class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(in_dim // 2, 1)
        )

    def forward(self, x, mask=None):
        if x.dim() == 2:
            return x
        scores = self.attn(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
        weights = F.softmax(scores / 0.7, dim=1).unsqueeze(-1)
        return torch.sum(x * weights, dim=1)


# ---------------------------
# 多尺度空洞 CNN
# ---------------------------
class MultiScaleDilatedCNN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(dim, dim, 3, padding=d, dilation=d, groups=dim),
                nn.Conv1d(dim, dim, 1),
                nn.GroupNorm(8, dim),
                nn.GELU()
            ) for d in [1, 2, 4]
        ])
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        res = x
        x = x.transpose(1, 2)
        outs = [conv(x) for conv in self.convs]
        out = torch.stack(outs, dim=0).mean(dim=0).transpose(1, 2)
        return self.norm(self.proj(out) + res)


# ---------------------------
# Token Cross-Attention
# ---------------------------
class TokenCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, q_seq, kv_seq, kv_mask):
        res = q_seq
        B, Lq, D = q_seq.shape
        Lk = kv_seq.shape[1]

        q = self.q_proj(q_seq).view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_seq).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_seq).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)

        attn_mask = (kv_mask[:, None, None, :]).expand(B, self.num_heads, Lq, Lk) == 0
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn_out = attn_out.transpose(1, 2).reshape(B, Lq, D)

        gate = torch.sigmoid(self.out_proj(attn_out))
        x = self.norm1(res + gate * attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


# ---------------------------
# DCI-DTA 完整模型
# ---------------------------
class DCI_DTA_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_proj = RobustProjector(PLM_DRUG_DIM, GNN_OUT)
        self.p_proj = RobustProjector(PLM_PROT_DIM, GNN_OUT)
        self.d_cnn = MultiScaleDilatedCNN(GNN_OUT)
        self.p_cnn = MultiScaleDilatedCNN(GNN_OUT)

        self.d_gnn = InteractionDynamicHyperEncoder(DRUG_FEAT_DIM_RAW, GNN_HID, GNN_OUT)
        self.p_gnn = GIN_GAT_Encoder(PROT_FEAT_DIM_RAW, GNN_HID, GNN_OUT)

        self.d_cross = TokenCrossAttention(GNN_OUT)
        self.p_cross = TokenCrossAttention(GNN_OUT)
        self.pool = AttentionPooling(GNN_OUT)

        self.classifier = nn.Sequential(
            nn.Linear(GNN_OUT * 6, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )

    def forward(self, d_plm, d_mask, p_plm, p_mask,
                dx, dadj_norm, dadj_bool, d_edge, dnode_mask,
                px, padj_norm, padj_bool, pnode_mask, d_hyper_index):
        # 1. Protein-guided drug extraction
        vec_p_init = self.pool(self.p_gnn(px, padj_norm, padj_bool, pnode_mask), pnode_mask)
        d_gnn_out, h_stat, h_dyn, d_scores = self.d_gnn(
            dx, dadj_norm, dadj_bool, d_edge, dnode_mask, d_hyper_index, vec_p_init
        )
        self.last_gate = self.d_gnn.last_gate
        vec_d_graph = self.pool(d_gnn_out, dnode_mask)

        # 2. Structure injection
        ed_fused = self.d_cnn(self.d_proj(d_plm))
        ep_fused = self.p_cnn(self.p_proj(p_plm))
        ep_pocket = ep_fused

        # Diversity loss
        div_loss = F.cosine_similarity(h_stat, h_dyn, dim=-1).mean()

        # 3. Cross-attention
        inter_d = self.d_cross(ed_fused, ep_pocket, p_mask)
        inter_p = self.p_cross(ep_pocket, ed_fused, d_mask)

        # Protein attention capture (inference only)
        if not self.training:
            with torch.no_grad():
                p_scores = self.pool.attn(inter_p).squeeze(-1)
                p_scores = p_scores.masked_fill(p_mask == 0, -1e4)
                p_weights = F.softmax(p_scores / 0.7, dim=1)
                self.last_prot_attention = p_weights.detach().cpu()

        fd = self.pool(inter_d, d_mask)
        fp = self.pool(inter_p, p_mask)

        combined = torch.cat([
            fd, fp,
            vec_d_graph, vec_p_init,
            fd * fp,
            torch.abs(fd - fp)
        ], dim=-1)

        return self.classifier(combined).squeeze(-1), fd, fp, div_loss, d_scores