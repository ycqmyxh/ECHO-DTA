# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import config as cfg
try:
    from torch_geometric.nn import HypergraphConv
except ImportError:
    HypergraphConv = None  
import torch._dynamo


class EMA:
    def __init__(self, model, decay=0.995):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        # 初始化影子权重
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(cfg.DEVICE)


    def state_dict(self):
            return self.shadow

        # 加载影子权重
    def load_state_dict(self, state_dict):
            for name, param in state_dict.items():
                if name in self.shadow:
                    # 获取模型第一个参数所在的设备
                    device = next(self.model.parameters()).device
                    self.shadow[name].copy_(param.to(device))
    @torch.no_grad()
    def update(self):
        """每个 step 结束后更新影子权重"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name].copy_(new_average)

    def apply_shadow(self):
        """验证/测试前：将平滑后的权重覆盖到模型上，并备份当前权重"""
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        """验证/测试后：恢复备份的原始训练权重"""
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


class RobustProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        bottleneck = 256
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, bottleneck),
            nn.GELU(),
            nn.LayerNorm(bottleneck),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(bottleneck, out_dim),
            nn.LayerNorm(out_dim)
        )
    def forward(self, x):
        return self.proj(x)


class GINLayer(nn.Module):
    def __init__(self, ind, outd):
        super().__init__()
        self.ind = ind
        self.outd = outd
        self.mlp = nn.Sequential(
            nn.Linear(ind if ind == outd else ind, outd),
            nn.LayerNorm(outd),
            nn.GELU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(outd, outd),
            nn.LayerNorm(outd),
            nn.GELU()
        )
        self.eps = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, adj_norm, node_mask):
        if x.dim() == 2: x = x.unsqueeze(0)
        mask = node_mask.unsqueeze(-1)
        x = x * mask
        neigh = torch.matmul(adj_norm, x)
        out = neigh + self.eps * x
        B, N, F = out.shape
        out = self.mlp(out.view(-1, F)).view(B, N, -1)
        if self.ind == self.outd: out = out + x
        return out * mask


class GATLayer(nn.Module):
    def __init__(self, ind, outd, dropout=cfg.DROPOUT, alpha=0.2):
        super().__init__()
        self.fc = nn.Linear(ind, outd, bias=False)
        self.attn_l = nn.Linear(outd, 1, bias=False)
        self.attn_r = nn.Linear(outd, 1, bias=False)
        self.leaky = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(outd)
        self.use_residual = (ind == outd)

    def forward(self, x, adj_bool, node_mask):
        if x.dim() == 2: x = x.unsqueeze(0)
        mask = node_mask.unsqueeze(-1)
        x = x * mask
        h = self.fc(x)
        e_l = self.attn_l(h)
        e_r = self.attn_r(h).permute(0, 2, 1)
        e = e_l + e_r
        e = self.leaky(e)

        node_pair_mask = (node_mask.unsqueeze(1) * node_mask.unsqueeze(2))
        allowed = (adj_bool > 0) & (node_pair_mask.bool())
        e_masked = e.masked_fill(~allowed, cfg._SOFTMAX_NEG_INF)

        attention = F.softmax(e_masked, dim=-1)
        attention = torch.nan_to_num(attention, nan=0.0)
        out = torch.matmul(self.dropout(attention), h)
        if self.use_residual: out = out + x
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


class RelationalGINLayer(nn.Module):
    def __init__(self, ind, outd, num_relations=4):
        super().__init__()
        # 为每种键类型学习独立的投影 (RGCN 思想)
        self.rel_weight = nn.Parameter(torch.Tensor(num_relations, ind, ind))
        nn.init.xavier_uniform_(self.rel_weight)

        self.mlp = nn.Sequential(
            nn.Linear(ind, outd),
            nn.LayerNorm(outd),
            nn.GELU()
        )
        self.eps = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, adj_bool, edge_feat, node_mask):
        # x: [B, N, D], edge_feat: [B, N, N, 4]
        B, N, D = x.shape

        # 1. 向量化投影所有关系的节点特征: [B, N, D] @ [4, D, D] -> [B, 4, N, D]
        x_rel = torch.einsum('bnd,rdk->brnk', x, self.rel_weight)

        # 2. 准备关系邻接矩阵: [B, N, N, 4] -> [B, 4, N, N]
        adj_rel = edge_feat.permute(0, 3, 1, 2) * adj_bool.unsqueeze(1).float()

        # 3. 批量矩阵乘法聚合邻居，并在关系维度求和 -> [B, N, D]
        neighbor_agg = torch.matmul(adj_rel, x_rel).sum(dim=1)

        out = self.mlp((1 + self.eps) * x + neighbor_agg)
        return out * node_mask.unsqueeze(-1)


class InteractionDynamicHyperEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.hid_dim = hid_dim
        self.out_dim = out_dim  #存储维度供 forward 使用

        self.rel_gin = RelationalGINLayer(in_dim, hid_dim)

        # 1. 强化版 Interaction Gating 
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
        # A. 基础骨架 (1-hop)
        x_base = self.rel_gin(x, adj_bool, edge_feat, node_mask)

        # B. Interaction Gating 调制 
        p_ctx = p_vec.unsqueeze(1).expand(-1, N, -1)
        gate = self.interaction_gate(torch.cat([x_base, p_ctx], dim=-1))
        x_aware = x_base * gate

        x_flat = x_aware.reshape(B * N, self.hid_dim)

        # C. 静态超图 
        h_static = torch.zeros((B * N, self.out_dim), device=x.device)
        if static_hyper_idx is not None and static_hyper_idx.numel() > 0:
            h_static = self.static_hyper_conv(x_flat, static_hyper_idx)

        # D. 真正动态多超边构建 
        scores = (x_aware * p_ctx).sum(-1)
        scores = scores / math.sqrt(x_aware.size(-1))
        scores = scores.masked_fill(node_mask == 0, -1e4)

        # 每个分子构建 4 个动态功能簇，每个簇 8 个原子
        num_edges_per_mol = 4

        node_counts = node_mask.sum(dim=1)  # [B]
        safe_k = torch.clamp(node_counts // 2, min=num_edges_per_mol * 2)
        safe_k = torch.maximum(safe_k, torch.tensor(num_edges_per_mol, device=x.device))
        safe_k = safe_k.int()
        k = torch.minimum(safe_k, torch.tensor(N, device=x.device))
        k = int(torch.median(k).item()) # batch 内统一
        _, topk_indices = torch.topk(scores, k, dim=1)

        # 矢量化构建索引
        k_per_edge = k // num_edges_per_mol
        topk_indices = topk_indices[:, :num_edges_per_mol * k_per_edge]
        topk_indices = topk_indices.view(B, num_edges_per_mol, k_per_edge)

        batch_offset = torch.arange(B, device=x.device).view(B, 1, 1) * N
        dynamic_node_idx = (topk_indices + batch_offset).flatten()
        edge_base = torch.arange(num_edges_per_mol, device=x.device).view(1, num_edges_per_mol, 1)
        edge_offset = (torch.arange(B, device=x.device) * num_edges_per_mol).view(B, 1, 1)
        # 将 [B, 4, 1] 显式扩展为 [B, 4, k_per_edge]
        dynamic_edge_idx = (edge_base + edge_offset).expand(-1, -1, k_per_edge).flatten()

        dynamic_hyper_idx = torch.stack([dynamic_node_idx, dynamic_edge_idx], dim=0)

 
        # 1. 验证被 topk 强行选出的节点，是否真的是有效原子 (node_mask == 1)
        valid_connection_mask = node_mask.reshape(-1)[dynamic_node_idx].bool()

        # 2. 过滤张量：仅保留真实原子的超图连接，彻底切断虚空 Padding 的拓扑污染
        dynamic_hyper_idx = dynamic_hyper_idx[:, valid_connection_mask]

        # 3. 安全输入卷积层
        if dynamic_hyper_idx.size(1) > 0:
            h_dynamic = self.dynamic_hyper_conv(x_flat, dynamic_hyper_idx)
        else:
            h_dynamic = torch.zeros_like(x_flat)
        # ---------------------------------------------

        # E. 融合
        gate_logits = self.adaptive_fusion_net(x_flat)  # [B*N, 3]
        w = torch.softmax(gate_logits, dim=-1)  # [B*N, 3]
        if not self.training and getattr(self, '_print_flag', True):
            w_mean_tensor = w.mean(dim=0).detach()
            print("\n[Adaptive Gate Avg] 基础RGNN / 静态超图 / 动态超图:", w_mean_tensor)
            self._print_flag = False  # 打印一次后关闭
        elif self.training:
            self._print_flag = True  # 训练时重置 flag，保证下一个验证 epoch 能打印
        # 2. 动态加权融合 (利用切片保持维度对齐)
        out = w[:, 0:1] * self.res_proj(x_flat) + \
              w[:, 1:2] * F.gelu(h_static) + \
              w[:, 2:3] * F.gelu(h_dynamic)
        # 原来只返回 out，现在多加两个
        return out.view(B, N, -1) * node_mask.unsqueeze(-1), h_static, h_dynamic


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
        if x.dim() == 2: return x
        scores = self.attn(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)

        #加入 0.7 的温度系数，让权重更聚焦
        weights = F.softmax(scores / 0.7, dim=1).unsqueeze(-1)
        return torch.sum(x * weights, dim=1)


class MultiScaleDilatedCNN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 使用不同空洞率捕获多尺度序列 Motif
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
        # x: [B, L, D]
        res = x
        x = x.transpose(1, 2)  # 转为 [B, D, L] 适配 Conv1d

        # 多尺度融合
        outs = [conv(x) for conv in self.convs]
        out = torch.stack(outs, dim=0).mean(dim=0).transpose(1, 2)  # [B, L, D]

        # 残差连接：T_motif = T_plm + CNN(T_plm)
        return self.norm(self.proj(out) + res)


class TokenCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        #核心稳定性组件
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, q_seq, kv_seq, kv_mask):
        res = q_seq
        B, Lq, D = q_seq.shape
        Lk = kv_seq.shape[1]

        q = self.q_proj(q_seq).view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_seq).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_seq).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)

        # PyTorch SDPA bool True = permitted key token, not padding.
        attn_mask = kv_mask[:, None, None, :].bool().expand(B, self.num_heads, Lq, Lk)

        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn_out = attn_out.transpose(1, 2).reshape(B, Lq, D)

        # Transformer 标准的双重残差结构
        gate = torch.sigmoid(self.out_proj(attn_out))
        x = self.norm1(res + gate * attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class DCI_DTA_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_proj = RobustProjector(cfg.PLM_DRUG_DIM, cfg.GNN_OUT)
        self.p_proj = RobustProjector(cfg.PLM_PROT_DIM, cfg.GNN_OUT)
        self.d_cnn = MultiScaleDilatedCNN(cfg.GNN_OUT)
        self.p_cnn = MultiScaleDilatedCNN(cfg.GNN_OUT)

   
        self.d_gnn = InteractionDynamicHyperEncoder(cfg.DRUG_FEAT_DIM_RAW, cfg.GNN_HID, cfg.GNN_OUT)
        self.p_gnn = GIN_GAT_Encoder(cfg.PROT_FEAT_DIM_RAW, cfg.GNN_HID, cfg.GNN_OUT)

        # 口袋筛选器与引导层
        # self.d_guide = nn.Linear(GNN_OUT, GNN_OUT)
        # self.soft_pocket_scorer = nn.Sequential(
        #     nn.Linear(GNN_OUT, 256),
        #     nn.GELU(),
        #     nn.Linear(256, 1)
        # )

        # self.d_inject = BroadcastFusion(GNN_OUT)
        # self.p_inject = BroadcastFusion(GNN_OUT)
        self.d_cross = TokenCrossAttention(cfg.GNN_OUT)
        self.p_cross = TokenCrossAttention(cfg.GNN_OUT)
        self.pool = AttentionPooling(cfg.GNN_OUT)


        self.classifier = nn.Sequential(
            nn.Linear(cfg.GNN_OUT * 6, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(cfg.DROPOUT),

            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(cfg.DROPOUT),

            nn.Linear(512, 256),
            nn.GELU(),

            nn.Linear(256, 1)
        ) # 保持你之前的 4 层分类器

    def forward(self, d_plm, d_mask, p_plm, p_mask,
                dx, dadj_norm, dadj_bool, d_edge, dnode_mask,
                px, padj_norm, padj_bool, pnode_mask, d_hyper_index):
        # 1. 蛋白引导药物提取
        vec_p_init = self.pool(self.p_gnn(px, padj_norm, padj_bool, pnode_mask), pnode_mask)
        d_gnn_out, h_stat, h_dyn = self.d_gnn(dx, dadj_norm, dadj_bool, d_edge, dnode_mask, d_hyper_index, vec_p_init)
        vec_d_graph = self.pool(d_gnn_out, dnode_mask)

        # 2. 结构注入与交互偏置 
        ed_fused = self.d_cnn(self.d_proj(d_plm))
        ep_fused = self.p_cnn(self.p_proj(p_plm))

        # 3. 引导式软掩码口袋 (温度 T=0.5)
        # d_ctx = self.d_guide(vec_d_graph).unsqueeze(1)
        # p_logits = self.soft_pocket_scorer(ep_fused + d_ctx)
        # p_logits = p_logits.masked_fill(p_mask.unsqueeze(-1) == 0, -1e4)
        # p_weights = torch.sigmoid(p_logits / 0.5)
        # ep_pocket = ep_fused * p_weights
        ep_pocket = ep_fused


        # [计算差异 Loss]
        # 我们希望相似度越小越好，所以这个值会加到总 loss 里
        # 使用 F.cosine_similarity 在特征维度计算每个原子的相似度
        div_loss = F.cosine_similarity(h_stat, h_dyn, dim=-1).mean()

        # 4. 交互与聚合
        inter_d = self.d_cross(ed_fused, ep_pocket, p_mask)
        inter_p = self.p_cross(ep_pocket, ed_fused, d_mask)

        fd = self.pool(inter_d, d_mask)
        fp = self.pool(inter_p, p_mask)

        combined = torch.cat([
            fd,  # 药物序列微观对齐特征
            fp,  # 蛋白序列微观对齐特征
            vec_d_graph,  # ⭐ 药物 2D/3D 宏观超图特征
            vec_p_init,  # ⭐ 蛋白 2D 宏观图特征
            fd * fp,
            torch.abs(fd - fp)
        ], dim=-1)

        return self.classifier(combined).squeeze(-1), fd, fp, div_loss
