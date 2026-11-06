import collections

import torch


def build_lookup(edge_index, n_users):
    table = collections.defaultdict(set)
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for u, c in zip(src, dst):
        table[u].add(c)
    return table


def evaluate(score_fn, bundle, split="test", k_values=(5, 10, 15, 20),
             device="cpu", batch_size=4096):
    n_users = bundle["n_users"]
    n_concepts = bundle["n_concepts"]
    kmax = max(k_values)

    seen_cpu = torch.zeros((n_users, n_concepts), dtype=torch.bool)
    tr_ei = bundle["train_ei"]
    if tr_ei.numel() > 0:
        seen_cpu[tr_ei[0].cpu(), tr_ei[1].cpu()] = True

    truth_cpu = torch.zeros((n_users, n_concepts), dtype=torch.bool)
    sp_ei = bundle[f"{split}_ei"]
    if sp_ei.numel() > 0:
        truth_cpu[sp_ei[0].cpu(), sp_ei[1].cpu()] = True

    t_cnt = truth_cpu.sum(dim=1)
    users = torch.nonzero(t_cnt > 0, as_tuple=False).squeeze(1)
    counted = int(users.numel())

    totals = {f"{m}@{k}": 0.0 for k in k_values for m in ("HR", "NDCG", "Precision", "Recall")}
    totals["MRR"] = 0.0
    if counted == 0:
        return totals, 0

    disc = 1.0 / torch.log2(torch.arange(2, kmax + 2, dtype=torch.float32))
    dcg_cum = torch.cumsum(disc, dim=0)
    k_eff = min(kmax, n_concepts)

    for start in range(0, counted, batch_size):
        chunk_u = users[start:start + batch_size]
        uu = chunk_u.to(device)
        scores = score_fn(uu).detach().cpu()
        scores.masked_fill_(seen_cpu[chunk_u], float("-inf"))
        topk = torch.topk(scores, k=k_eff, dim=1).indices
        hits = truth_cpu[chunk_u].gather(1, topk).float()

        any_hit = hits.sum(dim=1) > 0
        first_idx = torch.argmax(hits, dim=1)
        mrr_vec = torch.where(any_hit, 1.0 / (first_idx.float() + 1.0),
                              torch.zeros_like(first_idx, dtype=torch.float32))
        totals["MRR"] += float(mrr_vec.sum().item())

        u_tcnt = t_cnt[chunk_u].float()
        for k in k_values:
            hk = hits[:, :k]
            nhit = hk.sum(dim=1)
            totals[f"HR@{k}"] += float((nhit > 0).float().sum().item())
            totals[f"Precision@{k}"] += float((nhit / k).sum().item())
            totals[f"Recall@{k}"] += float((nhit / u_tcnt).sum().item())
            dcg = (hk * disc[:k]).sum(dim=1)
            idcg_idx = torch.clamp(t_cnt[chunk_u], max=k) - 1
            idcg = dcg_cum[idcg_idx]
            totals[f"NDCG@{k}"] += float((dcg / idcg).sum().item())

    return {m: v / counted for m, v in totals.items()}, counted


def format_metrics(metrics, keys=("NDCG@10", "HR@10", "MRR", "Recall@20", "Precision@10")):
    return "  ".join(f"{k}={metrics[k]:.4f}" for k in keys if k in metrics)
