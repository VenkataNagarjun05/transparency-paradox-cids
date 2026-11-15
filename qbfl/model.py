"""Batched MLP so many tenants train at once on one GPU.

Each tenant has its own weights, stored with a leading "client" dimension and
multiplied with bmm. Adam is element-wise, so training C clients together is
mathematically identical to training them one by one.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def layer_shapes(n_in: int, hidden=(128, 64), n_out: int = 2):
    dims = [n_in, *hidden, n_out]
    return [(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]


def n_params(shapes) -> int:
    return sum(i * o + o for i, o in shapes)


def init_theta(shapes, seed: int, device) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    parts = []
    for i, o in shapes:
        bound = 1 / math.sqrt(i)
        parts.append(((torch.rand(i * o, generator=g) * 2 - 1) * bound))
        parts.append(torch.zeros(o))
    return torch.cat(parts).to(device)


def unflatten(theta: torch.Tensor, shapes):
    """theta: [C, P] -> list of (W [C,i,o], b [C,1,o])."""
    C = theta.shape[0]
    out, k = [], 0
    for i, o in shapes:
        W = theta[:, k:k + i * o].reshape(C, i, o)
        k += i * o
        b = theta[:, k:k + o].reshape(C, 1, o)
        k += o
        out.append((W, b))
    return out


def forward(theta: torch.Tensor, shapes, x: torch.Tensor) -> torch.Tensor:
    """theta [C,P], x [C,B,in] (or [B,in] shared) -> logits [C,B,out]."""
    layers = unflatten(theta, shapes)
    h = x if x.dim() == 3 else x.unsqueeze(0).expand(theta.shape[0], -1, -1)
    for n, (W, b) in enumerate(layers):
        h = torch.baddbmm(b, h, W)
        if n < len(layers) - 1:
            h = F.relu(h)
    return h


@torch.no_grad()
def predict(theta: torch.Tensor, shapes, X: torch.Tensor, bs: int = 8192) -> torch.Tensor:
    """theta [C,P] (or [P]) on shared X [N,in] -> predicted class [C,N]."""
    if theta.dim() == 1:
        theta = theta.unsqueeze(0)
    outs = []
    for i in range(0, X.shape[0], bs):
        outs.append(forward(theta, shapes, X[i:i + bs]).argmax(-1))
    return torch.cat(outs, 1)


@torch.no_grad()
def predict_proba(theta: torch.Tensor, shapes, X: torch.Tensor, bs: int = 8192) -> torch.Tensor:
    """Probability of the attack class, [C,N]."""
    if theta.dim() == 1:
        theta = theta.unsqueeze(0)
    outs = [torch.softmax(forward(theta, shapes, X[i:i + bs]), -1)[..., 1] for i in range(0, X.shape[0], bs)]
    return torch.cat(outs, 1)


def local_train(theta0: torch.Tensor, shapes, X: torch.Tensor, y: torch.Tensor, epochs: int,
                bs: int, lr: float, gen: torch.Generator, w_pos: float = 1.0) -> torch.Tensor:
    """Train C clients in parallel from the same start.

    theta0 [P]; X [C,N,in]; y [C,N]. Returns the C trained parameter vectors [C,P].
    """
    C, N, _ = X.shape
    th = theta0.unsqueeze(0).repeat(C, 1).clone().requires_grad_(True)
    opt = torch.optim.Adam([th], lr=lr)
    wt = torch.tensor([1.0, w_pos], device=X.device)
    steps = math.ceil(N / bs)
    ar = torch.arange(C, device=X.device).unsqueeze(1)
    for _ in range(epochs):
        perm = torch.argsort(torch.rand(C, N, generator=gen, device=X.device), dim=1)
        for s in range(steps):
            idx = perm[:, s * bs:(s + 1) * bs]
            xb, yb = X[ar, idx], y[ar, idx]
            logit = forward(th, shapes, xb)
            loss = F.cross_entropy(logit.reshape(-1, logit.shape[-1]), yb.reshape(-1), weight=wt, reduction="none")
            loss = loss.reshape(C, -1).mean(1).sum()  # sum over clients keeps them independent
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return th.detach()


def metrics(pred: np.ndarray, y: np.ndarray, fam: np.ndarray, target: str = "dos", score=None) -> dict:
    """Binary IDS metrics plus per-family recall (fraction of attack flows flagged)."""
    tp = ((pred == 1) & (y == 1)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    fn = ((pred == 0) & (y == 1)).sum()
    tn = ((pred == 0) & (y == 0)).sum()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1_att = 2 * prec * rec / max(prec + rec, 1e-9)
    prec_b = tn / max(tn + fn, 1)
    rec_b = tn / max(tn + fp, 1)
    f1_ben = 2 * prec_b * rec_b / max(prec_b + rec_b, 1e-9)
    out = dict(acc=float((tp + tn) / max(len(y), 1)), macro_f1=float((f1_att + f1_ben) / 2),
               fpr=float(fp / max(fp + tn, 1)), recall=float(rec))
    for f in np.unique(fam):
        if f == "benign":
            continue
        m = fam == f
        out[f"rec_{f}"] = float((pred[m] == 1).mean()) if m.any() else float("nan")
    out["rec_target"] = out.get(f"rec_{target}", float("nan"))
    out["precision"] = float(prec)
    out["f1_attack"] = float(f1_att)
    if score is not None and len(np.unique(y)) == 2:
        from sklearn.metrics import roc_auc_score
        out["auc"] = float(roc_auc_score(y, score))
    return out
