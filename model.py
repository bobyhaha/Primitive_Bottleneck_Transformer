import os
import math
import json
import time
import copy
import random
import argparse
from dataclasses import dataclass, asdict, replace
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer


# ============================================================
# Config
# ============================================================

@dataclass
class LMConfig:
    # data
    dataset: str = "tinystories"                 # tinystories | tinyshakespeare | textfile
    data_root: str = "./data"
    text_path: str = ""
    tokenizer_name: str = "gpt2"
    seq_len: int = 128
    train_split: str = "train"
    val_split: str = "validation"
    train_fraction_for_textfile: float = 0.98
    num_proc: int = 1

    # optimization
    batch_size: int = 32
    epochs: int = 3
    lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    num_workers: int = 2

    # model
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1
    max_seq_len: int = 256
    nonlinearity: str = "gelu"

    # disentangled block
    head_mlp_alpha: float = 0.4
    head_mlp_layers: int = 1
    aggregation: str = "sum"                    # sum | mean | learned
    aggregation_scale: str = "sqrt"             # none | sqrt | n

    # primitive bottleneck block
    n_primitives: int = 128
    primitive_encoder_dim: int = 128
    primitive_activation: str = "relu"          # relu | gelu | softplus
    primitive_sparsity_lambda: float = 0.0
    primitive_usage_lambda: float = 0.0
    share_primitive_decoder: bool = True

    # experiment
    save_dir: str = "./runs_primitive_lm"
    save_checkpoints: bool = True

    def validate(self):
        assert self.d_model % self.n_heads == 0
        assert self.seq_len <= self.max_seq_len
        assert self.aggregation in {"sum", "mean", "learned"}
        assert self.aggregation_scale in {"none", "sqrt", "n"}
        assert self.nonlinearity in {"gelu", "relu", "silu"}
        assert self.primitive_activation in {"relu", "gelu", "softplus"}
        assert self.head_mlp_alpha >= 0.0
        assert self.head_mlp_layers >= 1

    @property
    def d_head(self):
        return self.d_model // self.n_heads

    @property
    def baseline_mlp_proj_budget_no_bias(self):
        d = self.d_model
        return d * d + 2 * d * self.d_ff

    @property
    def d_ff_h(self):
        alpha_budget = self.head_mlp_alpha * self.baseline_mlp_proj_budget_no_bias
        H = self.n_heads
        a = max(0, self.head_mlp_layers - 1) * H
        b = H * (self.d_head + self.d_model)
        c = -alpha_budget
        if a == 0:
            x = alpha_budget / max(1, b)
        else:
            disc = b * b - 4 * a * c
            x = (-b + math.sqrt(max(0.0, disc))) / (2 * a)
        return max(4, int(x))

    @property
    def d_ff_large(self):
        remaining = (1.0 - self.head_mlp_alpha) * self.baseline_mlp_proj_budget_no_bias
        x = remaining / max(1, 2 * self.d_model)
        return max(4, int(x))


# ============================================================
# Utilities
# ============================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.count)

    def update(self, value: float, n: int = 1):
        self.sum += value * n
        self.count += n


def get_act(name: str) -> nn.Module:
    return {
        "gelu": nn.GELU(),
        "relu": nn.ReLU(),
        "silu": nn.SiLU(),
    }[name]


def primitive_act(name: str, x: torch.Tensor) -> torch.Tensor:
    if name == "relu":
        return F.relu(x)
    if name == "gelu":
        return F.gelu(x)
    if name == "softplus":
        return F.softplus(x)
    raise ValueError(name)


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dropout_p: float = 0.0, training: bool = True):
    d = q.size(-1)
    t = q.size(-2)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)
    mask = torch.triu(torch.ones(t, t, device=q.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    if dropout_p > 0:
        attn = F.dropout(attn, p=dropout_p, training=training)
    return attn @ v


def append_row(csv_path: str, row: Dict[str, object]) -> None:
    import csv
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


# ============================================================
# Hugging Face data pipeline
# ============================================================

class HFCausalCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict[str, List[int]]]):
        input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
        labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
        return input_ids, labels


def get_text_column(dataset_name: str, split_dataset):
    candidates = ["text", "story", "content"]
    for c in candidates:
        if c in split_dataset.column_names:
            return c
    raise ValueError(f"Could not infer text column from columns: {split_dataset.column_names}")



def build_tokenizer(cfg: LMConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer



def chunk_examples(examples, text_column: str, tokenizer, seq_len: int):
    tokenized = tokenizer(examples[text_column], add_special_tokens=False, truncation=False)
    flat_ids = []
    for ids in tokenized["input_ids"]:
        flat_ids.extend(ids + [tokenizer.eos_token_id])

    total_len = (len(flat_ids) // (seq_len + 1)) * (seq_len + 1)
    if total_len == 0:
        return {"input_ids": [], "labels": []}
    flat_ids = flat_ids[:total_len]

    input_ids = []
    labels = []
    chunk_size = seq_len + 1
    for i in range(0, total_len, chunk_size):
        block = flat_ids[i:i + chunk_size]
        input_ids.append(block[:-1])
        labels.append(block[1:])
    return {"input_ids": input_ids, "labels": labels}



def load_raw_dataset(cfg: LMConfig):
    if cfg.dataset == "tinystories":
        # Requires internet on first run through Hugging Face cache.
        ds = load_dataset("roneneldan/TinyStories")
        train_split = cfg.train_split if cfg.train_split in ds else "train"
        val_split = cfg.val_split if cfg.val_split in ds else ("validation" if "validation" in ds else "train")
        return ds[train_split], ds[val_split]

    if cfg.dataset == "tinyshakespeare":
        path = os.path.join(cfg.data_root, "tinyshakespeare.txt")
        if not os.path.exists(path):
            import urllib.request
            os.makedirs(cfg.data_root, exist_ok=True)
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            print(f"Downloading Tiny Shakespeare to {path}")
            urllib.request.urlretrieve(url, path)
        ds = load_dataset("text", data_files={"train": path})
        full = ds["train"].train_test_split(test_size=1.0 - cfg.train_fraction_for_textfile, seed=cfg.seed)
        return full["train"], full["test"]

    if cfg.dataset == "textfile":
        if not cfg.text_path:
            raise ValueError("--text_path is required for dataset=textfile")
        ds = load_dataset("text", data_files={"train": cfg.text_path})
        full = ds["train"].train_test_split(test_size=1.0 - cfg.train_fraction_for_textfile, seed=cfg.seed)
        return full["train"], full["test"]

    raise ValueError(cfg.dataset)



def build_dataloaders(cfg: LMConfig):
    tokenizer = build_tokenizer(cfg)
    train_raw, val_raw = load_raw_dataset(cfg)

    train_text_col = get_text_column(cfg.dataset, train_raw)
    val_text_col = get_text_column(cfg.dataset, val_raw)

    train_tok = train_raw.map(
        lambda ex: chunk_examples(ex, train_text_col, tokenizer, cfg.seq_len),
        batched=True,
        remove_columns=train_raw.column_names,
        num_proc=cfg.num_proc,
        desc="Tokenizing train split",
    )
    val_tok = val_raw.map(
        lambda ex: chunk_examples(ex, val_text_col, tokenizer, cfg.seq_len),
        batched=True,
        remove_columns=val_raw.column_names,
        num_proc=cfg.num_proc,
        desc="Tokenizing val split",
    )

    collator = HFCausalCollator(pad_token_id=tokenizer.pad_token_id)

    train_dl = DataLoader(
        train_tok,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collator,
    )
    val_dl = DataLoader(
        val_tok,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collator,
    )
    return tokenizer, train_dl, val_dl


# ============================================================
# Core LM blocks
# ============================================================

class BaselineBlock(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        d = cfg.d_model
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.norm_attn = nn.LayerNorm(d)
        self.norm_mlp = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out_proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(
            nn.Linear(d, cfg.d_ff),
            get_act(cfg.nonlinearity),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, d),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor):
        b, t, _ = x.shape
        h = self.norm_attn(x)
        qkv = self.qkv(h).reshape(b, t, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(2)
        q, k, v = [z.permute(0, 2, 1, 3) for z in (q, k, v)]
        out = causal_attention(q, k, v, dropout_p=self.attn_drop.p, training=self.training)
        out = out.permute(0, 2, 1, 3).reshape(b, t, -1)
        x = x + self.out_proj(out)
        x = x + self.mlp(self.norm_mlp(x))
        aux = {"sparsity": x.new_tensor(0.0), "usage": x.new_tensor(0.0)}
        return x, aux


class DisentangledBlock(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        d = cfg.d_model
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        self.agg = cfg.aggregation
        self.agg_scale = cfg.aggregation_scale
        self.attn_drop = nn.Dropout(cfg.dropout)

        self.norm_attn = nn.LayerNorm(d)
        self.norm_mlp = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)

        self.small_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cfg.d_head, cfg.d_ff_h),
                get_act(cfg.nonlinearity),
                nn.Dropout(cfg.dropout),
                *sum([
                    [nn.Linear(cfg.d_ff_h, cfg.d_ff_h), get_act(cfg.nonlinearity), nn.Dropout(cfg.dropout)]
                    for _ in range(cfg.head_mlp_layers - 1)
                ], [])
            )
            for _ in range(cfg.n_heads)
        ])
        self.up_projs = nn.ModuleList([nn.Linear(cfg.d_ff_h, d) for _ in range(cfg.n_heads)])
        if self.agg == "learned":
            self.agg_w = nn.Parameter(torch.zeros(cfg.n_heads))
        self.large_mlp = nn.Sequential(
            nn.Linear(d, cfg.d_ff_large),
            get_act(cfg.nonlinearity),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff_large, d),
            nn.Dropout(cfg.dropout),
        )

    def _scale_sum(self, agg: torch.Tensor):
        if self.agg_scale == "none":
            return agg
        if self.agg_scale == "sqrt":
            return agg / math.sqrt(self.n_heads)
        if self.agg_scale == "n":
            return agg / self.n_heads
        raise ValueError(self.agg_scale)

    def forward(self, x: torch.Tensor):
        b, t, _ = x.shape
        h = self.norm_attn(x)
        qkv = self.qkv(h).reshape(b, t, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(2)
        q, k, v = [z.permute(0, 2, 1, 3) for z in (q, k, v)]
        head_out = causal_attention(q, k, v, dropout_p=self.attn_drop.p, training=self.training)

        projected = []
        for i in range(self.n_heads):
            z = self.small_mlps[i](head_out[:, i])
            z = self.up_projs[i](z)
            projected.append(z)
        ups = torch.stack(projected, dim=0)

        if self.agg == "sum":
            agg = self._scale_sum(ups.sum(0))
        elif self.agg == "mean":
            agg = ups.mean(0)
        else:
            w = F.softmax(self.agg_w, dim=0)
            agg = (ups * w[:, None, None, None]).sum(0)

        x = x + agg
        x = x + self.large_mlp(self.norm_mlp(x))
        aux = {"sparsity": x.new_tensor(0.0), "usage": x.new_tensor(0.0)}
        return x, aux


class PrimitiveBottleneckBlock(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        d = cfg.d_model
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        self.agg = cfg.aggregation
        self.agg_scale = cfg.aggregation_scale
        self.attn_drop = nn.Dropout(cfg.dropout)

        self.norm_attn = nn.LayerNorm(d)
        self.norm_mlp = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)

        self.head_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cfg.d_head, cfg.primitive_encoder_dim),
                get_act(cfg.nonlinearity),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.primitive_encoder_dim, cfg.n_primitives),
            )
            for _ in range(cfg.n_heads)
        ])

        if cfg.share_primitive_decoder:
            self.shared_decoder = nn.Linear(cfg.n_primitives, d, bias=False)
            self.head_decoders = None
        else:
            self.shared_decoder = None
            self.head_decoders = nn.ModuleList([
                nn.Linear(cfg.n_primitives, d, bias=False) for _ in range(cfg.n_heads)
            ])

        if self.agg == "learned":
            self.agg_w = nn.Parameter(torch.zeros(cfg.n_heads))

        self.large_mlp = nn.Sequential(
            nn.Linear(d, cfg.d_ff_large),
            get_act(cfg.nonlinearity),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff_large, d),
            nn.Dropout(cfg.dropout),
        )

    def _scale_sum(self, agg: torch.Tensor):
        if self.agg_scale == "none":
            return agg
        if self.agg_scale == "sqrt":
            return agg / math.sqrt(self.n_heads)
        if self.agg_scale == "n":
            return agg / self.n_heads
        raise ValueError(self.agg_scale)

    def forward(self, x: torch.Tensor):
        b, t, _ = x.shape
        h = self.norm_attn(x)
        qkv = self.qkv(h).reshape(b, t, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(2)
        q, k, v = [z.permute(0, 2, 1, 3) for z in (q, k, v)]
        head_out = causal_attention(q, k, v, dropout_p=self.attn_drop.p, training=self.training)

        projected = []
        primitive_codes = []
        for i in range(self.n_heads):
            code_logits = self.head_encoders[i](head_out[:, i])
            code = primitive_act(self.cfg.primitive_activation, code_logits)
            primitive_codes.append(code)
            if self.shared_decoder is not None:
                out_i = self.shared_decoder(code)
            else:
                out_i = self.head_decoders[i](code)
            projected.append(out_i)

        codes = torch.stack(primitive_codes, dim=0)   # [H,B,T,K]
        ups = torch.stack(projected, dim=0)           # [H,B,T,D]

        if self.agg == "sum":
            agg = self._scale_sum(ups.sum(0))
        elif self.agg == "mean":
            agg = ups.mean(0)
        else:
            w = F.softmax(self.agg_w, dim=0)
            agg = (ups * w[:, None, None, None]).sum(0)

        x = x + agg
        x = x + self.large_mlp(self.norm_mlp(x))

        sparsity = codes.abs().mean()
        mean_usage = codes.mean(dim=(0, 1, 2))
        probs = mean_usage / mean_usage.sum().clamp_min(1e-8)
        usage_entropy = -(probs * probs.clamp_min(1e-8).log()).sum()
        aux = {"sparsity": sparsity, "usage": -usage_entropy}
        return x, aux


# ============================================================
# LM model
# ============================================================

class TinyTransformerLM(nn.Module):
    def __init__(self, cfg: LMConfig, vocab_size: int, mode: str = "baseline"):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.mode = mode
        self.vocab_size = vocab_size

        self.token_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.max_seq_len, cfg.d_model))
        self.drop = nn.Dropout(cfg.dropout)

        if mode == "baseline":
            block_cls = BaselineBlock
        elif mode == "disentangled":
            block_cls = DisentangledBlock
        elif mode == "primitive":
            block_cls = PrimitiveBottleneckBlock
        else:
            raise ValueError(mode)

        self.blocks = nn.ModuleList([block_cls(cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx: torch.Tensor):
        b, t = idx.shape
        assert t <= self.cfg.max_seq_len
        x = self.token_emb(idx) + self.pos_emb[:, :t]
        x = self.drop(x)

        aux_list = []
        for block in self.blocks:
            x, aux = block(x)
            aux_list.append(aux)

        x = self.norm(x)
        logits = self.lm_head(x)

        aux_out = {
            "sparsity": torch.stack([a["sparsity"] for a in aux_list]).mean(),
            "usage": torch.stack([a["usage"] for a in aux_list]).mean(),
        }
        return logits, aux_out

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# Train / Eval
# ============================================================


def compute_loss(cfg: LMConfig, logits: torch.Tensor, targets: torch.Tensor, aux: Dict[str, torch.Tensor]):
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    loss = ce
    sparsity_term = logits.new_tensor(0.0)
    usage_term = logits.new_tensor(0.0)

    if cfg.primitive_sparsity_lambda > 0:
        sparsity_term = cfg.primitive_sparsity_lambda * aux["sparsity"]
        loss = loss + sparsity_term

    if cfg.primitive_usage_lambda > 0:
        usage_term = cfg.primitive_usage_lambda * aux["usage"]
        loss = loss + usage_term

    return loss, {
        "ce": float(ce.detach().item()),
        "sparsity": float(sparsity_term.detach().item()),
        "usage": float(usage_term.detach().item()),
        "total": float(loss.detach().item()),
    }


@torch.no_grad()
def evaluate(model: TinyTransformerLM, loader: DataLoader, cfg: LMConfig):
    model.eval()
    loss_meter = AverageMeter()
    ce_meter = AverageMeter()

    for x, y in loader:
        x = x.to(cfg.device, non_blocking=True)
        y = y.to(cfg.device, non_blocking=True)
        logits, aux = model(x)
        loss, stats = compute_loss(cfg, logits, y, aux)
        loss_meter.update(stats["total"], x.size(0))
        ce_meter.update(stats["ce"], x.size(0))

    ppl = math.exp(min(20.0, ce_meter.avg))
    return {"loss": loss_meter.avg, "ce": ce_meter.avg, "ppl": ppl}


def train_one_epoch(model: TinyTransformerLM, loader: DataLoader, optimizer, scheduler, cfg: LMConfig, epoch: int, variant_name: str):
    model.train()
    total_meter = AverageMeter()
    ce_meter = AverageMeter()
    sparse_meter = AverageMeter()
    usage_meter = AverageMeter()

    for step, (x, y) in enumerate(loader):
        x = x.to(cfg.device, non_blocking=True)
        y = y.to(cfg.device, non_blocking=True)

        logits, aux = model(x)
        loss, stats = compute_loss(cfg, logits, y, aux)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        total_meter.update(stats["total"], x.size(0))
        ce_meter.update(stats["ce"], x.size(0))
        sparse_meter.update(stats["sparsity"], x.size(0))
        usage_meter.update(stats["usage"], x.size(0))

        if step % 200 == 0:
            print(
                f"[{variant_name}] epoch={epoch} step={step}/{len(loader)} "
                f"loss={stats['total']:.4f} ce={stats['ce']:.4f} "
                f"sparse={stats['sparsity']:.4f} usage={stats['usage']:.4f}"
            )

    scheduler.step()
    return {
        "loss": total_meter.avg,
        "ce": ce_meter.avg,
        "ppl": math.exp(min(20.0, ce_meter.avg)),
        "sparsity": sparse_meter.avg,
        "usage": usage_meter.avg,
    }


# ============================================================
# Experiment suite / ablation
# ============================================================


def get_variants(base_cfg: LMConfig) -> List[Tuple[str, str, LMConfig]]:
    return [
        ("baseline", "baseline", replace(base_cfg, primitive_sparsity_lambda=0.0, primitive_usage_lambda=0.0)),
        ("disentangled", "disentangled", replace(base_cfg, primitive_sparsity_lambda=0.0, primitive_usage_lambda=0.0)),
        ("primitive_no_sparse", "primitive", replace(base_cfg, primitive_sparsity_lambda=0.0, primitive_usage_lambda=0.0)),
        ("primitive_sparse", "primitive", replace(base_cfg, primitive_sparsity_lambda=1e-5, primitive_usage_lambda=0.0)),
        ("primitive_sparse_usage", "primitive", replace(base_cfg, primitive_sparsity_lambda=1e-5, primitive_usage_lambda=1e-4)),
    ]



def run_variant(name: str, mode: str, cfg: LMConfig, vocab_size: int, train_dl: DataLoader, val_dl: DataLoader, suite_dir: str):
    os.makedirs(suite_dir, exist_ok=True)
    variant_dir = os.path.join(suite_dir, name)
    os.makedirs(variant_dir, exist_ok=True)

    set_seed(cfg.seed)
    model = TinyTransformerLM(cfg, vocab_size=vocab_size, mode=mode).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_val_ce = float("inf")
    best_state = None
    history_path = os.path.join(variant_dir, "history.csv")
    if os.path.exists(history_path):
        os.remove(history_path)

    start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        train_stats = train_one_epoch(model, train_dl, optimizer, scheduler, cfg, epoch, name)
        val_stats = evaluate(model, val_dl, cfg)

        row = {
            "variant": name,
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_ce": train_stats["ce"],
            "train_ppl": train_stats["ppl"],
            "train_sparsity": train_stats["sparsity"],
            "train_usage": train_stats["usage"],
            "val_loss": val_stats["loss"],
            "val_ce": val_stats["ce"],
            "val_ppl": val_stats["ppl"],
        }
        append_row(history_path, row)
        print(f"[{name}][val] ce={val_stats['ce']:.4f} ppl={val_stats['ppl']:.2f}")

        if val_stats["ce"] < best_val_ce:
            best_val_ce = val_stats["ce"]
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    final_val = evaluate(model, val_dl, cfg)
    elapsed = time.time() - start
    params = model.count_params()

    summary = {
        "variant": name,
        "mode": mode,
        "params": params,
        "best_val_ce": round(final_val["ce"], 6),
        "best_val_ppl": round(final_val["ppl"], 4),
        "elapsed_sec": round(elapsed, 2),
        "primitive_sparsity_lambda": cfg.primitive_sparsity_lambda,
        "primitive_usage_lambda": cfg.primitive_usage_lambda,
        "n_primitives": cfg.n_primitives,
        "share_primitive_decoder": cfg.share_primitive_decoder,
    }

    with open(os.path.join(variant_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if cfg.save_checkpoints:
        torch.save(
            {"model_state": model.state_dict(), "config": asdict(cfg), "summary": summary},
            os.path.join(variant_dir, "best.pt"),
        )

    return summary



def print_summary_table(rows: List[Dict[str, object]]):
    print("\nAblation summary:")
    header = f"{'variant':24s} {'params':>10s} {'val_ce':>10s} {'val_ppl':>10s} {'sp_lambda':>10s} {'u_lambda':>10s}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{str(row['variant']):24s} {int(row['params']):10d} {float(row['best_val_ce']):10.4f} "
            f"{float(row['best_val_ppl']):10.2f} {float(row['primitive_sparsity_lambda']):10.1e} "
            f"{float(row['primitive_usage_lambda']):10.1e}"
        )


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="tinystories", choices=["tinystories", "tinyshakespeare", "textfile"])
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--text_path", type=str, default="")
    parser.add_argument("--tokenizer_name", type=str, default="gpt2")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./runs_primitive_lm")
    parser.add_argument("--num_proc", type=int, default=1)

    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--nonlinearity", type=str, default="gelu", choices=["gelu", "relu", "silu"])

    parser.add_argument("--head_mlp_alpha", type=float, default=0.4)
    parser.add_argument("--head_mlp_layers", type=int, default=1)
    parser.add_argument("--aggregation", type=str, default="sum", choices=["sum", "mean", "learned"])
    parser.add_argument("--aggregation_scale", type=str, default="sqrt", choices=["none", "sqrt", "n"])

    parser.add_argument("--n_primitives", type=int, default=128)
    parser.add_argument("--primitive_encoder_dim", type=int, default=128)
    parser.add_argument("--primitive_activation", type=str, default="relu", choices=["relu", "gelu", "softplus"])
    parser.add_argument("--share_primitive_decoder", action="store_true")

    args = parser.parse_args()

    cfg = LMConfig(
        dataset=args.dataset,
        data_root=args.data_root,
        text_path=args.text_path,
        tokenizer_name=args.tokenizer_name,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        save_dir=args.save_dir,
        num_proc=args.num_proc,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        nonlinearity=args.nonlinearity,
        head_mlp_alpha=args.head_mlp_alpha,
        head_mlp_layers=args.head_mlp_layers,
        aggregation=args.aggregation,
        aggregation_scale=args.aggregation_scale,
        n_primitives=args.n_primitives,
        primitive_encoder_dim=args.primitive_encoder_dim,
        primitive_activation=args.primitive_activation,
        share_primitive_decoder=args.share_primitive_decoder,
    )
    cfg.validate()
    os.makedirs(cfg.save_dir, exist_ok=True)
    suite_dir = os.path.join(cfg.save_dir, f"suite_{cfg.dataset}_{cfg.tokenizer_name.replace('/', '_')}_seed{cfg.seed}")
    os.makedirs(suite_dir, exist_ok=True)

    print("Config:")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")

    tokenizer, train_dl, val_dl = build_dataloaders(cfg)
    vocab_size = len(tokenizer)
    print(f"Tokenizer: {cfg.tokenizer_name} | vocab_size={vocab_size}")

    variants = get_variants(cfg)
    summary_csv = os.path.join(suite_dir, "ablation_summary.csv")
    if os.path.exists(summary_csv):
        os.remove(summary_csv)

    rows = []
    for name, mode, variant_cfg in variants:
        summary = run_variant(name, mode, variant_cfg, vocab_size, train_dl, val_dl, suite_dir)
        append_row(summary_csv, summary)
        rows.append(summary)
        print_summary_table(rows)

    rows_sorted = sorted(rows, key=lambda x: float(x["best_val_ce"]))
    print("\nFinal sorted ablation summary:")
    print_summary_table(rows_sorted)
    with open(os.path.join(suite_dir, "ablation_summary_sorted.json"), "w", encoding="utf-8") as f:
        json.dump(rows_sorted, f, indent=2)

    print(f"\nSaved outputs under: {suite_dir}")


if __name__ == "__main__":
    main()
