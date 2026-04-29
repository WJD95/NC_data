#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from typing import List, Optional
from transformers.utils import logging as hf_logging

# ========= CONFIG =========
MODEL_NAME = "microsoft/deberta-v3-large"
OUT_DIR = Path("best_model_outputs_cv")          # where your fold_* dirs are
FOLD_IDS = [4]                                   # <<< ONLY FOLD 4
LABELS = ["A", "B", "C", "D"]
MAX_LEN = 128
BATCH_EVAL = 32
SEED = 42

# Input with the 4 columns
INPUT_PATH = "emotion.xlsx"  # set to your 60k file
CONTENT_COL = "text"
# ID_COLS = ["Account", "ID", "Comment ID", "Likes", "Replies", "Emojis", "Valence", "Arousal"]

ID_COLS = [
"number",	"classification"


    # "Year", "number", "like", "top comment", "favorite", "transfer",
    # "KC", "CV", "CA", "CE", "chengdu", "FA", "FV", "distance", "region", "quadrant",
    # "Facial Attention", "emotion_Angry", "emotion_Disgust", "emotion_Fear",
    # "emotion_Happy", "emotion_Neutral", "emotion_Sad", "emotion_Surprise",
    # "wish", "positivity", "Coherence", "Signaling", "Spatial contiguity", "p1",
    # "Segementing (1 minute)", "Pre-training", "p2", "Multimedia", "Personalization",
    # "Voice", "Image", "Embodiment", "p3", "p"
]

# Output
PRED_OUT = "emtion_prediction.xlsx"
# =========================

os.environ["TOKENIZERS_PARALLELISM"] = "false"
hf_logging.set_verbosity_error()

label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def find_best_checkpoint(fold_dir: Path) -> Optional[Path]:
    # use trainer_state.json best path if available; fallback to latest checkpoint-*
    state_path = fold_dir / "trainer_state.json"
    if state_path.exists():
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            best = state.get("best_model_checkpoint")
            if best:
                p = Path(best)
                if not p.exists():
                    p = fold_dir / Path(best).name
                if p.exists():
                    return p
        except Exception:
            pass
    candidates = sorted(fold_dir.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None

def load_selected_checkpoints(out_root: Path, fold_ids: List[int]) -> List[Path]:
    ckpts = []
    for fold in fold_ids:
        fold_dir = out_root / f"fold_{fold}"
        if not fold_dir.exists():
            print(f"[WARN] Missing {fold_dir}. Skipping.")
            continue
        best = find_best_checkpoint(fold_dir)
        if best is None:
            print(f"[WARN] No checkpoint in {fold_dir}. Skipping.")
            continue
        ckpts.append(best)
    if not ckpts:
        raise RuntimeError(f"No checkpoints found for folds {fold_ids}.")
    print(f"Using {len(ckpts)} checkpoint(s): {[str(p) for p in ckpts]}")
    return ckpts

def tokenize_fn(examples, tokenizer):
    texts = [str(x) for x in examples[CONTENT_COL]]
    return tokenizer(texts, truncation=True, max_length=MAX_LEN)

def main():
    set_seed(SEED)

    # --- Load input ---
    if not os.path.exists(INPUT_PATH):
        print(f"Input not found: {INPUT_PATH}")
        sys.exit(1)

    df = pd.read_excel(INPUT_PATH,sheet_name="Sheet1",engine="openpyxl")
    missing = [c for c in ID_COLS + [CONTENT_COL] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in INPUT: {missing}. Got: {df.columns.tolist()}")
    df[CONTENT_COL] = df[CONTENT_COL].astype(str)

    # --- HF dataset + tokenizer/collator ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds_raw = Dataset.from_pandas(df[ID_COLS + [CONTENT_COL]], preserve_index=True)
    ds_tok = ds_raw.map(lambda b: tokenize_fn(b, tokenizer),
                        batched=True,
                        remove_columns=[CONTENT_COL] + ID_COLS)
    ds_tok.set_format("torch")

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    eval_loader = DataLoader(
        ds_tok,
        batch_size=BATCH_EVAL,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,            # Windows-safe
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load ONLY fold 7 checkpoint ---
    ckpt_paths = load_selected_checkpoints(OUT_DIR, FOLD_IDS)

    num_samples = len(ds_tok)
    num_labels = len(LABELS)
    probs_accum = np.zeros((num_samples, num_labels), dtype=np.float32)

    for k, ckpt in enumerate(ckpt_paths, start=1):
        print(f"[{k}/{len(ckpt_paths)}] Loading checkpoint: {ckpt}")
        model = AutoModelForSequenceClassification.from_pretrained(
            ckpt, num_labels=num_labels, id2label=id2label, label2id=label2id
        ).to(device)
        model.eval()

        fold_logits_chunks = []
        with torch.inference_mode():
            for batch in eval_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                fold_logits_chunks.append(outputs.logits.detach().cpu().numpy())

        fold_logits = np.concatenate(fold_logits_chunks, axis=0)
        fold_probs = torch.softmax(torch.tensor(fold_logits), dim=-1).numpy()
        probs_accum += fold_probs

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # average (with one checkpoint it's the same as single model probs)
    probs = probs_accum / len(ckpt_paths)
    pred_ids = probs.argmax(axis=-1)
    pred_labels = [id2label[i] for i in pred_ids]
    pred_conf = probs.max(axis=-1)

    out = df.copy()
    out["pred_label"] = pred_labels
    out["pred_confidence"] = pred_conf
    for i, lab in enumerate(LABELS):
        out[f"prob_{lab}"] = probs[:, i]

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = OUT_DIR / PRED_OUT
    out.to_excel(out_path, index=False)
    print(f"✅ Saved predictions to: {out_path}")
    print(out.head())

if __name__ == "__main__":
    # Windows-safe
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
