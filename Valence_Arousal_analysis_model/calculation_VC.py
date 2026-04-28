import logging, math, torch
from contextlib import nullcontext

def pick_autobatch_size(model, tok, texts, device, max_length=200, start_bs=64, min_bs=4):
    """
    Probe the largest batch size that fits in VRAM using one warmup batch.
    Returns an integer batch size.
    """
    idxs = [i for i, t in enumerate(texts) if isinstance(t, str) and len(t) > 0]
    if not idxs:
        return min_bs
    probe_texts = [texts[idxs[0]]] * min(start_bs, 64)  # repeat a sample to get realistic shape

    bs = start_bs
    while bs >= min_bs:
        try:
            enc = tok(probe_texts[:bs], padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            with torch.no_grad():
                # use fp16 autocast if CUDA; otherwise do nothing
                autocast_ctx = torch.cuda.amp.autocast(dtype=torch.float16) if device == "cuda" else nullcontext()
                with autocast_ctx:
                    _ = model(**enc).logits  # forward only
            logging.info(f"Auto-batch probe OK at batch_size={bs}")
            return bs
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logging.info(f"OOM at batch_size={bs}, reducing...")
                torch.cuda.empty_cache()
                bs //= 2
            else:
                raise
    logging.info(f"Falling back to min batch_size={min_bs}")
    return min_bs

def run_inference_with_autobatch(df, text_col, tok, model, device, max_length=200, preset_bs=None):
    texts = df[text_col].astype(str).tolist()
    total = len(texts)

    # Decide batch size
    if preset_bs is not None:
        batch_size = preset_bs
        logging.info(f"Using user-specified batch_size={batch_size}")
    else:
        start_guess = 64 if device == "cuda" else 16
        batch_size = pick_autobatch_size(model, tok, texts, device, max_length, start_bs=start_guess, min_bs=4)
        logging.info(f"Selected batch_size={batch_size}")

    valence_preds, arousal_preds = [], []

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_texts = texts[start:end]
        logging.info(f"Processing rows {start}–{end-1} (batch={len(batch_texts)})")

        enc = tok(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        with torch.no_grad():
            autocast_ctx = torch.cuda.amp.autocast(dtype=torch.float16) if device == "cuda" else nullcontext()
            with autocast_ctx:
                logits = model(**enc).logits.detach().cpu().numpy()

        valence_preds.extend(logits[:, 0])
        arousal_preds.extend(logits[:, 1])

        if device == "cuda":
            torch.cuda.empty_cache()  # keep memory tidy between batches

    return valence_preds, arousal_preds


import torch, logging, pandas as pd
from transformers import AutoTokenizer
from models import XLMRobertaForSequenceClassificationSig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {DEVICE}")

checkpoint = r"XLM-RoBERTa-large MSE"
tok = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)
model = XLMRobertaForSequenceClassificationSig.from_pretrained(checkpoint, num_labels=2).to(DEVICE)
model.eval()

xlsx_path = "horizontal long videos_with_replies_cleaned.xlsx"
sheet_name = "Replys"
TEXT_COLUMN = "Reply Comment"  # adjust if needed
df = pd.read_excel(xlsx_path, sheet_name=sheet_name)

if TEXT_COLUMN not in df.columns:
    raise KeyError(f"'{TEXT_COLUMN}' not in columns: {list(df.columns)}")

from contextlib import nullcontext
# import the two functions from above here (or place them in a utils.py and import)

valence, arousal = run_inference_with_autobatch(
    df=df,
    text_col=TEXT_COLUMN,
    tok=tok,
    model=model,
    device=DEVICE,
    max_length=200,     # reduce to 128 if you need more memory headroom
    preset_bs=None      # set an integer to force a specific batch size
)

df["CV"] = valence
df["CA"] = arousal
out_path = "final_comments_with_predictions_hor.xlsx"
df.to_excel(out_path, index=False)
logging.info(f"✅ Saved predictions to {out_path}")
