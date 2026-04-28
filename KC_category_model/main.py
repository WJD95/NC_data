import os, json, numpy as np, pandas as pd, torch, torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
from pathlib import Path

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_fscore_support
)
from sklearn.model_selection import StratifiedKFold

# ===================== Config =====================
MODEL_NAME = "microsoft/deberta-v3-large"
LABELS = ["A", "B", "C", "D"]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}

MAX_LEN = 256           # Try 384 if VRAM allows
EPOCHS = 10
LR = 1e-5
WARMUP_RATIO = 0.10
WEIGHT_DECAY = 0.05
BATCH_TRAIN = 8
BATCH_EVAL = 16
GRAD_ACCUM = 1          # Increase (e.g., 2/4) for larger effective batch without OOM
LABEL_SMOOTH = 0.05
FOCAL_GAMMA = 2.0
USE_RDROP = True
RDROP_ALPHA = 1.0
EARLY_STOP_PATIENCE = 2
SEED = 42

N_SPLITS = 10           # K-folds
OUT_DIR = Path("best_model_outputs_cv")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_XLSX = "train_manually.xlsx"
VAL_XLSX = "val_manually.xlsx"
TEST_XLSX = "test_manually.xlsx"

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ===================== Utils =====================
def load_split(xlsx_path: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path)
    df = df[["Content", "Category"]].dropna().copy()
    df["Category"] = df["Category"].astype(str).str.strip().str.upper()
    df = df[df["Category"].isin(LABELS)]
    df["label"] = df["Category"].map(label2id)
    df["Content"] = df["Content"].astype(str)
    return df.reset_index(drop=True)

# ===================== Loss (Focal + LS) =====================
class FocalLossCE(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.05, class_weights=None):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if class_weights is not None and not isinstance(class_weights, torch.Tensor):
            class_weights = torch.tensor(class_weights, dtype=torch.float)
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)

    def forward(self, logits, target):
        device = logits.device
        weight = self.class_weights.to(device) if self.class_weights is not None else None
        ce = F.cross_entropy(
            logits, target, reduction="none",
            label_smoothing=self.label_smoothing,
            weight=weight
        )
        probs = F.softmax(logits, dim=-1)
        pt = probs[torch.arange(target.size(0), device=device), target]
        return ((1 - pt).pow(self.gamma) * ce).mean()

# ===================== Trainer with R-Drop =====================
class SmartTrainer(Trainer):
    def __init__(self, *args, focal_gamma=2.0, label_smoothing=0.05,
                 class_weights=None, use_rdrop=True, rdrop_alpha=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_main = FocalLossCE(
            gamma=focal_gamma, label_smoothing=label_smoothing, class_weights=class_weights
        )
        self.loss_main.to(self.model.device)
        self.use_rdrop = use_rdrop
        self.rdrop_alpha = rdrop_alpha

    # HF >= 4.46 has num_items_in_batch in signature
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        labels = inputs.pop("labels", inputs.pop("label", None))
        if labels is not None and labels.device != model.device:
            labels = labels.to(model.device)

        out1 = model(**inputs)
        logits1 = out1.logits
        loss = self.loss_main(logits1, labels)

        if self.use_rdrop and model.training:
            with torch.no_grad():
                out2 = model(**inputs)
                logits2 = out2.logits

            logp1 = F.log_softmax(logits1, dim=-1)
            p2 = F.softmax(logits2, dim=-1)
            logp2 = F.log_softmax(logits2, dim=-1)
            p1 = F.softmax(logits1, dim=-1)

            kl_12 = F.kl_div(logp1, p2, reduction="batchmean")
            kl_21 = F.kl_div(logp2, p1.detach(), reduction="batchmean")
            loss = loss + self.rdrop_alpha * 0.5 * (kl_12 + kl_21)

        return (loss, out1) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }

def main():
    set_seed(SEED)

    # ===================== Data =====================
    train_df = load_split(TRAIN_XLSX)
    val_df = load_split(VAL_XLSX)
    test_df = load_split(TEST_XLSX)

    # Combine train + val for CV
    cv_df = pd.concat([train_df, val_df], ignore_index=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        contents = [str(x) for x in batch["Content"]]
        return tokenizer(contents, truncation=True, max_length=MAX_LEN)

    # Tokenize combined (CV) and test once
    cv_ds_raw = Dataset.from_pandas(cv_df[["Content", "label"]], preserve_index=False)
    test_ds_raw = Dataset.from_pandas(test_df[["Content", "label"]], preserve_index=False)

    cv_ds = cv_ds_raw.map(tokenize, batched=True)
    test_ds = test_ds_raw.map(tokenize, batched=True)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    for ds_ in [cv_ds, test_ds]:
        ds_.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # Mixed precision: prefer bf16 if available
    try:
        BF16_AVAILABLE = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except AttributeError:
        BF16_AVAILABLE = False

    # ===================== K-fold Training =====================
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    fold_metrics = []
    oof_preds = np.full(len(cv_df), -1, dtype=int)
    oof_logits = np.zeros((len(cv_df), len(LABELS)), dtype=np.float32)
    test_logits_all_folds = []

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(cv_df)), cv_df["label"].values), start=1):
        print(f"\n===== Fold {fold}/{N_SPLITS} =====")
        fold_dir = OUT_DIR / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Per-fold class weights (inverse freq on training split)
        counts = pd.Series(cv_df["label"].values[tr_idx]).value_counts().reindex(range(len(LABELS))).fillna(0).values
        w = counts.sum() / np.maximum(counts, 1)
        class_weights = torch.tensor(w / w.mean(), dtype=torch.float)

        # Datasets for this fold
        train_dataset = cv_ds.select(tr_idx.tolist())
        eval_dataset = cv_ds.select(va_idx.tolist())

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=len(LABELS), id2label=id2label, label2id=label2id
        )
        model.gradient_checkpointing_disable()

        args = TrainingArguments(
            output_dir=str(fold_dir),
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            save_total_limit=1,

            learning_rate=LR,
            per_device_train_batch_size=BATCH_TRAIN,
            per_device_eval_batch_size=BATCH_EVAL,
            gradient_accumulation_steps=GRAD_ACCUM,
            num_train_epochs=EPOCHS,
            weight_decay=WEIGHT_DECAY,
            warmup_ratio=WARMUP_RATIO,
            lr_scheduler_type="cosine",
            optim="adamw_torch",

            bf16=BF16_AVAILABLE,
            fp16=(torch.cuda.is_available() and not BF16_AVAILABLE),
            logging_steps=50,
            report_to="none",
            seed=SEED + fold,                 # slight variation per fold
            dataloader_pin_memory=True,
            dataloader_num_workers=0,         # <<< IMPORTANT on Windows: avoid multiprocessing
            remove_unused_columns=False,
            gradient_checkpointing=False,     # keep OFF with R-Drop
        )

        callbacks = [EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE)]

        trainer = SmartTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            focal_gamma=FOCAL_GAMMA,
            label_smoothing=LABEL_SMOOTH,
            class_weights=class_weights,
            use_rdrop=USE_RDROP,
            rdrop_alpha=RDROP_ALPHA,
        )

        train_result = trainer.train()

        # Evaluate on this fold's validation split
        val_metrics = trainer.evaluate(eval_dataset)
        print("Fold val metrics:", val_metrics)

        # OOF predictions
        val_pred_logits = trainer.predict(eval_dataset).predictions
        val_preds = val_pred_logits.argmax(axis=-1)
        oof_preds[va_idx] = val_preds
        oof_logits[va_idx] = val_pred_logits

        # Save fold metrics
        fold_summary = {
            "fold": fold,
            "val_accuracy": float(val_metrics.get("eval_accuracy", np.nan)),
            "val_f1_macro": float(val_metrics.get("eval_f1_macro", np.nan)),
            "val_f1_weighted": float(val_metrics.get("eval_f1_weighted", np.nan)),
            "val_loss": float(val_metrics.get("eval_loss", np.nan)),
        }
        fold_metrics.append(fold_summary)
        with open(fold_dir / "val_metrics.json", "w", encoding="utf-8") as f:
            json.dump(fold_summary, f, indent=2)

        # Optional: quick report per fold
        y_true_va = cv_df["label"].values[va_idx]
        rep = classification_report(
            y_true_va, val_preds, target_names=[id2label[i] for i in range(len(LABELS))], digits=4
        )
        with open(fold_dir / "val_classification_report.txt", "w", encoding="utf-8") as f:
            f.write(rep)

        # Collect test logits for ensembling later
        test_pred_logits = trainer.predict(test_ds).predictions
        test_logits_all_folds.append(test_pred_logits)

        # Housekeeping
        del trainer, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ===================== Cross-validated Summary (OOF) =====================
    metrics_cv = {
        "oof_accuracy": accuracy_score(cv_df["label"].values, oof_preds),
        "oof_f1_macro": f1_score(cv_df["label"].values, oof_preds, average="macro"),
        "oof_f1_weighted": f1_score(cv_df["label"].values, oof_preds, average="weighted"),
    }
    print("\nOOF metrics:", metrics_cv)

    fold_metrics_df = pd.DataFrame(fold_metrics)
    fold_metrics_df.to_csv(OUT_DIR / "fold_val_metrics.csv", index=False)

    # Mean ± SD across folds
    summary = {
        "val_accuracy_mean": float(fold_metrics_df["val_accuracy"].mean()),
        "val_accuracy_sd": float(fold_metrics_df["val_accuracy"].std(ddof=1)),
        "val_f1_macro_mean": float(fold_metrics_df["val_f1_macro"].mean()),
        "val_f1_macro_sd": float(fold_metrics_df["val_f1_macro"].std(ddof=1)),
        "val_f1_weighted_mean": float(fold_metrics_df["val_f1_weighted"].mean()),
        "val_f1_weighted_sd": float(fold_metrics_df["val_f1_weighted"].std(ddof=1)),
        "oof_accuracy": metrics_cv["oof_accuracy"],
        "oof_f1_macro": metrics_cv["oof_f1_macro"],
        "oof_f1_weighted": metrics_cv["oof_f1_weighted"],
    }
    with open(OUT_DIR / "cv_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nCV Summary (mean ± SD):")
    for k in ["val_accuracy", "val_f1_macro", "val_f1_weighted"]:
        print(f"  {k}: {summary[f'{k}_mean']:.4f} ± {summary[f'{k}_sd']:.4f}")
    print(f"  OOF accuracy: {summary['oof_accuracy']:.4f}")
    print(f"  OOF macro-F1: {summary['oof_f1_macro']:.4f}")

    # Save OOF predictions and logits
    oof_df = cv_df.copy()
    oof_df["oof_pred"] = [id2label[i] for i in oof_preds]
    oof_df["gold"] = [id2label[i] for i in cv_df["label"].values]
    oof_df.to_csv(OUT_DIR / "oof_predictions.csv", index=False)
    np.save(OUT_DIR / "oof_logits.npy", oof_logits)

    # ===================== Test (Ensembled over folds) =====================
    # Average probabilities across folds for test set
    test_probs_ensemble = np.mean(
        [torch.softmax(torch.tensor(l), dim=-1).numpy() for l in test_logits_all_folds], axis=0
    )
    test_preds_ensemble = np.argmax(test_probs_ensemble, axis=-1)
    y_true_test = test_df["label"].to_numpy()

    test_metrics = {
        "accuracy": accuracy_score(y_true_test, test_preds_ensemble),
        "f1_macro": f1_score(y_true_test, test_preds_ensemble, average="macro"),
        "f1_weighted": f1_score(y_true_test, test_preds_ensemble, average="weighted"),
    }
    print("\nTest metrics (ensemble):", test_metrics)
    with open(OUT_DIR / "test_metrics_ensemble.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    # Save test predictions (labels)
    pred_labels = [id2label[i] for i in test_preds_ensemble]
    gold_labels = [id2label[i] for i in y_true_test]
    pd.DataFrame({"gold": gold_labels, "pred": pred_labels}).to_csv(OUT_DIR / "predictions_test_ensemble.csv", index=False)

    # Classification report
    rep_test = classification_report(
        y_true_test, test_preds_ensemble, target_names=[id2label[i] for i in range(len(LABELS))], output_dict=True, digits=4
    )
    pd.DataFrame(rep_test).transpose().to_csv(OUT_DIR / "classification_report_test_ensemble.csv", index=True)
    with open(OUT_DIR / "classification_report_test_ensemble.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(y_true_test, test_preds_ensemble, target_names=LABELS, digits=4))

    # ===================== Figures (Test ensemble) =====================
    cm = confusion_matrix(y_true_test, test_preds_ensemble, labels=list(range(len(LABELS))))
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, values_format="d")
    ax.set_title("Confusion Matrix (Counts) - Test (Ensemble)")
    plt.tight_layout(); plt.savefig(OUT_DIR / "cm_counts_test_ensemble.png", dpi=150); plt.close(fig)

    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cmn, interpolation='nearest')
    ax.set_title("Confusion Matrix (Row-Normalized) - Test (Ensemble)")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(range(len(LABELS))); ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=45); ax.set_yticklabels(LABELS)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.savefig(OUT_DIR / "cm_normalized_test_ensemble.png", dpi=150); plt.close(fig)

    # Per-class PRF on test (ensemble)
    prec, rec, f1c, supp = precision_recall_fscore_support(
        y_true_test, test_preds_ensemble, labels=list(range(len(LABELS))), zero_division=0
    )
    per_class = pd.DataFrame({
        "label": LABELS,
        "precision": prec,
        "recall": rec,
        "f1": f1c,
        "support": supp
    })
    per_class.to_csv(OUT_DIR / "per_class_metrics_test_ensemble.csv", index=False)

    x = np.arange(len(LABELS)); w = 0.25
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w, per_class["precision"].values, width=w, label="Precision")
    ax.bar(x, per_class["recall"].values, width=w, label="Recall")
    ax.bar(x + w, per_class["f1"].values, width=w, label="F1")
    ax.set_xticks(x); ax.set_xticklabels(LABELS)
    ax.set_ylim(0, 1.0); ax.set_ylabel("Score")
    ax.set_title("Per-class Precision / Recall / F1 (Test, Ensemble)")
    ax.legend()
    plt.tight_layout(); plt.savefig(OUT_DIR / "per_class_metrics_test_ensemble.png", dpi=150); plt.close(fig)

    print("\n✅ Done. Artifacts saved in:", OUT_DIR)

if __name__ == "__main__":
    # Windows-safe multiprocessing guard
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
