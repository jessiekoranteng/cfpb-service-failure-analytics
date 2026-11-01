"""Figures for the Model Validation Report, drawn from the deployed (calibrated) champion and the saved audit results,
so every plotted value matches the numbers quoted in the text.

    python -m src.modeling.report_figures
"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve

from src.modeling import charter as CH
from src.modeling import config as C
from src.modeling import data as D

TEAL, ORANGE, GREY, RED = "#0e7c86", "#f2a541", "#8a8f98", "#c8372d"
OUT = C.REPORTS


def performance_panel():
    a, _ = D.load()
    te = D.split(D.prepare(a), C.PRIMARY)["test"]
    y = te[C.PRIMARY].to_numpy()
    b = joblib.load(C.MODELS / f"risk_{C.PRIMARY}.joblib")
    s = b["calibrated"].predict_proba(te[C.RAW_INPUT])[:, 1]
    flags = CH.deployed_flags(b, te, s)
    fpr, tpr, _ = roc_curve(y, s)
    order = np.argsort(-s, kind="stable")
    cum = np.cumsum(y[order])
    recall = cum / y.sum()
    precision = cum / np.arange(1, len(y) + 1)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    ax[0].plot(fpr, tpr, color=TEAL, lw=2)
    ax[0].plot([0, 1], [0, 1], ls="--", color=GREY)
    ax[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title=f"ROC curve (AUC {roc_auc_score(y, s):.3f})")
    ax[1].plot(recall[50:], precision[50:], color=TEAL, lw=2)  # skip the first 50 ranks: precision there is a few cases, not a trend
    ax[1].axhline(y.mean(), ls="--", color=GREY)
    ax[1].text(0.02, y.mean() - 0.04, f"prevalence {y.mean():.3f}", color=GREY, fontsize=8)
    ax[1].set(xlabel="Recall", ylabel="Precision", ylim=(0, 0.8), title=f"Precision-recall (AP {average_precision_score(y, s):.3f})")
    cm = confusion_matrix(y, flags)
    ax[2].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax[2].text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax[2].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["not flagged", "flagged"], yticklabels=["no relief", "relief"],
              title=f"Deployed flags ({flags.mean():.1%} of complaints)")
    for a_ in ax[:2]:
        a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "report_performance.png", dpi=170)
    plt.close(fig)


def charter_panel():
    r = json.loads((OUT / "charter_checks.json").read_text())
    dep = r["deployed_region_thresholds"]["f2"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.5, 1]})
    labels = [("older_american", "Older American"), ("servicemember", "Servicemember"), ("four_group", "Four tag groups")]
    x = np.arange(len(labels))
    for off, m, col, name in ((-0.18, "tpr_gap", TEAL, "True-positive-rate gap"), (0.18, "fpr_gap", ORANGE, "False-positive-rate gap")):
        v = np.array([dep[k][m]["gap"] for k, _ in labels])
        lo = v - np.array([dep[k][m]["ci95"][0] for k, _ in labels])
        hi = np.array([dep[k][m]["ci95"][1] for k, _ in labels]) - v
        ax[0].bar(x + off, v, 0.34, color=col, label=name, yerr=[lo, hi], capsize=3, ecolor="#333333")
    ax[0].axhline(CH.F2_LIMIT, color=RED, ls="--", label="Charter limit 0.05")
    ax[0].set(xticks=x, xticklabels=[n for _, n in labels], ylabel="Gap (bars: 95% bootstrap interval)", title="Charter F2: error-rate gaps, deployed model")
    ax[0].legend(frameon=False, fontsize=8, loc="upper left")
    p = r["f3_proxy"]
    names = [("older_american", "Older American"), ("servicemember", "Servicemember")]
    for i, (k, n) in enumerate(names):
        auc, (lo, hi) = p[k]["auc_recover_tag_from_features"], p[k]["ci95"]
        ax[1].bar(i - 0.18, auc, 0.34, color=TEAL, yerr=[[auc - lo], [hi - auc]], capsize=3, label="From the nine inputs" if i == 0 else None)
        ax[1].bar(i + 0.18, p[k]["auc_relief_score_vs_tag"], 0.34, color=GREY, label="From the relief score" if i == 0 else None)
    ax[1].axhline(0.5, color=RED, ls="--", label="Chance (0.50)")
    ax[1].set(xticks=range(2), xticklabels=[n for _, n in names], ylim=(0.4, 0.75), ylabel="AUC for recovering the tag", title="Charter F3: proxy test")
    ax[1].legend(frameon=False, fontsize=8, loc="upper right")
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "report_charter.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    performance_panel()
    charter_panel()
    print("wrote report_performance.png, report_charter.png")
