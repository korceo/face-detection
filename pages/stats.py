# Streamlit page: Training stats and results for YOLO11 Nano
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

st.set_page_config(page_title="YOLO11 Nano — Training Stats", layout="wide")

# ------------------------------
# Constants
# ------------------------------
MODEL_NAME = "YOLO 11 Nano"
EPOCHS_TOTAL = 25
DATASET_SIZE = 16_700

PHASES = [
    {
        "title": "1) Head-only fine-tune (5 эпох)",
        "desc": "Разморозили head, тренировались 5 эпох.",
        "img": "results/yolo11n-head5.png",
        "csv": "results/yolo11n-head5.csv",
    },
    {
        "title": "2) Full backbone (ещё 10 эпох)",
        "desc": "Разморозили всё и учились 10 эпох.",
        "img": "results/yolo11n-back10.png",
        "csv": "results/yolo11n-back10.csv",
    },
    {
        "title": "3) Best weights → ещё 10 эпох (итого 25)",
        "desc": "Взяли лучшие веса и доучили 10 эпох.",
        "img": "results/yolo11n-back20.png",
        "csv": "results/yolo11n-back20.csv",
    },
]

FINAL_METRICS = {
    "Precision": 0.894,
    "Recall": 0.814,
    "mAP50": 0.891,
    "mAP50-95": 0.596,
}

# ------------------------------
# Helpers
# ------------------------------
@st.cache_data(show_spinner=False)
def _read_csv(path: str) -> Optional[pd.DataFrame]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        return df
    except Exception:
        return None


def _pick_epoch_index(df: pd.DataFrame) -> pd.Series:
    # Try common names; fallback to index
    for col in ["epoch", "Epoch", "ep", "step"]:
        if col in df.columns:
            return df[col]
    return pd.Series(range(len(df)))


_METRIC_CANDIDATES = {
    "Precision": [
        "metrics/precision(B)", "metrics/precision", "precision", "Precision",
    ],
    "Recall": [
        "metrics/recall(B)", "metrics/recall", "recall", "Recall",
    ],
    "mAP50": [
        "metrics/mAP50(B)", "metrics/mAP50", "map50", "mAP50",
    ],
    "mAP50-95": [
        "metrics/mAP50-95(B)", "metrics/mAP50-95", "map", "mAP50-95",
    ],
}


def _extract_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["epoch"] = _pick_epoch_index(df)
    for nice, cands in _METRIC_CANDIDATES.items():
        for c in cands:
            if c in df.columns:
                out[nice] = df[c]
                break
        if nice not in out.columns:
            # Try case-insensitive match
            lc = {c.lower(): c for c in df.columns}
            for cand in cands:
                if cand.lower() in lc:
                    out[nice] = df[lc[cand.lower()]]
                    break
    return out


def _phase_block(title: str, desc: str, img_path: str, csv_path: str) -> None:
    st.subheader(title)
    st.caption(desc)

    c1, c2 = st.columns([1.2, 1.8])
    with c1:
        if Path(img_path).exists():
            st.image(img_path, caption=Path(img_path).name, width='stretch')
        else:
            st.warning(f"Нет изображения: {img_path}")

    with c2:
        df = _read_csv(csv_path)
        if df is None:
            st.warning(f"Нет CSV: {csv_path}")
            return
        metr = _extract_metrics(df)
        if metr.shape[1] <= 1:
            st.info("В CSV не найдено стандартных метрик (Precision/Recall/mAP). Показан сырой DataFrame.")
            st.dataframe(df, width='stretch')
            return

        # Align index by epoch (if available)
        metr = metr.set_index("epoch", drop=False)
        st.markdown("**Метрики по эпохам**")
        st.line_chart(metr[[c for c in ["Precision", "Recall", "mAP50", "mAP50-95"] if c in metr.columns]])

        # Last row snapshot
        last = metr.iloc[-1][[c for c in ["Precision", "Recall", "mAP50", "mAP50-95"] if c in metr.columns]].to_frame(name="value")
        st.markdown("**Последняя эпоха (снимок):**")
        st.dataframe(last.T, width='stretch')


# ------------------------------
# Page body
# ------------------------------
st.title("Результаты обучения — YOLO11 Nano (face detection)")

colA, colB, colC = st.columns([1, 1, 1])
with colA:
    st.markdown(f"**Модель:** {MODEL_NAME}")
with colB:
    st.markdown(f"**Общее число эпох:** {EPOCHS_TOTAL}")
with colC:
    st.markdown(f"**Датасет (фото лиц):** {DATASET_SIZE:,}".replace(",", " "))

st.divider()

for ph in PHASES:
    _phase_block(ph["title"], ph["desc"], ph["img"], ph["csv"])
    st.markdown("")

st.divider()

st.subheader("Итоговые метрики (на лучших весах)")
ca, cb, cc, cd = st.columns(4)
ca.metric("Precision", f"{FINAL_METRICS['Precision']:.3f}")
cb.metric("Recall", f"{FINAL_METRICS['Recall']:.3f}")
cc.metric("mAP50", f"{FINAL_METRICS['mAP50']:.3f}")
cd.metric("mAP50-95", f"{FINAL_METRICS['mAP50-95']:.3f}")
