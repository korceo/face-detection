import io
import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import requests
from ultralytics import YOLO


# -------------------------------
# Constants and defaults
# -------------------------------
DEFAULT_IMAGES = [
    "imgs/stadium.png",
    "imgs/city.png",
    "imgs/subway.png",
]
MODEL_OPTIONS = {
    "YOLO11n (faces)": "weights/yolo11n-faces.pt",
    "YOLO11s WIDERFACE (faces)": "weights/yolo11s_widerface.pt",
}

# -------------------------------
# Utility functions
# -------------------------------
@st.cache_resource(show_spinner=False)
def load_model(weights_path: str) -> YOLO:
    """Cache model instance by weights path."""
    return YOLO(weights_path)


def _open_image_from_path(path: str) -> Optional[Image.Image]:
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception:
        return None


def _open_image_from_url(url: str, timeout: int = 10) -> Optional[Image.Image]:
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None


def _open_image_from_upload(upload) -> Optional[Image.Image]:
    try:
        return Image.open(upload).convert("RGB")
    except Exception:
        return None


from typing import Optional  # already imported at top

def pick_face_like_class(names: dict) -> Optional[str]:
    """Возвращаем только класс 'face' если он есть, иначе не фильтруем (None)."""
    inv = {int(k): v for k, v in names.items()} if isinstance(names, dict) else names
    if isinstance(inv, dict):
        return "face" if "face" in set(inv.values()) else None
    # inv is list/tuple
    return "face" if (isinstance(inv, (list, tuple)) and "face" in inv) else None


def run_detection(model: YOLO, image: Image.Image, conf: float, iou: float, target_class: Optional[str], suppress_superbox: bool, median_k: Optional[float] = None) -> Tuple[List[Tuple[int,int,int,int,float,str]], dict]:
    """
    Returns list of detections: [(x1,y1,x2,y2, conf, name), ...] and names map.
    """
    # Ultralytics expects numpy array or path; PIL is fine via numpy
    results = model.predict(image, conf=conf, iou=iou, verbose=False)
    r = results[0]
    names = r.names
    boxes_xyxy = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
    clses = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else np.zeros((0,), dtype=int)
    confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros((0,))

    dets: List[Tuple[int,int,int,int,float,str]] = []
    W, H = image.size

    for (x1, y1, x2, y2), c, p in zip(boxes_xyxy, clses, confs):
        name = names[int(c)] if isinstance(names, (list, tuple)) else names.get(int(c), str(c))
        if target_class is None or name == target_class:
            # Clamp to image bounds and convert to ints
            xi1 = int(max(0, min(W - 1, x1)))
            yi1 = int(max(0, min(H - 1, y1)))
            xi2 = int(max(0, min(W - 1, x2)))
            yi2 = int(max(0, min(H - 1, y2)))
            if xi2 > xi1 and yi2 > yi1:
                dets.append((xi1, yi1, xi2, yi2, float(p), str(name)))

    # ---------------- Super-box suppression ----------------
    if suppress_superbox and len(dets) >= 8:
        areas = np.array([(d[2] - d[0]) * (d[3] - d[1]) for d in dets], dtype=np.float64)
        med = float(np.median(areas)) if areas.size else 0.0
        if med > 0:
            centers = np.array([((d[0] + d[2]) * 0.5, (d[1] + d[3]) * 0.5) for d in dets], dtype=np.float64)
            keep: List[Tuple[int,int,int,int,float,str]] = []
            for i, (x1, y1, x2, y2, p, name) in enumerate(dets):
                area_i = areas[i]
                # Критерии: очень большой относительно медианы и содержит много других центров
                if area_i >= 8.0 * med:
                    cx, cy = centers[:, 0], centers[:, 1]
                    inside = (cx > x1) & (cx < x2) & (cy > y1) & (cy < y2)
                    count_inside = int(inside.sum()) - 1  # вычитаем сам бокс
                    if count_inside >= 6:
                        continue  # подавляем супер-бокс
                keep.append((x1, y1, x2, y2, p, name))
            dets = keep

    return dets, names


def draw_bboxes(image: Image.Image, dets: List[Tuple[int,int,int,int,float,str]]) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for (x1, y1, x2, y2, p, name) in dets:
        # rectangle
        draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 0), width=2)
        # label background
        label = f"{name} {p:.2f}"
        tw, th = draw.textbbox((0, 0), label, font=font)[2:]
        box_h = th + 4
        box_w = tw + 6
        draw.rectangle([(x1, max(0, y1 - box_h)), (x1 + box_w, y1)], fill=(0, 255, 0))
        draw.text((x1 + 3, y1 - box_h + 2), label, fill=(0, 0, 0), font=font)
    return out


def blur_regions(image: Image.Image, dets: List[Tuple[int,int,int,int,float,str]], radius: int = 16) -> Image.Image:
    if not dets:
        return image.copy()
    out = image.copy()
    for (x1, y1, x2, y2, _, _) in dets:
        crop = out.crop((x1, y1, x2, y2))
        crop_blur = crop.filter(ImageFilter.GaussianBlur(radius=radius))
        out.paste(crop_blur, (x1, y1))
    return out


def read_images(uploads, urls_text: str, use_defaults: bool) -> List[Tuple[str, Image.Image]]:
    images: List[Tuple[str, Image.Image]] = []

    # 1) Uploaded files
    for up in uploads or []:
        im = _open_image_from_upload(up)
        if im is not None:
            images.append((up.name, im))

    # 2) URLs (one per line)
    urls = [u.strip() for u in (urls_text or "").splitlines() if u.strip()]
    for url in urls:
        im = _open_image_from_url(url)
        if im is not None:
            images.append((url, im))

    # 3) Defaults
    if use_defaults or not images:
        for p in DEFAULT_IMAGES:
            im = _open_image_from_path(p)
            if im is not None:
                images.append((p, im))

    return images


# -------------------------------
# UI
# -------------------------------
st.set_page_config(
    page_title="Face Blur (YOLO11)",
    layout="wide",
)

st.title("Загрузка фото → детектирование → блюр лиц")

with st.sidebar:
    st.header("Настройки")
    model_label = st.selectbox("Модель", list(MODEL_OPTIONS.keys()), index=1)
    weights_path = MODEL_OPTIONS[model_label]

    conf = st.slider("Конфиденс (conf)", 0.05, 0.95, 0.35, 0.01)
    iou = st.slider("IoU NMS", 0.1, 0.9, 0.5, 0.05)
    blur_radius = st.slider("Сила блюра (px)", 4, 64, 18, 1)
    suppress_superbox = st.checkbox("Подавлять крупный общий бокс", value=True, help="Удалять огромные боксы, которые накрывают множество отдельных лиц.")

    st.caption("Детектируются только лица.")

col_u1, col_u2 = st.columns([2, 1])
with col_u1:
    uploads = st.file_uploader(
        "Загрузите изображения (можно несколько)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )
with col_u2:
    urls_text = st.text_area("Или вставьте ссылки (по одной на строку)")
    use_defaults = st.checkbox(
        "Использовать дефолтные изображения (imgs/stadium.png, imgs/city.png, imgs/subway.png)",
        value=True,
    )

run = st.button("Обработать", type="primary")

if run:
    # Load model once
    if not Path(weights_path).exists():
        st.error(f"Не найден файл весов: {weights_path}")
        st.stop()

    model = load_model(weights_path)

    images = read_images(uploads, urls_text, use_defaults)
    if not images:
        st.warning("Нет доступных изображений для обработки.")
        st.stop()

    # Decide target class once based on model names
    target = pick_face_like_class(model.model.names if hasattr(model, "model") else getattr(model, "names", {}))

    st.write(f"Класс для блюра: **{target}**")

    for idx, (src, img) in enumerate(images, start=1):
        st.markdown(f"### Изображение {idx}: `{src}`")
        dets, names = run_detection(model, img, conf=conf, iou=iou, target_class=target, suppress_superbox=suppress_superbox)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.caption("Оригинал")
            st.image(img, width='stretch')
        with c2:
            st.caption("Bounding boxes")
            boxed = draw_bboxes(img, dets)
            st.image(boxed, width='stretch')
            st.caption(f"Найдено: {len(dets)}")
        with c3:
            st.caption("Заблюрено")
            blurred = blur_regions(img, dets, radius=blur_radius)
            st.image(blurred, width='stretch')
            buf = io.BytesIO()
            blurred.save(buf, format="PNG")
            st.download_button(
                label="Скачать PNG",
                data=buf.getvalue(),
                file_name=f"blurred_{idx}.png",
                mime="image/png",
            )
else:
    st.info(
        "Загрузите файлы или вставьте ссылки, при необходимости оставьте дефолтные картинки. Затем нажмите **Обработать**."
    )

