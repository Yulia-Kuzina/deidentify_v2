import io
import zipfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from paddleocr import PaddleOCR
import easyocr

# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_paddle():
    return PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False, show_log=False)

@st.cache_resource
def get_easy():
    return easyocr.Reader(['en'], gpu=False)


# ══════════════════════════════════════════════════════════════════════════════
# PANELS pipeline
# ══════════════════════════════════════════════════════════════════════════════

def find_panels(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    hue = hsv[:, :, 0]
    ui_color_mask = (sat > 40) & (hue > 15) & (hue < 35)
    gray_filtered = gray.copy()
    gray_filtered[ui_color_mask] = 0

    col_mean = np.mean(gray_filtered, axis=0)
    row_mean = np.mean(gray_filtered, axis=1)
    threshold = 8
    col_bright = col_mean > threshold
    row_bright = row_mean > threshold

    def get_ranges(bright_arr, min_size=50):
        ranges = []
        in_range = False
        start = 0
        for i, b in enumerate(bright_arr):
            if b and not in_range:
                start = i
                in_range = True
            elif not b and in_range:
                if i - start > min_size:
                    ranges.append((start, i))
                in_range = False
        if in_range and len(bright_arr) - start > min_size:
            ranges.append((start, len(bright_arr)))
        return ranges

    col_ranges = get_ranges(col_bright, min_size=50)
    row_ranges = get_ranges(row_bright, min_size=50)
    if not row_ranges:
        return []

    y1, y2 = row_ranges[0][0], row_ranges[-1][1]
    panels = []
    for x1, x2 in col_ranges:
        region = gray_filtered[y1:y2, x1:x2]
        mean_brightness = np.mean(region)
        width = x2 - x1
        if mean_brightness < 12 or width < 80:
            continue
        height = y2 - y1
        if width > height * 1.5:
            continue
        panels.append((x1, y1, x2, y2))

    if panels:
        x1, _, x2, _ = panels[0]
        for y in range(y2, y1, -1):
            row_b = np.mean(gray_filtered[y, x1:x2])
            if row_b > 15:
                y2 = y + 5
                break
        panels = [(x1, y1, x2, y2) for (x1, y1, x2, _) in panels]

    trimmed_panels = []
    for (x1, y1, x2, y2) in panels:
        new_y1 = y1
        for y in range(y1, min(y1 + 40, y2)):
            row = img[y, x1:x2]
            hsv_row = cv2.cvtColor(row.reshape(1, -1, 3), cv2.COLOR_BGR2HSV)
            hue_r = hsv_row[0, :, 0]
            sat_r = hsv_row[0, :, 1]
            yellow_ratio = np.mean((sat_r > 40) & (hue_r > 15) & (hue_r < 35))
            if yellow_ratio > 0.1:
                new_y1 = y + 1
        trimmed_panels.append((x1, new_y1, x2, y2))
    return trimmed_panels


def build_panel_mask(img, panels):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for (x1, y1, x2, y2) in panels:
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return mask


def enhance_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return [
        img,
        cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR),
        cv2.bitwise_not(img),
        cv2.convertScaleAbs(img, alpha=1.5, beta=30),
    ]


def get_ocr_mask(crop):
    ocr = get_paddle()
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    for version in enhance_for_ocr(crop):
        result = ocr.ocr(version, cls=True)
        if not result or result[0] is None:
            continue
        for line in result[0]:
            if line[1][1] < 0.3:
                continue
            pts = np.array(line[0], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
    k = np.ones((7, 7), np.uint8)
    return cv2.dilate(mask, k, iterations=2)


def inpaint_simple(img, mask):
    return cv2.inpaint(img, mask, inpaintRadius=4, flags=cv2.INPAINT_TELEA)


def process_panels(img):
    panels = find_panels(img)
    if not panels:
        return img

    panel_mask = build_panel_mask(img, panels)
    result = img.copy()
    result[panel_mask == 0] = 0

    if len(panels) == 4:
        x1, y1, x2, y2 = panels[2]
        result[y1:y2, x1:x2] = 0
        panels = [panels[0], panels[1], panels[3]]

    for (x1, y1, x2, y2) in panels:
        crop = img[y1:y2, x1:x2].copy()
        text_mask = get_ocr_mask(crop)
        if cv2.countNonZero(text_mask) > 0:
            result[y1:y2, x1:x2] = inpaint_simple(crop, text_mask)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# FRAMES pipeline
# ══════════════════════════════════════════════════════════════════════════════

CONFIDENCE_THRESHOLD = 0.05
INPAINT_RADIUS       = 10


def remove_text(img):
    reader = get_easy()
    results = reader.readtext(img, low_text=0.3, text_threshold=0.5)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    n = 0
    for (bbox, text, conf) in results:
        if conf < CONFIDENCE_THRESHOLD or not text.strip():
            continue
        pts = np.array(bbox, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        n += 1
    if n == 0:
        return img
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.dilate(mask, kernel, iterations=2)
    return cv2.inpaint(img, mask, INPAINT_RADIUS, cv2.INPAINT_TELEA)


def crop_black_frame(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    _, mask = cv2.threshold(gray, 18, 255, cv2.THRESH_BINARY)
    ksize = max(40, int(min(H, W) * 0.05))
    ksize += (ksize % 2 == 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    if w * h < 0.30 * H * W:
        return img
    PAD = 3
    x  = max(0, x - PAD)
    y  = max(0, y - PAD)
    x2 = min(W, x + w + 2 * PAD)
    y2 = min(H, y + h + 2 * PAD)
    return img[y:y2, x:x2]


def process_frame(img):
    img = crop_black_frame(img)
    return remove_text(img)


# ══════════════════════════════════════════════════════════════════════════════
# Dispatch
# ══════════════════════════════════════════════════════════════════════════════

PANEL_THRESHOLD = 2

def process_image(img):
    panels = find_panels(img)
    if len(panels) >= PANEL_THRESHOLD:
        return process_panels(img), "panels"
    else:
        return process_frame(img), "frame"


def img_to_png_bytes(img):
    success, buf = cv2.imencode(".png", img)
    return buf.tobytes() if success else None


def bytes_to_cv2(data):
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ══════════════════════════════════════════════════════════════════════════════
# Streamlit UI
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Medical Image De-identifier",
    page_icon="🏥",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.banner {
    background: linear-gradient(90deg, #003366 0%, #005599 100%);
    padding: 2rem 2.5rem 1.5rem 2.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
}
.banner h1 {
    color: #ffffff;
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0 0 0.2rem 0;
    letter-spacing: 0.03em;
}
.banner p {
    color: #c8ddf2;
    font-size: 1.15rem;
    margin: 0;
}
.banner .subtitle {
    color: #a0c0e0;
    font-size: 1.5rem;
    margin-top: 0.3rem;
}
</style>
""", unsafe_allow_html=True)

# Centered content container
_, center, _ = st.columns([1, 3, 1])

with center:
    st.markdown("""
    <div class="banner">
        <h1>Clinicum Digitale</h1>
        <div class="subtitle">🏥 Medical Image De-identifier</div>
        <p>Automatically removes identifying text from medical images.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload image(s)",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("🚀 Process", type="primary"):
            results = []

            for uploaded in uploaded_files:
                with st.spinner(f"Processing {uploaded.name}…"):
                    raw = uploaded.read()
                    img = bytes_to_cv2(raw)
                    if img is None:
                        st.error(f"Could not read {uploaded.name}")
                        continue

                    out_img, mode = process_image(img)
                    out_bytes = img_to_png_bytes(out_img)
                    results.append((uploaded.name, raw, out_bytes, mode))

            if results:
                st.divider()

                for name, orig_bytes, out_bytes, mode in results:
                    st.subheader(name)
                    st.caption(f"Pipeline: **{mode}**")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Original**")
                        st.image(orig_bytes, use_container_width=True)
                    with col2:
                        st.markdown("**De-identified**")
                        st.image(out_bytes, use_container_width=True)

                    st.download_button(
                        label=f"⬇️ Download de-identified {name}",
                        data=out_bytes,
                        file_name=f"deidentified_{Path(name).stem}.png",
                        mime="image/png",
                        key=name,
                    )
                    st.divider()

                if len(results) > 1:
                    zip_buf = io.BytesIO()
                    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                        for name, _, out_bytes, _ in results:
                            zf.writestr(f"deidentified_{Path(name).stem}.png", out_bytes)
                    zip_buf.seek(0)
                    st.download_button(
                        label=f"⬇️ Download all ({len(results)} files as ZIP)",
                        data=zip_buf,
                        file_name="deidentified_images.zip",
                        mime="application/zip",
                        key="zip_all",
                    )