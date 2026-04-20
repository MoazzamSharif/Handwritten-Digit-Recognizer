import streamlit as st
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import tensorflow as tf
import matplotlib.pyplot as plt

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Digit Recognizer",
    page_icon="🔢",
    layout="centered"
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Background */
    .stApp { background: #0f0f1a; }

    /* Hide streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }

    /* Title */
    h1 {
        background: linear-gradient(90deg, #4f8bf9, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.6rem !important;
        font-weight: 800 !important;
        text-align: center;
        margin-bottom: 0 !important;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    /* Radio buttons */
    div[role="radiogroup"] {
        justify-content: center;
        gap: 12px;
    }
    div[role="radiogroup"] label {
        background: #1c1c2e;
        border: 1px solid #2d2d44;
        border-radius: 10px;
        padding: 8px 24px !important;
        color: #e0e0e0 !important;
        font-weight: 500;
        transition: all 0.2s;
    }
    div[role="radiogroup"] label:hover {
        border-color: #4f8bf9;
    }

    /* Predict button */
    div.stButton > button {
        background: linear-gradient(135deg, #4f8bf9, #7c3aed);
        color: white;
        font-size: 1.1rem;
        font-weight: 700;
        border: none;
        border-radius: 12px;
        padding: 0.65rem 1.5rem;
        width: 100%;
        transition: opacity 0.2s;
        letter-spacing: 0.5px;
    }
    div.stButton > button:hover { opacity: 0.88; }

    /* Divider */
    hr { border-color: #2d2d44 !important; }

    /* Card */
    .result-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1.5px solid #4f8bf9;
        border-radius: 20px;
        padding: 28px 20px;
        text-align: center;
        box-shadow: 0 0 30px rgba(79,139,249,0.15);
    }
    .result-digit {
        font-size: 90px;
        font-weight: 900;
        background: linear-gradient(135deg, #4f8bf9, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1;
    }
    .result-conf {
        color: #9ca3af;
        font-size: 0.95rem;
        margin-top: 8px;
    }

    /* Upload area */
    [data-testid="stFileUploader"] {
        background: #1c1c2e;
        border: 1.5px dashed #3d3d5c;
        border-radius: 14px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_model.keras")

try:
    model = load_model()
except Exception:
    st.error("❌ `mnist_model.keras` not found. Run `python train_and_save.py` first.")
    st.stop()


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(img: Image.Image, invert: bool = False) -> np.ndarray:
    """
    Resize to 28x28, optionally invert, normalize.
    MNIST = white digit on black background.
    """
    img = img.convert("L")                        # grayscale

    if invert:
        img = ImageOps.invert(img)

    # Crop to the bounding box of the digit (removes empty border)
    arr = np.array(img)
    rows = np.where(np.any(arr > 30, axis=1))[0]
    cols = np.where(np.any(arr > 30, axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        # Canvas is empty — return a blank array, prediction won't run
        return np.zeros((28, 28), dtype=np.float32)
    rmin, rmax = rows[0], rows[-1]
    cmin, cmax = cols[0], cols[-1]
    pad = 4
    rmin = max(0, rmin - pad)
    rmax = min(arr.shape[0] - 1, rmax + pad)
    cmin = max(0, cmin - pad)
    cmax = min(arr.shape[1] - 1, cmax + pad)
    img = img.crop((cmin, rmin, cmax, rmax))

    # Resize to 20x20 then pad to 28x28 (matches MNIST style)
    img = img.resize((20, 20), Image.LANCZOS)
    padded = Image.new("L", (28, 28), 0)
    padded.paste(img, (4, 4))

    arr = np.array(padded, dtype=np.float32) / 255.0
    return arr


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("<h1>🔢 Digit Recognizer</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">Draw a digit or upload an image — AI predicts it instantly</p>',
            unsafe_allow_html=True)

# ── Input method ──────────────────────────────────────────────────────────────
method = st.radio("", ["✏️ Draw", "📁 Upload Image"], horizontal=True, label_visibility="collapsed")
st.markdown("")

img_array = None

# ─── DRAW ─────────────────────────────────────────────────────────────────────
if method == "✏️ Draw":
    try:
        from streamlit_drawable_canvas import st_canvas

        col_c, col_hint = st.columns([3, 1])
        with col_c:
            canvas = st_canvas(
                fill_color="black",
                stroke_width=22,
                stroke_color="white",
                background_color="#111111",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="canvas",
            )
        with col_hint:
            st.markdown("""
            <div style="color:#6b7280; font-size:0.82rem; padding-top:20px; line-height:1.8;">
            ✏️ Draw a digit<br>
            0 – 9<br><br>
            Draw it <b style='color:#9ca3af'>large</b><br>
            and <b style='color:#9ca3af'>centered</b><br>
            for best results.
            </div>
            """, unsafe_allow_html=True)

        if canvas.image_data is not None:
            rgba = canvas.image_data.astype("uint8")
            img_pil = Image.fromarray(rgba, "RGBA").convert("L")
            img_array = preprocess(img_pil, invert=False)  # canvas: white on black ✓

    except ModuleNotFoundError:
        st.error("Run: `pip install streamlit-drawable-canvas` then restart Streamlit.")

# ─── UPLOAD ───────────────────────────────────────────────────────────────────
else:
    uploaded = st.file_uploader("", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

    if uploaded:
        img_pil = Image.open(uploaded).convert("L")
        arr_raw = np.array(img_pil)
        # If image is light background (black digit on white) → invert
        needs_invert = arr_raw.mean() > 127

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<p style="color:#9ca3af;text-align:center;font-size:0.85rem;">Original</p>',
                        unsafe_allow_html=True)
            st.image(img_pil, use_container_width=True)
        with col_b:
            processed = preprocess(img_pil, invert=needs_invert)
            preview = Image.fromarray((processed * 255).astype("uint8")).resize((140, 140), Image.NEAREST)
            st.markdown('<p style="color:#9ca3af;text-align:center;font-size:0.85rem;">Processed (28×28)</p>',
                        unsafe_allow_html=True)
            st.image(preview, use_container_width=True)

        img_array = processed

# ── Predict button ────────────────────────────────────────────────────────────
st.markdown("")
predict = st.button("🔮 Predict Digit", type="primary")

if predict:
    if img_array is None or img_array.max() < 0.05:
        st.warning("Please draw or upload a digit first.")
    else:
        probs = model.predict(img_array.reshape(1, 28, 28), verbose=0)[0]
        pred  = int(np.argmax(probs))
        conf  = float(probs[pred]) * 100

        st.markdown("---")

        col_res, col_chart = st.columns([1, 1.6])

        with col_res:
            st.markdown(f"""
            <div class="result-card">
                <div style="color:#6b7280;font-size:0.8rem;letter-spacing:2px;text-transform:uppercase;margin-bottom:6px;">Prediction</div>
                <div class="result-digit">{pred}</div>
                <div class="result-conf">{'⬛' * (round(conf/10))}{'░' * (10 - round(conf/10))}</div>
                <div class="result-conf" style="margin-top:6px;">{conf:.1f}% confidence</div>
            </div>
            """, unsafe_allow_html=True)

        with col_chart:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            fig.patch.set_facecolor("none")
            ax.set_facecolor("none")

            colors = ["#4f8bf9" if i == pred else "#2a2a3e" for i in range(10)]
            edge   = ["#7c3aed" if i == pred else "none" for i in range(10)]
            bars   = ax.barh(range(10), probs * 100, color=colors,
                             edgecolor=edge, linewidth=1.2, height=0.65)

            for bar, p in zip(bars, probs):
                if p * 100 > 1.5:
                    ax.text(p * 100 + 0.8, bar.get_y() + bar.get_height() / 2,
                            f"{p*100:.1f}%", va="center", color="white", fontsize=7.5)

            ax.set_yticks(range(10))
            ax.set_yticklabels([str(i) for i in range(10)], color="#9ca3af", fontsize=10)
            ax.set_xlim(0, 115)
            ax.tick_params(axis="x", colors="#3d3d5c")
            ax.spines[:].set_visible(False)
            ax.set_xlabel("Confidence (%)", color="#6b7280", fontsize=9)
            ax.xaxis.label.set_color("#6b7280")
            ax.tick_params(axis="x", labelcolor="#6b7280")

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
