"""
app.py
------
Streamlit front-end for the AI Detection project.
Thin layer — all ML logic lives in text_detector.py and image_detector.py.

Tabs:
  1. Text Detection  — Naive Bayes + TF-IDF (pickle)
  2. Image Detection — ResNet18 fine-tuned on CIFAKE (.pth)

Run with:
    streamlit run app.py
"""

import streamlit as st
from PIL import Image

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="AI Detector",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

    :root {
        --bg:          #0d0f14;
        --surface:     #161921;
        --surface2:    #1e2330;
        --border:      #2a2f3d;
        --text:        #e8eaf0;
        --muted:       #6b7280;
        --green:       #22c55e;
        --green-dim:   #14532d;
        --red:         #ef4444;
        --red-dim:     #7f1d1d;
        --yellow:      #eab308;
        --yellow-dim:  #713f12;
        --accent:      #6366f1;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background-color: var(--bg) !important;
        color: var(--text) !important;
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stHeader"]  { background: transparent !important; }
    [data-testid="stToolbar"] { display: none; }
    .block-container { padding-top: 2rem !important; max-width: 780px; }

    /* Tabs */
    [data-baseweb="tab-list"] {
        background: var(--surface) !important;
        border-radius: 12px;
        padding: 4px;
        border: 1px solid var(--border);
        gap: 4px;
    }
    [data-baseweb="tab"] {
        background: transparent !important;
        color: var(--muted) !important;
        border-radius: 8px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        padding: 8px 24px !important;
        border: none !important;
    }
    [aria-selected="true"] {
        background: var(--accent) !important;
        color: white !important;
    }
    [data-baseweb="tab-highlight"],
    [data-baseweb="tab-border"] { display: none !important; }

    /* Text area */
    textarea {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        color: var(--text) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.95rem !important;
    }
    textarea:focus { border-color: var(--accent) !important; }

    /* Buttons */
    .stButton > button {
        background: var(--accent) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 0.6rem 2rem !important;
        transition: opacity 0.2s;
        width: 100%;
    }
    .stButton > button:hover { opacity: 0.85; }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: var(--surface) !important;
        border: 2px dashed var(--border) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }

    /* Expander */
    details {
        background: var(--surface2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        padding: 0.5rem 1rem !important;
    }
    summary { color: var(--muted) !important; font-size: 0.9rem !important; }

    hr { border-color: var(--border) !important; }

    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align:center; padding: 1.5rem 0 1rem;">
        <span style="font-family:'Space Mono',monospace; font-size:1.8rem;
                     font-weight:700; color:#e8eaf0; letter-spacing:-1px;">
            AI<span style="color:#6366f1;">Detect</span>
        </span>
        <p style="color:#6b7280; font-size:0.9rem; margin-top:0.3rem;">
            Classify text and images as human-made or AI-generated
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Shared helpers ────────────────────────────────────────────────────────────

def verdict_colors(label, human_or_real_prob):
    """Return (color, bg, border, verdict_text) based on label and win probability."""
    UNCERTAIN = 0.60
    if human_or_real_prob < UNCERTAIN and (1 - human_or_real_prob) < UNCERTAIN:
        return "#eab308", "#1c1a0a", "#713f12", "UNCERTAIN"
    if label in ("Human", "Real"):
        suffix = "HUMAN-WRITTEN" if label == "Human" else "REAL PHOTO"
        return "#22c55e", "#071a0e", "#14532d", suffix
    return "#ef4444", "#1a0707", "#7f1d1d", "AI-GENERATED"


def render_verdict_banner(verdict_text, color, bg, border):
    st.markdown(
        f"""
        <div style="background:{bg}; border:1px solid {border};
                    border-radius:14px; padding:1.4rem 1.8rem;
                    margin:1.2rem 0 1rem; text-align:center;">
            <div style="font-family:'Space Mono',monospace; font-size:0.75rem;
                        letter-spacing:3px; color:{color}; opacity:0.75;
                        margin-bottom:0.3rem;">VERDICT</div>
            <div style="font-family:'Space Mono',monospace; font-size:1.7rem;
                        font-weight:700; color:{color};">{verdict_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prob_card(container, title, value, color, bg, border):
    container.markdown(
        f"""
        <div style="background:{bg}; border:1px solid {border};
                    border-radius:12px; padding:1rem 0.5rem; text-align:center;">
            <div style="font-size:0.72rem; letter-spacing:2px; color:{color};
                        opacity:0.7; font-family:'Space Mono',monospace;
                        margin-bottom:0.3rem;">{title}</div>
            <div style="font-family:'Space Mono',monospace; font-size:2rem;
                        font-weight:700; color:{color};">{value:.0%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_text, tab_image = st.tabs(["📝  Text Detection", "🖼️  Image Detection"])


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — TEXT DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_text:

    from text_detector import predict as predict_text

    st.markdown(
        "<p style='color:#6b7280; font-size:0.88rem; margin-bottom:1rem;'>"
        "Paste any text below — essay, article, email, social post — and the model "
        "will tell you whether it reads as human-written or AI-generated."
        "</p>",
        unsafe_allow_html=True,
    )

    user_text = st.text_area(
        label="Input text",
        placeholder="Paste your text here…",
        height=200,
        label_visibility="collapsed",
    )

    if st.button("Analyse Text", key="btn_text"):
        raw = user_text.strip()

        if not raw:
            st.warning("Please enter some text before analysing.")
        elif len(raw.split()) < 10:
            st.warning("Text is very short — try at least 10 words for a reliable result.")
        else:
            with st.spinner("Running model…"):
                result = predict_text(raw)

            label      = result["label"]
            ai_prob    = result["ai_prob"]
            human_prob = result["human_prob"]

            color, bg, border, verdict_text = verdict_colors(label, human_prob)

            render_verdict_banner(verdict_text, color, bg, border)

            # Two cards — confidence removed (it's just whichever prob is higher)
            col_human, col_ai = st.columns(2)
            render_prob_card(col_human, "HUMAN", human_prob, "#22c55e", "#071a0e", "#14532d")
            render_prob_card(col_ai,    "AI",    ai_prob,    "#ef4444", "#1a0707", "#7f1d1d")

            # Feature breakdown
            top_features = result.get("top_features", [])
            if top_features:
                with st.expander("🔍 Which words influenced this prediction?"):
                    st.markdown(
                        "<p style='color:#6b7280; font-size:0.85rem; margin-bottom:0.8rem;'>"
                        "Top TF-IDF weighted terms found in your text. "
                        "<span style='color:#22c55e;'>Green</span> = word leans Human-written. "
                        "<span style='color:#ef4444;'>Red</span> = word leans AI-generated."
                        "</p>",
                        unsafe_allow_html=True,
                    )
                    max_score = max(s for _, s, _ in top_features) or 1.0
                    for word, score, tendency in top_features:
                        bar_pct = int((score / max_score) * 100)
                        bar_color = "#22c55e" if tendency == "Human" else "#ef4444"
                        label_color = "#22c55e" if tendency == "Human" else "#ef4444"
                        st.markdown(
                            f"""
                            <div style="display:flex; align-items:center;
                                        gap:10px; margin-bottom:6px;">
                                <div style="font-family:'Space Mono',monospace;
                                            font-size:0.82rem; color:{label_color};
                                            width:140px; flex-shrink:0;">{word}</div>
                                <div style="flex:1; background:#1e2330;
                                            border-radius:4px; height:8px;">
                                    <div style="width:{bar_pct}%; background:{bar_color};
                                                height:8px; border-radius:4px;
                                                opacity:0.75;"></div>
                                </div>
                                <div style="font-size:0.78rem; color:#6b7280;
                                            width:44px; text-align:right;">
                                    {score:.3f}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            with st.expander("📊 About this model"):
                st.markdown(
                    """
                    <div style='color:#6b7280; font-size:0.85rem; line-height:1.7;'>
                    <b style='color:#e8eaf0;'>Model:</b> Multinomial Naive Bayes + TF-IDF (10k features, 1–3 n-grams)<br>
                    <b style='color:#e8eaf0;'>Dataset:</b> AI_Human — 487,235 samples (62.8% human, 37.2% AI)<br>
                    <b style='color:#e8eaf0;'>Test accuracy:</b> <span style='color:#22c55e;'>96%</span> on 97,447 held-out samples<br>
                    <b style='color:#e8eaf0;'>Precision / Recall / F1:</b> 0.96 / 0.96 / 0.96 (weighted avg)<br>
                    <b style='color:#e8eaf0;'>Limitation:</b> May be less reliable on heavily edited or very short AI text.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("<br>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — IMAGE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_image:

    from image_detector import predict as predict_image

    st.markdown(
        "<p style='color:#6b7280; font-size:0.88rem; margin-bottom:1rem;'>"
        "Upload an image and the model will determine whether it was photographed "
        "by a human or generated by AI."
        "</p>",
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        label="Upload image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.markdown("<div style='margin: 0.8rem 0;'>", unsafe_allow_html=True)
        st.image(image, use_container_width=True, caption="Uploaded image")
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("Analyse Image", key="btn_image"):
            with st.spinner("Running model…"):
                result = predict_image(image)

            label     = result["label"]      # "AI" or "Real"
            ai_prob   = result["ai_prob"]
            real_prob = result["real_prob"]

            color, bg, border, verdict_text = verdict_colors(label, real_prob)

            render_verdict_banner(verdict_text, color, bg, border)

            # Two probability cards
            col_real, col_ai = st.columns(2)
            render_prob_card(col_real, "REAL", real_prob, "#22c55e", "#071a0e", "#14532d")
            render_prob_card(col_ai,   "AI",   ai_prob,   "#ef4444", "#1a0707", "#7f1d1d")

            # Grad-CAM heatmap
            from image_detector import get_gradcam
            pred_class = 1 if label == "Real" else 0
            with st.spinner("Generating explanation heatmap…"):
                cam_image = get_gradcam(image, pred_class)

            if cam_image is not None:
                with st.expander("🔥 Why did the model predict this? (Grad-CAM)"):
                    st.markdown(
                        "<p style='color:#6b7280; font-size:0.85rem; margin-bottom:0.8rem;'>"
                        "Grad-CAM highlights the image regions that most influenced the prediction. "
                        "<span style='color:#ef4444;'>Red/warm areas</span> = high influence. "
                        "<span style='color:#6b7280;'>Blue/cool areas</span> = low influence."
                        "</p>",
                        unsafe_allow_html=True,
                    )
                    col_orig, col_cam = st.columns(2)
                    col_orig.image(image, caption="Original", use_container_width=True)
                    col_cam.image(cam_image, caption="Grad-CAM Heatmap", use_container_width=True)

            with st.expander("📊 About this model"):
                st.markdown(
                    """
                    <div style='color:#6b7280; font-size:0.85rem; line-height:1.7;'>
                    <b style='color:#e8eaf0;'>Model:</b> ResNet18 fine-tuned on CIFAKE (ImageNet pretrained)<br>
                    <b style='color:#e8eaf0;'>Dataset:</b> CIFAKE — real CIFAR-10 photos vs Stable Diffusion generated images<br>
                    <b style='color:#e8eaf0;'>Test accuracy:</b> <span style='color:#22c55e;'>98%</span> on held-out CIFAKE test set<br>
                    <b style='color:#e8eaf0;'>Architecture:</b> ResNet18, FC layer replaced (512 → 2 classes)<br>
                    <b style='color:#e8eaf0;'>Limitation:</b> Trained on 32×32 CIFAR-style images — may be less reliable on high-resolution photos.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    else:
        st.markdown(
            """
            <div style="background:#161921; border:1px dashed #2a2f3d;
                        border-radius:14px; padding:3rem 1.8rem;
                        text-align:center; margin-top:0.5rem;">
                <div style="font-size:2.5rem; margin-bottom:0.6rem;">📁</div>
                <div style="color:#4b5563; font-size:0.9rem;">
                    Drop a JPG, PNG, or WEBP image to get started
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <hr style="margin-top:2rem;">
    <div style="text-align:center; color:#374151; font-size:0.78rem;
                padding-bottom:1rem;">
        AI Detection Project · University Final Project
    </div>
    """,
    unsafe_allow_html=True,
)