"""
app.py
------
Streamlit front-end for the AI Detection project.
Thin layer — all ML logic lives in text_detector.py and image_detector.py.

Tabs:
  1. Text Detection  — Naive Bayes + TF-IDF (pickle)
  2. Image Detection — YOUR model here (see TODO comments)

Run with:
    streamlit run app.py
"""

import streamlit as st
from PIL import Image
import numpy as np

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
    /* ── Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

    /* ── Root palette ── */
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

    /* ── Base ── */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: var(--bg) !important;
        color: var(--text) !important;
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stHeader"] { background: transparent !important; }
    [data-testid="stToolbar"] { display: none; }
    .block-container { padding-top: 2rem !important; max-width: 780px; }

    /* ── Tabs ── */
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
    [data-baseweb="tab-highlight"] { display: none !important; }
    [data-baseweb="tab-border"]    { display: none !important; }

    /* ── Text area ── */
    textarea {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        color: var(--text) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.95rem !important;
    }
    textarea:focus { border-color: var(--accent) !important; }

    /* ── Buttons ── */
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

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        background: var(--surface) !important;
        border: 2px dashed var(--border) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }

    /* ── Expander ── */
    details {
        background: var(--surface2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        padding: 0.5rem 1rem !important;
    }
    summary { color: var(--muted) !important; font-size: 0.9rem !important; }

    /* ── Divider ── */
    hr { border-color: var(--border) !important; }

    /* ── Scrollbar ── */
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

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_text, tab_image = st.tabs(["📝  Text Detection", "🖼️  Image Detection"])


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — TEXT DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_text:

    # Lazy import so image-only users don't need NLP deps
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

    analyse_clicked = st.button("Analyse Text", key="btn_text")

    if analyse_clicked:
        raw = user_text.strip()

        if not raw:
            st.warning("Please enter some text before analysing.")
        elif len(raw.split()) < 10:
            st.warning("Text is very short — try at least 10 words for a reliable result.")
        else:
            with st.spinner("Running model…"):
                result = predict_text(raw)

            label      = result["label"]        # "AI" or "Human"
            ai_prob    = result["ai_prob"]
            human_prob = result["human_prob"]
            confidence = result["confidence"]

            # ── Colour logic ──────────────────────────────────────────────────
            # Green  = Human (confident)
            # Red    = AI    (confident)
            # Yellow = uncertain (confidence < 0.60)
            UNCERTAIN_THRESHOLD = 0.60

            if confidence < UNCERTAIN_THRESHOLD:
                verdict_color  = "#eab308"   # yellow
                verdict_bg     = "#1c1a0a"
                verdict_border = "#713f12"
                verdict_text   = "UNCERTAIN"
            elif label == "Human":
                verdict_color  = "#22c55e"   # green
                verdict_bg     = "#071a0e"
                verdict_border = "#14532d"
                verdict_text   = "HUMAN-WRITTEN"
            else:
                verdict_color  = "#ef4444"   # red
                verdict_bg     = "#1a0707"
                verdict_border = "#7f1d1d"
                verdict_text   = "AI-GENERATED"

            # ── Verdict banner ────────────────────────────────────────────────
            st.markdown(
                f"""
                <div style="
                    background: {verdict_bg};
                    border: 1px solid {verdict_border};
                    border-radius: 14px;
                    padding: 1.4rem 1.8rem;
                    margin: 1.2rem 0 1rem;
                    text-align: center;
                ">
                    <div style="
                        font-family: 'Space Mono', monospace;
                        font-size: 0.75rem;
                        letter-spacing: 3px;
                        color: {verdict_color};
                        opacity: 0.75;
                        margin-bottom: 0.3rem;
                    ">VERDICT</div>
                    <div style="
                        font-family: 'Space Mono', monospace;
                        font-size: 1.7rem;
                        font-weight: 700;
                        color: {verdict_color};
                        letter-spacing: -0.5px;
                    ">{verdict_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ── Big probability cards ─────────────────────────────────────────
            col_human, col_ai, col_conf = st.columns(3)

            def prob_card(container, title, value, color, bg, border):
                container.markdown(
                    f"""
                    <div style="
                        background: {bg};
                        border: 1px solid {border};
                        border-radius: 12px;
                        padding: 1rem 0.5rem;
                        text-align: center;
                    ">
                        <div style="
                            font-size: 0.72rem;
                            letter-spacing: 2px;
                            color: {color};
                            opacity: 0.7;
                            font-family: 'Space Mono', monospace;
                            margin-bottom: 0.3rem;
                        ">{title}</div>
                        <div style="
                            font-family: 'Space Mono', monospace;
                            font-size: 2rem;
                            font-weight: 700;
                            color: {color};
                        ">{value:.0%}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            prob_card(col_human, "HUMAN", human_prob, "#22c55e", "#071a0e", "#14532d")
            prob_card(col_ai,    "AI",    ai_prob,    "#ef4444", "#1a0707", "#7f1d1d")
            prob_card(col_conf,  "CONFIDENCE", confidence, verdict_color, verdict_bg, verdict_border)

            # ── Feature breakdown (expander) ──────────────────────────────────
            top_features = result.get("top_features", [])
            if top_features:
                with st.expander("🔍 Which words influenced this prediction?"):
                    st.markdown(
                        "<p style='color:#6b7280; font-size:0.85rem; margin-bottom:0.8rem;'>"
                        "Top TF-IDF weighted terms found in your text. Higher score = "
                        "more distinctive / influential for the model."
                        "</p>",
                        unsafe_allow_html=True,
                    )

                    max_score = max(s for _, s in top_features) or 1.0
                    for word, score in top_features:
                        bar_pct = int((score / max_score) * 100)
                        st.markdown(
                            f"""
                            <div style="display:flex; align-items:center;
                                        gap:10px; margin-bottom:6px;">
                                <div style="font-family:'Space Mono',monospace;
                                            font-size:0.82rem; color:#e8eaf0;
                                            width:140px; flex-shrink:0;">
                                    {word}
                                </div>
                                <div style="flex:1; background:#1e2330;
                                            border-radius:4px; height:8px;">
                                    <div style="width:{bar_pct}%;
                                                background:{verdict_color};
                                                height:8px; border-radius:4px;
                                                opacity:0.75;">
                                    </div>
                                </div>
                                <div style="font-size:0.78rem; color:#6b7280;
                                            width:44px; text-align:right;">
                                    {score:.3f}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            st.markdown("<br>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — IMAGE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_image:

    # ── TODO: Import your image detector here ─────────────────────────────────
    # from image_detector import predict as predict_image
    # ─────────────────────────────────────────────────────────────────────────

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

        # Display uploaded image
        st.markdown("<div style='margin: 0.8rem 0;'>", unsafe_allow_html=True)
        st.image(image, use_container_width=True, caption="Uploaded image")
        st.markdown("</div>", unsafe_allow_html=True)

        analyse_img = st.button("Analyse Image", key="btn_image")

        if analyse_img:
            with st.spinner("Running model…"):

                # ── TODO: Replace this block with your real model call ─────────
                #
                # Step 1 — preprocess the PIL image for your model, e.g.:
                #   img_tensor = your_transform(image).unsqueeze(0)
                #
                # Step 2 — run inference, e.g.:
                #   result = predict_image(image)
                #   label      = result["label"]       # "AI" or "Real"
                #   ai_prob    = result["ai_prob"]
                #   real_prob  = result["real_prob"]
                #   confidence = result["confidence"]
                #
                # Step 3 — delete the placeholder below and plug in your values.
                # ─────────────────────────────────────────────────────────────

                # PLACEHOLDER — remove once your model is wired up
                label      = "PLACEHOLDER"
                ai_prob    = 0.0
                real_prob  = 0.0
                confidence = 0.0
                _is_placeholder = True

            if _is_placeholder:
                st.markdown(
                    """
                    <div style="
                        background: #161921;
                        border: 1px dashed #2a2f3d;
                        border-radius: 14px;
                        padding: 2rem 1.8rem;
                        margin: 1.2rem 0;
                        text-align: center;
                    ">
                        <div style="font-size: 2rem; margin-bottom:0.5rem;">🔧</div>
                        <div style="font-family:'Space Mono',monospace;
                                    font-size:0.85rem; color:#6b7280;
                                    letter-spacing:1px;">
                            IMAGE MODEL NOT WIRED UP YET
                        </div>
                        <div style="color:#4b5563; font-size:0.8rem; margin-top:0.5rem;">
                            See the TODO block in app.py → tab_image
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            else:
                # ── This block renders once you wire up your real model ────────
                UNCERTAIN_THRESHOLD = 0.60

                if confidence < UNCERTAIN_THRESHOLD:
                    verdict_color  = "#eab308"
                    verdict_bg     = "#1c1a0a"
                    verdict_border = "#713f12"
                    verdict_label  = "UNCERTAIN"
                elif label == "Real":
                    verdict_color  = "#22c55e"
                    verdict_bg     = "#071a0e"
                    verdict_border = "#14532d"
                    verdict_label  = "REAL PHOTO"
                else:
                    verdict_color  = "#ef4444"
                    verdict_bg     = "#1a0707"
                    verdict_border = "#7f1d1d"
                    verdict_label  = "AI-GENERATED"

                st.markdown(
                    f"""
                    <div style="
                        background: {verdict_bg};
                        border: 1px solid {verdict_border};
                        border-radius: 14px;
                        padding: 1.4rem 1.8rem;
                        margin: 1.2rem 0 1rem;
                        text-align: center;
                    ">
                        <div style="font-family:'Space Mono',monospace;
                                    font-size:0.75rem; letter-spacing:3px;
                                    color:{verdict_color}; opacity:0.75;
                                    margin-bottom:0.3rem;">VERDICT</div>
                        <div style="font-family:'Space Mono',monospace;
                                    font-size:1.7rem; font-weight:700;
                                    color:{verdict_color};">
                            {verdict_label}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                col_real, col_ai_img, col_conf_img = st.columns(3)

                def prob_card_img(container, title, value, color, bg, border):
                    container.markdown(
                        f"""
                        <div style="background:{bg}; border:1px solid {border};
                                    border-radius:12px; padding:1rem 0.5rem;
                                    text-align:center;">
                            <div style="font-size:0.72rem; letter-spacing:2px;
                                        color:{color}; opacity:0.7;
                                        font-family:'Space Mono',monospace;
                                        margin-bottom:0.3rem;">{title}</div>
                            <div style="font-family:'Space Mono',monospace;
                                        font-size:2rem; font-weight:700;
                                        color:{color};">{value:.0%}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                prob_card_img(col_real,    "REAL",       real_prob,  "#22c55e", "#071a0e", "#14532d")
                prob_card_img(col_ai_img,  "AI",         ai_prob,    "#ef4444", "#1a0707", "#7f1d1d")
                prob_card_img(col_conf_img,"CONFIDENCE", confidence, verdict_color, verdict_bg, verdict_border)

    else:
        # Empty state
        st.markdown(
            """
            <div style="
                background: #161921;
                border: 1px dashed #2a2f3d;
                border-radius: 14px;
                padding: 3rem 1.8rem;
                text-align: center;
                margin-top: 0.5rem;
            ">
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
    <div style="text-align:center; color:#374151; font-size:0.78rem; padding-bottom:1rem;">
        AI Detection Project · University Final Project
    </div>
    """,
    unsafe_allow_html=True,
)
