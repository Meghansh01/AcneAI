import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import json
import os

CLASS_NAMES = ["Blackheads", "Whiteheads", "Papules", "Pustules", "Cyst"]
SEVERITY = {
    "Blackheads":("Mild",     "#27AE60","🟢"),
    "Whiteheads":("Mild",     "#27AE60","🟢"),
    "Papules":   ("Moderate", "#F39C12","🟡"),
    "Pustules":  ("Moderate", "#F39C12","🟡"),
    "Cyst":      ("Severe",   "#E74C3C","🔴"),
}
ADVICE = {
    "Blackheads": "Use salicylic acid cleanser daily. Avoid heavy face oils.",
    "Whiteheads": "Non-comedogenic moisturizer. Gentle cleanser. Try retinoids.",
    "Papules":    "Apply benzoyl peroxide. Do NOT squeeze or pop.",
    "Pustules":   "Topical antibiotics. Keep hands off face. See dermatologist.",
    "Cyst":       "See dermatologist immediately. May need professional drainage."
}
DESCRIPTION = {
    "Blackheads": "Open comedones. Dark spots caused by oxidized melanin.",
    "Whiteheads": "Closed comedones. Small white bumps under skin surface.",
    "Papules":    "Inflamed red bumps without pus. Tender to touch.",
    "Pustules":   "Pus-filled red bumps. Classic pimples.",
    "Cyst":       "Deep pus-filled lesion. Most severe. Can cause scarring."
}

st.set_page_config(page_title="AcneAI Fusion", page_icon="🧴", layout="wide")

st.markdown("""
<style>
.main-title{font-size:2.4rem;font-weight:bold;text-align:center;
            background:linear-gradient(135deg,#667eea,#764ba2);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.sub{text-align:center;color:#888;margin-bottom:1rem;font-size:1rem;}
.fusion-badge{background:linear-gradient(135deg,#667eea,#764ba2);
              color:white;padding:0.3rem 1rem;border-radius:20px;
              font-weight:bold;font-size:0.85rem;display:inline-block;}
.advice{background:#EBF5FB;border-left:4px solid #3498DB;
        padding:1rem;border-radius:0 10px 10px 0;margin:0.8rem 0;}
.model-card{background:white;padding:0.8rem;border-radius:10px;
            text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.08);
            margin:0.3rem 0;}
</style>
""", unsafe_allow_html=True)

# ── Project paths ──
MODELS_DIR = r"C:/Users/Kartik/OneDrive/Desktop/acne_project/models"

# ── Load fusion config ──
with open(MODELS_DIR + "/fusion_config.json") as f:
    fusion_config = json.load(f)

w_b0     = fusion_config["models"]["efficientnetb0"]["weight"]
w_b2     = fusion_config["models"]["efficientnetb2"]["weight"]
w_resnet = fusion_config["models"]["resnet50"]["weight"]

@st.cache_resource
def load_all_models():
    b0     = tf.keras.models.load_model(MODELS_DIR + "/best_acne_model.keras")
    b2     = tf.keras.models.load_model(MODELS_DIR + "/efficientnetb2_acne.keras")
    resnet = tf.keras.models.load_model(MODELS_DIR + "/resnet50_acne.keras")
    return b0, b2, resnet

def preprocess(image, size):
    img = image.resize(size)
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, 0)

def fusion_predict(image, model_b0, model_b2, model_resnet):
    img_224 = preprocess(image, (224, 224))
    img_260 = preprocess(image, (260, 260))
    p_b0     = model_b0.predict(img_224, verbose=0)[0]
    p_b2     = model_b2.predict(img_260, verbose=0)[0]
    p_resnet = model_resnet.predict(img_224, verbose=0)[0]
    # Weighted fusion
    fused    = (w_b0 * p_b0) + (w_b2 * p_b2) + (w_resnet * p_resnet)
    return fused, p_b0, p_b2, p_resnet

# ── Sidebar ──
with st.sidebar:
    st.markdown("## 🧴 AcneAI")
    st.markdown('<span class="fusion-badge">⚡ FUSION MODEL</span>', unsafe_allow_html=True)
    st.markdown("---")
    threshold = st.slider("Confidence Threshold", 0.3, 0.95, 0.5, 0.05)
    show_individual = st.checkbox("Show individual model predictions", value=True)
    st.markdown("---")
    st.markdown("**Ensemble Models**")

    model_weights = {
        "EfficientNetB0": (w_b0,     fusion_config["models"]["efficientnetb0"]["accuracy"]),
        "EfficientNetB2": (w_b2,     fusion_config["models"]["efficientnetb2"]["accuracy"]),
        "ResNet50":       (w_resnet, fusion_config["models"]["resnet50"]["accuracy"]),
    }
    for name, (w, acc) in model_weights.items():
        st.markdown(f"""
        <div class="model-card">
            <b>{name}</b><br>
            <small>Weight: {w:.3f} | Acc: {acc*100:.1f}%</small>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"**Fusion Accuracy:** `{fusion_config['best_accuracy']*100:.2f}%`")
    st.caption("⚠️ For educational purposes only.")

# ── Main UI ──
st.markdown('<div class="main-title">🧴 AcneAI — Multi-Fusion Detection</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub">EfficientNetB0 + EfficientNetB2 + ResNet50 — Weighted Ensemble</div>',
    unsafe_allow_html=True
)

uploaded = st.file_uploader("📤 Upload a skin image", type=["jpg","jpeg","png"])

if uploaded:
    with st.spinner("Loading models..."):
        model_b0, model_b2, model_resnet = load_all_models()

    image = Image.open(uploaded).convert("RGB")

    with st.spinner("🔍 Running fusion analysis..."):
        fused_probs, p_b0, p_b2, p_resnet = fusion_predict(
            image, model_b0, model_b2, model_resnet
        )

    pred_idx   = np.argmax(fused_probs)
    pred_cls   = CLASS_NAMES[pred_idx]
    confidence = float(fused_probs[pred_idx]) * 100
    sev, color, emoji = SEVERITY[pred_cls]

    st.markdown("---")
    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        st.markdown(
            f"<h2 style='color:{color};'>{emoji} {pred_cls}</h2>",
            unsafe_allow_html=True
        )
        st.markdown(f"**Fusion Confidence:** {confidence:.1f}%")
        st.progress(int(confidence))
        st.markdown(
            f"**Severity:** <span style='color:{color};font-weight:bold;'> {sev}</span>",
            unsafe_allow_html=True
        )
        st.info(f"📋 {DESCRIPTION[pred_cls]}")
        st.markdown(
            f'<div class="advice">💡 <b>Advice:</b> {ADVICE[pred_cls]}</div>',
            unsafe_allow_html=True
        )
        if confidence < threshold * 100:
            st.warning("⚠️ Low confidence. Please consult a dermatologist.")

    # Probability chart
    st.markdown("---")
    st.subheader("📊 Fusion Model — Class Probabilities")

    bar_colors = [color if i == pred_idx else "#BDC3C7" for i in range(len(CLASS_NAMES))]
    fig = go.Figure(go.Bar(
        x=CLASS_NAMES,
        y=[float(p)*100 for p in fused_probs],
        marker_color=bar_colors,
        text=[f"{float(p)*100:.1f}%" for p in fused_probs],
        textposition="outside",
    ))
    fig.update_layout(
        yaxis=dict(range=[0,115], title="Probability (%)"),
        xaxis_title="Acne Type",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=360, showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    # Individual model predictions
    if show_individual:
        st.markdown("---")
        st.subheader("🔬 Individual Model Predictions")

        m_col1, m_col2, m_col3 = st.columns(3)
        model_preds = [
            ("EfficientNetB0", p_b0,     "#3498DB", w_b0),
            ("EfficientNetB2", p_b2,     "#2ECC71", w_b2),
            ("ResNet50",       p_resnet, "#E67E22", w_resnet),
        ]

        for col, (mname, probs_m, mcolor, weight) in zip([m_col1,m_col2,m_col3], model_preds):
            m_pred  = CLASS_NAMES[np.argmax(probs_m)]
            m_conf  = float(np.max(probs_m)) * 100
            m_emoji = SEVERITY[m_pred][2]
            with col:
                st.markdown(f"**{mname}**")
                st.markdown(f"Weight: `{weight:.3f}`")
                fig_m = go.Figure(go.Bar(
                    x=CLASS_NAMES,
                    y=[float(p)*100 for p in probs_m],
                    marker_color=[mcolor if i == np.argmax(probs_m) else "#E8E8E8"
                                  for i in range(5)],
                    text=[f"{float(p)*100:.0f}%" for p in probs_m],
                    textposition="outside"
                ))
                fig_m.update_layout(
                    height=250, showlegend=False,
                    yaxis=dict(range=[0,120]),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=10,b=10,l=10,r=10)
                )
                st.plotly_chart(fig_m, use_container_width=True)
                st.markdown(
                    f"**Pred:** {m_emoji} {m_pred} ({m_conf:.1f}%)"
                )

    # Top 3 and download
    st.markdown("---")
    st.subheader("🏆 Top 3 Fusion Predictions")
    top3 = np.argsort(fused_probs)[-3:][::-1]
    medals = ["🥇","🥈","🥉"]
    c1,c2,c3 = st.columns(3)
    for col, idx, medal in zip([c1,c2,c3], top3, medals):
        s, sc, em = SEVERITY[CLASS_NAMES[idx]]
        with col:
            st.metric(
                label=f"{medal} {CLASS_NAMES[idx]}",
                value=f"{float(fused_probs[idx])*100:.1f}%",
                delta=f"{s} severity"
            )

    result = {
        "fusion_prediction":  pred_cls,
        "fusion_confidence":  round(confidence, 2),
        "severity":           sev,
        "fusion_probs":       {CLASS_NAMES[i]: round(float(p)*100,2) for i,p in enumerate(fused_probs)},
        "individual": {
            "efficientnetb0": {CLASS_NAMES[i]: round(float(p)*100,2) for i,p in enumerate(p_b0)},
            "efficientnetb2": {CLASS_NAMES[i]: round(float(p)*100,2) for i,p in enumerate(p_b2)},
            "resnet50":       {CLASS_NAMES[i]: round(float(p)*100,2) for i,p in enumerate(p_resnet)},
        }
    }
    st.download_button(
        "📥 Download Full Result (JSON)",
        json.dumps(result, indent=2),
        "acne_fusion_result.json", "application/json"
    )

else:
    st.info("👆 Upload a skin image to run the fusion model analysis")
    st.markdown("---")
    st.subheader("⚡ How Fusion Works")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="model-card">
            <div style="font-size:1.5rem">🔵</div>
            <b>EfficientNetB0</b><br>
            <small>224×224 input<br>Fast & efficient</small>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="model-card">
            <div style="font-size:1.5rem">🟢</div>
            <b>EfficientNetB2</b><br>
            <small>260×260 input<br>Higher resolution</small>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="model-card">
            <div style="font-size:1.5rem">🟠</div>
            <b>ResNet50</b><br>
            <small>224×224 input<br>Deep residual features</small>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="model-card" style="border:2px solid #9B59B6;">
            <div style="font-size:1.5rem">⚡</div>
            <b>Weighted Fusion</b><br>
            <small>Combined accuracy:<br><b>{fusion_config['best_accuracy']*100:.2f}%</b></small>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📚 Acne Type Reference")
    cols = st.columns(5)
    for i, cls in enumerate(CLASS_NAMES):
        sev, color, emoji = SEVERITY[cls]
        with cols[i]:
            st.markdown(f"""
            <div style="background:#fafafa;padding:1rem;border-radius:10px;
                        border-top:4px solid {color};text-align:center;">
                <div style="font-size:1.5rem">{emoji}</div>
                <b style="color:{color};">{cls}</b><br>
                <small style="color:#888;">{sev}</small><br>
                <small>{DESCRIPTION[cls]}</small>
            </div>""", unsafe_allow_html=True)

