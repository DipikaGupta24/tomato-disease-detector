"""
Tomato Leaf Disease Detection + Generative AI Report
-----------------------------------------------------
Run locally with:      streamlit run app.py
Deploy for free on:    https://share.streamlit.io

Files needed in the same folder:
    - tomato_model_advanced.h5
    - class_indices.json

Gemini API key: https://aistudio.google.com/app/apikey
"""

import os
import json
import numpy as np
import cv2
import streamlit as st
import tensorflow as tf
from PIL import Image
import google.generativeai as genai

IMG_SIZE    = 224
MODEL_PATH  = "tomato_model_advanced.h5"
CLASS_INDEX = "class_indices.json"

st.set_page_config(
    page_title="Tomato Disease Detector",
    page_icon="🍅",
    layout="wide",
)

st.markdown("""
<style>
.section-card {
    background: #1e1e1e;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 18px;
    border: 1px solid #2e2e2e;
}
.section-title {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: #888;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.disease-name {
    font-size: 28px;
    font-weight: 700;
    color: #f5f5f5;
    margin: 4px 0 12px 0;
}
.confidence-badge-green  { color: #4ade80; font-size: 18px; font-weight: 700; }
.confidence-badge-yellow { color: #facc15; font-size: 18px; font-weight: 700; }
.confidence-badge-red    { color: #f87171; font-size: 18px; font-weight: 700; }
.conf-msg { color: #aaa; font-size: 14px; margin-top: 6px; }
.top3-label { font-size: 14px; color: #ccc; margin-bottom: 2px; }
.chat-user { background:#2d4a6e; border-radius:10px; padding:10px 14px;
             margin:6px 0; color:#e8f4fd; font-size:14px; }
.chat-ai   { background:#1e3a2e; border-radius:10px; padding:10px 14px;
             margin:6px 0; color:#d4edda; font-size:14px; }
.divider   { border-top: 1px solid #2e2e2e; margin: 20px 0; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Per-disease subtitle shown under the disease name
# ──────────────────────────────────────────────────────────────────────────────
DISEASE_SUBTITLES = {
    "Tomato___Bacterial_spot":
        "Caused by Xanthomonas bacteria — spreads rapidly in warm, wet conditions.",
    "Tomato___Early_blight":
        "Caused by Alternaria fungus — typically attacks older leaves first, moving upward.",
    "Tomato___Late_blight":
        "Caused by Phytophthora infestans — the same pathogen behind the Irish Potato Famine. Acts fast.",
    "Tomato___Leaf_Mold":
        "Caused by Passalora fulva fungus — thrives in high humidity, common in greenhouses.",
    "Tomato___Septoria_leaf_spot":
        "Caused by Septoria lycopersici fungus — small circular spots with dark borders, spreads through rain splash.",
    "Tomato___Spider_mites Two-spotted_spider_mite":
        "Caused by Tetranychus urticae mites — not a fungus or bacteria; tiny pests that suck cell sap.",
    "Tomato___Target_Spot":
        "Caused by Corynespora cassiicola fungus — produces distinctive concentric-ring lesions.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":
        "A viral disease spread by whiteflies — no chemical cure once infected; prevention is key.",
    "Tomato___Tomato_mosaic_virus":
        "A highly contagious plant virus — spreads through touch, tools, and infected seeds.",
    "Tomato___healthy":
        "No disease detected — your plant looks healthy. Keep up the good care!",
}

# Per-disease Gemini prompt (so each report is tailored, not generic)
DISEASE_PROMPTS = {
    "Tomato___Bacterial_spot": """
A tomato leaf was diagnosed with **Bacterial Spot** (Xanthomonas bacteria), confidence {conf:.1f}%.
Write a structured farmer-friendly report. Use exactly these bold headers:
**Disease** — explain it is bacterial (not fungal/viral), caused by Xanthomonas, spreads via water splash and infected tools.
**Symptoms** — small, water-soaked spots becoming dark/raised with yellow halo; spots may merge in severe cases.
**Organic Treatment** — copper-based sprays, neem oil, remove infected leaves.
**Chemical Treatment** — copper hydroxide or copper oxychloride bactericides; mancozeb combinations.
**Prevention** — avoid overhead watering, use certified disease-free seeds, rotate crops.
Keep each section 1-3 sentences. Simple language. Under 220 words.""",

    "Tomato___Early_blight": """
A tomato leaf was diagnosed with **Early Blight** (Alternaria solani fungus), confidence {conf:.1f}%.
Write a structured farmer-friendly report. Use exactly these bold headers:
**Disease** — fungal disease starting on older/lower leaves, moves upward as season progresses.
**Symptoms** — dark brown spots with distinctive concentric target-like rings, surrounded by yellow tissue; leaves eventually yellow and drop.
**Organic Treatment** — remove infected lower leaves, apply neem oil or copper spray, mulch around base to prevent soil splash.
**Chemical Treatment** — chlorothalonil, mancozeb, or azoxystrobin fungicides applied preventatively every 7-10 days.
**Prevention** — water at base (avoid wetting leaves), space plants for airflow, rotate crops yearly.
Keep each section 1-3 sentences. Simple language. Under 220 words.""",

    "Tomato___Late_blight": """
A tomato leaf was diagnosed with **Late Blight** (Phytophthora infestans), confidence {conf:.1f}%.
Write a structured farmer-friendly report with urgency — Late Blight spreads extremely fast. Use exactly these bold headers:
**Disease** — water mold (not a true fungus) that spreads rapidly in cool, wet weather; can destroy a crop in days.
**Symptoms** — large, greasy-looking dark lesions on leaves and stems, white fuzzy growth on leaf undersides in humid conditions, fruit develops firm brown rot.
**Organic Treatment** — copper-based fungicide (most effective organic option), immediately remove and destroy (do not compost) infected plant parts.
**Chemical Treatment** — metalaxyl, cymoxanil, or mandipropamid; apply every 5-7 days during wet weather.
**Prevention** — use resistant varieties, avoid overhead irrigation, do not leave infected debris in field, monitor forecasts for cool/wet spells.
Keep each section 1-3 sentences. Under 220 words. Stress urgency.""",

    "Tomato___Leaf_Mold": """
A tomato leaf was diagnosed with **Leaf Mold** (Passalora fulva fungus), confidence {conf:.1f}%.
Write a structured farmer-friendly report. Use exactly these bold headers:
**Disease** — fungal disease that thrives in high humidity (above 85%); very common in greenhouses and polytunnels.
**Symptoms** — yellow patches on upper leaf surface, corresponding olive-green to brown velvety mold growth on underside; severe cases cause leaf curl and drop.
**Organic Treatment** — improve ventilation, reduce humidity, apply copper fungicide or potassium bicarbonate spray.
**Chemical Treatment** — chlorothalonil or mancozeb-based fungicides; apply to underside of leaves where the mold grows.
**Prevention** — maintain humidity below 85%, ensure good plant spacing and airflow, avoid wetting foliage when watering.
Keep each section 1-3 sentences. Under 220 words.""",

    "Tomato___Septoria_leaf_spot": """
A tomato leaf was diagnosed with **Septoria Leaf Spot** (Septoria lycopersici fungus), confidence {conf:.1f}%.
Write a structured farmer-friendly report. Use exactly these bold headers:
**Disease** — fungal disease that spreads through rain splash and infected soil; starts on lower leaves after first fruit set.
**Symptoms** — many small (3-6mm) circular spots with dark brown borders and light gray or tan centers; tiny black specks (pycnidia) may be visible inside spots.
**Organic Treatment** — remove and destroy infected leaves immediately, copper-based fungicide, neem oil spray.
**Chemical Treatment** — chlorothalonil, mancozeb, or copper hydroxide; spray every 7-10 days, especially after rain.
**Prevention** — mulch to prevent soil splash onto leaves, stake plants to improve airflow, avoid working with wet plants.
Keep each section 1-3 sentences. Under 220 words.""",

    "Tomato___Spider_mites Two-spotted_spider_mite": """
A tomato leaf was diagnosed with **Spider Mite damage** (Tetranychus urticae), confidence {conf:.1f}%.
Write a structured farmer-friendly report. Use exactly these bold headers:
**Disease** — not a fungal or bacterial disease; tiny 8-legged mites that suck sap from leaf cells. Thrives in hot, dry conditions.
**Symptoms** — fine yellow stippling/speckling across leaves, bronzed appearance, fine webbing visible on underside in heavy infestations; leaves eventually dry and drop.
**Organic Treatment** — spray plants with strong jets of water to dislodge mites, apply neem oil or insecticidal soap spray to leaf undersides; introduce predatory mites (Phytoseiidae) if available.
**Chemical Treatment** — miticide/acaricide sprays (abamectin, bifenazate); do NOT use regular insecticides — they often kill mite predators and worsen outbreaks.
**Prevention** — maintain adequate moisture (dry conditions favor mites), avoid dusty conditions, inspect plants regularly, avoid broad-spectrum insecticides.
Keep each section 1-3 sentences. Under 220 words.""",

    "Tomato___Target_Spot": """
A tomato leaf was diagnosed with **Target Spot** (Corynespora cassiicola fungus), confidence {conf:.1f}%.
Write a structured farmer-friendly report. Use exactly these bold headers:
**Disease** — fungal disease that affects leaves, stems, and fruit; named after the concentric-ring pattern on lesions.
**Symptoms** — brown lesions with distinctive concentric rings (target-like appearance), yellow halo around spots; lesions on fruit appear sunken.
**Organic Treatment** — remove infected plant material, copper-based fungicide, improve air circulation around plants.
**Chemical Treatment** — azoxystrobin, tebuconazole, or chlorothalonil; begin preventative spraying when conditions are warm and humid.
**Prevention** — avoid dense planting, stake and prune for airflow, avoid leaf wetness from overhead irrigation.
Keep each section 1-3 sentences. Under 220 words.""",

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": """
A tomato leaf was diagnosed with **Tomato Yellow Leaf Curl Virus (TYLCV)**, confidence {conf:.1f}%.
Write a structured farmer-friendly report. Use exactly these bold headers:
**Disease** — a viral disease transmitted by silverleaf whitefly (Bemisia tabaci); no chemical cure once the plant is infected.
**Symptoms** — upward curling of leaves, yellowing of leaf margins, stunted growth, flowers may drop; young leaves are most affected.
**Organic Treatment** — remove and destroy infected plants immediately to prevent spread; use yellow sticky traps to monitor/reduce whitefly populations; apply neem oil to deter whiteflies.
**Chemical Treatment** — control whitefly vectors with imidacloprid or thiamethoxam systemic insecticides on surrounding healthy plants; no antiviral treatment exists.
**Prevention** — use TYLCV-resistant tomato varieties, install insect-proof netting, plant reflective mulch to deter whiteflies, avoid planting near other infected crops.
Keep each section 1-3 sentences. Emphasise that infected plants cannot be cured. Under 220 words.""",

    "Tomato___Tomato_mosaic_virus": """
A tomato leaf was diagnosed with **Tomato Mosaic Virus (ToMV)**, confidence {conf:.1f}%.
Write a structured farmer-friendly report. Use exactly these bold headers:
**Disease** — one of the most contagious plant viruses; spreads through touch, contaminated tools, infected seeds, and even tobacco products; no cure once infected.
**Symptoms** — mottled mosaic pattern of light and dark green patches on leaves, distorted/curled leaves, stunted growth; fruit may show uneven ripening.
**Organic Treatment** — remove and destroy infected plants; disinfect hands and tools with diluted bleach or soap between plants; do not smoke near plants.
**Chemical Treatment** — no antiviral chemicals exist; focus entirely on preventing spread and protecting healthy plants.
**Prevention** — use certified virus-free seeds, wash hands before handling plants, sterilize tools between uses, control aphids (which can spread related viruses).
Keep each section 1-3 sentences. Stress hygiene and prevention since there is no cure. Under 220 words.""",

    "Tomato___healthy": """
A tomato leaf was classified as **Healthy** with {conf:.1f}% confidence.
Write a warm, encouraging 3-4 sentence note for a farmer confirming the plant looks healthy.
Then give exactly 3 specific preventive care tips to keep it that way — be specific to tomatoes, not generic gardening advice.
Use a friendly, supportive tone. Under 150 words.""",
}


# ──────────────────────────────────────────────────────────────────────────────
# ML helpers
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_classes():
    mdl = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_INDEX) as f:
        idx_to_class = json.load(f)
    return mdl, idx_to_class


def crop_to_leaf(img_uint8):
    hsv  = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (10, 20, 20), (100, 255, 255))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        lc = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(lc)
        if w > img_uint8.shape[1] * 0.2 and h > img_uint8.shape[0] * 0.2:
            return cv2.resize(img_uint8[y:y+h, x:x+w], (IMG_SIZE, IMG_SIZE))
    return cv2.resize(img_uint8, (IMG_SIZE, IMG_SIZE))


def make_gradcam_heatmap(img_batch, mdl, layer="Conv_1", pred_index=None):
    gm = tf.keras.models.Model([mdl.inputs], [mdl.get_layer(layer).output, mdl.output])
    with tf.GradientTape() as tape:
        conv_out, predictions = gm(img_batch)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        cc = predictions[:, pred_index]
    grads   = tape.gradient(cc, conv_out)
    pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_out[0] @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_index)


def overlay_gradcam(img_uint8, heatmap):
    h = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    h = cv2.applyColorMap(np.uint8(255 * h), cv2.COLORMAP_JET)
    h = cv2.cvtColor(h, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_uint8, 0.6, h, 0.4, 0)


def confidence_badge(conf):
    if conf >= 80:
        return "🟢", "confidence-badge-green",  "High Confidence",   "The model is highly confident. Detected symptoms strongly match this disease."
    elif conf >= 55:
        return "🟡", "confidence-badge-yellow", "Medium Confidence", "This image appears somewhat challenging. Consider uploading another photo in natural daylight."
    else:
        return "🔴", "confidence-badge-red",    "Low Confidence",    "The image may be blurry, poorly lit, or contain multiple overlapping leaves. Please upload a clearer photo."


def clean(label):
    return label.replace("Tomato___", "").replace("_", " ")


def get_disease_emoji(label):
    if "healthy"  in label.lower(): return "🌿"
    if "yellow"   in label.lower() or "curl" in label.lower(): return "🟡"
    if "mosaic"   in label.lower(): return "🔵"
    if "mite"     in label.lower(): return "🕷️"
    if "blight"   in label.lower(): return "🍂"
    return "🔴"


def call_gemini(prompt, api_key):
    genai.configure(api_key=api_key)
    m = genai.GenerativeModel("gemini-3.5-flash")
    return m.generate_content(prompt).text


def generate_report(disease_name, confidence, api_key):
    prompt_template = DISEASE_PROMPTS.get(disease_name, DISEASE_PROMPTS["Tomato___healthy"])
    prompt = prompt_template.format(conf=confidence)
    return call_gemini(prompt, api_key)


def ask_followup(disease_name, confidence, question, api_key):
    prompt = f"""Context:
- Disease detected: {clean(disease_name)}
- Model confidence: {confidence:.1f}%
- User question: {question}

Answer the question clearly in 2-4 sentences. Be practical and farmer-friendly."""
    return call_gemini(prompt, api_key)


# ──────────────────────────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────────────────────────
for key, default in [
    ("chat_history",    []),
    ("report_text",     ""),
    ("predicted_class", None),
    ("confidence",      None),
    ("cleared",         False),
    ("last_file",       None),
    ("input_key",       0),       # incremented to clear the text input widget
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — API key + Clear button
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    api_key_input = st.text_input(
        "Gemini API key",
        value=os.environ.get("GEMINI_API_KEY", ""),
        type="password",
        help="Get a free key at https://aistudio.google.com/app/apikey",
    )
    st.caption("Your key is only used in this session and is never stored.")

    st.markdown("---")
    if st.button("🗑️ Clear Everything", use_container_width=True, type="secondary"):
        st.session_state.chat_history    = []
        st.session_state.report_text     = ""
        st.session_state.predicted_class = None
        st.session_state.confidence      = None
        st.session_state.last_file       = None
        st.session_state.cleared         = True
        st.session_state.input_key      += 1
        st.rerun()

    st.markdown("---")
    st.markdown("### How to use")
    st.markdown("""
1. Enter your Gemini API key above
2. Upload a tomato leaf image
3. Wait for prediction + AI report
4. Ask follow-up questions below
5. Click **Clear Everything** to start fresh
""")


# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("# 🍅 Tomato Leaf Disease Detector")
st.markdown(
    "Upload a close-up photo of a tomato leaf. "
    "The model predicts the disease, explains its reasoning with Grad-CAM, "
    "and generates a tailored AI treatment report you can ask follow-up questions about."
)
st.divider()


# ──────────────────────────────────────────────────────────────────────────────
# Upload
# ──────────────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a tomato leaf image",
    type=["jpg", "jpeg", "png"],
    key=f"uploader_{st.session_state.input_key}",
    label_visibility="collapsed",
)

if uploaded_file is None:
    st.info("📂 Upload an image above to get started.")
    st.stop()

# Reset state when a new image is uploaded
if st.session_state.last_file != uploaded_file.name:
    st.session_state.chat_history = []
    st.session_state.report_text  = ""
    st.session_state.last_file    = uploaded_file.name

if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_INDEX):
    st.error(f"Missing '{MODEL_PATH}' or '{CLASS_INDEX}'. Place both files next to app.py.")
    st.stop()


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────
model, idx_to_class = load_model_and_classes()

image        = Image.open(uploaded_file).convert("RGB")
raw_resized  = np.uint8(np.array(image.resize((IMG_SIZE, IMG_SIZE))))
img_cropped  = crop_to_leaf(raw_resized)
img_batch    = np.expand_dims(img_cropped / 255.0, axis=0)

preds           = model.predict(img_batch, verbose=0)[0]
pred_index      = int(np.argmax(preds))
confidence      = float(preds[pred_index]) * 100
predicted_class = idx_to_class[str(pred_index)]

st.session_state.predicted_class = predicted_class
st.session_state.confidence      = confidence

heatmap, _  = make_gradcam_heatmap(img_batch, model, pred_index=pred_index)
gradcam_img = overlay_gradcam(img_cropped, heatmap)

emoji, badge_cls, conf_label, conf_msg = confidence_badge(confidence)
disease_emoji   = get_disease_emoji(predicted_class)
disease_clean   = clean(predicted_class)
disease_subtitle = DISEASE_SUBTITLES.get(predicted_class, "")


# ──────────────────────────────────────────────────────────────────────────────
# ① Prediction Card
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Prediction</div>', unsafe_allow_html=True)
st.markdown(f'<div class="disease-name">{disease_emoji} {disease_clean}</div>', unsafe_allow_html=True)
if disease_subtitle:
    st.markdown(f'<div class="conf-msg" style="margin-bottom:10px">{disease_subtitle}</div>', unsafe_allow_html=True)
st.markdown(
    f'<span class="{badge_cls}">{emoji} {confidence:.1f}% — {conf_label}</span>'
    f'<div class="conf-msg">{conf_msg}</div>',
    unsafe_allow_html=True,
)
st.markdown('</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# ② Top-3 with progress bars
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Top Predictions</div>', unsafe_allow_html=True)
for idx in np.argsort(preds)[::-1][:3]:
    n   = clean(idx_to_class[str(idx)])
    pct = float(preds[idx]) * 100
    st.markdown(f'<div class="top3-label">{n} &nbsp; <b>{pct:.1f}%</b></div>', unsafe_allow_html=True)
    st.progress(int(pct))
st.markdown('</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# ③ Visual Analysis
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Visual Analysis</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    st.caption("Uploaded Leaf (after crop)")
    st.image(img_cropped, use_container_width=True)
with col2:
    st.caption("Grad-CAM — red = highest model attention")
    st.image(gradcam_img, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# ④ AI Diagnosis
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📋 AI Diagnosis</div>', unsafe_allow_html=True)

if not api_key_input:
    st.info("Enter your Gemini API key in the sidebar to generate the diagnosis.")
else:
    if not st.session_state.report_text:
        with st.spinner("Generating AI diagnosis..."):
            try:
                st.session_state.report_text = generate_report(
                    predicted_class, confidence, api_key_input
                )
            except Exception as e:
                st.error(f"Couldn't generate report: {e}")

    if st.session_state.report_text:
        st.markdown(st.session_state.report_text)

st.markdown('</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# ⑤ Ask AI — chat
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">💬 Ask AI about this Disease</div>', unsafe_allow_html=True)

if not api_key_input:
    st.info("Enter your Gemini API key in the sidebar to enable the Ask AI feature.")
else:
    # Existing chat history
    for turn in st.session_state.chat_history:
        st.markdown(f'<div class="chat-user">🧑 {turn["question"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-ai">🤖 {turn["answer"]}</div>',     unsafe_allow_html=True)

    st.caption("Examples: Can I eat these tomatoes? · Can this spread to other plants? · Which fungicide should I use? · How often should I spray?")

    # Key increments after each question → clears the widget automatically
    user_q = st.text_input(
        "Ask anything",
        key=f"ask_{st.session_state.input_key}_{len(st.session_state.chat_history)}",
        label_visibility="collapsed",
        placeholder="Type your question here and press Ask...",
    )

    if st.button("Ask Gemini", type="primary") and user_q.strip():
        with st.spinner("Thinking..."):
            try:
                answer = ask_followup(predicted_class, confidence, user_q.strip(), api_key_input)
                st.session_state.chat_history.append({"question": user_q.strip(), "answer": answer})
                st.rerun()   # rerun re-renders the input with new key → blank box
            except Exception as e:
                st.error(f"Couldn't get answer: {e}")

st.markdown('</div>', unsafe_allow_html=True)