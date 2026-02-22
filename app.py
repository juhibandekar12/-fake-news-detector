import streamlit as st
from src.model import load_model
from src.predict import predict
from config import APP_TITLE, APP_ICON, APP_DESCRIPTION

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="centered"
)

@st.cache_resource
def get_model():
    return load_model()

model, tokenizer = get_model()

st.title("📰 Fake News Detector")
st.write("Powered by BERT — Enter a news article to check if it's real or fake.")
st.markdown("---")  # ← replace st.divider()

news_input = st.text_area(
    "Paste your news article here:",
    height=200,
    placeholder="Enter news article text..."
)

if st.button("🔍 Analyze", use_container_width=True):
    if news_input.strip() == "":
        st.warning("⚠️ Please enter some text first.")
    else:
        with st.spinner("Analyzing article..."):
            result = predict(news_input, model, tokenizer)

        st.markdown("---")  # ← replace st.divider()

        if result['label'] == 'FAKE':
            st.error("🚨 FAKE NEWS")
        else:
            st.success("✅ REAL NEWS")

        st.subheader("Confidence Scores")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("🚨 Fake", f"{result['fake_prob']:.2f}%")
            st.progress(result['fake_prob'] / 100)

        with col2:
            st.metric("✅ Real", f"{result['real_prob']:.2f}%")
            st.progress(result['real_prob'] / 100)