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
        with st.spinner("Analyzing article & Verifying Factuality..."):
            result = predict(news_input, model, tokenizer)
            
            # --- CROSS-CHECK WITH NEWS API / WEB SEARCH ---
            from services.fact_checker import cross_reference_news
            title_guess = " ".join(news_input.split()[:15])  # Take first 15 words as search query
            verification = cross_reference_news(title_guess)
            
            is_real = result.get('label') == 'REAL'
            
            # Override if News API verifies the article
            if not is_real and verification.get("is_verified"):
                result['label'] = 'REAL'
                # Swap visual probabilities to match REAL outcome
                fake_prob = result.get('fake_prob', 0)
                real_prob = result.get('real_prob', 0)
                result['real_prob'] = max(fake_prob, 85.0)
                result['fake_prob'] = min(real_prob, 15.0)

        st.markdown("---")  # ← replace st.divider()

        if result['label'] == 'FAKE':
            st.error("🚨 FAKE NEWS")
        else:
            st.success("✅ REAL NEWS")
            if verification.get("is_verified"):
                source = verification.get("source", "NewsAPI/Web Search")
                st.info(f"📰 Verified by: **{source}**")

        st.subheader("Confidence Scores")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("🚨 Fake", f"{result['fake_prob']:.2f}%")
            st.progress(min(result['fake_prob'] / 100, 1.0))

        with col2:
            st.metric("✅ Real", f"{result['real_prob']:.2f}%")
            st.progress(min(result['real_prob'] / 100, 1.0))