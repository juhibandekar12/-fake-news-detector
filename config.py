# config.py

# Model settings
MODEL_SOURCE = "huggingface"  # change to "local" for local testing

HUGGINGFACE_MODEL_PATH = "avivaaaaa123/fake-news-detector"
LOCAL_MODEL_PATH = "./model/bert_model"
MAX_LENGTH = 128

# App settings
APP_TITLE = "Fake News Detector"
APP_ICON = "📰"
APP_DESCRIPTION = "Powered by BERT — Enter a news article to check if it's real or fake."