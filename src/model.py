from transformers import TFBertForSequenceClassification, BertTokenizer
from config import MODEL_SOURCE, HUGGINGFACE_MODEL_PATH, LOCAL_MODEL_PATH

def load_model():
    # Load from local folder
    model_path = LOCAL_MODEL_PATH if MODEL_SOURCE == "local" else HUGGINGFACE_MODEL_PATH
    
    print("Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(model_path)
    
    print("Loading model...")
    model = TFBertForSequenceClassification.from_pretrained(model_path)
    
    print("✅ Model loaded!")
    return model, tokenizer