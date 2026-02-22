import tensorflow as tf
import numpy as np
from config import MAX_LENGTH

def predict(text, model, tokenizer, max_length=MAX_LENGTH):
    inputs = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='tf'
    )

    outputs  = model(inputs)
    probs    = tf.nn.softmax(outputs.logits, axis=1).numpy()[0]
    prob_0   = float(probs[0])
    prob_1   = float(probs[1])

    # Debug — remove after confirming
    print(f"Prob[0]: {prob_0:.4f} | Prob[1]: {prob_1:.4f}")

    label = 'REAL' if prob_1 > prob_0 else 'FAKE'

    return {
        'label':     label,
        'real_prob': max(prob_0, prob_1) * 100 if label == 'REAL' else min(prob_0, prob_1) * 100,
        'fake_prob': max(prob_0, prob_1) * 100 if label == 'FAKE' else min(prob_0, prob_1) * 100
    }