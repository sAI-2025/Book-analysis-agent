# from transformers import pipeline

# def detect_emotion(passage: str):
#     emotion_pipeline = pipeline(
#         "text-classification",
#         model="j-hartmann/emotion-english-distilroberta-base",
#         return_all_scores=False
#     )
#     result = emotion_pipeline(passage[:512])[0]
#     return result["label"], result["score"]
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# === Step 1: Load model once globally (on startup) ===
def load_emotion_model():
    MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

    # Efficient model loading for low memory systems
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Select device automatically
    device = 0 if torch.cuda.is_available() else -1

    # Build the pipeline with preloaded model
    emotion_pipeline = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        truncation=True,
        max_length=512
    )
    return emotion_pipeline

# Load globally once at startup
emotion_cls = load_emotion_model()

# === Step 2: Create the production‑ready function ===
def detect_emotion(passage: str):
    # Lightweight CPU‑friendly inference
    truncated_text = passage.strip()[:512]  # safety truncation
    with torch.no_grad():                   # disable gradient tracking
        result = emotion_cls(truncated_text)[0]
    return result["label"], round(result["score"], 3)
