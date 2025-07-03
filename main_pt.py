# main_pt.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import os
import logging
from transformers import BertTokenizer, BertForSequenceClassification, logging as hf_log

# ───────────────────── Configuration ────────────────────── #
MODEL_PATH = "model/bert.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hf_log.set_verbosity_error()
logging.basicConfig(level=logging.INFO)

# ─────────────────────── FastAPI app ─────────────────────── #
app = FastAPI(title="Tweet Classifier API (.pt version)")
model = None
tokenizer = None

@app.on_event("startup")
def load_artifacts():
    """Charge le modèle .pt et le tokenizer"""
    global model, tokenizer

    try:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE).eval()
        logging.info("✅ Modèle et tokenizer chargés depuis .pt")
    except Exception as exc:
        raise RuntimeError(f"Erreur lors du chargement du modèle .pt : {exc}") from exc

# ─────────────────────── Schéma entrée ───────────────────── #
class TweetInput(BaseModel):
    text: str

# ─────────────────────── Endpoints ───────────────────────── #
@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(inp: TweetInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    try:
        enc = tokenizer(
            inp.text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=64
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=1)[0]

        return {
            "predicted_class": int(torch.argmax(probs)),
            "confidence": float(torch.max(probs)),
            "probabilities": probs.cpu().tolist(),
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {exc}")

# ─────────────────────── Lancement local ─────────────────── #
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    import uvicorn
    uvicorn.run("main_pt:app", host="0.0.0.0", port=port)
