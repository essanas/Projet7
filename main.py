# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow, mlflow.pytorch
import torch, os, uvicorn, logging
from transformers import BertTokenizer, logging as hf_log

# ───────────────────── Configuration ────────────────────── #
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_URI           = os.getenv("MODEL_URI",           # ex. models:/bert/Production
                                "runs:/76ddea8898814ad6a4f882756e39615d/BERT")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hf_log.set_verbosity_error()          # réduit le bruit de HuggingFace
logging.basicConfig(level=logging.INFO)

# ────────────────────── FastAPI app ─────────────────────── #
app       = FastAPI(title="Tweet Classifier API")
model     = None
tokenizer = None

@app.on_event("startup")
def load_artifacts() -> None:
    """Se connecte au tracking-server et charge modèle + tokenizer."""
    global model, tokenizer

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logging.info("Connexion MLflow : %s", MLFLOW_TRACKING_URI)

    # ─ Vérification facultative du run (si MODEL_URI commence par runs:/) ─ #
    if MODEL_URI.startswith("runs:/"):
        run_id = MODEL_URI.split("/")[1]
        try:
            run = mlflow.get_run(run_id)
            logging.info("Run trouvé : %s — artifact_uri=%s",
                         run.info.run_id, run.info.artifact_uri)
        except Exception as exc:
            raise RuntimeError(
                f"Run {run_id} introuvable sur {MLFLOW_TRACKING_URI}"
            ) from exc

    # ─ Chargement du modèle ─ #
    try:
        model = mlflow.pytorch.load_model(MODEL_URI).to(DEVICE).eval()
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        logging.info("Modèle et tokenizer chargés.")
    except Exception as exc:
        raise RuntimeError(
            f"Échec lors du chargement de '{MODEL_URI}' : {exc}"
        ) from exc

# ─────────────────────── Schéma entrée ───────────────────── #
class TweetInput(BaseModel):
    text: str

# ─────────────────────── Endpoints ───────────────────────── #
@app.get("/health")
async def health() -> dict:
    return {"status": "healthy"}

@app.post("/predict")
async def predict(inp: TweetInput) -> dict:
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    try:
        enc = tokenizer(
            inp.text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=64,
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}

        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits, dim=1)[0]

        return {
            "predicted_class": int(torch.argmax(probs)),
            "confidence": float(torch.max(probs)),
            "probabilities": probs.cpu().tolist(),
        }

    except Exception as exc:
        raise HTTPException(status_code=500,
                            detail=f"Erreur de prédiction : {exc}") from exc

# ─────────────────────── Lancement local ─────────────────── #
if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 8080))  # 8080 par défaut sur cloud
    uvicorn.run("main:app",
                host="0.0.0.0",
                port=port)

