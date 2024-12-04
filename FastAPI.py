#!/usr/bin/env python
# coding: utf-8

# !pip install fastapi uvicorn
# 

# In[2]:


get_ipython().system('pip show nest_asyncio')


# In[3]:


import nest_asyncio
nest_asyncio.apply()


# In[4]:


import mlflow.pytorch

# Configurer l'URI de suivi MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Chemin du modèle
model_path = "runs:/b9edb24e191946b98da2cd76fbb0d5c1/bert_model"  # Remplacez par le bon run_id

try:
    model = mlflow.pytorch.load_model(model_path)
    print("Modèle chargé avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")


# In[5]:


from transformers import BertTokenizer
import torch
import mlflow.pytorch

# Configurer l'URI de suivi MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Charger le modèle et le tokenizer
model_path = "runs:/b9edb24e191946b98da2cd76fbb0d5c1/bert_model"  # Remplacez par le bon run_id
try:
    model = mlflow.pytorch.load_model(model_path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print("Modèle et tokenizer chargés avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle ou du tokenizer : {e}")
    raise

# Tester une prédiction
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

text = "This is a test tweet"
encoded_input = tokenizer(
    text,
    padding=True,
    truncation=True,
    max_length=64,
    return_tensors="pt"
)
input_ids = encoded_input['input_ids'].to(device)
attention_mask = encoded_input['attention_mask'].to(device)

with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1).cpu().numpy()

print(f"Probabilités : {probs}")
predicted_class = int(probs.argmax(axis=1)[0])
print(f"Classe prédite : {predicted_class}")


# In[ ]:


from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pytorch
import torch
from transformers import BertTokenizer
import uvicorn
import nest_asyncio

# Appliquer nest_asyncio pour les environnements interactifs
nest_asyncio.apply()

# Configurer l'URI de suivi MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Charger le modèle et le tokenizer
try:
    model_path = "runs:/b9edb24e191946b98da2cd76fbb0d5c1/bert_model"  
    model = mlflow.pytorch.load_model(model_path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print("Modèle et tokenizer chargés avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle ou du tokenizer : {e}")
    raise

# Définir le dispositif d'exécution (GPU ou CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Créer l'application FastAPI
app = FastAPI()

# Endpoint racine
@app.get("/")
def read_root():
    """
    Affiche un message d'accueil pour la racine de l'API.
    """
    return {"message": "Bienvenue sur l'API FastAPI avec MLflow"}

# Modèle de données pour l'API
class TweetInput(BaseModel):
    text: str

# Endpoint pour effectuer une prédiction
@app.post("/predict")
def predict_tweet(input: TweetInput):
    """
    Prend un tweet en entrée et retourne la prédiction de la classe avec confiance.
    """
    try:
        # Prétraiter le texte
        encoded_input = tokenizer(
            input.text,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )
        input_ids = encoded_input['input_ids'].to(device)
        attention_mask = encoded_input['attention_mask'].to(device)

        # Faire des prédictions
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        # Retourner la classe prédite et les probabilités associées
        predicted_class = int(probs.argmax(axis=1)[0])
        confidence = float(probs.max(axis=1)[0])

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": probs.tolist()
        }
    except Exception as e:
        return {"error": f"Erreur lors de la prédiction : {e}"}

# Endpoint pour vérifier la santé de l'API
@app.get("/health")
def health_check():
    """
    Vérifie la santé de l'API.
    """
    return {"status": "healthy"}

# Lancer le serveur localement
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)


# In[ ]:




