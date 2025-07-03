# ───────────────── Dockerfile ───────────────── #

# 1. Image de base
FROM python:3.10-slim

# 2. Répertoire de travail
WORKDIR /app

# 3. Copie des fichiers
COPY . /app

# 4. Installation des dépendances
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# 5. Téléchargement du tokenizer BERT (cache dans l’image)
RUN python -c "from transformers import BertTokenizer; BertTokenizer.from_pretrained('bert-base-uncased')"

# 6. Port exposé pour Cloud Run
EXPOSE 8080

# 7. Commande de lancement
CMD ["uvicorn", "main_pt:app", "--host", "0.0.0.0", "--port", "8080"]
