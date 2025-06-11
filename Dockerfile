# Base image officielle
FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers
COPY . /app

# Installer les dépendances
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Exposer le port utilisé par Uvicorn
EXPOSE 8080

# Lancer l'app FastAPI avec Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
