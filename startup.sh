#!/bin/bash

# Active l'environnement virtuel
source venv/bin/activate

# Lance l'application FastAPI avec Uvicorn
uvicorn main:app --host 0.0.0.0 --port 80
