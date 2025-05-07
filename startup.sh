#!/bin/bash

# Active l'environnement virtuel
source venv/bin/activate

# Lance l'application avec gunicorn et uvicorn
gunicorn -k uvicorn.workers.UvicornWorker main:app --host 0.0.0.0 --port 8000
