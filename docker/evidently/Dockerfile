FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY configs/ ./configs/

# Sert les rapports HTML générés dans reports/ sur le port 8050
EXPOSE 8050

# Génère un rapport puis sert le dossier reports/ en statique.
# En production, ce service peut aussi être déclenché en tâche planifiée
# (cron / .github/workflows/retrain.yml) plutôt qu'en process persistant.
CMD ["sh", "-c", "python -m src.monitoring.drift_report --config configs/config.yaml; python -m http.server 8050 --directory reports"]
