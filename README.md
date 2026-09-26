# projet-immobilier

Pipeline MLOps de bout en bout pour prédire le prix d'un bien immobilier à
partir des données **DVF** (Demandes de Valeurs Foncières), publiées en open
data par la DGFiP sur [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres/).

```
Données DVF → DVC (versioning) → scikit-learn (entraînement) → MLflow (tracking)
     → FastAPI (API) → GitHub Actions (CI/CD) → Docker (build) → Déploiement
     → Evidently AI (monitoring) → boucle de feedback vers l'entraînement
```

## Sommaire

1. [Architecture](#architecture)
2. [Démarrage rapide](#démarrage-rapide)
3. [Pipeline de données](#pipeline-de-données)
4. [Entraînement & tracking MLflow](#entraînement--tracking-mlflow)
5. [API](#api)
6. [Monitoring](#monitoring)
7. [Docker / docker-compose](#docker--docker-compose)
8. [CI/CD](#cicd)
9. [Structure du dépôt](#structure-du-dépôt)

## Architecture

| Étape | Outil | Rôle |
|---|---|---|
| 1. Entraînement | scikit-learn, DVC, MLflow | Entraîne le modèle, versionne données/modèle, trace les runs |
| 2. Code & API | FastAPI, Evidently AI | Sert le modèle via une API REST, valide qualité données/modèle |
| 3. CI/CD | GitHub Actions | Tests, lint, déclenchement du build à chaque push |
| 4. Build | Docker | Empaquette l'API dans une image reproductible |
| 5. Déploiement | docker-compose / registre d'images | Met l'API en production |
| 6. Monitoring | Evidently AI | Détecte la dérive des données/du modèle, boucle vers l'étape 1 |

## Démarrage rapide

```bash
git clone <votre-fork>
cd projet-immobilier
python -m venv .venv && source .venv/bin/activate
make install

# Pipeline complet : ingestion -> prétraitement -> features -> entraînement -> évaluation
make pipeline

# Lancer l'API en local
make api        # http://localhost:8000/docs
```

Ou avec DVC (reproductibilité garantie par le graphe de dépendances) :

```bash
dvc repro
```

## Pipeline de données

Les données proviennent des fichiers DVF redistribués par data.gouv.fr /
Etalab, un CSV par département et par année :

```
https://files.data.gouv.fr/geo-dvf/latest/csv/{annee}/departements/{dept}.csv.gz
```

> ⚠️ Cette URL peut évoluer. En cas d'échec de téléchargement, vérifiez
> l'adresse courante sur la
> [page du jeu de données](https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres/)
> et mettez à jour `data.base_url` dans `configs/config.yaml`.

Les départements et années couverts sont configurables dans
`configs/config.yaml` (`data.departements`, `data.years`).

```bash
make ingest       # télécharge et consolide les CSV bruts -> data/raw/
make preprocess   # nettoyage, filtrage, prix au m² -> data/processed/
make features     # encodage, split train/test -> data/features/
```

## Entraînement & tracking MLflow

```bash
make train        # entraîne un RandomForestRegressor, log params/metrics/modèle
make mlflow-ui     # http://localhost:5000
```

Les hyperparamètres sont dans `configs/model_params.yaml` — aucune
modification de code n'est nécessaire pour les ajuster.

Le modèle est à la fois :
- **loggé dans MLflow** (`mlflow.sklearn.log_model`, avec model registry),
- **sauvegardé localement** en `models/model.pkl` (versionné par DVC),
- optionnellement **publié sur Hugging Face Hub** via
  `python -m src.models.push_hf` (nécessite `HF_TOKEN`).

## API

```bash
make api
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"surface_reelle_bati": 65, "type_local": "Appartement", "code_postal": "75015", "nombre_pieces_principales": 3}'
```

Documentation interactive : `http://localhost:8000/docs`.

## Monitoring

```bash
make monitor
```

Génère `reports/drift_report.html` (Evidently AI), en comparant les données
d'entraînement (référence) aux données de production accumulées (courant).
Un dépassement du seuil `monitoring.drift_share_threshold`
(`configs/config.yaml`) signale qu'un ré-entraînement est recommandé — c'est
la boucle de feedback vers l'étape 1.

## Docker / docker-compose

```bash
make docker-build
make docker-up
```

Services exposés derrière Nginx sur `http://localhost:8080` :
- `/` → API de prédiction
- `/mlflow/` → interface MLflow
- `/monitoring/` → rapports Evidently AI

## CI/CD

- **`.github/workflows/ci.yml`** — lint (ruff) + tests (pytest) à chaque push.
- **`.github/workflows/cd.yml`** — build & push de l'image Docker sur GHCR
  après un CI réussi sur `main`, ou sur un tag `vX.Y.Z`.
- **`.github/workflows/retrain.yml`** — pipeline complet planifié (chaque
  lundi) : ingestion, prétraitement, entraînement, évaluation, monitoring,
  versioning DVC et publication Hugging Face.

## Structure du dépôt

```
projet-immobilier/
├── .github/workflows/     CI, CD, ré-entraînement planifié
├── notebooks/             Exploration pas à pas (miroir de src/)
├── src/
│   ├── data/               ingestion + prétraitement
│   ├── features/           feature engineering
│   ├── models/              train / evaluate / predict / push_hf
│   ├── api/                 FastAPI (schemas, model_loader, main)
│   └── monitoring/          rapport de dérive Evidently
├── tests/                  tests unitaires (data, modèle, API)
├── docker/                 Dockerfiles (api, mlflow, evidently) + nginx.conf
├── configs/                config.yaml, model_params.yaml
├── data/                   raw / processed / features (versionné DVC)
├── models/                 model.pkl, preprocessor.pkl (versionné DVC)
├── reports/                rapports Evidently + métriques
├── dvc.yaml                pipeline DVC (ingest → preprocess → features → train → evaluate)
├── docker-compose.yml
├── Makefile
├── requirements.txt
└── pyproject.toml
```

## Licence des données

Les données DVF sont publiées sous
[Licence Ouverte / Open Licence](https://www.etalab.gouv.fr/licence-ouverte-open-licence/).
Le fichier DVF contient des données à caractère personnel : consultez les
conditions générales d'utilisation avant toute réutilisation en production
(pas de ré-identification indirecte, pas d'indexation par des moteurs de
recherche externes).
