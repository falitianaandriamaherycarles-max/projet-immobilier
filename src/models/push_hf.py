from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import yaml
from huggingface_hub import HfApi, create_repo

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    hf_cfg = cfg["huggingface"]

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError(
            "La variable d'environnement HF_TOKEN est requise pour publier sur Hugging Face Hub."
        )

    api = HfApi(token=token)
    repo_id = hf_cfg["repo_id"]

    create_repo(repo_id, token=token, private=hf_cfg.get("private", True), exist_ok=True)

    for artefact in [cfg["train"]["model_path"], cfg["train"]["preprocessor_path"], config_path]:
        path = Path(artefact)
        if not path.exists():
            logger.warning("Artefact introuvable, ignoré : %s", path)
            continue
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=repo_id,
            token=token,
        )
        logger.info("Publié sur Hugging Face Hub : %s -> %s/%s", path, repo_id, path.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publication du modèle sur Hugging Face Hub")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
