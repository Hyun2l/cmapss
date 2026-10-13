"""configs/config.yaml 로드. 경로는 프로젝트 루트 기준 절대경로로 변환."""
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path=DEFAULT_CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {k: PROJECT_ROOT / v for k, v in cfg["paths"].items()}
    return cfg


def raw_files(cfg: dict) -> dict:
    """dataset 이름(FD001 등)에 맞는 train/test/RUL 원본 파일 경로."""
    raw, ds = cfg["paths"]["raw_dir"], cfg["dataset"]
    return {
        "train": raw / f"train_{ds}.txt",
        "test": raw / f"test_{ds}.txt",
        "rul": raw / f"RUL_{ds}.txt",
    }
