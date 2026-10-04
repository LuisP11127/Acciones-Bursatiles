"""Versionado de modelos y reglas de promoción a producción.

Cada candidato recibe una versión (model_v001, model_v002...) con su fecha,
periodo de entrenamiento, versión de features y métricas. Solo los modelos
promovidos guardan sus pesos; un candidato que empeora claramente al modelo en
producción queda registrado como "rejected" junto con los motivos.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..data.storage import read_json, write_json
from ..logging_utils import get_logger

log = get_logger("MODEL")


def _num(valor) -> float | None:
    try:
        v = float(valor)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def evaluate_promotion(candidate: dict, production: dict | None, cfg: dict,
                       features_version: str) -> tuple[bool, list[dict]]:
    """Devuelve (promover, comprobaciones). Cada comprobación: name, passed, detail."""
    checks: list[dict] = []
    wf = (candidate.get("walk_forward") or {}).get("aggregate") or {}
    auc = _num(wf.get("roc_auc"))
    n = int(wf.get("n") or 0)

    def check(nombre: str, ok: bool, detalle: str) -> None:
        checks.append({"name": nombre, "passed": bool(ok), "detail": detalle})

    check("muestras_test", n >= int(cfg["min_test_samples"]),
          f"{n} predicciones fuera de muestra (mínimo {cfg['min_test_samples']})")
    check("auc_minimo", auc is not None and auc >= float(cfg["min_auc"]),
          f"AUC fuera de muestra {auc if auc is None else round(auc, 4)} (mínimo {cfg['min_auc']})")
    lift = _num(wf.get("top_decile_lift"))
    check("lift_decil_superior", lift is not None and lift >= float(cfg["min_top_decile_lift"]),
          f"lift del decil superior {lift if lift is None else round(lift, 3)} (mínimo {cfg['min_top_decile_lift']})")
    bases = {k: v for k, v in ((candidate.get("walk_forward") or {}).get("baselines") or {}).items()
             if k != "random" and _num(v) is not None}
    if bases and auc is not None:
        mejor = max(bases, key=bases.get)
        ventaja = auc - bases[mejor]
        check("supera_benchmarks", ventaja >= float(cfg["min_auc_advantage_vs_baselines"]),
              f"AUC {auc:.4f} vs mejor benchmark sencillo {mejor} {bases[mejor]:.4f} (ventaja {ventaja:+.4f})")
    else:
        check("supera_benchmarks", False, "sin benchmarks comparables")
    if production and production.get("features_version") == features_version:
        prod_wf = (production.get("walk_forward") or {}).get("aggregate") or {}
        prod_auc = _num(prod_wf.get("roc_auc"))
        if prod_auc is not None and auc is not None:
            check("no_empeora_auc", auc >= prod_auc - float(cfg["max_auc_drop_vs_production"]),
                  f"AUC {auc:.4f} vs producción {production['model_version']} {prod_auc:.4f}")
        cand_sharpe = _num(((candidate.get("backtest") or {}).get("combined") or {}).get("sharpe"))
        prod_sharpe = _num(((production.get("backtest") or {}).get("combined") or {}).get("sharpe"))
        if cand_sharpe is not None and prod_sharpe is not None:
            check("no_empeora_sharpe", cand_sharpe >= prod_sharpe - float(cfg["max_sharpe_drop_vs_production"]),
                  f"Sharpe backtest {cand_sharpe:.3f} vs producción {prod_sharpe:.3f}")
    return all(c["passed"] for c in checks), checks


class ModelRegistry:
    def __init__(self, models_dir: Path):
        self.dir = Path(models_dir)
        self.path = self.dir / "registry.json"
        self.data = read_json(self.path, {"production": None, "versions": []})

    @property
    def production_version(self) -> str | None:
        return self.data.get("production")

    def get(self, version: str | None) -> dict | None:
        if not version:
            return None
        return next((v for v in self.data["versions"] if v["model_version"] == version), None)

    def production(self) -> dict | None:
        return self.get(self.production_version)

    def next_version(self) -> str:
        numeros = [int(m.group(1)) for v in self.data["versions"]
                   if (m := re.match(r"model_v(\d+)", v["model_version"]))]
        return f"model_v{(max(numeros) + 1 if numeros else 1):03d}"

    def register(self, meta: dict, model=None, promote: bool = False) -> dict:
        ahora = datetime.now(timezone.utc).isoformat(timespec="seconds")
        meta = dict(meta)
        meta["registered_at"] = ahora
        if promote:
            import torch

            destino = self.dir / meta["model_version"]
            destino.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), destino / "model.pt")
            write_json(destino / "meta.json", meta)
            anterior = self.production()
            if anterior:
                anterior["status"] = "archived"
            meta["status"] = "production"
            meta["promoted_at"] = ahora
            self.data["production"] = meta["model_version"]
        else:
            meta["status"] = "rejected"
        self.data["versions"].append(meta)
        self.save()
        log.info("Modelo %s registrado como %s", meta["model_version"], meta["status"])
        return meta

    def save(self) -> None:
        write_json(self.path, self.data)

    def load_production(self):
        """(modelo, meta) del modelo en producción o (None, None)."""
        meta = self.production()
        if not meta:
            return None, None
        ruta = self.dir / meta["model_version"] / "model.pt"
        if not ruta.exists():
            log.warning("Faltan los pesos del modelo en producción %s", meta["model_version"])
            return None, None
        try:
            import torch

            from .model import OpportunityNet

            hp = meta["hyperparameters"]
            model = OpportunityNet(len(meta["feature_names"]), int(hp["hidden_size"]), int(hp["num_layers"]),
                                   float(hp["dropout"]), str(hp["architecture"]))
            model.load_state_dict(torch.load(ruta, map_location="cpu", weights_only=True))
            model.eval()
            meta = dict(meta)
            meta["_mean"] = np.asarray(meta["scaler"]["mean"], dtype=np.float32)
            meta["_std"] = np.asarray(meta["scaler"]["std"], dtype=np.float32)
            return model, meta
        except ImportError:
            log.warning("PyTorch no está instalado: la red neuronal no se usa en esta ejecución")
        except Exception as exc:
            log.warning("No se pudo cargar el modelo %s: %s", meta["model_version"], exc)
        return None, None
