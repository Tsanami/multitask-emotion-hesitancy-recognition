"""
Лёгкая обёртка над MLflow для трекинга экспериментов НИРа.

Зачем: ablation E3–E8 × сиды [42,0,123] = много runs. MLflow даёт таблицу
сравнения по гиперпараметрам (thr, lambda, seed), кривые метрик по эпохам и
хранение артефактов (чекпойнты, конфиги, истории) — локально, без аккаунта.

Запуск дашборда:
    mlflow ui --backend-store-uri sqlite:///mehr/results/mlflow.db
    # затем http://127.0.0.1:5000

Если mlflow не установлен или use_mlflow=False — всё работает как no-op.
"""
import os
import contextlib
from dataclasses import asdict, is_dataclass

try:
    import mlflow
    _HAS_MLFLOW = True
except ImportError:
    _HAS_MLFLOW = False

from configs.configs import RESULTS_DIR

# SQLite-бэкенд (file-store в новых mlflow в maintenance mode), артефакты — рядом
TRACKING_URI    = "sqlite:///" + os.path.join(RESULTS_DIR, "mlflow.db")
ARTIFACT_DIR    = os.path.join(RESULTS_DIR, "mlartifacts")
EXPERIMENT_NAME = "ssl_mepr"


def _params_from_cfg(cfg) -> dict:
    """Достаёт скалярные поля конфига как гиперпараметры run-а."""
    d = asdict(cfg) if is_dataclass(cfg) else dict(vars(cfg))
    return {k: v for k, v in d.items() if isinstance(v, (int, float, str, bool))}


def _numeric(d: dict) -> dict:
    """Оставляет только числовые метрики (без списков вроде pseudo_emo_hist)."""
    return {k: float(v) for k, v in d.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)}


class _NullRun:
    """Заглушка, когда трекинг выключен."""
    def log_metrics(self, metrics, step=None): pass
    def log_artifact(self, path):             pass


class _MlflowRun:
    def log_metrics(self, metrics, step=None):
        m = _numeric(metrics)
        if m:
            mlflow.log_metrics(m, step=step)

    def log_artifact(self, path):
        if path and os.path.exists(path):
            mlflow.log_artifact(path)


@contextlib.contextmanager
def track_run(run_name, cfg, seed, enabled=True, extra_params=None):
    """
    Контекст одного MLflow-run. Логирует параметры конфига + seed на входе.
    Внутри пользуйся tracker.log_metrics(...) / tracker.log_artifact(...).
    """
    if not (enabled and _HAS_MLFLOW):
        if enabled and not _HAS_MLFLOW:
            print("[tracking] mlflow не установлен — трекинг пропущен (pip install mlflow)")
        yield _NullRun()
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(
            EXPERIMENT_NAME, artifact_location="file:" + os.path.abspath(ARTIFACT_DIR))
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=run_name):
        params = _params_from_cfg(cfg)
        params["seed"] = seed
        if extra_params:
            params.update(extra_params)
        mlflow.log_params(params)
        yield _MlflowRun()
