import json
import os


class History:
    """Накапливает метрики по эпохам и сохраняет в JSON после каждой."""

    KEYS = [
        "train_loss", "train_loss_emo_sup", "train_loss_ah_sup",
        "train_loss_emo_ssl", "train_loss_ah_ssl",
        "train_emo_mf1", "train_ah_mf1",
        # SSL pseudo-label статистика
        "train_n_ssl_emo", "train_cov_ssl_emo",
        "train_n_ssl_ah",  "train_cov_ssl_ah",
        "train_pseudo_emo_hist",
        "val_loss", "val_loss_emo_sup", "val_loss_ah_sup",
        "val_emo_mf1", "val_emo_uar",
        "val_ah_mf1",  "val_ah_uar",
        "val_overall_f1", "lr",
    ]

    def __init__(self, path: str):
        self.path = path
        self.data = {k: [] for k in self.KEYS}

    def update(self, train_log: dict, val_log: dict, optimizer):
        self.data["train_loss"].append(train_log["loss"])
        self.data["train_loss_emo_sup"].append(train_log["loss_emo_sup"])
        self.data["train_loss_ah_sup"].append(train_log["loss_ah_sup"])
        self.data["train_loss_emo_ssl"].append(train_log["loss_emo_ssl"])
        self.data["train_loss_ah_ssl"].append(train_log["loss_ah_ssl"])
        self.data["train_emo_mf1"].append(train_log["emo_mf1"])
        self.data["train_ah_mf1"].append(train_log["ah_mf1"])

        # SSL статистика (с .get для совместимости с GradNorm-логом)
        self.data["train_n_ssl_emo"].append(train_log.get("n_ssl_emo", 0))
        self.data["train_cov_ssl_emo"].append(train_log.get("cov_ssl_emo", 0.0))
        self.data["train_n_ssl_ah"].append(train_log.get("n_ssl_ah", 0))
        self.data["train_cov_ssl_ah"].append(train_log.get("cov_ssl_ah", 0.0))
        self.data["train_pseudo_emo_hist"].append(train_log.get("pseudo_emo_hist", [0]*7))

        self.data["val_loss"].append(val_log["loss"])
        self.data["val_loss_emo_sup"].append(val_log["loss_emo_sup"])
        self.data["val_loss_ah_sup"].append(val_log["loss_ah_sup"])
        self.data["val_emo_mf1"].append(val_log["emo_mf1"])
        self.data["val_emo_uar"].append(val_log["emo_uar"])
        self.data["val_ah_mf1"].append(val_log["ah_mf1"])
        self.data["val_ah_uar"].append(val_log["ah_uar"])
        self.data["val_overall_f1"].append(val_log["overall_f1"])
        self.data["lr"].append(optimizer.param_groups[0]["lr"])

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=4)
