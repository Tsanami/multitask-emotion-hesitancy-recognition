from sklearn.metrics import classification_report, mean_absolute_error
import numpy as np

def mf1(targets: list[np.ndarray] | np.ndarray,
                         predicts: list[np.ndarray] | np.ndarray,
                         return_scores: bool = False) -> float | tuple[float, list[float]]:
    """Calculates mean Macro F1 score (emotional multilabel mMacroF1)

    Args:
        targets: Targets array (ground truth)
        predicts: Predicts array (model predictions)
        return_scores: If True, returns both mean and per-class scores

    Returns:
        float: Mean Macro F1 score across all classes
        or
        tuple[float, list[float]]: If return_scores=True, returns (mean, per_class_scores)
    """
    targets = np.array(targets)
    predicts = np.array(predicts)

    f1_macro_scores = []
    for i in range(predicts.shape[1]):
        cr = classification_report(targets[:, i], predicts[:, i],
                                         output_dict=True, zero_division=0)
        f1_macro_scores.append(cr['macro avg']['f1-score'])

    if return_scores:
        return np.mean(f1_macro_scores), f1_macro_scores
    return np.mean(f1_macro_scores)


def uar(targets: list[np.ndarray] | np.ndarray,
                    predicts: list[np.ndarray] | np.ndarray,
                    return_scores: bool = False) -> float | tuple[float, list[float]]:
    """Calculates mean Unweighted Average Recall (emotional multilabel mUAR)

    Args:
        targets: Targets array (ground truth)
        predicts: Predicts array (model predictions)
        return_scores: If True, returns both mean and per-class scores

    Returns:
        float: Mean UAR across all classes
        or
        tuple[float, list[float]]: If return_scores=True, returns (mean, per_class_scores)
    """
    targets = np.array(targets)
    predicts = np.array(predicts)

    uar_scores = []
    for i in range(predicts.shape[1]):
        cr = classification_report(targets[:, i], predicts[:, i],
                                         output_dict=True, zero_division=0)
        uar_scores.append(cr['macro avg']['recall'])

    if return_scores:
        return np.mean(uar_scores), uar_scores
    return np.mean(uar_scores)

def mwacc(targets: list[np.ndarray] | np.ndarray,
          predicts: list[np.ndarray] | np.ndarray,
          return_scores: bool = False) -> float | tuple[float, list[float]]:
    """Mean weighted accuracy (для multilabel-эмоций).

    mwacc = 1/C * sum_{i=1}^C 1/2 * ( TP/(TP+FN) + TN/(TN+FP) )

    Для каждой бинарной колонки-класса i считается balanced accuracy
    (полусумма sensitivity и specificity), затем среднее по C колонкам.
    C = число колонок (для эмоций = 6, Neutral отброшен ещё на этапе trues/preds).

    Совпадает по смыслу с macro-avg recall (uar), но считается напрямую
    из счётчиков ошибок, без classification_report. Деление на ноль → 0.0.
    """
    targets = np.array(targets)
    predicts = np.array(predicts)

    scores = []
    for i in range(predicts.shape[1]):
        t = targets[:, i]
        p = predicts[:, i]
        tp = int(np.sum((t == 1) & (p == 1)))
        fn = int(np.sum((t == 1) & (p == 0)))
        tn = int(np.sum((t == 0) & (p == 0)))
        fp = int(np.sum((t == 0) & (p == 1)))
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # recall / sensitivity
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0   # specificity
        scores.append(0.5 * (sens + spec))

    if return_scores:
        return float(np.mean(scores)), scores
    return float(np.mean(scores))


def wf1(targets: list[np.ndarray] | np.ndarray,
        predicts: list[np.ndarray] | np.ndarray) -> float:
    """Weighted-average F1 (взвешенный по поддержке классов). Для бинарного BAH.

    В отличие от mf1 (macro, per-column-then-mean), это одна F1, усреднённая
    по классам с весами = доля примеров класса. Принимает плоские метки 0/1.
    """
    targets = np.array(targets).ravel()
    predicts = np.array(predicts).ravel()
    cr = classification_report(targets, predicts, output_dict=True, zero_division=0)
    return float(cr['weighted avg']['f1-score'])


def acc_func(trues, preds):
    # print('acc', trues, preds)
    acc = []
    for i in range(5):
        acc.append(mean_absolute_error(trues[:, i], preds[:, i]))
    acc = 1 - np.asarray(acc)
    return np.mean(acc)