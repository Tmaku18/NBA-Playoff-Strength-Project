"""Multi-temperature score aggregation for Model A inference.

When attention.multi_temp_enabled is true, Model A is run at several temperatures
(e.g. 1, 5, 10). This module combines the per-temperature scores into a single
score per team and a per-team confidence (e.g. from agreement across temps).
"""

from __future__ import annotations

import numpy as np


def aggregate_multi_temp_scores(
    scores_by_temp: dict[int, np.ndarray],
    base_weights: dict[int, float],
    starter_avail: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine per-temperature Model A scores into final score and confidence per team.

    Args:
        scores_by_temp: Map temperature -> (n_teams,) array of strength scores.
        base_weights: Map temperature -> weight (e.g. {1: 0.85, 5: 1.0, 10: 0.7}).
        starter_avail: Optional (n_teams,) availability; if set, used to scale confidence.

    Returns:
        s_final_batch: (n_teams,) weighted average of scores across temperatures.
        c_A_batch: (n_teams,) confidence in [0,1]; higher when scores agree across temps.
    """
    temps = list(scores_by_temp.keys())
    if not temps:
        raise ValueError("scores_by_temp must not be empty")
    first = np.asarray(scores_by_temp[temps[0]]).ravel()
    n_teams = len(first)
    for t in temps:
        arr = np.asarray(scores_by_temp[t]).ravel()
        if len(arr) != n_teams:
            arr = np.resize(arr, n_teams)
        scores_by_temp[t] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    total_w = sum(base_weights.get(int(t), 1.0) for t in temps)
    if total_w <= 0:
        total_w = 1.0
    s_final = np.zeros(n_teams, dtype=np.float64)
    for t in temps:
        w = base_weights.get(int(t), 1.0) / total_w
        s_final += w * scores_by_temp[t]

    # Confidence: high when scores agree across temps (low std)
    stack = np.stack([scores_by_temp[t] for t in temps], axis=0)
    std_per_team = np.std(stack, axis=0)
    std_max = np.max(std_per_team) if std_per_team.size else 1.0
    if std_max < 1e-9:
        std_max = 1.0
    c_A = 1.0 - np.minimum(1.0, std_per_team / std_max)
    c_A = np.clip(c_A, 0.0, 1.0).astype(np.float64)
    if starter_avail is not None and len(starter_avail) == n_teams:
        avail = np.asarray(starter_avail, dtype=np.float64)
        avail = np.clip(np.nan_to_num(avail, nan=0.0), 0.0, 1.0)
        c_A = c_A * avail
    return s_final, c_A
