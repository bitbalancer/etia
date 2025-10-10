"""Bootstrap-based edge confidence utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from ...data.Dataset import Dataset


def is_consistent_edge_L(m1_ij, m1_ji, m2_ij, m2_ji):
    """Check if two edges are consistent based on their orientation codes."""

    if m1_ij == m2_ij and m1_ji == m2_ji:
        return True

    if m1_ij == 1 and m1_ji == 1 and m2_ij != 0 and m2_ji != 0:
        return True
    if m1_ij == 2 and m1_ji == 1:
        return m2_ij == 2 and m2_ji in {1, 2, 3} or (m2_ij, m2_ji) == (1, 1)
    if m1_ij == 1 and m1_ji == 2:
        return m2_ji == 2 and m2_ij in {1, 2, 3} or (m2_ij, m2_ji) == (1, 1)
    if m1_ij == 2 and m1_ji == 3:
        return (m2_ij, m2_ji) in {(2, 1), (1, 1)}
    if m1_ij == 3 and m1_ji == 2:
        return (m2_ij, m2_ji) in {(1, 2), (1, 1)}
    if m1_ij == 2 and m1_ji == 2:
        return (m2_ij, m2_ji) in {(1, 1), (2, 1), (1, 2)}
    if m1_ij == 0 or m2_ij == 0:
        return False
    return False


@dataclass(frozen=True)
class _EdgeStats:
    source_index: int
    target_index: int
    forward_code: int
    backward_code: int
    consistency: float
    similarity: float


def calculate_confidence(
    dataset: Dataset,
    opt_conf: Mapping[str, Any] | Any,
    n_bootstraps: int = 50,
    *,
    sample_frac: float = 1.0,
    random_state: Optional[int] = None,
    progress: bool = True,
):
    """Estimate edge stability via non-parametric bootstrap.

    Parameters
    ----------
    dataset
        ``Dataset`` instance used to learn the optimal configuration.
    opt_conf
        Optimal configuration returned by OCT. Can be either a dictionary or an
        object exposing the same attributes (as in legacy versions).
    n_bootstraps
        Number of bootstrap repetitions.
    sample_frac
        Fraction of rows to draw (with replacement) for each bootstrap sample.
    random_state
        Optional seed for reproducibility.
    progress
        If ``True``, prints lightweight progress information.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Edge consistency and similarity counts (one row per edge, matching the
        original implementation for API compatibility).
    """

    if not isinstance(dataset, Dataset):
        raise TypeError("dataset must be an instance of ETIA.data.Dataset")
    if opt_conf is None:
        raise RuntimeError(
            "You need to have an optimal configuration before you can calculate "
            "the edge confidences"
        )
    if n_bootstraps < 1:
        raise ValueError("n_bootstraps must be at least 1")
    if not 0 < sample_frac <= 1.0:
        raise ValueError("sample_frac must be within (0, 1]")

    cfg_dict = _config_to_dict(opt_conf)
    model = cfg_dict.get("model")
    if model is None:
        raise ValueError("opt_conf must contain a 'model' instance")

    rng = np.random.default_rng(random_state)
    base_matrix, boot_mecs, _ = _bootstrap_matrices(
        dataset=dataset,
        cfg_dict=cfg_dict,
        model=model,
        n_bootstraps=n_bootstraps,
        sample_frac=sample_frac,
        rng=rng,
        progress=progress,
    )

    edge_stats = _collect_edge_stats(base_matrix, boot_mecs)
    consistency = np.array([[stat.consistency] for stat in edge_stats])
    similarity = np.array([[stat.similarity] for stat in edge_stats])
    return consistency, similarity


def edge_metrics_on_bootstraps(best_mec_matrix, bootstrapped_mec_matrix):
    """Backward-compatible wrapper around the refactored implementation."""

    best = np.asarray(best_mec_matrix)
    boots = [np.asarray(m) for m in bootstrapped_mec_matrix]
    stats = _collect_edge_stats(best, boots)
    consistency = np.array([[stat.consistency] for stat in stats])
    similarity = np.array([[stat.similarity] for stat in stats])
    return consistency, similarity


def bootstrapping_causal_graph(n_bootstraps, dataset, tiers, best_config, is_cat_var):
    """Return the bootstrapped MEC and graph matrices (legacy API)."""

    _ = tiers
    _ = is_cat_var

    cfg_dict = _config_to_dict(best_config)
    model = cfg_dict.get("model")
    if model is None:
        raise ValueError("best_config must contain a 'model' instance")

    rng = np.random.default_rng()
    _, boot_mecs, boot_graphs = _bootstrap_matrices(
        dataset=dataset,
        cfg_dict=cfg_dict,
        model=model,
        n_bootstraps=n_bootstraps,
        sample_frac=1.0,
        rng=rng,
        progress=False,
    )

    return boot_mecs, boot_graphs


def _collect_edge_stats(best_matrix: np.ndarray, boot_matrices: Sequence[np.ndarray]) -> List[_EdgeStats]:
    if not boot_matrices:
        raise RuntimeError("No bootstrap matrices supplied")

    n_boot = len(boot_matrices)
    n_nodes = best_matrix.shape[0]
    stats: List[_EdgeStats] = []

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            forward = int(best_matrix[i, j])
            backward = int(best_matrix[j, i])
            if forward == 0 and backward == 0:
                continue

            consistency = 0
            similarity = 0
            for boot in boot_matrices:
                if is_consistent_edge_L(forward, backward, int(boot[i, j]), int(boot[j, i])):
                    consistency += 1
                if boot[i, j] == forward and boot[j, i] == backward:
                    similarity += 1

            stats.append(
                _EdgeStats(
                    source_index=i,
                    target_index=j,
                    forward_code=forward,
                    backward_code=backward,
                    consistency=consistency / n_boot,
                    similarity=similarity / n_boot,
                )
            )

    return stats


def _config_to_dict(opt_conf: Mapping[str, Any] | Any) -> MutableMapping[str, Any]:
    if isinstance(opt_conf, Mapping):
        return dict(opt_conf)

    # Fall back to attribute introspection for legacy _ConfigWrapper usage.
    attrs = {
        key: getattr(opt_conf, key)
        for key in dir(opt_conf)
        if not key.startswith("_") and not callable(getattr(opt_conf, key))
    }
    return dict(attrs)


def _config_for_dataset(opt_conf: Mapping[str, Any], dataset: Dataset) -> MutableMapping[str, Any]:
    config = dict(opt_conf)
    config["var_type"] = dataset.get_data_type_info()["var_type"]
    config.pop("matrix_mec_graph", None)
    config.pop("matrix_graph", None)
    config.pop("indexes", None)
    return config


def _bootstrap_sample(dataframe, rng: np.random.Generator, sample_frac: float):
    n_rows = len(dataframe)
    sample_size = max(1, int(round(n_rows * sample_frac)))
    indices = rng.integers(0, n_rows, size=sample_size)
    return dataframe.iloc[indices].reset_index(drop=True)


def _dataset_clone(base_dataset: Dataset, sample_df, suffix: str) -> Dataset:
    return Dataset(
        data=sample_df,
        data_time_info=dict(base_dataset.get_data_time_info()),
        time_series=base_dataset.time_series,
        dataset_name=f"{base_dataset.dataset_name}_bootstrap_{suffix}",
    )


def _bootstrap_matrices(
    dataset: Dataset,
    cfg_dict: Mapping[str, Any],
    model: Any,
    n_bootstraps: int,
    sample_frac: float,
    rng: np.random.Generator,
    progress: bool,
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    base_config = _config_for_dataset(cfg_dict, dataset)
    base_value = cfg_dict.get("matrix_mec_graph")
    if base_value is not None:
        base_mec = _to_numpy(base_value)
    else:
        base_mec_df, _, _ = model.run(dataset, base_config, prepare_data=True)
        base_mec = base_mec_df.to_numpy()

    boot_mecs: List[np.ndarray] = []
    boot_graphs: List[np.ndarray] = []

    for b in range(n_bootstraps):
        if progress:
            print(f"Bootstrap {b + 1}/{n_bootstraps}...")
        sampled_df = _bootstrap_sample(dataset.get_dataset(), rng, sample_frac)
        boot_dataset = _dataset_clone(dataset, sampled_df, suffix=str(b))
        boot_config = _config_for_dataset(cfg_dict, boot_dataset)
        boot_mec_df, boot_graph_df, _ = model.run(boot_dataset, boot_config, prepare_data=True)
        boot_mecs.append(boot_mec_df.to_numpy())
        boot_graphs.append(_to_numpy(boot_graph_df))

    return base_mec, boot_mecs, boot_graphs


def _to_numpy(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "to_numpy"):
        return matrix.to_numpy()
    return np.asarray(matrix)
