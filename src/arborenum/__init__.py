import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from matplotlib import colormaps
from ._core import (
    ArborEnum as _ArborEnumCore,
    rid_subtractive_model_reliance as _rid_subtractive_core,
    rid_subtractive_model_reliance_continuous as _rid_subtractive_continuous_core,
)
from ._threshold_guessing import ThresholdGuessBinarizer

import json
from pathlib import Path
from importlib.resources import files

DEFER_PREDICTION = -1

__all__ = ["ArborEnum", "ThresholdGuessBinarizer", "DEFER_PREDICTION"]

def _json_safe(x):
    if isinstance(x, np.integer):
        return int(x)

    if isinstance(x, np.floating):
        return float(x)

    if isinstance(x, np.ndarray):
        return x.tolist()

    if isinstance(x, dict):
        return {
            str(k): _json_safe(v)
            for k, v in x.items()
        }

    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]

    return x

def _normalize_key(s: str) -> str:
    # lower, trim, and make separators uniform
    s = str(s).strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    # collapse repeats
    while "__" in s:
        s = s.replace("__", "_")
    return s

_PROXY_STYLE_MAP = {
    # explicit canonical names
    "recursively_choose_best_split": 0,
    "block_split": 1,
    "split_without_postprocessing": 3,

    # synonyms
    "licketysplit": 0,
    "lickety_split": 0,
    "lickety": 0,
    "greedy": 0,
    "default": 0,
    "recursive": 0,
    "recursively_choose": 0,
    "choose_best_split": 0,

    "block": 1,
    "cyclic": 1,
    "cyclic_k": 1,

    "no_postprocessing": 3,
    "no_postprocess": 3,
    "without_postprocessing": 3,
    "split_no_postprocessing": 3,
    "split": 3,
}

_KEY_MODE_MAP = {
    "hash": "hash",
    "hash64": "hash",
    "hash_64": "hash",
    "64": "hash",
    "64bit": "hash",
    "64_bit": "hash",
    "fingerprint64": "hash",
    "fingerprint_64": "hash",

    "hash128": "hash128",
    "hash_128": "hash128",
    "128": "hash128",
    "128bit": "hash128",
    "128_bit": "hash128",
    "fingerprint128": "hash128",
    "fingerprint_128": "hash128",

    "exact": "exact",
    "bitvector": "exact",
    "bit_vector": "exact",

    "literal": "lits_exact",
    "lits": "lits_exact",
    "lits_exact": "lits_exact",
    "itemset": "lits_exact",
}


def parse_key_mode(key_mode):
    if isinstance(key_mode, (int, np.integer)):
        v = int(key_mode)
        if v == 64:
            return "hash"
        if v == 128:
            return "hash128"
        raise ValueError("key_mode as an integer must be 64 or 128.")

    key = _normalize_key(key_mode)
    if key in _KEY_MODE_MAP:
        return _KEY_MODE_MAP[key]

    allowed = sorted(set(_KEY_MODE_MAP.keys()))
    raise ValueError(
        f"Unknown key_mode='{key_mode}'. "
        f"Supported: 64, 128, 'hash', 'hash128', 'exact', or 'lits_exact'. "
        f"Aliases: {allowed}"
    )


def parse_proxy_style(proxy_style):
    # accepts int or string
    if isinstance(proxy_style, (int, np.integer)):
        v = int(proxy_style)
        if v in (0, 1, 3):
            return v
        raise ValueError(
            f"proxy_style={v} is not supported. "
            f"Supported oracle styles are 0, 1, 3."
        )

    key = _normalize_key(proxy_style)
    if key in _PROXY_STYLE_MAP:
        return _PROXY_STYLE_MAP[key]

    allowed = sorted(set(_PROXY_STYLE_MAP.keys()))
    raise ValueError(
        f"Unknown proxy_style='{proxy_style}'. "
        f"Supported: oracle_style 0/1/3, or one of: {allowed}"
    )

_GREEDY_HEURISTIC_MAP = {
    "entropy": 0,
    "info_gain": 0,
    "information_gain": 0,
    "ig": 0,

    "entropy_depth1_exact": 1,
    "entropy_with_depth1_exact": 1,
    "depth1_exact": 1,
    "default": 1,

    "best_split_for_leaves": 2,
    "min_child_leaf_objective": 2,
    "min_child_objective": 2,
    "always_misclassification_minimizing": 2,
    "misclassification_minimizing": 2,
    "misclassification_based": 2,
}

_GREEDY_CONTINUOUS_MODE_MAP = {
    "binary": "binary",
    "binarized": "binary",
    "threshold": "binary",
    "thresholds": "binary",
    "current": "binary",
    "default": "binary",

    "numerical": "numerical",
    "numeric": "numerical",
    "sorted": "numerical",
    "cart": "numerical",
    "cart_style": "numerical",
}


def parse_greedy_continuous_mode(greedy_continuous_mode):
    if isinstance(greedy_continuous_mode, (int, np.integer)):
        v = int(greedy_continuous_mode)
        if v == 0:
            return "binary"
        if v == 1:
            return "numerical"
        raise ValueError(
            f"greedy_continuous_mode={v} is invalid. Supported: 0/1."
        )

    key = _normalize_key(greedy_continuous_mode)
    if key in _GREEDY_CONTINUOUS_MODE_MAP:
        return _GREEDY_CONTINUOUS_MODE_MAP[key]

    allowed = sorted(set(_GREEDY_CONTINUOUS_MODE_MAP.keys()))
    raise ValueError(
        f"Unknown greedy_continuous_mode='{greedy_continuous_mode}'. "
        f"Supported: 0/1, or one of: {allowed}"
    )

def parse_heuristic_for_greedy(heuristic_for_greedy):
    # accepts int or string
    if isinstance(heuristic_for_greedy, (int, np.integer)):
        v = int(heuristic_for_greedy)
        if v in (0, 1, 2):
            return v
        raise ValueError(
            f"heuristic_for_greedy={v} is invalid. Supported: 0,1,2."
        )

    key = _normalize_key(heuristic_for_greedy)
    if key in _GREEDY_HEURISTIC_MAP:
        return _GREEDY_HEURISTIC_MAP[key]

    allowed = sorted(set(_GREEDY_HEURISTIC_MAP.keys()))
    raise ValueError(
        f"Unknown heuristic_for_greedy='{heuristic_for_greedy}'. "
        f"Supported: 0/1/2, or one of: {allowed}"
    )


_PROXY_MODE_MAP = {
    "continuous": "continuous",
    "full": "continuous",
    "unrestricted": "continuous",

    "hybrid": "hybrid",
    "lickety_continuous": "hybrid",
    "continuous_lickety": "hybrid",

    "binarized": "binarized",
    "binary": "binarized",
    "restricted": "binarized",
}

_PROXY_REFINEMENT_MAP = {
    "off": 0,
    "none": 0,
    "never": 0,
    "on": 1,
    "always": 1,
    "auto": 2,
    "automatic": 2,
}


def parse_proxy_mode(proxy_mode):
    key = _normalize_key(proxy_mode)
    if key not in _PROXY_MODE_MAP:
        raise ValueError(
            "proxy_mode must be 'continuous', 'hybrid', or 'binarized'."
        )
    return _PROXY_MODE_MAP[key]


def parse_proxy_refinement(proxy_refinement):
    if isinstance(proxy_refinement, (int, np.integer)):
        value = int(proxy_refinement)
        if value in (0, 1, 2):
            return value
        raise ValueError(
            "proxy_refinement as an integer must be 0, 1, or 2."
        )

    key = _normalize_key(proxy_refinement)
    if key not in _PROXY_REFINEMENT_MAP:
        raise ValueError(
            "proxy_refinement must be 'off', 'on', or 'auto'."
        )
    return _PROXY_REFINEMENT_MAP[key]


def _proxy_mode_settings(proxy_mode):
    mode = parse_proxy_mode(proxy_mode)

    if mode == "continuous":
        return {
            "mode": mode,
            "restrict_lickety": False,
            "restrict_depthd": False,
            "restrict_greedy": False,
            "continuous_lickety": True,
            "continuous_depthd": True,
            "continuous_greedy": True,
            "needs_proxy_binarization": False,
        }

    if mode == "hybrid":
        return {
            "mode": mode,
            "restrict_lickety": False,
            "restrict_depthd": True,
            "restrict_greedy": True,
            "continuous_lickety": True,
            "continuous_depthd": False,
            "continuous_greedy": False,
            "needs_proxy_binarization": True,
        }

    return {
        "mode": mode,
        "restrict_lickety": True,
        "restrict_depthd": True,
        "restrict_greedy": True,
        "continuous_lickety": False,
        "continuous_depthd": False,
        "continuous_greedy": False,
        "needs_proxy_binarization": True,
    }

def _resolve_rid_root_budget(root_budget):
    if root_budget is None:
        return -1

    if isinstance(root_budget, (bool, np.bool_)) or not isinstance(
        root_budget,
        (int, np.integer),
    ):
        raise TypeError(
            "root_budget must be an integer or None."
        )

    root_budget = int(root_budget)

    if root_budget < 0:
        raise ValueError(
            "root_budget must be nonnegative or None."
        )

    return root_budget


def _as_2d_numeric_array(X, name):
    if hasattr(X, "to_numpy"):
        X = X.to_numpy()

    X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {X.shape}")

    try:
        X = X.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must contain only numerical values."
        ) from exc

    if not np.all(np.isfinite(X)):
        raise ValueError(
            f"{name} contains NaN or infinite values."
        )

    return X

def _shift_column_values_to_midpoints(column):
    # replace each observed value except the maximum with the midpoint
    # between it and the next distinct observed value.
    # used for training because splits are directly on unique values in the cpp

    column = np.asarray(column, dtype=np.float64)
    unique = np.unique(column)

    if unique.size <= 1:
        return column

    left = unique[:-1]
    right = unique[1:]

    midpoints = left + 0.5 * (right - left)

    valid = (
        np.isfinite(midpoints)
        & (midpoints > left)
        & (midpoints < right)
    )

    shifted_unique = unique.copy()
    shifted_unique[:-1] = np.where(
        valid,
        midpoints,
        left,
    )

    positions = np.searchsorted(unique, column)
    return shifted_unique[positions]

def _split_binary_and_continuous(
    X,
    binary_unique_threshold=7,
):
    X = _as_2d_numeric_array(X, "X")

    binary_columns = []
    binary_feature_specs = []

    numerical_columns = []
    numerical_feature_indices = []

    for j in range(X.shape[1]):
        column = X[:, j]
        unique = np.unique(column)

        if unique.size <= 1:
            continue

        left = unique[:-1]
        right = unique[1:]

        thresholds = left + 0.5 * (right - left)

        valid = (
            np.isfinite(thresholds)
            & (thresholds > left)
            & (thresholds < right)
        )

        thresholds = np.where(
            valid,
            thresholds,
            left,
        )

        if unique.size <= binary_unique_threshold:
            # python directly creates the midpoint threshold columns for low cardinality features
            for threshold in thresholds:
                binary_columns.append(
                    (column <= threshold).astype(np.uint8)
                )

                binary_feature_specs.append(
                    {
                        "original_feature": j,
                        "cutpoint": float(threshold),
                    }
                )

        else:
            # shift each observed level so the existing C++ threshold
            # generation (split on unique values) produces the midpoint cutpoints.
            shifted_column = _shift_column_values_to_midpoints(
                column
            )

            numerical_columns.append(
                shifted_column.astype(
                    np.float64,
                    copy=False,
                )
            )

            numerical_feature_indices.append(j)

    n = X.shape[0]

    X_bin = (
        np.column_stack(binary_columns).astype(
            np.uint8,
            copy=False,
        )
        if binary_columns
        else np.empty((n, 0), dtype=np.uint8)
    )

    X_num = (
        np.column_stack(numerical_columns).astype(
            np.float64,
            copy=False,
        )
        if numerical_columns
        else np.empty((n, 0), dtype=np.float64)
    )

    return (
        X,
        X_bin,
        X_num,
        binary_feature_specs,
        numerical_feature_indices,
    )


def _optional_binary_matrix(X, n, name):
    if X is None:
        return np.empty((n, 0), dtype=np.uint8)

    X = np.asarray(X, dtype=np.uint8)

    if X.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {X.shape}")

    if X.shape[0] != n:
        raise ValueError(
            f"{name} rows must match X rows: got {X.shape[0]} vs {n}"
        )

    return X


def _resolve_proxy_matrix(
    X_original,
    y,
    X_proxy,
    proxy_settings,
    proxy_features_n_estimators,
    proxy_features_max_depth,
    proxy_features_random_state,
    proxy_features_column_elimination,
):
    X_proxy = _optional_binary_matrix(
        X_proxy,
        X_original.shape[0],
        "X_proxy",
    )

    if (
        proxy_settings["needs_proxy_binarization"]
        and X_proxy.shape[1] < 2
    ):
        X_proxy = ThresholdGuessBinarizer(
            n_estimators=int(proxy_features_n_estimators),
            max_depth=int(proxy_features_max_depth),
            random_state=int(proxy_features_random_state),
            column_elimination=bool(proxy_features_column_elimination),
        ).fit_transform(
            X_original,
            y,
        ).astype(np.uint8, copy=False)

    return X_proxy

def _prepare_bb_pred(
    bb_pred,
    n_rows,
    *,
    use_deferral=False,
    n_classes=None,
):
    if bb_pred is None:
        bb_pred_vec = np.empty(0, dtype=np.int32)
    else:
        bb_pred_arr = np.asarray(bb_pred)

        if bb_pred_arr.ndim != 1:
            raise ValueError(
                "bb_pred must be 1D, "
                f"got shape {bb_pred_arr.shape}"
            )

        if bb_pred_arr.shape[0] != n_rows:
            raise ValueError(
                "bb_pred length must match the number of rows: "
                f"got {bb_pred_arr.shape[0]} vs {n_rows}"
            )

        if not np.issubdtype(bb_pred_arr.dtype, np.integer):
            if (
                np.issubdtype(bb_pred_arr.dtype, np.floating)
                and np.all(np.isfinite(bb_pred_arr))
                and np.all(bb_pred_arr == np.floor(bb_pred_arr))
            ):
                bb_pred_arr = bb_pred_arr.astype(np.int32)
            else:
                raise ValueError(
                    "bb_pred must contain integer class predictions."
                )

        bb_pred_vec = np.asarray(
            bb_pred_arr,
            dtype=np.int32,
        )

        if np.any(bb_pred_vec < 0):
            raise ValueError(
                "bb_pred must contain nonnegative class predictions."
            )

        if (
            n_classes is not None
            and np.any(bb_pred_vec >= int(n_classes))
        ):
            raise ValueError(
                "bb_pred contains class predictions outside "
                f"[0, {int(n_classes)})."
            )

    if use_deferral and bb_pred_vec.size == 0:
        raise ValueError(
            "bb_pred is required when use_deferral=True."
        )

    return bb_pred_vec

def _prepare_matched_groups_by_variable(
    matched_groups_by_variable,
    n_rows,
    n_variables,
):
    if matched_groups_by_variable is None:
        return []

    partitions = list(matched_groups_by_variable)

    # empty outer list means ordinary permutation importance.
    if len(partitions) == 0:
        return []

    if len(partitions) != int(n_variables):
        raise ValueError(
            "matched_groups_by_variable must contain exactly one "
            f"partition per variable: expected {n_variables}, "
            f"got {len(partitions)}."
        )

    out = []

    for variable_index, variable_groups in enumerate(partitions):
        variable_groups = list(variable_groups)

        if len(variable_groups) == 0:
            raise ValueError(
                f"matched_groups_by_variable[{variable_index}] "
                "contains no groups."
            )

        seen = np.zeros(int(n_rows), dtype=bool)
        groups = []

        for group_index, group in enumerate(variable_groups):
            rows = [int(row) for row in group]

            if len(rows) == 0:
                raise ValueError(
                    "matched_groups_by_variable"
                    f"[{variable_index}][{group_index}] is empty."
                )

            for row in rows:
                if row < 0 or row >= n_rows:
                    raise ValueError(
                        "matched_groups_by_variable contains row "
                        f"{row}, but valid row indices are "
                        f"[0, {n_rows - 1}]."
                    )

                if seen[row]:
                    raise ValueError(
                        f"Row {row} appears in more than one matched "
                        f"group for variable {variable_index}."
                    )

                seen[row] = True

            groups.append(rows)

        if not np.all(seen):
            missing = np.flatnonzero(~seen)

            raise ValueError(
                f"Matched groups for variable {variable_index} must "
                "partition all rows. Missing rows: "
                f"{missing.tolist()}"
            )

        out.append(groups)

    return out

class ArborEnum:
    def __init__(self):
        self._model = _ArborEnumCore()
        self._rid_out = None
        self._rid_feature_indices = None

        self.binary_feature_specs_ = None
        self.continuous_feature_indices_ = None
        self.n_features_in_ = None

    def fit(
        self,
        X,
        y,
        *,
        X_proxy=None,
        X_initial=None,
        early_stopping=False,
        proxy_mode="hybrid",
        greedy_continuous_mode="binary",
        binary_unique_threshold=2,
        proxy_features_n_estimators=150,
        proxy_features_max_depth=2,
        proxy_features_random_state=0,
        proxy_features_column_elimination=False,
        lambda_reg=0.01,
        depth_budget=5,
        rashomon_mult=0.01,
        second_rashomon_mult=None,
        multiplier_step_size=0.01,
        lookahead_k=1,
        proxy_refinement="auto",
        refinement_width=1,
        max_refinement_rounds=-1,
        runtime_limit_seconds=-1.0,
        memory_limit_mb=-1.0,
        max_number_thresholds_per_feature=None,
        multiplicative_slack=0.0,
        key_mode="hash",
        proxy_style=0,
        use_budget_refinement=True,
        guarantee_rule_list_recovery=False,
        majority_leaf_only=False,
        cache_early_exits=False,
        heuristic_for_greedy=1,
        proxy_caching=True,
        trie_cache_enabled=True,
        stronger_rollout=False,
        root_budget=None,
        rashomon_mode=True,
        use_deferral=False,
        eta_defer=0.0,
        bb_pred=None,
        additive=False,
    ):
        # fit directly from a mixed numerical matrix.
        # columns with at most binary_unique_threshold unique values are
        # threshold-binarized and treated as ordinary binary features. columns
        # with more unique values are treated as continuous features.
        # proxy_mode: continuous, hybrid (greedy subroutine is binarized), or binarized (everything is binarized)
        # early_stopping=False runs the deterministic continuous algorithm.
        # early_stopping=True runs the anytime algorithm (with some small overhead)
        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = [
                f"f{j}" for j in range(np.asarray(X).shape[1])
            ]

        (X_original, X_bin, X_num, self.binary_feature_specs_, self.continuous_feature_indices_,) = _split_binary_and_continuous(
            X,
            binary_unique_threshold=binary_unique_threshold,
        )

        y = np.asarray(y, dtype=int)
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}")
        if y.shape[0] != X_original.shape[0]:
            raise ValueError(
                f"y length must match X rows: got {y.shape[0]} vs "
                f"{X_original.shape[0]}"
            )

        proxy_settings = _proxy_mode_settings(proxy_mode)

        X_proxy = _resolve_proxy_matrix(
            X_original=X_original,
            y=y,
            X_proxy=X_proxy,
            proxy_settings=proxy_settings,
            proxy_features_n_estimators=proxy_features_n_estimators,
            proxy_features_max_depth=proxy_features_max_depth,
            proxy_features_random_state=proxy_features_random_state,
            proxy_features_column_elimination=proxy_features_column_elimination,
        )

        n = X_original.shape[0]

        self.root_n_ = int(n)
        self.lambda_reg_ = float(lambda_reg)
        self.gamma_ = int(round(float(lambda_reg) * int(n)))
        self.depth_budget_ = int(depth_budget)
        self.rashomon_mult_ = float(rashomon_mult)
        self.multiplicative_slack_ = float(multiplicative_slack)
        self.lookahead_k_ = int(lookahead_k)
        self.eta_defer_ = float(eta_defer)

        if X_initial is None:
            if early_stopping and X_num.shape[1] > 0:
                if proxy_settings["mode"] == "binarized":
                    X_initial = X_proxy
                else:
                    X_initial = y.reshape(-1, 1).astype(np.uint8)
            else:
                X_initial = np.empty((n, 0), dtype=np.uint8)
        else:
            X_initial = _optional_binary_matrix(
                X_initial,
                n,
                "X_initial",
            )

        if not early_stopping and (
            float(runtime_limit_seconds) >= 0.0
            or float(memory_limit_mb) >= 0.0
        ):
            raise ValueError(
                "runtime_limit_seconds and memory_limit_mb are only "
                "available when early_stopping=True."
            )

        self.prepare_continuous_data(
            X_num=X_num,
            X_bin=X_bin,
            y=y,
            X_initial_active=X_initial,
            X_proxy_active=X_proxy,
            max_number_thresholds_per_feature=(
                max_number_thresholds_per_feature
            ),
            bb_pred=bb_pred,
        )

        self.n_features_in_ = X_original.shape[1]

        if early_stopping:
            return self.fit_prepared_anytime(
                lambda_reg=lambda_reg,
                depth_budget=depth_budget,
                rashomon_mult=rashomon_mult,
                second_rashomon_mult=second_rashomon_mult,
                multiplier_step_size=multiplier_step_size,
                multiplicative_slack=multiplicative_slack,
                key_mode=key_mode,
                lookahead_k=lookahead_k,
                proxy_style=proxy_style,
                use_budget_refinement=use_budget_refinement,
                guarantee_rule_list_recovery=(
                    guarantee_rule_list_recovery
                ),
                majority_leaf_only=majority_leaf_only,
                cache_early_exits=cache_early_exits,
                heuristic_for_greedy=heuristic_for_greedy,
                greedy_continuous_mode=greedy_continuous_mode,
                proxy_caching=proxy_caching,
                proxy_threshold_features=None,
                initial_active_threshold_features=None,
                refinement_width=refinement_width,
                max_refinement_rounds=max_refinement_rounds,
                proxy_refinement_mode=parse_proxy_refinement(
                    proxy_refinement
                ),
                continuous_proxy_in_lickety=(
                    proxy_settings["continuous_lickety"]
                ),
                continuous_proxy_in_depthd_exact=(
                    proxy_settings["continuous_depthd"]
                ),
                continuous_proxy_in_greedy=(
                    proxy_settings["continuous_greedy"]
                ),
                trie_cache_enabled=trie_cache_enabled,
                runtime_limit_seconds=runtime_limit_seconds,
                memory_limit_mb=memory_limit_mb,
                use_deferral=use_deferral,
                eta_defer=eta_defer,
                
            )

        return self.fit_prepared(
            lambda_reg=lambda_reg,
            depth_budget=depth_budget,
            rashomon_mult=rashomon_mult,
            multiplicative_slack=multiplicative_slack,
            key_mode=key_mode,
            lookahead_k=lookahead_k,
            proxy_style=proxy_style,
            root_budget=root_budget,
            use_budget_refinement=use_budget_refinement,
            guarantee_rule_list_recovery=guarantee_rule_list_recovery,
            majority_leaf_only=majority_leaf_only,
            cache_early_exits=cache_early_exits,
            heuristic_for_greedy=heuristic_for_greedy,
            greedy_continuous_mode=greedy_continuous_mode,
            proxy_caching=proxy_caching,
            restrict_proxy_in_lickety=(
                proxy_settings["restrict_lickety"]
            ),
            restrict_proxy_in_depthd_exact=(
                proxy_settings["restrict_depthd"]
            ),
            restrict_proxy_in_greedy=(
                proxy_settings["restrict_greedy"]
            ),
            rashomon_mode=rashomon_mode,
            trie_cache_enabled=trie_cache_enabled,
            stronger_rollout=stronger_rollout,
            use_deferral=use_deferral,
            eta_defer=eta_defer,
        )

    

    def fit_binarized(
        self,
        X,
        y,
        lambda_reg=0.01,
        depth_budget=5,
        rashomon_mult=0.01,
        multiplicative_slack=0.0,
        key_mode="hash",
        lookahead_k=1,
        proxy_style=0, 
        root_budget=None, 
        use_budget_refinement=True, 
        guarantee_rule_list_recovery=False,
        majority_leaf_only=False,
        cache_early_exits=False,
        heuristic_for_greedy=1,
        greedy_continuous_mode="binary",
        proxy_caching=True,
        allowed_proxy_features=None,
        restrict_proxy_in_lickety=False,
        restrict_proxy_in_depthd_exact=False,
        restrict_proxy_in_greedy=False,
        rashomon_mode=True,
        continuous_starts=None,
        trie_cache_enabled=True,
        stronger_rollout=False,
        use_deferral=False,
        eta_defer=0.0,
        bb_pred=None,
    ):
        X = np.asarray(X, dtype=np.uint8)
        y = np.asarray(y, dtype=int)
        if X.ndim != 2:
            raise ValueError(
                f"X must be 2D, got shape {X.shape}"
            )

        if y.ndim != 1:
            raise ValueError(
                f"y must be 1D, got shape {y.shape}"
            )

        if y.shape[0] != X.shape[0]:
            raise ValueError(
                "y length must match X rows: "
                f"got {y.shape[0]} vs {X.shape[0]}"
            )

        eta_defer = float(eta_defer)

        if not np.isfinite(eta_defer) or eta_defer < 0.0:
            raise ValueError(
                "eta_defer must be finite and nonnegative."
            )

        bb_pred_vec = _prepare_bb_pred(
            bb_pred,
            X.shape[0],
            use_deferral=bool(use_deferral),
        )



        if continuous_starts is None:
            continuous_starts_vec = []
        else:
            continuous_starts_vec = [int(v) for v in continuous_starts]
            continuous_starts_vec = sorted(set(continuous_starts_vec))

            n_features = X.shape[1]
            bad = [v for v in continuous_starts_vec if v < 0 or v >= n_features]
            if bad:
                raise ValueError(
                    f"continuous_starts contains invalid feature indices {bad}; "
                    f"valid range is [0, {n_features - 1}]"
                )
        if allowed_proxy_features is None:
            allowed_proxy_features_vec = []
        else:
            allowed_proxy_features_vec = [int(v) for v in allowed_proxy_features]
            allowed_proxy_features_vec = sorted(set(allowed_proxy_features_vec))

            n_features = X.shape[1]
            bad = [v for v in allowed_proxy_features_vec if v < 0 or v >= n_features]
            if bad:
                raise ValueError(
                    f"allowed_proxy_features contains invalid feature indices {bad}; "
                    f"valid range is [0, {n_features - 1}]"
                )

        proxy_style_int = parse_proxy_style(proxy_style)
        greedy_heur_int = parse_heuristic_for_greedy(heuristic_for_greedy)
        greedy_cont_mode = parse_greedy_continuous_mode(greedy_continuous_mode)
        
        if root_budget is None:
            root_budget_int = -1
        else:
            root_budget_int = int(root_budget)

        key_mode_parsed = parse_key_mode(key_mode)
        self._model.fit(
            X,
            y,
            lambda_reg,
            depth_budget,
            rashomon_mult,
            multiplicative_slack,
            key_mode_parsed,
            bool(trie_cache_enabled),
            lookahead_k,
            root_budget_int,
            bool(use_budget_refinement), 
            bool(guarantee_rule_list_recovery), 
            int(proxy_style_int), 
            bool(majority_leaf_only),
            bool(cache_early_exits),
            int(greedy_heur_int),
            greedy_cont_mode,
            bool(proxy_caching),
            allowed_proxy_features_vec,
            bool(restrict_proxy_in_lickety),
            bool(restrict_proxy_in_depthd_exact),
            bool(restrict_proxy_in_greedy),
            bool(rashomon_mode),
            continuous_starts_vec,
            bool(stronger_rollout),
            bool(use_deferral),
            float(eta_defer),
            bb_pred_vec,
        )

        return self


    def fit_binarized_repeated_subsamples(
        self,
        X,
        y,
        *,
        lambda_reg=0.01,
        depth_budget=5,
        rashomon_mult=0.01,
        multiplicative_slack=0.0,
        key_mode="hash",
        lookahead_k=1,
        proxy_style=0,
        use_budget_refinement=True,
        guarantee_rule_list_recovery=False,
        majority_leaf_only=False,
        cache_early_exits=False,
        heuristic_for_greedy=1,
        greedy_continuous_mode="binary",
        proxy_caching=True,
        allowed_proxy_features=None,
        restrict_proxy_in_lickety=False,
        restrict_proxy_in_depthd_exact=False,
        restrict_proxy_in_greedy=False,
        continuous_starts=None,
        trie_cache_enabled=True,
        stronger_rollout=False,
        use_deferral=False,
        eta_defer=0.0,
        bb_pred=None,
        subsample_fraction=0.8,
        num_subsamples=50,
        seed=0,
        reuse_caches_between_subsamples=False,
    ):
        X = np.asarray(X, dtype=np.uint8)
        y = np.asarray(y, dtype=int)

        if X.ndim != 2:
            raise ValueError(
                f"X must be 2D, got shape {X.shape}"
            )

        if y.ndim != 1:
            raise ValueError(
                f"y must be 1D, got shape {y.shape}"
            )

        if y.shape[0] != X.shape[0]:
            raise ValueError(
                "y length must match X rows: "
                f"got {y.shape[0]} vs {X.shape[0]}"
            )

        subsample_fraction = float(subsample_fraction)

        if (
            not np.isfinite(subsample_fraction)
            or subsample_fraction <= 0.0
            or subsample_fraction > 1.0
        ):
            raise ValueError(
                "subsample_fraction must lie in (0, 1]."
            )

        num_subsamples = int(num_subsamples)

        if num_subsamples <= 0:
            raise ValueError(
                "num_subsamples must be positive."
            )

        seed = int(seed)

        if seed < 0 or seed > (2**64 - 1):
            raise ValueError(
                "seed must lie in [0, 2**64 - 1]."
            )

        eta_defer = float(eta_defer)

        if not np.isfinite(eta_defer) or eta_defer < 0.0:
            raise ValueError(
                "eta_defer must be finite and nonnegative."
            )

        bb_pred_vec = _prepare_bb_pred(
            bb_pred,
            X.shape[0],
            use_deferral=bool(use_deferral),
        )

        n_features = X.shape[1]

        if continuous_starts is None:
            continuous_starts_vec = []
        else:
            continuous_starts_vec = sorted(
                set(int(v) for v in continuous_starts)
            )

            bad = [
                v
                for v in continuous_starts_vec
                if v < 0 or v >= n_features
            ]

            if bad:
                raise ValueError(
                    "continuous_starts contains invalid feature "
                    f"indices {bad}; valid range is "
                    f"[0, {n_features - 1}]"
                )

        if allowed_proxy_features is None:
            allowed_proxy_features_vec = []
        else:
            allowed_proxy_features_vec = sorted(
                set(int(v) for v in allowed_proxy_features)
            )

            bad = [
                v
                for v in allowed_proxy_features_vec
                if v < 0 or v >= n_features
            ]

            if bad:
                raise ValueError(
                    "allowed_proxy_features contains invalid feature "
                    f"indices {bad}; valid range is "
                    f"[0, {n_features - 1}]"
                )

        proxy_style_int = parse_proxy_style(proxy_style)

        greedy_heur_int = parse_heuristic_for_greedy(
            heuristic_for_greedy
        )

        greedy_cont_mode = parse_greedy_continuous_mode(
            greedy_continuous_mode
        )

        key_mode_parsed = parse_key_mode(key_mode)

        if (
            reuse_caches_between_subsamples
            and not trie_cache_enabled
        ):
            raise ValueError(
                "trie_cache_enabled must be True when "
                "reuse_caches_between_subsamples=True if you want "
                "the full cache-reuse experiment."
            )

        cumulative_times = self._model.fit_repeated_subsamples(
            X,
            y,
            float(lambda_reg),
            int(depth_budget),
            float(rashomon_mult),
            float(multiplicative_slack),
            key_mode_parsed,
            bool(trie_cache_enabled),
            int(lookahead_k),
            bool(use_budget_refinement),
            bool(guarantee_rule_list_recovery),
            int(proxy_style_int),
            bool(majority_leaf_only),
            bool(cache_early_exits),
            int(greedy_heur_int),
            greedy_cont_mode,
            bool(proxy_caching),
            allowed_proxy_features_vec,
            bool(restrict_proxy_in_lickety),
            bool(restrict_proxy_in_depthd_exact),
            bool(restrict_proxy_in_greedy),
            continuous_starts_vec,
            float(subsample_fraction),
            int(num_subsamples),
            int(seed),
            bool(reuse_caches_between_subsamples),
            bool(stronger_rollout),
            bool(use_deferral),
            float(eta_defer),
            bb_pred_vec,
        )

        self.subsample_cumulative_times_ = np.asarray(
            cumulative_times,
            dtype=float,
        )

        return self

    def fit_then_extend(
        self,
        X,
        y,
        lambda_reg=0.01,
        depth_budget=5,
        first_rashomon_mult=0.01,
        second_rashomon_mult=0.03,
        multiplier_step_size=0.01,
        multiplicative_slack=0.0,
        key_mode="hash",
        lookahead_k=1,
        proxy_style=0,
        use_budget_refinement=True,
        guarantee_rule_list_recovery=False,
        majority_leaf_only=False,
        cache_early_exits=False,
        heuristic_for_greedy=1,
        greedy_continuous_mode="binary",
        proxy_caching=True,
        allowed_proxy_features=None,
        restrict_proxy_in_lickety=False,
        restrict_proxy_in_depthd_exact=False,
        restrict_proxy_in_greedy=False,
        continuous_starts=None,
        trie_cache_enabled=True,
        stronger_rollout=False,
        runtime_limit_seconds=-1.0,
        memory_limit_mb=-1.0,
        use_deferral=False,
        eta_defer=0.0,
        bb_pred=None,
        additive=False,
    ):
        X = np.asarray(X, dtype=np.uint8)
        y = np.asarray(y, dtype=int)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")

        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}")

        if y.shape[0] != X.shape[0]:
            raise ValueError(
                f"y length must match X rows: got len(y)={y.shape[0]}, "
                f"X rows={X.shape[0]}"
            )

        eta_defer = float(eta_defer)

        if not np.isfinite(eta_defer) or eta_defer < 0.0:
            raise ValueError(
                "eta_defer must be finite and nonnegative."
            )

        bb_pred_vec = _prepare_bb_pred(
            bb_pred,
            X.shape[0],
            use_deferral=bool(use_deferral),
        )

        first_rashomon_mult = float(first_rashomon_mult)
        second_rashomon_mult = float(second_rashomon_mult)

        if first_rashomon_mult < 0:
            raise ValueError(
                "first_rashomon_mult must be nonnegative."
            )

        if second_rashomon_mult < 0:
            raise ValueError(
                "second_rashomon_mult must be nonnegative."
            )

        multiplier_step_size = float(multiplier_step_size)

        if (
            second_rashomon_mult > first_rashomon_mult
            and multiplier_step_size <= 0
        ):
            raise ValueError(
                "multiplier_step_size must be positive when "
                "second_rashomon_mult is larger than first_rashomon_mult."
            )

        n_features = X.shape[1]

        if continuous_starts is None:
            continuous_starts_vec = []
        else:
            continuous_starts_vec = sorted(
                set(int(v) for v in continuous_starts)
            )

            bad = [
                v for v in continuous_starts_vec
                if v < 0 or v >= n_features
            ]

            if bad:
                raise ValueError(
                    "continuous_starts contains invalid feature "
                    f"indices {bad}; valid range is "
                    f"[0, {n_features - 1}]"
                )

        if allowed_proxy_features is None:
            allowed_proxy_features_vec = []
        else:
            allowed_proxy_features_vec = sorted(
                set(int(v) for v in allowed_proxy_features)
            )

            bad = [
                v for v in allowed_proxy_features_vec
                if v < 0 or v >= n_features
            ]

            if bad:
                raise ValueError(
                    "allowed_proxy_features contains invalid feature "
                    f"indices {bad}; valid range is "
                    f"[0, {n_features - 1}]"
                )

        proxy_style_int = parse_proxy_style(proxy_style)

        greedy_heur_int = parse_heuristic_for_greedy(
            heuristic_for_greedy
        )

        greedy_cont_mode = parse_greedy_continuous_mode(
            greedy_continuous_mode
        )

        key_mode_parsed = parse_key_mode(key_mode)

        self._model.fit_then_extend(
            X,
            y,
            float(lambda_reg),
            int(depth_budget),
            first_rashomon_mult,
            second_rashomon_mult,
            multiplier_step_size,
            float(multiplicative_slack),
            key_mode_parsed,
            bool(trie_cache_enabled),
            int(lookahead_k),
            bool(use_budget_refinement),
            bool(guarantee_rule_list_recovery),
            int(proxy_style_int),
            bool(majority_leaf_only),
            bool(cache_early_exits),
            int(greedy_heur_int),
            greedy_cont_mode,
            bool(proxy_caching),
            allowed_proxy_features_vec,
            bool(restrict_proxy_in_lickety),
            bool(restrict_proxy_in_depthd_exact),
            bool(restrict_proxy_in_greedy),
            continuous_starts_vec,
            bool(stronger_rollout),
            float(runtime_limit_seconds),
            float(memory_limit_mb),
            bool(use_deferral),
            float(eta_defer),
            bb_pred_vec,
            bool(additive)
        )

        return self

    def fit_anytime(
        self,
        X,
        y,
        lambda_reg=0.01,
        depth_budget=5,
        rashomon_mult=0.01,
        second_rashomon_mult=None,
        multiplier_step_size=0.01,
        multiplicative_slack=0.0,
        key_mode="hash",
        lookahead_k=1,
        proxy_style=0,
        use_budget_refinement=True,
        guarantee_rule_list_recovery=False,
        majority_leaf_only=False,
        cache_early_exits=False,
        heuristic_for_greedy=1,
        greedy_continuous_mode="binary",
        proxy_caching=True,
        proxy_threshold_features=None,
        initial_active_threshold_features=None,
        refinement_width=1,
        max_refinement_rounds=-1,
        proxy_refinement_mode=0,
        continuous_proxy_in_lickety=False,
        continuous_proxy_in_depthd_exact=False,
        continuous_proxy_in_greedy=False,
        continuous_starts=None,
        trie_cache_enabled=True,
        runtime_limit_seconds=-1.0,
        memory_limit_mb=-1.0,
        use_deferral=False,
        eta_defer=0.0,
        bb_pred=None,
    ):
        X = np.asarray(X, dtype=np.uint8)
        y = np.asarray(y, dtype=int)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}")
        if y.shape[0] != X.shape[0]:
            raise ValueError(
                f"y length must match X rows: got len(y)={y.shape[0]}, "
                f"X rows={X.shape[0]}"
            )

        eta_defer = float(eta_defer)

        if not np.isfinite(eta_defer) or eta_defer < 0.0:
            raise ValueError(
                "eta_defer must be finite and nonnegative."
            )

        bb_pred_vec = _prepare_bb_pred(
            bb_pred,
            X.shape[0],
            use_deferral=bool(use_deferral),
        )

        rashomon_mult = float(rashomon_mult)

        if second_rashomon_mult is None:
            second_rashomon_mult = rashomon_mult
        else:
            second_rashomon_mult = float(second_rashomon_mult)

        multiplier_step_size = float(multiplier_step_size)

        if rashomon_mult < 0:
            raise ValueError("rashomon_mult must be nonnegative.")

        if second_rashomon_mult < 0:
            raise ValueError("second_rashomon_mult must be nonnegative.")

        if (
            second_rashomon_mult > rashomon_mult
            and multiplier_step_size <= 0
        ):
            raise ValueError(
                "multiplier_step_size must be positive when "
                "second_rashomon_mult exceeds rashomon_mult."
            )

        proxy_refinement_mode = int(proxy_refinement_mode)
        if proxy_refinement_mode not in (0, 1, 2):
            raise ValueError(
                "proxy_refinement_mode must be 0, 1, or 2."
            )

        n_features = X.shape[1]

        if continuous_starts is None:
            continuous_starts_vec = []
        else:
            continuous_starts_vec = sorted(
                set(int(v) for v in continuous_starts)
            )

            bad = [
                v for v in continuous_starts_vec
                if v < 0 or v >= n_features
            ]
            if bad:
                raise ValueError(
                    f"continuous_starts contains invalid feature indices {bad}; "
                    f"valid range is [0, {n_features - 1}]"
                )

        if proxy_threshold_features is None:
            proxy_threshold_features_vec = []
        else:
            proxy_threshold_features_vec = sorted(
                set(int(v) for v in proxy_threshold_features)
            )

            bad = [
                v for v in proxy_threshold_features_vec
                if v < 0 or v >= n_features
            ]
            if bad:
                raise ValueError(
                    "proxy_threshold_features contains invalid feature "
                    f"indices {bad}; valid range is [0, {n_features - 1}]"
                )

        if initial_active_threshold_features is None:
            initial_active_threshold_features_vec = []
        else:
            initial_active_threshold_features_vec = sorted(
                set(int(v) for v in initial_active_threshold_features)
            )

            bad = [
                v for v in initial_active_threshold_features_vec
                if v < 0 or v >= n_features
            ]
            if bad:
                raise ValueError(
                    "initial_active_threshold_features contains invalid "
                    f"feature indices {bad}; valid range is "
                    f"[0, {n_features - 1}]"
                )

        proxy_style_int = parse_proxy_style(proxy_style)
        greedy_heur_int = parse_heuristic_for_greedy(
            heuristic_for_greedy
        )
        greedy_cont_mode = parse_greedy_continuous_mode(
            greedy_continuous_mode
        )
        key_mode_parsed = parse_key_mode(key_mode)

        self._model.fit_anytime(
            X,
            y,
            float(lambda_reg),
            int(depth_budget),
            rashomon_mult,
            second_rashomon_mult,
            multiplier_step_size,
            float(multiplicative_slack),
            key_mode_parsed,
            bool(trie_cache_enabled),
            int(lookahead_k),
            bool(use_budget_refinement),
            bool(guarantee_rule_list_recovery),
            int(proxy_style_int),
            bool(majority_leaf_only),
            bool(cache_early_exits),
            int(greedy_heur_int),
            greedy_cont_mode,
            bool(proxy_caching),
            proxy_threshold_features_vec,
            initial_active_threshold_features_vec,
            int(refinement_width),
            int(max_refinement_rounds),
            proxy_refinement_mode,
            bool(continuous_proxy_in_lickety),
            bool(continuous_proxy_in_depthd_exact),
            bool(continuous_proxy_in_greedy),
            continuous_starts_vec,
            float(runtime_limit_seconds),
            float(memory_limit_mb),
            bool(use_deferral),
            float(eta_defer),
            bb_pred_vec,
        )

        return self

    def prepare_continuous_data(
        self,
        X_num,
        y,
        X_bin=None,
        X_initial_active=None,
        X_proxy_active=None,
        max_number_thresholds_per_feature=None,
        bb_pred=None,
    ):
        # X_num: Row-major numeric matrix, shape (n_samples, n_numeric_features).
        # by default, these columns are exhaustively threshold-binarized in C++.
        # if max_number_thresholds_per_feature is set, features with more than that
        # many candidate thresholds use quantile-spaced thresholds instead.
        # y: labels, shape (n_samples,).
        # X_bin: optional row-major already-binary matrix, shape (n_samples, n_binary_features).
        # X_initial_active: optional binary columns used to seed anytime enumeration
        # X_proxy_active: optional row-major binary matrix to use in proxies
        # each X_active column is mapped in C++ to the nearest full binarized feature by Hamming distance.

        X_num = np.asarray(X_num, dtype=np.float64)
        y = np.asarray(y, dtype=int)

        if X_num.ndim != 2:
            raise ValueError(f"X_num must be 2D, got shape {X_num.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}")

        n = X_num.shape[0]
        if y.shape[0] != n:
            raise ValueError(
                f"y length must match X_num rows: got len(y)={y.shape[0]}, "
                f"X_num rows={n}"
            )

        if X_bin is None:
            X_bin = np.empty((n, 0), dtype=np.uint8)
        else:
            X_bin = np.asarray(X_bin, dtype=np.uint8)
            if X_bin.ndim != 2:
                raise ValueError(f"X_bin must be 2D, got shape {X_bin.shape}")
            if X_bin.shape[0] != n:
                raise ValueError(
                    f"X_bin rows must match X_num rows: got {X_bin.shape[0]} vs {n}"
                )

        if X_initial_active is None:
            X_initial_active = np.empty((n, 0), dtype=np.uint8)
        else:
            X_initial_active = np.asarray(
                X_initial_active,
                dtype=np.uint8,
            )
            if X_initial_active.ndim != 2:
                raise ValueError(
                    "X_initial_active must be 2D, "
                    f"got shape {X_initial_active.shape}"
                )
            if X_initial_active.shape[0] != n:
                raise ValueError(
                    "X_initial_active rows must match X_num rows: "
                    f"got {X_initial_active.shape[0]} vs {n}"
                )

        if X_proxy_active is None:
            X_proxy_active = np.empty((n, 0), dtype=np.uint8)
        else:
            X_proxy_active = np.asarray(
                X_proxy_active,
                dtype=np.uint8,
            )
            if X_proxy_active.ndim != 2:
                raise ValueError(
                    "X_proxy_active must be 2D, "
                    f"got shape {X_proxy_active.shape}"
                )
            if X_proxy_active.shape[0] != n:
                raise ValueError(
                    "X_proxy_active rows must match X_num rows: "
                    f"got {X_proxy_active.shape[0]} vs {n}"
                )

        if max_number_thresholds_per_feature is None:
            max_thresholds = -1
        else:
            max_thresholds = int(max_number_thresholds_per_feature)
            if max_thresholds <= 0:
                raise ValueError(
                    "max_number_thresholds_per_feature must be positive or None."
                )

        bb_pred_vec = _prepare_bb_pred(
            bb_pred,
            n,
        )

        self._model.prepare_continuous_data(
            X_num,
            X_bin,
            y,
            X_initial_active,
            X_proxy_active,
            max_thresholds,
            bb_pred_vec,
        )
        
        return self

    def fit_prepared(
        self,
        lambda_reg=0.01,
        depth_budget=5,
        rashomon_mult=0.01,
        multiplicative_slack=0.0,
        key_mode="hash",
        lookahead_k=1,
        proxy_style=0,
        root_budget=None,
        use_budget_refinement=True,
        guarantee_rule_list_recovery=False,
        majority_leaf_only=False,
        cache_early_exits=False,
        heuristic_for_greedy=1,
        greedy_continuous_mode="binary",
        proxy_caching=True,
        restrict_proxy_in_lickety=False,
        restrict_proxy_in_depthd_exact=False,
        restrict_proxy_in_greedy=False,
        rashomon_mode=True,
        trie_cache_enabled=True,
        stronger_rollout=False,
        use_deferral=False,
        eta_defer=0.0,
    ):
        proxy_style_int = parse_proxy_style(proxy_style)
        greedy_heur_int = parse_heuristic_for_greedy(heuristic_for_greedy)
        greedy_cont_mode = parse_greedy_continuous_mode(greedy_continuous_mode)

        eta_defer = float(eta_defer)

        if not np.isfinite(eta_defer) or eta_defer < 0.0:
            raise ValueError(
                "eta_defer must be finite and nonnegative."
            )

        if root_budget is None:
            root_budget_int = -1
        else:
            root_budget_int = int(root_budget)

        key_mode_parsed = parse_key_mode(key_mode)
        self._model.fit_prepared(
            float(lambda_reg),
            int(depth_budget),
            float(rashomon_mult),
            float(multiplicative_slack),
            key_mode_parsed,
            bool(trie_cache_enabled),
            int(lookahead_k),
            int(root_budget_int),
            bool(use_budget_refinement),
            bool(guarantee_rule_list_recovery),
            int(proxy_style_int),
            bool(majority_leaf_only),
            bool(cache_early_exits),
            int(greedy_heur_int),
            greedy_cont_mode,
            bool(proxy_caching),
            bool(restrict_proxy_in_lickety),
            bool(restrict_proxy_in_depthd_exact),
            bool(restrict_proxy_in_greedy),
            bool(rashomon_mode),
            bool(stronger_rollout),
            bool(use_deferral),
            float(eta_defer),
        )

        return self

    def fit_prepared_repeated_subsamples(
        self,
        *,
        lambda_reg=0.01,
        depth_budget=5,
        rashomon_mult=0.01,
        multiplicative_slack=0.0,
        key_mode="hash",
        lookahead_k=1,
        proxy_style=0,
        use_budget_refinement=True,
        guarantee_rule_list_recovery=False,
        majority_leaf_only=False,
        cache_early_exits=False,
        heuristic_for_greedy=1,
        greedy_continuous_mode="binary",
        proxy_caching=True,
        restrict_proxy_in_lickety=False,
        restrict_proxy_in_depthd_exact=False,
        restrict_proxy_in_greedy=False,
        trie_cache_enabled=True,
        stronger_rollout=False,
        use_deferral=False,
        eta_defer=0.0,
        subsample_fraction=0.8,
        num_subsamples=50,
        seed=0,
        reuse_caches_between_subsamples=False,
    ):
        subsample_fraction = float(subsample_fraction)

        if (
            not np.isfinite(subsample_fraction)
            or subsample_fraction <= 0.0
            or subsample_fraction > 1.0
        ):
            raise ValueError(
                "subsample_fraction must lie in (0, 1]."
            )

        num_subsamples = int(num_subsamples)

        if num_subsamples <= 0:
            raise ValueError(
                "num_subsamples must be positive."
            )

        seed = int(seed)

        if seed < 0 or seed > 2**64 - 1:
            raise ValueError(
                "seed must lie in [0, 2**64 - 1]."
            )

        eta_defer = float(eta_defer)

        if not np.isfinite(eta_defer) or eta_defer < 0.0:
            raise ValueError(
                "eta_defer must be finite and nonnegative."
            )

        proxy_style_int = parse_proxy_style(proxy_style)

        greedy_heur_int = parse_heuristic_for_greedy(
            heuristic_for_greedy
        )

        greedy_cont_mode = parse_greedy_continuous_mode(
            greedy_continuous_mode
        )

        key_mode_parsed = parse_key_mode(key_mode)

        cumulative_times = (
            self._model.fit_prepared_repeated_subsamples(
                float(lambda_reg),
                int(depth_budget),
                float(rashomon_mult),
                float(multiplicative_slack),
                key_mode_parsed,
                bool(trie_cache_enabled),
                int(lookahead_k),
                bool(use_budget_refinement),
                bool(guarantee_rule_list_recovery),
                int(proxy_style_int),
                bool(majority_leaf_only),
                bool(cache_early_exits),
                int(greedy_heur_int),
                greedy_cont_mode,
                bool(proxy_caching),
                bool(restrict_proxy_in_lickety),
                bool(restrict_proxy_in_depthd_exact),
                bool(restrict_proxy_in_greedy),
                float(subsample_fraction),
                int(num_subsamples),
                int(seed),
                bool(reuse_caches_between_subsamples),
                bool(stronger_rollout),
                bool(use_deferral),
                float(eta_defer),
            )
        )

        self.subsample_cumulative_times_ = np.asarray(
            cumulative_times,
            dtype=float,
        )

        return self

    def fit_prepared_then_extend(
        self,
        lambda_reg=0.01,
        depth_budget=5,
        first_rashomon_mult=0.01,
        second_rashomon_mult=0.03,
        multiplier_step_size=0.01,
        multiplicative_slack=0.0,
        key_mode="hash",
        lookahead_k=1,
        proxy_style=0,
        use_budget_refinement=True,
        guarantee_rule_list_recovery=False,
        majority_leaf_only=False,
        cache_early_exits=False,
        heuristic_for_greedy=1,
        greedy_continuous_mode="binary",
        proxy_caching=True,
        restrict_proxy_in_lickety=False,
        restrict_proxy_in_depthd_exact=False,
        restrict_proxy_in_greedy=False,
        trie_cache_enabled=True,
        stronger_rollout=False,
        runtime_limit_seconds=-1.0,
        memory_limit_mb=-1.0,
        use_deferral=False,
        eta_defer=0.0,
        additive=False,
    ):
        first_rashomon_mult = float(first_rashomon_mult)
        second_rashomon_mult = float(second_rashomon_mult)

        if first_rashomon_mult < 0:
            raise ValueError(
                "first_rashomon_mult must be nonnegative."
            )

        if second_rashomon_mult < 0:
            raise ValueError(
                "second_rashomon_mult must be nonnegative."
            )

        multiplier_step_size = float(multiplier_step_size)

        if (
            second_rashomon_mult > first_rashomon_mult
            and multiplier_step_size <= 0
        ):
            raise ValueError(
                "multiplier_step_size must be positive when "
                "second_rashomon_mult is larger than first_rashomon_mult."
            )

        proxy_style_int = parse_proxy_style(proxy_style)

        greedy_heur_int = parse_heuristic_for_greedy(
            heuristic_for_greedy
        )

        greedy_cont_mode = parse_greedy_continuous_mode(
            greedy_continuous_mode
        )

        key_mode_parsed = parse_key_mode(key_mode)

        self._model.fit_prepared_then_extend(
            float(lambda_reg),
            int(depth_budget),
            first_rashomon_mult,
            second_rashomon_mult,
            multiplier_step_size,
            float(multiplicative_slack),
            key_mode_parsed,
            bool(trie_cache_enabled),
            int(lookahead_k),
            bool(use_budget_refinement),
            bool(guarantee_rule_list_recovery),
            int(proxy_style_int),
            bool(majority_leaf_only),
            bool(cache_early_exits),
            int(greedy_heur_int),
            greedy_cont_mode,
            bool(proxy_caching),
            bool(restrict_proxy_in_lickety),
            bool(restrict_proxy_in_depthd_exact),
            bool(restrict_proxy_in_greedy),
            bool(stronger_rollout),
            float(runtime_limit_seconds),
            float(memory_limit_mb),
            bool(use_deferral),
            float(eta_defer),
            bool(additive)
        )

        return self

    def fit_prepared_anytime(
        self,
        lambda_reg=0.01,
        depth_budget=5,
        rashomon_mult=0.01,
        second_rashomon_mult=None,
        multiplier_step_size=0.01,
        multiplicative_slack=0.0,
        key_mode="hash",
        lookahead_k=1,
        proxy_style=0,
        use_budget_refinement=True,
        guarantee_rule_list_recovery=False,
        majority_leaf_only=False,
        cache_early_exits=False,
        heuristic_for_greedy=1,
        greedy_continuous_mode="binary",
        proxy_caching=True,
        proxy_threshold_features=None,
        initial_active_threshold_features=None,
        refinement_width=1,
        max_refinement_rounds=-1,
        proxy_refinement_mode=0,
        continuous_proxy_in_lickety=False,
        continuous_proxy_in_depthd_exact=False,
        continuous_proxy_in_greedy=False,
        trie_cache_enabled=True,
        runtime_limit_seconds=-1.0,
        memory_limit_mb=-1.0,
        use_deferral=False,
        eta_defer=0.0,
    ):
        rashomon_mult = float(rashomon_mult)

        if second_rashomon_mult is None:
            second_rashomon_mult = rashomon_mult
        else:
            second_rashomon_mult = float(second_rashomon_mult)

        multiplier_step_size = float(multiplier_step_size)

        if rashomon_mult < 0:
            raise ValueError("rashomon_mult must be nonnegative.")

        if second_rashomon_mult < 0:
            raise ValueError(
                "second_rashomon_mult must be nonnegative."
            )

        if (
            second_rashomon_mult > rashomon_mult
            and multiplier_step_size <= 0
        ):
            raise ValueError(
                "multiplier_step_size must be positive when "
                "second_rashomon_mult exceeds rashomon_mult."
            )

        proxy_refinement_mode = int(proxy_refinement_mode)
        if proxy_refinement_mode not in (0, 1, 2):
            raise ValueError(
                "proxy_refinement_mode must be 0, 1, or 2."
            )

        proxy_style_int = parse_proxy_style(proxy_style)
        greedy_heur_int = parse_heuristic_for_greedy(
            heuristic_for_greedy
        )
        greedy_cont_mode = parse_greedy_continuous_mode(
            greedy_continuous_mode
        )

        if proxy_threshold_features is None:
            proxy_threshold_features_vec = []
        else:
            proxy_threshold_features_vec = sorted(
                set(int(v) for v in proxy_threshold_features)
            )

        if initial_active_threshold_features is None:
            initial_active_threshold_features_vec = []
        else:
            initial_active_threshold_features_vec = sorted(
                set(int(v) for v in initial_active_threshold_features)
            )

        key_mode_parsed = parse_key_mode(key_mode)

        self._model.fit_prepared_anytime(
            float(lambda_reg),
            int(depth_budget),
            rashomon_mult,
            second_rashomon_mult,
            multiplier_step_size,
            float(multiplicative_slack),
            key_mode_parsed,
            bool(trie_cache_enabled),
            int(lookahead_k),
            bool(use_budget_refinement),
            bool(guarantee_rule_list_recovery),
            int(proxy_style_int),
            bool(majority_leaf_only),
            bool(cache_early_exits),
            int(greedy_heur_int),
            greedy_cont_mode,
            bool(proxy_caching),
            proxy_threshold_features_vec,
            initial_active_threshold_features_vec,
            int(refinement_width),
            int(max_refinement_rounds),
            proxy_refinement_mode,
            bool(continuous_proxy_in_lickety),
            bool(continuous_proxy_in_depthd_exact),
            bool(continuous_proxy_in_greedy),
            float(runtime_limit_seconds),
            float(memory_limit_mb),
            bool(use_deferral),
            float(eta_defer),
        )

        return self

    def set_greedy_continuous_mode(self, mode="binary"):
        mode = parse_greedy_continuous_mode(mode)
        self._model.set_greedy_continuous_mode(mode)
        return self

    def count_trees(self):
        return self._model.count_trees()

    def count_distinct_or_nodes(self):
        return self._model.count_distinct_or_nodes()

    def count_graph_features(self):
        return self._model.count_graph_features()


    def get_min_objective(self):
        return self._model.get_min_objective()

    def get_root_histogram(self):
        return self._model.get_root_histogram()
    
    def get_tree_objective(self, tree_index: int):
        obj, obj_norm = self._model.get_tree_objective(int(tree_index))
        return obj, obj_norm

    def get_tree_num_leaves(self, tree_index: int) -> int:
        paths, _ = self.get_tree_paths(tree_index)
        return len(paths)
        
    def count_trees_within_mult(self, mult: float) -> int:
        hist = self.get_root_histogram()
        min_obj = self.get_min_objective()
        thresh = round((1.0 + mult) * min_obj)
        return sum(cnt for obj, cnt in hist if obj <= thresh)

    def alternating_optimization(self, max_iterations=10):
        return int(
            self._model.alternating_optimization(
                int(max_iterations)
            )
        )

    def get_continuous_starts(self):
        # return the internal column index at which each continuous threshold group begins.
        return list(self._model.get_continuous_starts())


    def get_num_continuous_groups(self):
        return int(self._model.get_num_continuous_groups())


    def get_continuous_group_end(self, continuous_group):
        return int(
            self._model.get_continuous_group_end(
                int(continuous_group)
            )
        )


    def get_continuous_cutpoints(self, continuous_group):
        # return the ordered cutpoints for one continuous feature group.
        # internal offset j represents: value <= cutpoints[j]
        return np.asarray(
            self._model.get_continuous_cutpoints(
                int(continuous_group)
            ),
            dtype=np.float64,
        )


    def get_continuous_threshold_info(self, internal_feature):
        # return continuous_group, offset_within_group, cutpoint
        group, offset, cutpoint = (
            self._model.get_continuous_threshold_info(
                int(internal_feature)
            )
        )

        return int(group), int(offset), float(cutpoint)


    def get_internal_feature_info(self, internal_feature):
        # return a dictionary describing one internal binary feature
        is_continuous, group, offset, cutpoint = (
            self._model.get_internal_feature_info(
                int(internal_feature)
            )
        )

        if not is_continuous:
            return {
                "kind": "binary",
                "internal_feature": int(internal_feature),
                "continuous_group": None,
                "offset": None,
                "cutpoint": None,
            }

        return {
            "kind": "continuous_threshold",
            "internal_feature": int(internal_feature),
            "continuous_group": int(group),
            "offset": int(offset),
            "cutpoint": float(cutpoint),
        }


    def encode_continuous_value(self, continuous_group, value):
        # encode one numerical value over all cutpoints for the selected continuous group.

        return np.asarray(
            self._model.encode_continuous_value(
                int(continuous_group),
                float(value),
            ),
            dtype=np.uint8,
        )

    def transform(self, X):
        if (
            self.binary_feature_specs_ is None
            or self.continuous_feature_indices_ is None
        ):
            raise RuntimeError(
                "Raw-data transformation is available only after "
                "calling the high-level fit(X, y, ...)."
            )

        X = _as_2d_numeric_array(X, "X")

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but the model was fitted "
                f"with {self.n_features_in_}."
            )

        binary_parts = []

        for spec in self.binary_feature_specs_:
            column = spec["original_feature"]
            cutpoint = spec["cutpoint"]

            binary_parts.append(
                (X[:, column] <= cutpoint).astype(np.uint8)
            )

        continuous_parts = []

        for group, original_column in enumerate(
            self.continuous_feature_indices_
        ):
            cutpoints = self.get_continuous_cutpoints(group)

            values = X[:, original_column]

            continuous_parts.append(
                (values[:, None] <= cutpoints[None, :]).astype(
                    np.uint8
                )
            )

        pieces = []

        if binary_parts:
            pieces.append(np.column_stack(binary_parts))

        if continuous_parts:
            pieces.extend(continuous_parts)

        if not pieces:
            return np.empty((X.shape[0], 0), dtype=np.uint8)

        return np.column_stack(pieces).astype(
            np.uint8,
            copy=False,
        )

    def get_exact_replacement_importance_intervals(
        self,
        X_eval,
        y_eval,
        budget_override=None,
    ):
        """
        Exact global [min, max] subtractive model-reliance interval
        over the fitted Rashomon set, evaluated on X_eval, y_eval.
        X_eval may be completely different from the training data.
        """

        if self.n_features_in_ is None:
            raise RuntimeError("Fit the model first.")

        X_internal = np.ascontiguousarray(
            self.transform(X_eval),
            dtype=np.uint8,
        )

        y_eval = np.ascontiguousarray(
            np.asarray(y_eval, dtype=np.int32)
        )

        if y_eval.ndim != 1:
            raise ValueError("y_eval must be 1D.")

        if y_eval.shape[0] != X_internal.shape[0]:
            raise ValueError(
                "y_eval must have the same number of rows as X_eval."
            )

        # map each original feature to its internal threshold columns.
        variable_columns = [
            [] for _ in range(self.n_features_in_)
        ]

        # low-cardinality features created directly in python.
        for internal_feature, spec in enumerate(
            self.binary_feature_specs_
        ):
            original_feature = int(spec["original_feature"])
            variable_columns[original_feature].append(
                int(internal_feature)
            )

        # continuous threshold groups created in cpp
        starts = self.get_continuous_starts()

        for group, original_feature in enumerate(
            self.continuous_feature_indices_
        ):
            original_feature = int(original_feature)

            start = int(starts[group])
            end = int(self.get_continuous_group_end(group))

            variable_columns[original_feature].extend(
                range(start, end)
            )

        # constant training features have no internal columns and
        # therefore importance exactly zero for every fitted tree.
        active_features = [
            j
            for j, cols in enumerate(variable_columns)
            if cols
        ]

        active_variable_columns = [
            variable_columns[j]
            for j in active_features
        ]

        budget = (
            -1
            if budget_override is None
            else int(budget_override)
        )

        active_intervals = np.asarray(
            self._model
                .get_exact_replacement_importance_intervals_packed_trie(
                    X_internal,
                    y_eval,
                    budget,
                    active_variable_columns,
                    np.empty(0, dtype=np.int32),
                    False, # global min/max, not per-sample
                ),
            dtype=float,
        )

        intervals = np.zeros(
            (self.n_features_in_, 2),
            dtype=float,
        )

        if active_features:
            intervals[active_features] = active_intervals

        return intervals


    # WARNING: 1-indexed unlike features
    def get_tree_paths(self, tree_index: int):
        """
        returns (paths, predictions):
        - paths: list of lists of signed feature indices. these are 1-indexed but features are 0-indexed so must subtract 1.
          +f means "go left / True on feature f-1"
          -f means "go right / False on feature f-1".
        - predictions: list of class labels for each leaf, -1 denotes defer.
        """
        return self._model.get_tree_paths(int(tree_index))
    
    def get_tree_paths_str(self, tree_index: int):
        """
        returns (paths_str, predictions) where:
        - paths_str is a list of strings like "[+0, -1, +2]"
        - indices are shifted by -1 so features are 0-indexed as one would expect
        """
        paths, preds = self.get_tree_paths(tree_index)

        out = []
        for p in paths:
            converted = []
            for v in p:
                if v >= 0:
                    converted.append(f"+{v - 1}")
                else:
                    converted.append(f"-{abs(v) - 1}")
            path_str = "[" + ", ".join(converted) + "]"
            out.append(path_str)

        return out, preds
    
    def get_predictions(
        self,
        tree_index: int,
        X,
        *,
        bb_pred=None,
        defer_placeholder=99,
    ):
        X_internal = self.transform(X)

        if bb_pred is None:
            bb_pred_vec = np.empty(0, dtype=np.int32)
        else:
            bb_pred_vec = np.asarray(bb_pred, dtype=np.int32)

            if bb_pred_vec.ndim != 1:
                raise ValueError(
                    f"bb_pred must be 1D, got shape {bb_pred_vec.shape}"
                )

            if bb_pred_vec.shape[0] != X_internal.shape[0]:
                raise ValueError(
                    "bb_pred length must match X rows: "
                    f"got {bb_pred_vec.shape[0]} vs "
                    f"{X_internal.shape[0]}"
                )

        preds = np.asarray(
            self._model.get_predictions(
                int(tree_index),
                X_internal,
                bb_pred_vec,
                int(defer_placeholder),
            ),
            dtype=np.int16,
        )

        preds[preds == 255] = -1
        return preds


    def get_all_predictions(
        self,
        X,
        *,
        bb_pred=None,
        defer_placeholder=99,
        stack=False,
    ):
        X_internal = self.transform(X)

        if bb_pred is None:
            bb_pred_vec = np.empty(0, dtype=np.int32)
        else:
            bb_pred_vec = np.asarray(bb_pred, dtype=np.int32)

            if bb_pred_vec.ndim != 1:
                raise ValueError(
                    f"bb_pred must be 1D, got shape {bb_pred_vec.shape}"
                )

            if bb_pred_vec.shape[0] != X_internal.shape[0]:
                raise ValueError(
                    "bb_pred length must match X rows: "
                    f"got {bb_pred_vec.shape[0]} vs "
                    f"{X_internal.shape[0]}"
                )

        preds = np.asarray(
            self._model.get_all_predictions(
                X_internal,
                bb_pred_vec,
                int(defer_placeholder),
                bool(stack),
            ),
            dtype=np.int16,
        )

        preds[preds == 255] = -1
        return preds
    
    def get_internal_feature_names(self, feature_names=None):
        if feature_names is None:
            feature_names = [
                f"f{j}" for j in range(self.n_features_in_)
            ]
        else:
            feature_names = list(feature_names)

        if len(feature_names) != self.n_features_in_:
            raise ValueError(
                f"feature_names must have length "
                f"{self.n_features_in_}."
            )

        names = []

        # low-cardinality threshold columns created in Python.
        for spec in self.binary_feature_specs_:
            original_feature = int(spec["original_feature"])
            cutpoint = float(spec["cutpoint"])

            names.append(
                f"{feature_names[original_feature]} <= {cutpoint:g}"
            )

        # continuous threshold columns created in C++.
        for group, original_feature in enumerate(
            self.continuous_feature_indices_
        ):
            cutpoints = self.get_continuous_cutpoints(group)

            for cutpoint in cutpoints:
                names.append(
                    f"{feature_names[original_feature]} <= {cutpoint:g}"
                )

        return names

    def export_andor_graph(self, as_dict=True):
        g = self._model.export_andor_graph()

        if not as_dict:
            return g

        return {
            "root_trie_id": int(g.root_trie_id),

            "trie_nodes": [
                {
                    "id": int(node.id),
                    "budget": int(node.budget),
                    "min_objective": int(node.min_objective),
                    "subproblem_size": int(
                        node.subproblem_size
                    ),
                    "leaf_ids": [
                        int(x)
                        for x in node.leaf_ids
                    ],
                    "split_ids": [
                        int(x)
                        for x in node.split_ids
                    ],
                }
                for node in g.trie_nodes
            ],

            "split_nodes": [
                {
                    "id": int(split.id),
                    "parent_trie_id": int(
                        split.parent_trie_id
                    ),
                    "feature": int(split.feature),
                    "left_trie_id": int(
                        split.left_trie_id
                    ),
                    "right_trie_id": int(
                        split.right_trie_id
                    ),
                    "min_objective": int(
                        split.min_objective
                    ),
                }
                for split in g.split_nodes
            ],

            "leaf_nodes": [
                {
                    "id": int(leaf.id),
                    "parent_trie_id": int(
                        leaf.parent_trie_id
                    ),
                    "prediction": int(
                        leaf.prediction
                    ),
                    "loss": int(leaf.loss),
                    "subproblem_size": int(
                        leaf.subproblem_size
                    ),
                }
                for leaf in g.leaf_nodes
            ],
        }

    def _get_builder_feature_metadata(
        self,
        feature_names=None,
    ):
        if self.binary_feature_specs_ is None:
            raise RuntimeError(
                "Feature metadata is unavailable. "
                "Fit ArborEnum through fit(X, y, ...) first."
            )

        if self.continuous_feature_indices_ is None:
            raise RuntimeError(
                "Continuous feature metadata is unavailable."
            )

        if self.n_features_in_ is None:
            raise RuntimeError(
                "Original feature count is unavailable."
            )

        if feature_names is None:
            feature_names = list(self.feature_names_in_)
        else:
            feature_names = list(feature_names)

        if len(feature_names) != self.n_features_in_:
            raise ValueError(
                "feature_names must contain one name for "
                "every original input feature."
            )

        # exactly aligned with our internal ids
        internal_feature_names = (
            self.get_internal_feature_names(feature_names)
        )

        thresholds = {}
        feature_registry = []

        # low cardinality features
        for internal_feature, spec in enumerate(
            self.binary_feature_specs_
        ):
            original_feature = int(
                spec["original_feature"]
            )

            cutpoint = float(spec["cutpoint"])

            thresholds[internal_feature] = cutpoint

            feature_registry.append(
                {
                    "internalFeature": internal_feature,
                    "originalFeature": original_feature,
                    "originalName": str(
                        feature_names[original_feature]
                    ),
                    "threshold": cutpoint,
                    "kind": "binary_threshold",
                    "continuousGroup": None,
                }
            )

        # cpp generated continuous groups
        continuous_groups = {}

        starts = [
            int(x)
            for x in self._model.get_continuous_starts()
        ]

        for group, original_feature in enumerate(
            self.continuous_feature_indices_
        ):
            original_feature = int(original_feature)

            cutpoints = [
                float(x)
                for x in self.get_continuous_cutpoints(group)
            ]

            start = starts[group]

            internal_columns = []

            for offset, cutpoint in enumerate(cutpoints):
                internal_feature = start + offset

                thresholds[internal_feature] = cutpoint
                internal_columns.append(internal_feature)

                feature_registry.append(
                    {
                        "internalFeature": internal_feature,
                        "originalFeature": original_feature,
                        "originalName": str(
                            feature_names[original_feature]
                        ),
                        "threshold": cutpoint,
                        "kind": "continuous_threshold",
                        "continuousGroup": group,
                    }
                )

            continuous_groups[
                str(feature_names[original_feature])
            ] = internal_columns

        # feature_registry[i] describes split.feature == i. this should already be the case by how we went over.
        feature_registry.sort(
            key=lambda x: x["internalFeature"]
        )

        for expected, entry in enumerate(feature_registry):
            if entry["internalFeature"] != expected:
                raise RuntimeError(
                    "Builder feature metadata is not aligned "
                    "with ArborEnum internal feature indices: "
                    f"expected {expected}, got "
                    f"{entry['internalFeature']}."
                )

        if len(internal_feature_names) != len(
            feature_registry
        ):
            raise RuntimeError(
                "Internal feature-name count does not match "
                "the builder feature registry."
            )

        if len(starts) != len(
            self.continuous_feature_indices_
        ):
            raise RuntimeError(
                "Continuous-group metadata is inconsistent: "
                "the number of C++ continuous groups does not "
                "match continuous_feature_indices_."
            )


        # feature_registry says for each threshold
        '''
        internalFeature: 0,
        originalFeature: 1,
        originalName: sex,
        threshold: 0.5,
        kind: binary_threshold,
        continuousGroup: None, 
        '''

        return {
            "featureRegistry": feature_registry,
        }
        

    def save_builder_payload(
        self,
        path,
        *,
        feature_names=None,
        feature_descriptions=None,
        lambda_reg=None,
        depth_budget=None,
        rashomon_mult=None,
        multiplicative_slack=None,
        lookahead_k=None,
        root_n=None,
        gamma=None,
        eta_defer=None,
        indent=2,
    ):
        graph = self.export_andor_graph(
            as_dict=True
        )

        feature_meta = (
            self._get_builder_feature_metadata(
                feature_names=feature_names
            )
        )

        if lambda_reg is None:
            lambda_reg = getattr(
                self,
                "lambda_reg_",
                None,
            )

        if depth_budget is None:
            depth_budget = getattr(
                self,
                "depth_budget_",
                None,
            )

        if rashomon_mult is None:
            rashomon_mult = getattr(
                self,
                "rashomon_mult_",
                None,
            )

        if multiplicative_slack is None:
            multiplicative_slack = getattr(
                self,
                "multiplicative_slack_",
                None,
            )

        if lookahead_k is None:
            lookahead_k = getattr(
                self,
                "lookahead_k_",
                None,
            )

        if eta_defer is None:
            eta_defer = getattr(
                self,
                "eta_defer_",
                None,
            )

        if root_n is None:
            root_n = getattr(
                self,
                "root_n_",
                None,
            )

        if gamma is None:
            gamma = getattr(
                self,
                "gamma_",
                None,
            )

        meta = {
            **feature_meta,

            "featureDescriptions": feature_descriptions,

            "lambda_reg": lambda_reg,
            "depth_budget": depth_budget,
            "rashomon_mult": rashomon_mult,
            "multiplicative_slack": multiplicative_slack,
            "lookahead_k": lookahead_k,
            "root_n": root_n,
            "gamma": gamma,
            "eta_defer": eta_defer,
        }

        root_id = int(
            graph.get("root_trie_id", 0)
        )

        root = next(
            (
                node
                for node in graph.get(
                    "trie_nodes",
                    [],
                )
                if int(node.get("id", -1)) == root_id
            ),
            None,
        )

        if root is not None:
            meta["root_budget"] = root.get(
                "budget"
            )

            meta["root_min_objective"] = root.get(
                "min_objective"
            )

            meta["root_subproblem_size"] = root.get(
                "subproblem_size"
            )

            if meta.get("root_n") is None:
                meta["root_n"] = root.get(
                    "subproblem_size"
                )

        if (
            meta.get("gamma") is None
            and lambda_reg is not None
            and meta.get("root_n") is not None
        ):
            meta["gamma"] = int(
                round(
                    float(lambda_reg)
                    * int(meta["root_n"])
                )
            )

        payload = {
            "graph": graph,
            "meta": {
                k: v
                for k, v in meta.items()
                if v is not None
            },
        }

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            json.dumps(
                _json_safe(payload),
                indent=indent,
            ),
            encoding="utf-8",
        )

        return path

    def save_builder_html(
        self,
        path,
        **kwargs,
    ):
        payload_path = (
            Path(path).with_suffix(
                ".payload.json"
            )
        )

        self.save_builder_payload(
            payload_path,
            **kwargs,
        )

        payload = json.loads(
            payload_path.read_text(
                encoding="utf-8"
            )
        )

        payload_path.unlink(
            missing_ok=True
        )

        payload_json = json.dumps(payload)

        # prevent accidental closing of the script tag
        payload_json = payload_json.replace(
            "</",
            "<\\/",
        )

        html = (
            files(__package__)
            .joinpath(
                "builder_static/index.html"
            )
            .read_text(
                encoding="utf-8"
            )
        )

        inject = f"""
        <script>
        window.ARBORENUM_BUILDER_PAYLOAD = {payload_json};
        </script>
        """

        if "</head>" in html:
            html = html.replace(
                "</head>",
                inject + "\n</head>",
                1,
            )
        else:
            html = inject + html

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            html,
            encoding="utf-8",
        )

        return path

    
    def plot_tree(self, tree_index: int, feature_names=None, figsize=(8, 6), ax=None, title=None, show=True):
        paths, preds = self.get_tree_paths(tree_index)

        if feature_names is None:
            if (
                self.binary_feature_specs_ is not None
                and self.continuous_feature_indices_ is not None
            ):
                feature_names = self.get_internal_feature_names(self.feature_names_in_)
            else:
                # feature names if not given
                encodings = [
                    abs(v)
                    for path in paths
                    for v in path
                ]

                max_f = max(encodings) - 1 if encodings else -1  # convert back to 0-based now that we don't need sign
                feature_names = [
                    f"f{j}" for j in range(max_f + 1)
                ]
        else:
            # public users will definitely pass original raw feature names.
            if len(feature_names) == self.n_features_in_:
                feature_names = self.get_internal_feature_names(
                    feature_names
                )
            else:
                # for advanced users
                feature_names = list(feature_names)

        # convert path representation into an explicit tree structure
        class Node:
            __slots__ = ("feature", "left", "right", "prediction")
            def __init__(self):
                self.feature = None
                self.left = None
                self.right = None
                self.prediction = None

        root = Node()

        # build tree
        for path, pred in zip(paths, preds):
            cur = root
            for signed_f in path:
                f = abs(signed_f) - 1  # 0-based
                go_left = signed_f > 0  # + => left / True, - => right / False

                if cur.feature is None:
                    cur.feature = f
                    cur.left = Node()
                    cur.right = Node()

                cur = cur.left if go_left else cur.right

            cur.prediction = pred

        def collect_leaves_in_order(node, leaves):
            if node is None:
                return
            if node.prediction is not None or (node.left is None and node.right is None):
                leaves.append(node)
                return
            collect_leaves_in_order(node.left, leaves)
            collect_leaves_in_order(node.right, leaves)

        def tree_depth(node):
            if node is None:
                return 0
            if node.prediction is not None:
                return 1
            return 1 + max(tree_depth(node.left), tree_depth(node.right))

        def assign_positions_tree(root, positions):
            leaves = []
            collect_leaves_in_order(root, leaves)
            if not leaves:
                leaves = [root]

            leaf_x = {leaf: i for i, leaf in enumerate(leaves)}

            def dfs(node, depth):
                if node is None:
                    return
                if node.prediction is not None or (node.left is None and node.right is None):
                    x = leaf_x[node]
                    positions[node] = (x, -depth)
                    return
                dfs(node.left, depth + 1)
                dfs(node.right, depth + 1)
                x_left, _ = positions[node.left]
                x_right, _ = positions[node.right]
                positions[node] = (0.5 * (x_left + x_right), -depth)

            dfs(root, 0)
            return len(leaves)

        positions = {}
        n_leaves = assign_positions_tree(root, positions)
        depth = tree_depth(root)

        x_scale = 3.2
        y_scale = 2.2

        for node, (x, y) in list(positions.items()):
            positions[node] = (x * x_scale, y * y_scale)

        if ax is None:
            width = max(figsize[0], 1.6 * n_leaves)
            height = max(figsize[1], 1.4 * depth)
            fig, ax = plt.subplots(figsize=(width, height))
        else:
            fig = ax.figure

        ax.set_axis_off()

        def _edge_label_pos(x1, y1, x2, y2, frac=0.52, base_offset=0.28, side_sign=+1.0):
            mx = x1 + frac * (x2 - x1)
            my = y1 + frac * (y2 - y1)
            dx = x2 - x1
            dy = y2 - y1
            dist = (dx * dx + dy * dy) ** 0.5
            if dist == 0:
                return mx, my
            nx = -dy / dist
            ny =  dx / dist

            # more offset on short edges, less on long edges
            scale = min(1.8, max(0.9, 1.2 / (dist ** 0.5)))
            offset = base_offset * scale

            return mx + side_sign * offset * nx, my + side_sign * offset * ny


        def _edge_label(parent_feature_idx, is_left_branch):
            name = feature_names[parent_feature_idx]
            # left branch: feature is True => no prefix
            # right branch: feature is False => prefix "!" # no ! prefix anymore 
            return name if is_left_branch else f"{name}"
        
        def _shrink_segment(x1, y1, x2, y2, r1, r2):
            dx = x2 - x1
            dy = y2 - y1
            dist = (dx * dx + dy * dy) ** 0.5
            if dist == 0:
                return x1, y1, x2, y2
            ux = dx / dist
            uy = dy / dist
            return (
                x1 + ux * r1,
                y1 + uy * r1,
                x2 - ux * r2,
                y2 - uy * r2,
            )


        def draw_node(node):
            x, y = positions[node]
            internal_r = 0.34
            leaf_r = 0.40

            # draw edges + labels + recurse
            if node.left is not None:
                x2, y2 = positions[node.left]
                # ax.add_line(Line2D([x, x2], [y, y2], color="black", linewidth=2.2))
                r_parent = internal_r
                r_child = leaf_r if node.left.prediction is not None else internal_r

                sx, sy, ex, ey = _shrink_segment(x, y, x2, y2, r_parent, r_child)
                ax.add_line(Line2D([sx, ex], [sy, ey], color="#4D4D4D", linewidth=2.2))
                
                # if node.feature is not None and node.prediction is None:
                #     tx, ty = _edge_label_pos(x, y, x2, y2, frac=0.52, base_offset=0.28, side_sign=+1.0)
                #     ax.text(
                #         tx, ty,
                #         _edge_label(node.feature, True),
                #         ha="center", va="center", fontsize=11,
                #         bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.9),
                #     )
                draw_node(node.left)

            if node.right is not None:
                x2, y2 = positions[node.right]
                #ax.add_line(Line2D([x, x2], [y, y2], color="black", linewidth=2.2))
                r_parent = internal_r
                r_child = leaf_r if node.right.prediction is not None else internal_r

                sx, sy, ex, ey = _shrink_segment(x, y, x2, y2, r_parent, r_child)
                ax.add_line(Line2D([sx, ex], [sy, ey], color="#4D4D4D", linewidth=2.2))

                # if node.feature is not None and node.prediction is None:
                #     tx, ty = _edge_label_pos(x, y, x2, y2, frac=0.52, base_offset=0.28, side_sign=-1.0)
                #     ax.text(
                #         tx, ty,
                #         _edge_label(node.feature, False),
                #         ha="center", va="center", fontsize=11,
                #         bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.9),
                #     )
                draw_node(node.right)

            # draw node
            if node.prediction is None:
                face = "#DCEAF4"
                edge = "#4D4D4D"
                text_color = "#222222"
                radius = internal_r
                label = None
            else:
                prediction = int(node.prediction)

                if prediction == -1:
                    face = "#7A5195"
                    label = "D"
                elif prediction == 0:
                    face = "#E69F00"
                    label = "0"
                else:
                    face = "#009E73"
                    label = str(prediction)

             
                edge = "#4D4D4D"
                text_color = "#111111"
                radius = leaf_r

            circ = Circle(
                (x, y),
                radius,
                facecolor=face,
                edgecolor=edge,
                linewidth=1.6
            )
            ax.add_patch(circ)

            if node.prediction is None and node.feature is not None:
                # feature name above internal node
                ax.text(
                    x, y + radius + 0.22,
                    feature_names[node.feature],
                    ha="center", va="bottom",
                    fontsize=11,
                    color="#222222",
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.95),
                    zorder=10,
                )

            if label is not None:
                ax.text(
                    x, y,
                    label,
                    ha="center", va="center",
                    fontsize=12,
                    fontweight="bold",
                    color="white",
                    zorder=10,
                )
        draw_node(root)

        xs, ys = zip(*positions.values())
        pad_x = 1.6
        pad_y = 1.6
        ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
        ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()
        ax.set_title(f"ArborEnum Tree {tree_index}" if title is None else str(title))
        if show:
            plt.show()
        return fig, ax


        
    def get_tree_frontier_scores(self, tree_index: int, depth_budget: int):
        # returns a list of (depth_from_root, frontier_score) for each internal node of the specified tree. Root has depth 0.
        return self._model.get_tree_frontier_scores(int(tree_index), int(depth_budget))

    def root_lickety_objective_lookahead1(self, depth_budget: int):
        return int(self._model.root_lickety_objective_lookahead1(int(depth_budget)))

    def reachable_actions_for_training_sample(self, sample_idx: int):
        actions = self._model.reachable_actions_for_training_sample(
            int(sample_idx)
        )

        return {
            "class_mask": int(actions["class_mask"]),
            "can_defer": bool(actions["can_defer"]),
        }


    def training_sample_can_defer(self, sample_idx: int):
        return bool(
            self._model.training_sample_can_defer(
                int(sample_idx)
            )
        )

    def training_sample_has_multiple_reachable_predictions(self, sample_idx: int):
        return self._model.training_sample_has_multiple_reachable_predictions(int(sample_idx))

    def training_samples_with_multiple_reachable_predictions(self):
        return self._model.training_samples_with_multiple_reachable_predictions()

    @staticmethod
    def _prepare_rid_input(X):
        X = _as_2d_numeric_array(X, "X")

        constant_indices = []
        binary_indices = []
        continuous_indices = []

        for j in range(X.shape[1]):
            n_unique = np.unique(X[:, j]).size

            if n_unique <= 1:
                constant_indices.append(j)
            elif n_unique == 2:
                binary_indices.append(j)
            else:
                continuous_indices.append(j)

        # matching internal order
        retained_indices = binary_indices + continuous_indices

        if not retained_indices:
            raise ValueError(
                "RID requires at least one nonconstant feature."
            )

        X_rid = X[:, retained_indices]

        return (
            X,
            X_rid,
            retained_indices,
            constant_indices,
            binary_indices,
            continuous_indices,
        )

    def _resolve_rid_feature_names(self, feature_names=None):
        self._require_rid()

        if feature_names is None:
            return list(self.rid_feature_names_)

        feature_names = list(feature_names)

        if len(feature_names) != self.rid_n_original_features_:
            raise ValueError(
                "feature_names must contain one name for every column "
                "in the original dataset, in the original dataset order. "
                f"Expected {self.rid_n_original_features_} names, "
                f"received {len(feature_names)}."
            )

        return [
            feature_names[j]
            for j in self.rid_feature_indices_
        ]

    def _resolve_rid_feature(self, feature, feature_names):
        if isinstance(feature, (int, np.integer)):
            feature = int(feature)

            if feature < 0 or feature >= len(feature_names):
                raise IndexError(
                    f"RID feature index {feature} is outside "
                    f"[0, {len(feature_names) - 1}]."
                )

            return feature

        feature = str(feature)

        matches = [
            j
            for j, name in enumerate(feature_names)
            if name == feature
        ]

        if not matches:
            raise ValueError(
                f"Unknown RID feature {feature!r}. "
                f"Available features are: {feature_names}"
            )

        if len(matches) > 1:
            raise ValueError(
                f"RID feature name {feature!r} is not unique."
            )

        return matches[0]

    def compute_rid(
        self,
        X,
        y,
        *,
        X_proxy=None,
        X_initial=None,
        early_stopping=False,
        proxy_mode="hybrid",
        greedy_continuous_mode="binary",
        proxy_features_n_estimators=150,
        proxy_features_max_depth=2,
        proxy_features_random_state=0,
        proxy_features_column_elimination=False,
        n_boot=10,
        n_scramble_evals=5,
        lambda_reg=0.01,
        depth_budget=5,
        rashomon_mult=0.03,
        root_budget=None,
        additive=False,
        second_rashomon_mult=None,
        multiplier_step_size=0.01,
        lookahead_k=1,
        seed=0,
        memory_efficient=False,
        key_mode="hash",
        trie_cache_enabled=True,
        proxy_refinement="auto",
        refinement_width=1,
        max_refinement_rounds=-1,
        runtime_limit_seconds=-1.0,
        memory_limit_mb=-1.0,
        max_number_thresholds_per_feature=None,
        use_budget_refinement=True,
        guarantee_rule_list_recovery=False,
        proxy_style=0,
        majority_leaf_only=False,
        cache_early_exits=False,
        heuristic_for_greedy=1,
        proxy_caching=True,
        use_deferral=False,
        eta_defer=0.0,
        bb_pred=None,
        return_joint_samples=False,
        lossless=True,
        matched_groups_by_variable=None,
        importance_interval_mode=0,
        subsample=-1,
    ):

        if hasattr(X, "columns"):
            original_feature_names = list(X.columns)
        else:
            original_feature_names = [
                f"f{j}" for j in range(np.asarray(X).shape[1])
            ]

        self.rid_n_original_features_ = len(original_feature_names)
        
        (
            X_original,
            X_rid,
            self.rid_feature_indices_,
            self.rid_constant_feature_indices_,
            rid_binary_original_indices,
            rid_continuous_original_indices,
        ) = self._prepare_rid_input(X)

        self.rid_feature_names_ = [
            original_feature_names[j]
            for j in self.rid_feature_indices_
        ]

        self.rid_constant_feature_names_ = [
            original_feature_names[j]
            for j in self.rid_constant_feature_indices_
        ]

        (
            _,
            X_bin,
            X_num,
            rid_binary_feature_specs,
            rid_continuous_feature_indices,
        ) = _split_binary_and_continuous(
            X_rid,
            binary_unique_threshold=2,
        )



        y = np.asarray(y, dtype=int)
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}")
        if y.shape[0] != X_original.shape[0]:
            raise ValueError(
                f"y length must match X rows: got {y.shape[0]} vs "
                f"{X_original.shape[0]}"
            )

        n_boot = int(n_boot)
        n_scramble_evals = int(n_scramble_evals)

        importance_interval_mode = int(importance_interval_mode)
        subsample = int(subsample)

        if importance_interval_mode not in (0, 1, 2):
            raise ValueError(
                "importance_interval_mode must be 0, 1, or 2."
            )

        if importance_interval_mode != 0 and not lossless:
            raise ValueError(
                "importance_interval_mode requires lossless=True."
            )

        if importance_interval_mode != 0 and return_joint_samples:
            raise ValueError(
                "return_joint_samples is incompatible with "
                "importance_interval_mode != 0."
            )

        if subsample != -1 and (
            subsample <= 0 or subsample > X_original.shape[0]
        ):
            raise ValueError(
                "subsample must be -1 or an integer in "
                f"[1, {X_original.shape[0]}]."
            )

        root_budget_int = _resolve_rid_root_budget(
            root_budget
        )

        if root_budget_int >= 0 and early_stopping:
            raise ValueError(
                "root_budget is not currently supported with "
                "early_stopping=True."
            )

        if n_boot <= 0:
            raise ValueError("n_boot must be positive.")

        if not lossless and n_scramble_evals <= 0:
            raise ValueError(
                "n_scramble_evals must be positive."
            )

        eta_defer = float(eta_defer)

        if not np.isfinite(eta_defer) or eta_defer < 0.0:
            raise ValueError(
                "eta_defer must be finite and nonnegative."
            )

        matched_groups_by_original_variable = (
            _prepare_matched_groups_by_variable(
                matched_groups_by_variable,
                X_original.shape[0],
                X_original.shape[1],
            )
        )

        if matched_groups_by_original_variable:
            matched_groups_by_variable_vec = [
                matched_groups_by_original_variable[j]
                for j in self.rid_feature_indices_
            ]
        else:
            matched_groups_by_variable_vec = []

        

        n_classes = int(np.max(y)) + 1

        bb_pred_vec = _prepare_bb_pred(
            bb_pred,
            X_original.shape[0],
            use_deferral=bool(use_deferral),
            n_classes=n_classes,
        )

        proxy_settings = _proxy_mode_settings(proxy_mode)

        X_proxy = _resolve_proxy_matrix(
            X_original=X_rid,
            y=y,
            X_proxy=X_proxy,
            proxy_settings=proxy_settings,
            proxy_features_n_estimators=(
                proxy_features_n_estimators
            ),
            proxy_features_max_depth=(
                proxy_features_max_depth
            ),
            proxy_features_random_state=(
                proxy_features_random_state
            ),
            proxy_features_column_elimination=(
                proxy_features_column_elimination
            ),
        )

        n = X_original.shape[0]

        if X_initial is None:
            if early_stopping and X_num.shape[1] > 0:
                if proxy_settings["mode"] == "binarized":
                    X_initial = X_proxy
                else:
                    X_initial = y.reshape(-1, 1).astype(np.uint8)
            else:
                X_initial = np.empty((n, 0), dtype=np.uint8)
        else:
            X_initial = _optional_binary_matrix(
                X_initial,
                n,
                "X_initial",
            )

        if not early_stopping and (
            float(runtime_limit_seconds) >= 0.0
            or float(memory_limit_mb) >= 0.0
        ):
            raise ValueError(
                "runtime_limit_seconds and memory_limit_mb are only "
                "available when early_stopping=True."
            )

        rashomon_mult = float(rashomon_mult)

        if second_rashomon_mult is None:
            second_rashomon_mult = rashomon_mult
        else:
            second_rashomon_mult = float(
                second_rashomon_mult
            )

        multiplier_step_size = float(multiplier_step_size)

        if rashomon_mult < 0.0:
            raise ValueError(
                "rashomon_mult must be nonnegative."
            )

        if second_rashomon_mult < 0.0:
            raise ValueError(
                "second_rashomon_mult must be nonnegative."
            )

        if (
            second_rashomon_mult > rashomon_mult
            and multiplier_step_size <= 0.0
        ):
            raise ValueError(
                "multiplier_step_size must be positive when "
                "second_rashomon_mult exceeds rashomon_mult."
            )

        if max_number_thresholds_per_feature is None:
            max_thresholds = -1
        else:
            max_thresholds = int(
                max_number_thresholds_per_feature
            )
            if max_thresholds <= 0:
                raise ValueError(
                    "max_number_thresholds_per_feature must be "
                    "positive or None."
                )

        expected_rid_features = (
            X_bin.shape[1] + X_num.shape[1]
        )

        if expected_rid_features != len(self.rid_feature_indices_):
            raise RuntimeError(
                "Internal RID feature grouping does not match the "
                "retained original features."
            )

        self._rid_out = _rid_subtractive_continuous_core(
            X_num,
            X_bin,
            y,
            X_proxy,
            X_initial,
            int(n_boot),
            int(n_scramble_evals),
            float(lambda_reg),
            int(depth_budget),
            float(rashomon_mult),
            float(second_rashomon_mult),
            float(multiplier_step_size),
            int(lookahead_k),
            int(seed),
            bool(memory_efficient),
            parse_key_mode(key_mode),
            bool(trie_cache_enabled),
            bool(early_stopping),
            int(refinement_width),
            int(max_refinement_rounds),
            parse_proxy_refinement(proxy_refinement),
            bool(proxy_settings["continuous_lickety"]),
            bool(proxy_settings["continuous_depthd"]),
            bool(proxy_settings["continuous_greedy"]),
            bool(use_budget_refinement),
            bool(guarantee_rule_list_recovery),
            int(parse_proxy_style(proxy_style)),
            bool(majority_leaf_only),
            bool(cache_early_exits),
            int(parse_heuristic_for_greedy(heuristic_for_greedy)),
            parse_greedy_continuous_mode(greedy_continuous_mode),
            bool(proxy_caching),
            int(max_thresholds),
            float(runtime_limit_seconds),
            float(memory_limit_mb),
            bool(use_deferral),
            float(eta_defer),
            bb_pred_vec,
            bool(return_joint_samples),
            bool(lossless),
            matched_groups_by_variable_vec,
            bool(additive),
            int(importance_interval_mode),
            int(subsample),
            int(root_budget_int),
        )

        if (
            bool(return_joint_samples)
            and "feature_importance_weight_samples" in self._rid_out
        ):
            self._rid_out[
                "feature_importance_weight_samples"
            ] = np.asarray(
                self._rid_out[
                    "feature_importance_weight_samples"
                ],
                dtype=float,
            )

        if "bootstrap_importance_intervals" in self._rid_out:
            self._rid_out["bootstrap_importance_intervals"] = np.asarray(
                self._rid_out["bootstrap_importance_intervals"],
                dtype=float,
            )

        return self._rid_out

    def compute_rid_binarized(
        self,
        X,
        y,
        n_boot=10,
        n_scramble_evals=5,
        lambda_reg=0.01,
        depth_budget=5,
        rashomon_mult=0.03,
        root_budget=None,
        lookahead_k=1,
        seed=0,
        memory_efficient=False,
        key_mode="hash",
        trie_cache_enabled=True,
        binning_map=None,
        use_deferral=False,
        eta_defer=0.0,
        bb_pred=None,
        return_joint_samples=False,
        lossless=True,
        matched_groups_by_variable=None,
        additive=False,
        importance_interval_mode=0,
        subsample=-1,
    ):
        X = np.asarray(X, dtype=np.uint8)
        y = np.asarray(y, dtype=int)

        if X.ndim != 2:
            raise ValueError(
                f"X must be 2D, got shape {X.shape}"
            )

        if y.ndim != 1:
            raise ValueError(
                f"y must be 1D, got shape {y.shape}"
            )

        if y.shape[0] != X.shape[0]:
            raise ValueError(
                "y length must match X rows: "
                f"got {y.shape[0]} vs {X.shape[0]}"
            )

        n_boot = int(n_boot)
        n_scramble_evals = int(n_scramble_evals)

        if n_boot <= 0:
            raise ValueError("n_boot must be positive.")

        importance_interval_mode = int(importance_interval_mode)
        subsample = int(subsample)

        if importance_interval_mode not in (0, 1, 2):
            raise ValueError(
                "importance_interval_mode must be 0, 1, or 2."
            )

        if importance_interval_mode != 0 and not lossless:
            raise ValueError(
                "importance_interval_mode requires lossless=True."
            )

        if importance_interval_mode != 0 and return_joint_samples:
            raise ValueError(
                "return_joint_samples is incompatible with "
                "importance_interval_mode != 0."
            )

        if subsample != -1 and (
            subsample <= 0 or subsample > X.shape[0]
        ):
            raise ValueError(
                "subsample must be -1 or an integer in "
                f"[1, {X.shape[0]}]."
            )

        root_budget_int = _resolve_rid_root_budget(
            root_budget
        )

        if not lossless and n_scramble_evals <= 0:
            raise ValueError(
                "n_scramble_evals must be positive."
            )

        eta_defer = float(eta_defer)

        if not np.isfinite(eta_defer) or eta_defer < 0.0:
            raise ValueError(
                "eta_defer must be finite and nonnegative."
            )

        if binning_map is None:
            n_rid_variables = X.shape[1]
        else:
            n_rid_variables = len(binning_map)

        matched_groups_by_variable_vec = (
            _prepare_matched_groups_by_variable(
                matched_groups_by_variable,
                X.shape[0],
                n_rid_variables,
            )
        )
        n_classes = int(np.max(y)) + 1

        bb_pred_vec = _prepare_bb_pred(
            bb_pred,
            X.shape[0],
            use_deferral=bool(use_deferral),
            n_classes=n_classes,
        )

        self._rid_out = _rid_subtractive_core(
            X,
            y,
            int(n_boot),
            int(n_scramble_evals),
            float(lambda_reg),
            int(depth_budget),
            float(rashomon_mult),
            int(lookahead_k),
            int(seed),
            bool(memory_efficient),
            parse_key_mode(key_mode),
            bool(trie_cache_enabled),
            binning_map,
            bool(use_deferral),
            float(eta_defer),
            bb_pred_vec,
            bool(return_joint_samples),
            bool(lossless),
            matched_groups_by_variable_vec,
            bool(additive),
            int(importance_interval_mode),
            int(subsample),
            int(root_budget_int),
        )

        if (
            bool(return_joint_samples)
            and "feature_importance_weight_samples" in self._rid_out
        ):
            self._rid_out[
                "feature_importance_weight_samples"
            ] = np.asarray(
                self._rid_out[
                    "feature_importance_weight_samples"
                ],
                dtype=float,
            )
        
        if "bootstrap_importance_intervals" in self._rid_out:
            self._rid_out["bootstrap_importance_intervals"] = np.asarray(
                self._rid_out["bootstrap_importance_intervals"],
                dtype=float,
            )

        return self._rid_out

    def compute_rid_continuous_low_level(
        self,
        X_num,
        y,
        X_bin=None,
        X_proxy_active=None,
        X_initial_active=None,
        n_boot=10,
        n_scramble_evals=5,
        lambda_reg=0.01,
        depth_budget=5,
        rashomon_mult=0.03,
        root_budget=None,
        second_rashomon_mult=None,
        multiplier_step_size=0.01,
        lookahead_k=1,
        seed=0,
        memory_efficient=False,
        key_mode="hash",
        trie_cache_enabled=True,
        use_anytime_fit=False,
        refinement_width=1,
        max_refinement_rounds=-1,
        proxy_refinement_mode=0,
        continuous_proxy_in_lickety=True,
        continuous_proxy_in_depthd_exact=True,
        continuous_proxy_in_greedy=True,
        use_budget_refinement=True,
        guarantee_rule_list_recovery=False,
        proxy_style=0,
        majority_leaf_only=False,
        cache_early_exits=False,
        heuristic_for_greedy=1,
        greedy_continuous_mode="binary",
        proxy_caching=True,
        max_number_thresholds_per_feature=None,
        runtime_limit_seconds=-1.0,
        memory_limit_mb=-1.0,
        use_deferral=False,
        eta_defer=0.0,
        bb_pred=None,
        return_joint_samples=False,
        lossless=True,
        matched_groups_by_variable=None,
        additive=False,
        importance_interval_mode=0,
        subsample=-1,
    ):
        X_num = np.asarray(X_num, dtype=np.float64)
        y = np.asarray(y, dtype=int)

        if X_num.ndim != 2:
            raise ValueError(
                f"X_num must be 2D, got shape {X_num.shape}"
            )

        if y.ndim != 1:
            raise ValueError(
                f"y must be 1D, got shape {y.shape}"
            )

        n = X_num.shape[0]

        if y.shape[0] != n:
            raise ValueError(
                f"y length must match X_num rows: got len(y)={y.shape[0]}, "
                f"X_num rows={n}"
            )

        n_boot = int(n_boot)
        n_scramble_evals = int(n_scramble_evals)

        if n_boot <= 0:
            raise ValueError("n_boot must be positive.")

        importance_interval_mode = int(importance_interval_mode)
        subsample = int(subsample)

        if importance_interval_mode not in (0, 1, 2):
            raise ValueError(
                "importance_interval_mode must be 0, 1, or 2."
            )

        if importance_interval_mode != 0 and not lossless:
            raise ValueError(
                "importance_interval_mode requires lossless=True."
            )

        if importance_interval_mode != 0 and return_joint_samples:
            raise ValueError(
                "return_joint_samples is incompatible with "
                "importance_interval_mode != 0."
            )

        if subsample != -1 and (
            subsample <= 0 or subsample > n
        ):
            raise ValueError(
                "subsample must be -1 or an integer in "
                f"[1, {n}]."
            )

        root_budget_int = _resolve_rid_root_budget(
            root_budget
        )

        if root_budget_int >= 0 and use_anytime_fit:
            raise ValueError(
                "root_budget is not currently supported with "
                "use_anytime_fit=True."
            )

        if not lossless and n_scramble_evals <= 0:
            raise ValueError(
                "n_scramble_evals must be positive."
            )

        eta_defer = float(eta_defer)

        if not np.isfinite(eta_defer) or eta_defer < 0.0:
            raise ValueError(
                "eta_defer must be finite and nonnegative."
            )

        n_classes = int(np.max(y)) + 1

        bb_pred_vec = _prepare_bb_pred(
            bb_pred,
            n,
            use_deferral=bool(use_deferral),
            n_classes=n_classes,
        )

        if X_bin is None:
            X_bin = np.empty((n, 0), dtype=np.uint8)
        else:
            X_bin = np.asarray(X_bin, dtype=np.uint8)

            if X_bin.ndim != 2:
                raise ValueError(
                    f"X_bin must be 2D, got shape {X_bin.shape}"
                )

            if X_bin.shape[0] != n:
                raise ValueError(
                    f"X_bin rows must match X_num rows: "
                    f"got {X_bin.shape[0]} vs {n}"
                )

        if X_proxy_active is None:
            X_proxy_active = np.empty(
                (n, 0),
                dtype=np.uint8,
            )
        else:
            X_proxy_active = np.asarray(
                X_proxy_active,
                dtype=np.uint8,
            )

            if X_proxy_active.ndim != 2:
                raise ValueError(
                    "X_proxy_active must be 2D, "
                    f"got shape {X_proxy_active.shape}"
                )

            if X_proxy_active.shape[0] != n:
                raise ValueError(
                    "X_proxy_active rows must match X_num rows: "
                    f"got {X_proxy_active.shape[0]} vs {n}"
                )

        if X_initial_active is None:
            X_initial_active = np.empty(
                (n, 0),
                dtype=np.uint8,
            )
        else:
            X_initial_active = np.asarray(
                X_initial_active,
                dtype=np.uint8,
            )

            if X_initial_active.ndim != 2:
                raise ValueError(
                    "X_initial_active must be 2D, "
                    f"got shape {X_initial_active.shape}"
                )

            if X_initial_active.shape[0] != n:
                raise ValueError(
                    "X_initial_active rows must match X_num rows: "
                    f"got {X_initial_active.shape[0]} vs {n}"
                )

        rashomon_mult = float(rashomon_mult)

        if second_rashomon_mult is None:
            second_rashomon_mult = rashomon_mult
        else:
            second_rashomon_mult = float(
                second_rashomon_mult
            )

        multiplier_step_size = float(multiplier_step_size)

        if rashomon_mult < 0.0:
            raise ValueError(
                "rashomon_mult must be nonnegative."
            )

        if second_rashomon_mult < 0.0:
            raise ValueError(
                "second_rashomon_mult must be nonnegative."
            )

        if (
            second_rashomon_mult > rashomon_mult
            and multiplier_step_size <= 0
        ):
            raise ValueError(
                "multiplier_step_size must be positive when "
                "second_rashomon_mult exceeds rashomon_mult."
            )

        proxy_refinement_mode = int(proxy_refinement_mode)

        if proxy_refinement_mode not in (0, 1, 2):
            raise ValueError(
                "proxy_refinement_mode must be 0, 1, or 2."
            )

        proxy_style_int = parse_proxy_style(proxy_style)

        if max_number_thresholds_per_feature is None:
            max_thresholds = -1
        else:
            max_thresholds = int(
                max_number_thresholds_per_feature
            )
            if max_thresholds <= 0:
                raise ValueError(
                    "max_number_thresholds_per_feature must be "
                    "positive or None."
                )

        n_rid_variables = X_bin.shape[1] + X_num.shape[1]

        matched_groups_by_variable_vec = (
            _prepare_matched_groups_by_variable(
                matched_groups_by_variable,
                n,
                n_rid_variables,
            )
        )

        self._rid_out = _rid_subtractive_continuous_core(
            X_num,
            X_bin,
            y,
            X_proxy_active,
            X_initial_active,
            int(n_boot),
            int(n_scramble_evals),
            float(lambda_reg),
            int(depth_budget),
            rashomon_mult,
            second_rashomon_mult,
            multiplier_step_size,
            int(lookahead_k),
            int(seed),
            bool(memory_efficient),
            parse_key_mode(key_mode),
            bool(trie_cache_enabled),
            bool(use_anytime_fit),
            int(refinement_width),
            int(max_refinement_rounds),
            proxy_refinement_mode,
            bool(continuous_proxy_in_lickety),
            bool(continuous_proxy_in_depthd_exact),
            bool(continuous_proxy_in_greedy),
            bool(use_budget_refinement),
            bool(guarantee_rule_list_recovery),
            int(proxy_style_int),
            bool(majority_leaf_only),
            bool(cache_early_exits),
            int(parse_heuristic_for_greedy(heuristic_for_greedy)),
            parse_greedy_continuous_mode(
                greedy_continuous_mode
            ),
            bool(proxy_caching),
            int(max_thresholds),
            float(runtime_limit_seconds),
            float(memory_limit_mb),
            bool(use_deferral),
            float(eta_defer),
            bb_pred_vec,
            bool(return_joint_samples),
            bool(lossless),
            matched_groups_by_variable_vec,
            bool(additive),
            int(importance_interval_mode),
            int(subsample),
            int(root_budget_int),
        )

        if "bootstrap_importance_intervals" in self._rid_out:
            self._rid_out["bootstrap_importance_intervals"] = np.asarray(
                self._rid_out["bootstrap_importance_intervals"],
                dtype=float,
            )

        if (
            bool(return_joint_samples)
            and "feature_importance_weight_samples" in self._rid_out
        ):
            self._rid_out[
                "feature_importance_weight_samples"
            ] = np.asarray(
                self._rid_out[
                    "feature_importance_weight_samples"
                ],
                dtype=float,
            )

        return self._rid_out
    
    def _require_rid(self):
        if self._rid_out is None:
            raise RuntimeError("RID not computed. Call compute_rid(...) first.")
        return self._rid_out
    
    def rid_plot_mean(self, feature_names=None, **kwargs):
        rid_out = self._require_rid()
        feature_names = self._resolve_rid_feature_names(
            feature_names
        )

        return rid_plot_mean(
            rid_out,
            feature_names=feature_names,
            **kwargs,
        )


    def rid_plot_violin(self, feature_names=None, **kwargs):
        rid_out = self._require_rid()
        feature_names = self._resolve_rid_feature_names(
            feature_names
        )

        return rid_plot_violin(
            rid_out,
            feature_names=feature_names,
            **kwargs,
        )


    def rid_plot_cdfs(self, feature_names=None, **kwargs):
        rid_out = self._require_rid()
        feature_names = self._resolve_rid_feature_names(
            feature_names
        )

        return rid_plot_cdfs(
            rid_out,
            feature_names=feature_names,
            **kwargs,
        )

    def rid_plot_pair(
        self,
        feature_a,
        feature_b,
        feature_names=None,
        **kwargs,
    ):
        rid_out = self._require_rid()

        feature_names = self._resolve_rid_feature_names(
            feature_names
        )

        feature_a_index = self._resolve_rid_feature(
            feature_a,
            feature_names,
        )

        feature_b_index = self._resolve_rid_feature(
            feature_b,
            feature_names,
        )

        if feature_a_index == feature_b_index:
            raise ValueError(
                "feature_a and feature_b must refer to different features."
            )

        return rid_plot_pair(
            rid_out,
            feature_a_index,
            feature_b_index,
            feature_names=feature_names,
            **kwargs,
        )


    def rid_plot_all_pairs(
        self,
        feature_names=None,
        **kwargs,
    ):
        rid_out = self._require_rid()

        feature_names = self._resolve_rid_feature_names(
            feature_names
        )

        return rid_plot_all_pairs(
            rid_out,
            feature_names=feature_names,
            **kwargs,
        )
        
    @staticmethod
    def _require_binary_predictions(preds):
        a = np.asarray(preds)
        u = np.unique(a)
        ok = np.all((u == 0) | (u == 1))
        if not ok:
            raise ValueError(f"We require predictions in {{0,1}}.")

    def get_p_per_sample(self, X, tree_indices=None):
        # returns p_i per sample, the proportion of models predicting 1 - require binary predicitons
        # tree_indices : iterable[int] | None. if none, uses all trees, otherwise averages over the tree indices provided.
       

        if tree_indices is None:
            P = self.get_all_predictions(X, stack=True)
        else:
            idxs = list(tree_indices)
            if len(idxs) == 0:
                raise ValueError("tree_indices is empty.")
            preds_list = [self.get_predictions(int(t), X) for t in idxs]
            P = np.stack(preds_list, axis=0)

        self._require_binary_predictions(P)

        return P.mean(axis=0)

    def get_variance_per_sample(self, X, tree_indices=None):
        # returns per-sample variance of hard predictions across trees: p_i(1-p_i)

        if tree_indices is None:
            P = self.get_all_predictions(X, stack=True)  # (T, N)
        else:
            idxs = list(tree_indices)
            preds_list = [self.get_predictions(int(t), X) for t in idxs]
            P = np.stack(preds_list, axis=0)

        P = np.asarray(P)
        self._require_binary_predictions(P)

        return P.var(axis=0, ddof=0)

    def get_avg_variance_across_samples(self, X, tree_indices=None):
        v = self.get_variance_per_sample(X, tree_indices=tree_indices)
        return float(np.mean(v))

    def plot_disagreement_cdf(self, X, tree_indices=None, ax=None, figsize=(6.5, 4.0), title="Disagreement across samples", show=True, label=None):
        # plots proportion of points where variance is at most threshold t.
        v = self.get_variance_per_sample(X, tree_indices=tree_indices)
        v = np.asarray(v, float)
        n = v.size
    
        xs = np.sort(v)
        F = (np.arange(1, n + 1, dtype=float) / float(n))

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        ax.step(xs, F, where="post", linewidth=2.0, label=label)
        ax.set_ylabel("Proportion with var ≤ t")

        ax.set_xlabel("Variance threshold t")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(title)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.25)

        if label is not None:
            ax.legend(frameon=False)

        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

def _rid_style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25)


def _rid_feature_names(feature_names, V):
    if feature_names is None:
        return [f"f{j}" for j in range(V)]
    if len(feature_names) != V:
        raise ValueError(f"feature_names must have length {V}, got {len(feature_names)}")
    return list(feature_names)


def _rid_sorted_xy(xs, ps):
    xs = np.asarray(xs, float)
    ps = np.asarray(ps, float)
    if xs.size == 0:
        return xs, ps
    idx = np.argsort(xs)
    xs, ps = xs[idx], ps[idx]
    ps = np.clip(ps, 0.0, 1.0)
    ps = np.maximum.accumulate(ps)
    return xs, ps


def _rid_pmf_from_cdf(xs, ps):
    if xs.size == 0:
        return xs, np.asarray([], float)
    pprev = np.concatenate([[0.0], ps[:-1]])
    w = ps - pprev
    w = np.maximum(w, 0.0)
    s = w.sum()
    if s <= 0:
        w = np.ones_like(w) / max(1, w.size)
    else:
        w = w / s
    return xs, w


def rid_plot_mean(
    rid_out,
    feature_names=None,
    figsize=(10, 3),
    ax=None,
    title="RID mean reliance per feature",
    show=True,
):
    mean = np.asarray(rid_out["mean_sub_mr"], dtype=float)
    V = int(mean.size)
    feature_names = _rid_feature_names(feature_names, V)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    x = np.arange(V)
    ax.scatter(x, mean, s=30)
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_xlabel("feature")
    ax.set_ylabel("mean reliance\n(average accuracy drop when scrambled)")
    ax.set_title(title)
    _rid_style_ax(ax)

    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def rid_plot_violin(
    rid_out,
    feature_names=None,
    samples_per_feature=4000,
    seed=123,
    figsize=(10, 6),
    ax=None,
    title="RID distribution per feature",
    show=True,
):
    mean = np.asarray(rid_out["mean_sub_mr"], dtype=float)
    cdf_x = rid_out["cdf_x"]
    cdf_p = rid_out["cdf_p"]

    V = int(mean.size)
    feature_names = _rid_feature_names(feature_names, V)

    order = np.argsort(-mean)
    mean_s = mean[order]
    names_s = [feature_names[j] for j in order]

    cdf_pairs = []
    xmin, xmax = 0.0, 0.0
    for j in range(V):
        xs, ps = _rid_sorted_xy(cdf_x[j], cdf_p[j])
        cdf_pairs.append((xs, ps))
        if xs.size:
            xmin = min(xmin, float(xs[0]))
            xmax = max(xmax, float(xs[-1]))

    rng = np.random.default_rng(seed)
    samples_sorted = []
    for j in order:
        xs, ps = cdf_pairs[int(j)]
        xs, w = _rid_pmf_from_cdf(xs, ps)
        if xs.size == 0:
            samples_sorted.append(np.zeros(int(samples_per_feature), float))
        else:
            samples_sorted.append(
                rng.choice(xs, size=int(samples_per_feature), replace=True, p=w)
            )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    parts = ax.violinplot(
        samples_sorted,
        positions=np.arange(V),
        vert=False,
        widths=0.85,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body in parts["bodies"]:
        body.set_alpha(0.65)

    med_sorted = np.array([np.median(s) for s in samples_sorted])
    # ax.scatter(med_sorted, np.arange(V), s=18, zorder=3, label="median")
    # ax.scatter(mean_s, np.arange(V), s=22, zorder=3, marker="x", label="mean")
    # median: white dot with black outline (high contrast on top of violin)
    ax.scatter(
        med_sorted, np.arange(V),
        s=44, zorder=4,
        facecolors="white", edgecolors="black", linewidths=1.2,
        label="median",
    )
    ax.scatter(
        mean_s, np.arange(V),
        s=46, zorder=4,
        marker="x", c="black", linewidths=1.6,
        label="mean",
    )


    ax.set_yticks(np.arange(V))
    ax.set_yticklabels(names_s)
    ax.set_xlabel("Accuracy drop when scrambled")
    ax.set_ylabel("feature")
    ax.set_title(title)
    ax.set_xlim(xmin, xmax)
    _rid_style_ax(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def rid_plot_cdfs(
    rid_out,
    feature_names=None,
    figsize=(10, 4.5),
    ax=None,
    title="RID CDFs (all features overlaid)",
    cmap_name="tab20",
    legend_ncol=5,
    legend_fontsize=9,
    show=True,
):
    cdf_x = rid_out["cdf_x"]
    cdf_p = rid_out["cdf_p"]
    mean = np.asarray(rid_out["mean_sub_mr"], dtype=float)

    V = int(mean.size)
    feature_names = _rid_feature_names(feature_names, V)

    cdf_pairs = []
    xmin, xmax = 0.0, 0.0
    for j in range(V):
        xs, ps = _rid_sorted_xy(cdf_x[j], cdf_p[j])
        cdf_pairs.append((xs, ps))
        if xs.size:
            xmin = min(xmin, float(xs[0]))
            xmax = max(xmax, float(xs[-1]))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    cmap = colormaps[cmap_name]
    colors = [cmap(i % 20) for i in range(V)]

    for j in range(V):
        xs, ps = cdf_pairs[j]
        if xs.size == 0:
            continue
        ax.plot(xs, ps, linewidth=1.8, alpha=0.9, color=colors[j], label=feature_names[j])

    ax.set_title(title)
    ax.set_xlabel("Accuracy drop when scrambled")
    ax.set_ylabel("P[Δ accuracy ≤ t]")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.02, 1.02)
    _rid_style_ax(ax)

    ax.legend(
        ncol=int(legend_ncol),
        fontsize=float(legend_fontsize),
        frameon=False,
        handlelength=1.8,
        columnspacing=1.1,
    )

    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax

def _rid_joint_samples(rid_out):
    key = "feature_importance_weight_samples"

    if key not in rid_out:
        raise RuntimeError(
            "Joint RID samples were not returned. "
            "Call compute_rid(..., return_joint_samples=True) first."
        )

    samples = np.asarray(
        rid_out[key],
        dtype=float,
    )

    if samples.ndim != 2 or samples.shape[1] < 2:
        raise ValueError(
            "feature_importance_weight_samples must be a 2D array "
            "with feature columns followed by one weight column."
        )

    return samples


def _rid_feature_index(
    feature,
    feature_names,
    n_features,
):
    if isinstance(feature, (int, np.integer)):
        feature_index = int(feature)

        if (
            feature_index < 0
            or feature_index >= n_features
        ):
            raise IndexError(
                f"feature index {feature_index} is outside "
                f"[0, {n_features})."
            )

        return feature_index

    if feature_names is None:
        raise ValueError(
            "String feature names require feature_names "
            "to be provided."
        )

    feature = str(feature)

    if feature not in feature_names:
        raise ValueError(
            f"Unknown feature '{feature}'. "
            f"Available features: {feature_names}"
        )

    return int(feature_names.index(feature))


def rid_plot_pair(
    rid_out,
    feature_a,
    feature_b,
    feature_names=None,
    figsize=(5.5, 5.0),
    ax=None,
    title=None,
    alpha=0.35,
    s=14,
    show=True,
):
    samples = _rid_joint_samples(rid_out)

    n_features = samples.shape[1] - 1

    feature_names = _rid_feature_names(
        feature_names,
        n_features,
    )

    feature_a = _rid_feature_index(
        feature_a,
        feature_names,
        n_features,
    )

    feature_b = _rid_feature_index(
        feature_b,
        feature_names,
        n_features,
    )

    x = samples[:, feature_a]
    y = samples[:, feature_b]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.scatter(
        x,
        y,
        s=s,
        alpha=alpha,
    )

    ax.set_xlabel(
        f"RID importance: {feature_names[feature_a]}"
    )

    ax.set_ylabel(
        f"RID importance: {feature_names[feature_b]}"
    )

    if title is None:
        title = (
            "RID pair distribution: "
            f"{feature_names[feature_a]} vs "
            f"{feature_names[feature_b]}"
        )

    ax.set_title(title)

    _rid_style_ax(ax)
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def rid_plot_all_pairs(
    rid_out,
    feature_names=None,
    max_features=None,
    order_by_mean=True,
    figsize_per_panel=(3.2, 3.0),
    alpha=0.25,
    s=8,
    show=True,
):
    samples = _rid_joint_samples(rid_out)

    n_features = samples.shape[1] - 1

    feature_names = _rid_feature_names(
        feature_names,
        n_features,
    )

    if max_features is None:
        features = list(range(n_features))
    else:
        max_features = int(max_features)

        if max_features < 2:
            raise ValueError(
                "max_features must be at least 2."
            )

        if (
            order_by_mean
            and "mean_sub_mr" in rid_out
        ):
            mean = np.asarray(
                rid_out["mean_sub_mr"],
                dtype=float,
            )

            features = list(
                np.argsort(-mean)[:max_features]
            )
        else:
            features = list(
                range(
                    min(
                        n_features,
                        max_features,
                    )
                )
            )

    n_selected = len(features)

    if n_selected < 2:
        raise ValueError(
            "Need at least two features to plot pairs."
        )

    n_pairs = n_selected * (n_selected - 1) // 2

    ncols = int(np.ceil(np.sqrt(n_pairs)))
    nrows = int(np.ceil(n_pairs / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(
            figsize_per_panel[0] * ncols,
            figsize_per_panel[1] * nrows,
        ),
        squeeze=False,
    )

    pair_index = 0

    for a in range(n_selected):
        for b in range(a + 1, n_selected):
            feature_a = features[a]
            feature_b = features[b]

            row = pair_index // ncols
            col = pair_index % ncols

            ax = axes[row][col]

            ax.scatter(
                samples[:, feature_a],
                samples[:, feature_b],
                s=s,
                alpha=alpha,
            )

            ax.set_xlabel(
                feature_names[feature_a]
            )

            ax.set_ylabel(
                feature_names[feature_b]
            )

            ax.set_title(
                f"{feature_names[feature_a]} vs "
                f"{feature_names[feature_b]}",
                fontsize=10,
            )

            _rid_style_ax(ax)

            pair_index += 1

    for index in range(
        pair_index,
        nrows * ncols,
    ):
        row = index // ncols
        col = index % ncols

        axes[row][col].set_axis_off()

    fig.suptitle(
        "Pairwise RID feature-importance distributions",
        y=1.02,
    )

    fig.tight_layout()

    if show:
        plt.show()

    return fig, axes