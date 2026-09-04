// implementation for the Rashomon importance distribution - not one of our
// contributions but still integrated with the method.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <vector>
#include <chrono>
#include <iomanip>
#include <limits>
#include <utility>
#include <atomic>
#include <thread>

using std::cout;

struct PeakMemorySampler {
    std::atomic<bool> stop_requested{false};
    std::thread worker;
    double peak_mb = 0.0;
    bool running = false;

    void start() {
        stop_requested.store(false, std::memory_order_relaxed);
        peak_mb = ArborEnum::current_memory_mb();
        running = true;

        worker = std::thread([this]() {
            while (!stop_requested.load(std::memory_order_relaxed)) {
                peak_mb = std::max(
                    peak_mb,
                    ArborEnum::current_memory_mb()
                );

                std::this_thread::sleep_for(
                    std::chrono::milliseconds(20)
                );
            }

            peak_mb = std::max(
                peak_mb,
                ArborEnum::current_memory_mb()
            );
        });
    }

    double finish() {
        if (running) {
            stop_requested.store(true, std::memory_order_relaxed);

            if (worker.joinable()) {
                worker.join();
            }

            running = false;
        }

        return peak_mb;
    }

    ~PeakMemorySampler() {
        finish();
    }
};

struct RIDResult {
    std::vector<double> mean_sub_mr;
    std::vector<std::vector<double>> cdf_x;
    std::vector<std::vector<double>> cdf_p;

    // interval-only output [bootstrap][variable]
    // importance_interval_mode=1: [min_f Phi_j(f), max_f Phi_j(f)].
    // importance_interval_mode=2: [(1/n)sum_i min_f phi_ij(f),
    //                              (1/n)sum_i max_f phi_ij(f)].
    std::vector<std::vector<std::pair<double, double>>>
        bootstrap_importance_intervals;

    // optional output. when return_joint_samples=true, there is one row per
    // tree per bootstrap. columns 0..V-1 are the feature importances and the
    // final column is the tree's probability weight.
    std::vector<std::vector<double>> feature_importance_weight_samples;
};

static inline void bootstrap_indices(
    int n,
    int subsample,
    std::mt19937_64& rng,
    std::vector<int>& idx
) {
    // standard bootstrap: n draws with replacement.
    if (subsample == -1) {
        std::uniform_int_distribution<int> unif(0, n - 1);
        idx.resize((std::size_t)n);

        for (int i = 0; i < n; ++i) {
            idx[(std::size_t)i] = unif(rng);
        }

        return;
    }

    // subsample without replacement.
    idx.resize((std::size_t)n);

    for (int i = 0; i < n; ++i) {
        idx[(std::size_t)i] = i;
    }

    // only randomize enough positions to choose the first subsample samples.
    for (int i = 0; i < subsample; ++i) {
        std::uniform_int_distribution<int> unif(i, n - 1);
        const int j = unif(rng);
        std::swap(idx[(std::size_t)i], idx[(std::size_t)j]);
    }

    idx.resize((std::size_t)subsample);
}

static inline void make_bootstrap_dataset(
    const std::vector<std::vector<uint8_t>>& X,
    const std::vector<int>& y,
    const std::vector<int>& idx,
    std::vector<std::vector<uint8_t>>& Xb,
    std::vector<int>& yb
) {
    const int n = (int)idx.size();
    const int d = (int)X[0].size();

    Xb.assign((std::size_t)n, std::vector<uint8_t>((std::size_t)d));
    yb.assign((std::size_t)n, 0);

    for (int i = 0; i < n; ++i) {
        const int source = idx[(std::size_t)i];
        Xb[(std::size_t)i] = X[(std::size_t)source];
        yb[(std::size_t)i] = y[(std::size_t)source];
    }
}

static inline void make_bootstrap_vector_int(
    const std::vector<int>& values,
    const std::vector<int>& idx,
    std::vector<int>& bootstrapped_values
) {
    const int n = (int)idx.size();
    bootstrapped_values.assign((std::size_t)n, 0);

    for (int i = 0; i < n; ++i) {
        bootstrapped_values[(std::size_t)i] =
            values[(std::size_t)idx[(std::size_t)i]];
    }
}

static inline void rowmajor_to_colmajor_bool(
    const std::vector<std::vector<uint8_t>>& X_row,
    std::vector<std::vector<bool>>& X_col
) {
    const int n = (int)X_row.size();
    const int d = (int)X_row[0].size();

    X_col.assign(
        (std::size_t)d,
        std::vector<bool>((std::size_t)n, false)
    );

    for (int i = 0; i < n; ++i) {
        const auto& row = X_row[(std::size_t)i];

        for (int j = 0; j < d; ++j) {
            X_col[(std::size_t)j][(std::size_t)i] =
                row[(std::size_t)j] != 0;
        }
    }
}

static inline void make_permutation(
    int n,
    std::mt19937_64& rng,
    std::vector<int>& permutation
) {
    permutation.resize((std::size_t)n);

    for (int i = 0; i < n; ++i) {
        permutation[(std::size_t)i] = i;
    }

    std::shuffle(permutation.begin(), permutation.end(), rng);
}

static inline void make_groupwise_permutation(
    int n,
    const std::vector<std::vector<int>>& groups,
    std::mt19937_64& rng,
    std::vector<int>& permutation,
    std::vector<int>& scratch
) {
    permutation.resize((std::size_t)n);

    for (int i = 0; i < n; ++i) {
        permutation[(std::size_t)i] = i;
    }

    for (const auto& group : groups) {
        scratch.assign(group.begin(), group.end());
        std::shuffle(scratch.begin(), scratch.end(), rng);

        for (std::size_t t = 0; t < group.size(); ++t) {
            const int target_row = group[t];
            const int replacement_row = scratch[t];

            permutation[(std::size_t)target_row] =
                replacement_row;
        }
    }
}

// scramble one original variable represented by one or more binary columns.
// every column in the block receives the same row permutation, preserving the
// internal consistency of a threshold-binarized continuous variable.
static inline void scramble_block_inplace(
    std::vector<std::vector<uint8_t>>& X,
    const std::vector<int>& columns,
    const std::vector<int>& permutation,
    std::vector<std::vector<uint8_t>>& saved_columns
) {
    const int n = (int)X.size();

    saved_columns.assign(
        columns.size(),
        std::vector<uint8_t>((std::size_t)n)
    );

    for (std::size_t column_position = 0;
         column_position < columns.size();
         ++column_position) {
        const int column = columns[column_position];

        for (int i = 0; i < n; ++i) {
            saved_columns[column_position][(std::size_t)i] =
                X[(std::size_t)i][(std::size_t)column];
        }
    }

    for (std::size_t column_position = 0;
         column_position < columns.size();
         ++column_position) {
        const int column = columns[column_position];

        for (int i = 0; i < n; ++i) {
            X[(std::size_t)i][(std::size_t)column] =
                saved_columns[column_position]
                    [(std::size_t)permutation[(std::size_t)i]];
        }
    }
}

static inline void restore_block_inplace(
    std::vector<std::vector<uint8_t>>& X,
    const std::vector<int>& columns,
    const std::vector<std::vector<uint8_t>>& saved_columns
) {
    const int n = (int)X.size();

    for (std::size_t column_position = 0;
         column_position < columns.size();
         ++column_position) {
        const int column = columns[column_position];

        for (int i = 0; i < n; ++i) {
            X[(std::size_t)i][(std::size_t)column] =
                saved_columns[column_position][(std::size_t)i];
        }
    }
}

// evaluation-time objective used by subtractive model reliance.
// the training leaf penalty is unchanged by evaluation-time permutation and
// therefore cancels in scrambled minus original loss. With deferral enabled,
// the remaining evaluation objective is:
// misclassifications + round(eta_defer * number_of_deferrals).
static inline int rid_eval_objective_from_mis_def(
    int misclassifications,
    int deferrals,
    bool use_deferral,
    double eta_defer
) {
    if (!use_deferral) {
        return misclassifications;
    }

    return misclassifications
        + (int)std::llround(eta_defer * (double)deferrals);
}

RIDResult compute_rid_subtractive_mr_bootstrap(
    const std::vector<std::vector<uint8_t>>& X_row_major,
    const std::vector<int>& y,
    int n_bootstraps,
    int n_scramble_evals,
    double lambda,
    int depth_budget,
    double rashomon_mult,
    int lookahead_k,
    uint64_t seed,
    bool memory_efficient,
    ArborEnum::KeyMode key_mode = ArborEnum::KeyMode::HASH64,
    bool trie_cache_enabled = true,
    const std::vector<std::vector<int>>& binning_map_vars = {},
    const std::vector<int>& continuous_starts = {},
    bool use_anytime_fit = false,
    double second_rashomon_mult = -1.0,
    double multiplier_step_size = 0.01,
    const std::vector<int>& proxy_threshold_features = {},
    const std::vector<int>& initial_active_threshold_features = {},
    int refinement_width = 1,
    int max_refinement_rounds = -1,
    bool use_multipass = true,
    bool rule_list_mode = false,
    int proxy_style = 0,
    bool majority_leaf_only = false,
    bool cache_cheap_subproblems = false,
    int greedy_split_mode = 1,
    int greedy_continuous_mode = 0,
    bool proxy_caching = true,
    int proxy_refinement_mode = 0,
    bool continuous_proxy_in_lickety = true,
    bool continuous_proxy_in_depthd_exact = true,
    bool continuous_proxy_in_greedy = true,
    double runtime_limit_seconds = -1.0,
    double memory_limit_mb = -1.0,
    bool use_deferral = false,
    double eta_defer = 0.0,
    const std::vector<int>& bb_pred = {},
    bool return_joint_samples = false,
    bool lossless = false,
    const std::vector<std::vector<std::vector<int>>>&
        matched_groups_by_variable = {},
    bool additive = false,
    // 0 = full RID distribution (existing behavior)
    // 1 = interval only: min/max global importance over models
    // 2 = interval only: sum of per-sample min/max local importances
    int importance_interval_mode = 0,
    int subsample = -1,
    int root_budget = -1
) {
    (void)memory_efficient;

    const int n_full = (int)X_row_major.size();

    if (n_full == 0) {
        throw std::runtime_error(
            "compute_rid_subtractive_mr_bootstrap: X is empty."
        );
    }

    if (subsample != -1 &&
        (subsample <= 0 || subsample > n_full)) {
        throw std::runtime_error(
            "compute_rid_subtractive_mr_bootstrap: "
            "subsample must be -1 or an integer in [1, n]."
        );
    }

    if (n_bootstraps <= 0) {
        throw std::runtime_error(
            "compute_rid_subtractive_mr_bootstrap: "
            "n_bootstraps must be positive."
        );
    }

    if (root_budget < -1) {
        throw std::runtime_error(
            "compute_rid_subtractive_mr_bootstrap: "
            "root_budget must be -1 or a nonnegative integer."
        );
    }

    if (root_budget >= 0 && use_anytime_fit) {
        throw std::runtime_error(
            "compute_rid_subtractive_mr_bootstrap: "
            "root_budget is not currently supported with "
            "use_anytime_fit=true."
        );
    }

    if (!lossless && n_scramble_evals <= 0) {
        throw std::runtime_error(
            "compute_rid_subtractive_mr_bootstrap: "
            "n_scramble_evals must be positive in Monte Carlo mode."
        );
    }

    if (importance_interval_mode < 0 || importance_interval_mode > 2) {
        throw std::runtime_error(
            "compute_rid_subtractive_mr_bootstrap: "
            "importance_interval_mode must be 0, 1, or 2."
        );
    }

    if (importance_interval_mode != 0 && !lossless) {
        throw std::runtime_error(
            "compute_rid_subtractive_mr_bootstrap: "
            "importance interval modes are supported only with lossless=true."
        );
    }

    if (importance_interval_mode != 0 && return_joint_samples) {
        throw std::runtime_error(
            "compute_rid_subtractive_mr_bootstrap: "
            "return_joint_samples is incompatible with interval-only mode."
        );
    }

    if (!std::isfinite(lambda) || lambda < 0.0) {
        throw std::runtime_error(
            "compute_rid_subtractive_mr_bootstrap: "
            "lambda must be finite and nonnegative."
        );
    }

    if (!std::isfinite(rashomon_mult) || rashomon_mult < 0.0) {
        throw std::runtime_error(
            "compute_rid_subtractive_mr_bootstrap: "
            "rashomon_mult must be finite and nonnegative."
        );
    }

    const int d = (int)X_row_major[0].size();

    if (d == 0) {
        throw std::runtime_error(
            "compute_rid_subtractive_mr_bootstrap: X has zero columns."
        );
    }

    for (int i = 0; i < n_full; ++i) {
        if ((int)X_row_major[(std::size_t)i].size() != d) {
            throw std::runtime_error(
                "compute_rid_subtractive_mr_bootstrap: "
                "X must be rectangular."
            );
        }
    }

    if ((int)y.size() != n_full) {
        throw std::runtime_error(
            "compute_rid_subtractive_mr_bootstrap: "
            "y has a different number of rows than X."
        );
    }

    if (use_deferral) {
        if ((int)bb_pred.size() != n_full) {
            throw std::runtime_error(
                "compute_rid_subtractive_mr_bootstrap: "
                "use_deferral=true requires bb_pred with the same "
                "number of rows as X and y."
            );
        }

        if (!std::isfinite(eta_defer) || eta_defer < 0.0) {
            throw std::runtime_error(
                "compute_rid_subtractive_mr_bootstrap: "
                "eta_defer must be finite and nonnegative."
            );
        }
    }

    const double resolved_second_rashomon_mult =
        second_rashomon_mult < 0.0
            ? rashomon_mult
            : second_rashomon_mult;

    if (!std::isfinite(resolved_second_rashomon_mult) ||
        resolved_second_rashomon_mult < rashomon_mult) {
        throw std::runtime_error(
            "compute_rid_subtractive_mr_bootstrap: "
            "second_rashomon_mult must be at least rashomon_mult."
        );
    }

    // build original-variable -> internal binary-column mapping. When no map is
    // provided, every binary column is treated as a separate variable.
    std::vector<std::vector<int>> variable_columns;

    if (!binning_map_vars.empty()) {
        variable_columns = binning_map_vars;
    } else {
        variable_columns.resize((std::size_t)d);

        for (int j = 0; j < d; ++j) {
            variable_columns[(std::size_t)j] = {j};
        }
    }

    for (const auto& columns : variable_columns) {
        if (columns.empty()) {
            throw std::runtime_error(
                "compute_rid_subtractive_mr_bootstrap: "
                "binning_map_vars contains an empty variable block."
            );
        }

        for (int column : columns) {
            if (column < 0 || column >= d) {
                throw std::runtime_error(
                    "compute_rid_subtractive_mr_bootstrap: "
                    "binning_map_vars contains an out-of-range column."
                );
            }
        }
    }

    const int number_of_variables =
        (int)variable_columns.size();

    const bool use_matched_groups =
        !matched_groups_by_variable.empty();

    std::vector<std::vector<int>>
        original_group_id_by_variable;

    if (use_matched_groups) {
        if (
            (int)matched_groups_by_variable.size() !=
            number_of_variables
        ) {
            throw std::runtime_error(
                "matched_groups_by_variable must contain exactly "
                "one matched-group partition per RID variable."
            );
        }

        original_group_id_by_variable.assign(
            (std::size_t)number_of_variables,
            std::vector<int>(
                (std::size_t)n_full,
                -1
            )
        );

        for (int variable = 0;
            variable < number_of_variables;
            ++variable) {

            const auto& groups =
                matched_groups_by_variable[
                    (std::size_t)variable
                ];

            if (groups.empty()) {
                throw std::runtime_error(
                    "Each RID variable must have a nonempty "
                    "matched-group partition."
                );
            }

            auto& group_ids =
                original_group_id_by_variable[
                    (std::size_t)variable
                ];

            for (std::size_t group_index = 0;
                group_index < groups.size();
                ++group_index) {

                const auto& group = groups[group_index];

                if (group.empty()) {
                    throw std::runtime_error(
                        "matched_groups_by_variable contains "
                        "an empty group."
                    );
                }

                for (int row : group) {
                    if (row < 0 || row >= n_full) {
                        throw std::runtime_error(
                            "matched_groups_by_variable contains "
                            "an out-of-range row index."
                        );
                    }

                    if (group_ids[(std::size_t)row] != -1) {
                        throw std::runtime_error(
                            "A variable's matched groups are not "
                            "disjoint."
                        );
                    }

                    group_ids[(std::size_t)row] =
                        static_cast<int>(group_index);
                }
            }

            for (int row = 0; row < n_full; ++row) {
                if (group_ids[(std::size_t)row] == -1) {
                    throw std::runtime_error(
                        "Each variable's matched groups must "
                        "partition every row."
                    );
                }
            }
        }
    }

        
    std::mt19937_64 bootstrap_rng(seed);
    std::mt19937_64 scramble_rng(seed ^ 0x9E3779B97F4A7C15ULL);

    RIDResult output;

    if (importance_interval_mode == 0) {
        output.mean_sub_mr.assign(
            (std::size_t)number_of_variables,
            0.0
        );
        output.cdf_x.assign(
            (std::size_t)number_of_variables,
            {}
        );
        output.cdf_p.assign(
            (std::size_t)number_of_variables,
            {}
        );
    } else {
        const double nan =
            std::numeric_limits<double>::quiet_NaN();

        output.bootstrap_importance_intervals.assign(
            (std::size_t)n_bootstraps,
            std::vector<std::pair<double, double>>(
                (std::size_t)number_of_variables,
                {nan, nan}
            )
        );
    }

    // each map stores probability mass directly on the normalized,
    // scramble-averaged importance values.
    std::vector<std::map<double, double>> mass_by_importance(
        (std::size_t)number_of_variables
    );

    int successful_bootstraps = 0;

    for (int bootstrap = 0; bootstrap < n_bootstraps; ++bootstrap) {

        const auto bootstrap_start = std::chrono::steady_clock::now();

        std::vector<int> bootstrap_idx;
        bootstrap_indices(
            n_full,
            subsample,
            bootstrap_rng,
            bootstrap_idx
        );

        std::vector<std::vector<uint8_t>> X_bootstrap;
        std::vector<int> y_bootstrap;

        make_bootstrap_dataset(
            X_row_major,
            y,
            bootstrap_idx,
            X_bootstrap,
            y_bootstrap
        );

        std::vector<int> bb_pred_bootstrap;
        if (use_deferral) {
            make_bootstrap_vector_int(
                bb_pred,
                bootstrap_idx,
                bb_pred_bootstrap
            );
        }

        const int n = (int)X_bootstrap.size();

        // matched_group_of_bootstrap_row_by_variable[j][i] = matched-group ID of bootstrap row i for variable j
        // matched_group_size_bootstrap_by_variable[j][g] = number of bootstrap rows in group g for variable j

        std::vector<std::vector<int>>
            matched_group_of_bootstrap_row_by_variable;

        std::vector<std::vector<int>>
            matched_group_size_bootstrap_by_variable;

        std::vector<std::vector<std::vector<int>>>
            matched_groups_bootstrap_by_variable;

        if (use_matched_groups) {
            matched_group_of_bootstrap_row_by_variable.resize(
                (std::size_t)number_of_variables
            );

            matched_group_size_bootstrap_by_variable.resize(
                (std::size_t)number_of_variables
            );

            // explicit row lists are only needed by the monte carlo error path.
            if (!lossless) {
                matched_groups_bootstrap_by_variable.resize(
                    (std::size_t)number_of_variables
                );
            }

            for (int variable = 0;
                variable < number_of_variables;
                ++variable) {

                const int number_of_groups =
                    static_cast<int>(
                        matched_groups_by_variable[
                            (std::size_t)variable
                        ].size()
                    );

                const auto& original_group_ids =
                    original_group_id_by_variable[
                        (std::size_t)variable
                    ];

                auto& bootstrap_group_of_row =
                    matched_group_of_bootstrap_row_by_variable[
                        (std::size_t)variable
                    ];

                auto& bootstrap_group_sizes =
                    matched_group_size_bootstrap_by_variable[
                        (std::size_t)variable
                    ];

                bootstrap_group_of_row.assign(
                    (std::size_t)n,
                    -1
                );

                bootstrap_group_sizes.assign(
                    (std::size_t)number_of_groups,
                    0
                );

                if (!lossless) {
                    matched_groups_bootstrap_by_variable[
                        (std::size_t)variable
                    ].assign(
                        (std::size_t)number_of_groups,
                        {}
                    );
                }

                for (int bootstrap_row = 0;
                    bootstrap_row < n;
                    ++bootstrap_row) {

                    const int original_row =
                        bootstrap_idx[
                            (std::size_t)bootstrap_row
                        ];

                    const int group =
                        original_group_ids[
                            (std::size_t)original_row
                        ];

                    bootstrap_group_of_row[
                        (std::size_t)bootstrap_row
                    ] = group;

                    ++bootstrap_group_sizes[
                        (std::size_t)group
                    ];

                    if (!lossless) {
                        matched_groups_bootstrap_by_variable[
                            (std::size_t)variable
                        ][
                            (std::size_t)group
                        ].push_back(bootstrap_row);
                    }
                }
            }
        }

                
        
        std::vector<std::vector<bool>> X_col_major;
        rowmajor_to_colmajor_bool(X_bootstrap, X_col_major);

        ArborEnum model;
        model.set_key_mode(key_mode);
        model.set_trie_cache_enabled(trie_cache_enabled);
        model.set_greedy_split_mode(greedy_split_mode);
        model.set_greedy_continuous_mode(greedy_continuous_mode);
        model.set_additive(additive);

        const bool any_proxy_is_restricted =
            !continuous_proxy_in_lickety ||
            !continuous_proxy_in_depthd_exact ||
            !continuous_proxy_in_greedy;

        if (!continuous_starts.empty() &&
            any_proxy_is_restricted &&
            proxy_threshold_features.empty()) {
            throw std::runtime_error(
                "Continuous RID requires nonempty proxy_threshold_features "
                "when any proxy component is restricted."
            );
        }

        PeakMemorySampler training_memory;
        training_memory.start();

        const auto training_start = std::chrono::steady_clock::now();

        if (use_anytime_fit) {
            model.fit_anytime(
                X_col_major,
                y_bootstrap,
                lambda,
                static_cast<int8_t>(depth_budget),
                rashomon_mult,
                resolved_second_rashomon_mult,
                multiplier_step_size,
                static_cast<int8_t>(lookahead_k),
                use_multipass,
                rule_list_mode,
                proxy_style,
                majority_leaf_only,
                cache_cheap_subproblems,
                proxy_caching,
                proxy_threshold_features,
                initial_active_threshold_features,
                refinement_width,
                max_refinement_rounds,
                proxy_refinement_mode,
                continuous_proxy_in_lickety,
                continuous_proxy_in_depthd_exact,
                continuous_proxy_in_greedy,
                continuous_starts,
                runtime_limit_seconds,
                memory_limit_mb,
                use_deferral,
                eta_defer,
                bb_pred_bootstrap
            );
        } else {
            model.fit(
                X_col_major,
                y_bootstrap,
                lambda,
                static_cast<int8_t>(depth_budget),
                rashomon_mult,
                static_cast<int8_t>(lookahead_k),
                root_budget,
                use_multipass,
                rule_list_mode,
                proxy_style,
                majority_leaf_only,
                cache_cheap_subproblems,
                proxy_caching,
                proxy_threshold_features,
                !continuous_proxy_in_lickety,
                !continuous_proxy_in_depthd_exact,
                !continuous_proxy_in_greedy,
                true,
                continuous_starts,
                false,
                use_deferral,
                eta_defer,
                bb_pred_bootstrap
            );
        }

        const auto training_end = std::chrono::steady_clock::now();

        const double training_peak_mb =
            training_memory.finish();

        if (!model.result) {
            cout << "RID bootstrap " << (bootstrap + 1)
                 << ": skipped because training returned no result\n";
            continue;
        }

        if (
            root_budget >= 0 &&
            model.result->min_objective > root_budget
        ) {
            cout << "RID bootstrap " << (bootstrap + 1)
                 << ": skipped because minimum objective "
                 << model.result->min_objective
                 << " exceeds root budget "
                 << root_budget
                 << "\n";
            continue;
        }


        const double training_seconds =
            std::chrono::duration<double>(
                training_end - training_start
            ).count();

        cout << std::fixed << std::setprecision(3)
            << "RID bootstrap " << (bootstrap + 1)
            << ": Rashomon training = "
            << training_seconds << " s\n"
            << "RID bootstrap " << (bootstrap + 1)
            << ": Rashomon training peak memory = "
            << training_peak_mb << " MB\n";

                int budget_override;

        if (root_budget >= 0) {
            // absolute integer user-specified budget.
            // we are not rescaling for bootstrap/subsample size.
            budget_override = root_budget;
        } else {
            // standard RID behavior
            const int requested_budget =
                additive
                    ? static_cast<int>(std::llround(
                        static_cast<double>(
                            model.result->min_objective
                        )
                        + rashomon_mult *
                            static_cast<double>(n)
                    ))
                    : static_cast<int>(std::llround(
                        (1.0 + rashomon_mult) *
                        static_cast<double>(
                            model.result->min_objective
                        )
                    ));

            budget_override = std::min(
                model.result->budget,
                requested_budget
            );
        }

        const uint64_t trees_at_budget =
            model.result->count_leq(budget_override);

        if (trees_at_budget == 0) {
            cout << "RID bootstrap " << (bootstrap + 1)
                 << ": skipped because no trees are within budget "
                 << budget_override
                 << "\n";
            continue;
        }

        ++successful_bootstraps;
     

        const double pre_importance_memory_mb =
            ArborEnum::current_memory_mb();

        cout << "RID bootstrap " << (bootstrap + 1)
            << ": memory before feature importance = "
            << pre_importance_memory_mb << " MB\n";

        PeakMemorySampler importance_memory;
        importance_memory.start();

        const auto post_training_start =
            std::chrono::steady_clock::now();

        

        // lossless permutation importance: compute the exact expected replacement mistakes for
        // every variable and every tree in one packed graph traversal.
        if (lossless) {
            if (use_deferral) {
                throw std::runtime_error(
                    "lossless RID currently supports use_deferral=false only."
                );
            }

            if (importance_interval_mode != 0) {
                const auto exact_start =
                    std::chrono::steady_clock::now();

                const uint64_t number_of_trees = trees_at_budget;

                if (number_of_trees == 0) {
                    continue;
                }

                // auto intervals =
                //     model.get_exact_replacement_importance_intervals_packed_trie(
                //         X_bootstrap,
                //         y_bootstrap,
                //         budget_override,
                //         variable_columns,
                //         {},
                //         matched_group_of_bootstrap_row_by_variable,
                //         matched_group_size_bootstrap_by_variable,
                //         importance_interval_mode == 2
                //     );

                const bool use_new_cached_frontier_method = false;

                std::vector<ArborEnum::ExactImportanceInterval> intervals;

                if (
                    use_new_cached_frontier_method &&
                    importance_interval_mode == 1
                ) {
                    intervals =
                        model.get_exact_replacement_importance_intervals_cached_frontier_packed_trie(
                            X_bootstrap,
                            y_bootstrap,
                            budget_override,
                            variable_columns,
                            {},
                            matched_group_of_bootstrap_row_by_variable,
                            matched_group_size_bootstrap_by_variable
                        );
                } else {
                    intervals =
                        model.get_exact_replacement_importance_intervals_packed_trie(
                            X_bootstrap,
                            y_bootstrap,
                            budget_override,
                            variable_columns,
                            {},
                            matched_group_of_bootstrap_row_by_variable,
                            matched_group_size_bootstrap_by_variable,
                            importance_interval_mode == 2
                        );
                }

                if (
                    intervals.size() !=
                    (std::size_t)number_of_variables
                ) {
                    throw std::runtime_error(
                        "Lossless RID interval extraction returned the wrong "
                        "number of variables."
                    );
                }

                output.bootstrap_importance_intervals[
                    (std::size_t)bootstrap
                ] = std::move(intervals);

                cout << "Finished RID bootstrap: "
                     << (bootstrap + 1)
                     << " / "
                     << n_bootstraps
                     << " with "
                     << number_of_trees
                     << " trees represented (not enumerated)\n";

                const auto exact_end =
                    std::chrono::steady_clock::now();

                const double exact_seconds =
                    std::chrono::duration<double>(
                        exact_end - exact_start
                    ).count();

                const double importance_peak_mb =
                    importance_memory.finish();

                const double importance_extra_peak_mb =
                    std::max(
                        0.0,
                        importance_peak_mb - pre_importance_memory_mb
                    );

                const auto bootstrap_end =
                    std::chrono::steady_clock::now();

                const double post_training_seconds =
                    std::chrono::duration<double>(
                        bootstrap_end - post_training_start
                    ).count();

                const double bootstrap_seconds =
                    std::chrono::duration<double>(
                        bootstrap_end - bootstrap_start
                    ).count();

                cout << std::fixed << std::setprecision(3)
                    << "RID bootstrap " << (bootstrap + 1)
                    << ": Fast Graph interval evaluation = "
                    << exact_seconds << " s\n"

                    << "RID bootstrap " << (bootstrap + 1)
                    << ": memory before feature importance = "
                    << pre_importance_memory_mb << " MB\n"

                    << "RID bootstrap " << (bootstrap + 1)
                    << ": feature importance peak memory = "
                    << importance_peak_mb << " MB\n"

                    << "RID bootstrap " << (bootstrap + 1)
                    << ": feature importance extra peak memory = "
                    << importance_extra_peak_mb << " MB\n"

                    << "RID bootstrap " << (bootstrap + 1)
                    << ": post-training RID = "
                    << post_training_seconds << " s\n"

                    << "RID bootstrap " << (bootstrap + 1)
                    << ": total = "
                    << bootstrap_seconds << " s\n"

                    << "RID bootstrap " << (bootstrap + 1)
                    << ": training fraction = "
                    << (100.0 * training_seconds / bootstrap_seconds)
                    << "%, post-training fraction = "
                    << (100.0 * post_training_seconds / bootstrap_seconds)
                    << "%\n";

                continue;
            }

            const auto exact_start =
                std::chrono::steady_clock::now();

            auto exact =
                model.get_all_exact_replacement_misclassifications_packed_trie(
                    X_bootstrap,
                    y_bootstrap,
                    budget_override,
                    variable_columns,
                    {},
                    matched_group_of_bootstrap_row_by_variable,
                    matched_group_size_bootstrap_by_variable
                );

            const uint64_t number_of_trees =
                static_cast<uint64_t>(exact.size());

            if (number_of_trees == 0) {
                continue;
            }

            cout << "Finished RID bootstrap: "
                 << (bootstrap + 1)
                 << " / "
                 << n_bootstraps
                 << " with "
                 << number_of_trees
                 << " trees\n";

            const double tree_weight =
                1.0 /
                ((double)n_bootstraps * (double)number_of_trees);

            std::size_t joint_sample_offset = 0;

            if (return_joint_samples) {
                joint_sample_offset =
                    output.feature_importance_weight_samples.size();

                output.feature_importance_weight_samples.resize(
                    joint_sample_offset + (std::size_t)number_of_trees,
                    std::vector<double>(
                        (std::size_t)number_of_variables + 1,
                        0.0
                    )
                );

                for (uint64_t tree = 0;
                     tree < number_of_trees;
                     ++tree) {
                    output.feature_importance_weight_samples[
                        joint_sample_offset + (std::size_t)tree
                    ][(std::size_t)number_of_variables] = tree_weight;
                }
            }

            for (uint64_t tree = 0;
                 tree < number_of_trees;
                 ++tree) {

                const auto& row = exact[(std::size_t)tree];

                if (
                    row.mistakes.size() !=
                    (std::size_t)number_of_variables + 1
                ) {
                    throw std::runtime_error(
                        "Lossless RID returned the wrong number of "
                        "mistake columns."
                    );
                }

                const double original_mistakes =
                    row.mistakes[0];

                for (int variable = 0;
                     variable < number_of_variables;
                     ++variable) {

                    const double importance =
                        (
                            row.mistakes[
                                (std::size_t)variable + 1
                            ]
                            - original_mistakes
                        )
                        / (double)n;

                    output.mean_sub_mr[
                        (std::size_t)variable
                    ] +=
                        tree_weight * importance;

                    mass_by_importance[
                        (std::size_t)variable
                    ][importance] +=
                        tree_weight;

                    if (return_joint_samples) {
                        output.feature_importance_weight_samples[
                            joint_sample_offset +
                            (std::size_t)tree
                        ][
                            (std::size_t)variable
                        ] = importance;
                    }
                }
            }

            const auto exact_end =
                std::chrono::steady_clock::now();

            const double exact_seconds =
                std::chrono::duration<double>(
                    exact_end - exact_start
                ).count();

            // peak reached at any point while computing feature importance
            const double importance_peak_mb =
                importance_memory.finish();

            const double importance_extra_peak_mb =
                std::max(
                    0.0,
                    importance_peak_mb - pre_importance_memory_mb
                );

            const auto bootstrap_end =
                std::chrono::steady_clock::now();

            const double post_training_seconds =
                std::chrono::duration<double>(
                    bootstrap_end - post_training_start
                ).count();

            const double bootstrap_seconds =
                std::chrono::duration<double>(
                    bootstrap_end - bootstrap_start
                ).count();

            cout << std::fixed << std::setprecision(3)
                << "RID bootstrap " << (bootstrap + 1)
                << ": Fast Graph feature evaluation = "
                << exact_seconds << " s\n"

                << "RID bootstrap " << (bootstrap + 1)
                << ": memory before feature importance = "
                << pre_importance_memory_mb << " MB\n"

                << "RID bootstrap " << (bootstrap + 1)
                << ": feature importance peak memory = "
                << importance_peak_mb << " MB\n"

                << "RID bootstrap " << (bootstrap + 1)
                << ": feature importance extra peak memory = "
                << importance_extra_peak_mb << " MB\n"

                << "RID bootstrap " << (bootstrap + 1)
                << ": post-training RID = "
                << post_training_seconds << " s\n"

                << "RID bootstrap " << (bootstrap + 1)
                << ": total = "
                << bootstrap_seconds << " s\n"

                << "RID bootstrap " << (bootstrap + 1)
                << ": training fraction = "
                << (100.0 * training_seconds / bootstrap_seconds)
                << "%, post-training fraction = "
                << (100.0 * post_training_seconds / bootstrap_seconds)
                << "%\n";

            continue;
        }

        // legacy Monte Carlo RID path.
        std::vector<int> original_misclassifications;

        if (use_deferral) {
            original_misclassifications =
                model.get_all_misclassifications_packed_trie(
                    X_bootstrap,
                    y_bootstrap,
                    budget_override,
                    bb_pred_bootstrap
                );
        } else {
            original_misclassifications =
                model.get_all_misclassifications_packed_trie(
                    X_bootstrap,
                    y_bootstrap,
                    budget_override
                );
        }

        std::vector<int> original_deferrals;

        if (use_deferral) {
            original_deferrals =
                model.get_all_deferrals_packed_trie(
                    X_bootstrap,
                    budget_override
                );

            if (original_deferrals.size() !=
                original_misclassifications.size()) {
                throw std::runtime_error(
                    "RID deferral extraction returned different numbers of "
                    "misclassification and deferral entries."
                );
            }
        }

        const uint64_t number_of_trees =
            (uint64_t)original_misclassifications.size();

        if (number_of_trees == 0) {
            continue;
        }

        cout << "Finished RID bootstrap: "
             << (bootstrap + 1)
             << " / "
             << n_bootstraps
             << " with "
             << number_of_trees
             << " trees\n";

        std::vector<int> original_eval_objectives(
            (std::size_t)number_of_trees,
            0
        );

        for (uint64_t tree = 0; tree < number_of_trees; ++tree) {
            const int deferrals = use_deferral
                ? original_deferrals[(std::size_t)tree]
                : 0;

            original_eval_objectives[(std::size_t)tree] =
                rid_eval_objective_from_mis_def(
                    original_misclassifications[(std::size_t)tree],
                    deferrals,
                    use_deferral,
                    eta_defer
                );
        }

        const double tree_weight =
            1.0 /
            ((double)n_bootstraps * (double)number_of_trees);

        std::size_t joint_sample_offset = 0;

        if (return_joint_samples) {
            joint_sample_offset =
                output.feature_importance_weight_samples.size();

            output.feature_importance_weight_samples.resize(
                joint_sample_offset + (std::size_t)number_of_trees,
                std::vector<double>(
                    (std::size_t)number_of_variables + 1,
                    0.0
                )
            );

            for (uint64_t tree = 0; tree < number_of_trees; ++tree) {
                output.feature_importance_weight_samples[
                    joint_sample_offset + (std::size_t)tree
                ][(std::size_t)number_of_variables] = tree_weight;
            }
        }


            // reuse these buffers across every variable and scramble.
            std::vector<std::vector<uint8_t>> saved_columns;
            std::vector<int> permutation;
            std::vector<int> group_permutation_scratch;

            for (int variable = 0;
                variable < number_of_variables;
                ++variable) {
                const std::vector<int>& columns =
                    variable_columns[(std::size_t)variable];

                // O(number_of_trees) memory, independent of n_scramble_evals.
                std::vector<int64_t> summed_objective_differences(
                    (std::size_t)number_of_trees,
                    0
                );

                for (int scramble = 0;
                    scramble < n_scramble_evals;
                    ++scramble) {
                    if (use_matched_groups) {
                        make_groupwise_permutation(
                            n,
                            matched_groups_bootstrap_by_variable[
                                (std::size_t)variable
                            ],
                            scramble_rng,
                            permutation,
                            group_permutation_scratch
                        );
                    } else {
                        make_permutation(
                            n,
                            scramble_rng,
                            permutation
                        );
                    }
                    
                    scramble_block_inplace(
                        X_bootstrap,
                        columns,
                        permutation,
                        saved_columns
                    );

                    std::vector<int> scrambled_misclassifications;

                    if (use_deferral) {
                        scrambled_misclassifications =
                            model.get_all_misclassifications_packed_trie(
                                X_bootstrap,
                                y_bootstrap,
                                budget_override,
                                bb_pred_bootstrap
                            );
                    } else {
                        scrambled_misclassifications =
                            model.get_all_misclassifications_packed_trie(
                                X_bootstrap,
                                y_bootstrap,
                                budget_override
                            );
                    }

                    std::vector<int> scrambled_deferrals;

                    if (use_deferral) {
                        scrambled_deferrals =
                            model.get_all_deferrals_packed_trie(
                                X_bootstrap,
                                budget_override
                            );

                        if (scrambled_deferrals.size() !=
                            scrambled_misclassifications.size()) {
                            throw std::runtime_error(
                                "RID deferral extraction returned different "
                                "numbers of scrambled misclassification and "
                                "deferral entries."
                            );
                        }
                    }

                    if ((uint64_t)scrambled_misclassifications.size() !=
                        number_of_trees) {
                        throw std::runtime_error(
                            "RID scalar extraction returned a different tree "
                            "count after scrambling. The original and scrambled "
                            "collectors must use identical traversal order."
                        );
                    }

                    for (uint64_t tree = 0;
                        tree < number_of_trees;
                        ++tree) {
                        const int deferrals = use_deferral
                            ? scrambled_deferrals[(std::size_t)tree]
                            : 0;

                        const int scrambled_eval_objective =
                            rid_eval_objective_from_mis_def(
                                scrambled_misclassifications[(std::size_t)tree],
                                deferrals,
                                use_deferral,
                                eta_defer
                            );

                        summed_objective_differences[(std::size_t)tree] +=
                            (int64_t)scrambled_eval_objective
                            - (int64_t)original_eval_objectives
                                [(std::size_t)tree];
                    }

                    restore_block_inplace(
                        X_bootstrap,
                        columns,
                        saved_columns
                    );
                }

                for (uint64_t tree = 0;
                    tree < number_of_trees;
                    ++tree) {
                    const double importance =
                        (double)summed_objective_differences[(std::size_t)tree]
                        /
                        ((double)n * (double)n_scramble_evals);

                    output.mean_sub_mr[(std::size_t)variable] +=
                        tree_weight * importance;

                    mass_by_importance[(std::size_t)variable][importance] +=
                        tree_weight;

                    if (return_joint_samples) {
                        output.feature_importance_weight_samples[
                            joint_sample_offset + (std::size_t)tree
                        ][(std::size_t)variable] = importance;
                    }
                }
            }

        const auto bootstrap_end = std::chrono::steady_clock::now();

        const double post_training_seconds =
            std::chrono::duration<double>(
                bootstrap_end - post_training_start
            ).count();

        const double bootstrap_seconds =
            std::chrono::duration<double>(
                bootstrap_end - bootstrap_start
            ).count();

        cout << std::fixed << std::setprecision(3)
            << "RID bootstrap " << (bootstrap + 1)
            << ": post-training RID = "
            << post_training_seconds << " s\n"
            << "RID bootstrap " << (bootstrap + 1)
            << ": total = "
            << bootstrap_seconds << " s\n"
            << "RID bootstrap " << (bootstrap + 1)
            << ": training fraction = "
            << (100.0 * training_seconds / bootstrap_seconds)
            << "%, post-training fraction = "
            << (100.0 * post_training_seconds / bootstrap_seconds)
            << "%\n";
    }

    if (importance_interval_mode == 0) {
        for (int variable = 0;
             variable < number_of_variables;
             ++variable) {
            const auto& mass =
                mass_by_importance[(std::size_t)variable];

            output.cdf_x[(std::size_t)variable].reserve(mass.size());
            output.cdf_p[(std::size_t)variable].reserve(mass.size());

            double cumulative_probability = 0.0;

            for (const auto& [importance, probability] : mass) {
                cumulative_probability += probability;

                output.cdf_x[(std::size_t)variable].push_back(importance);
                output.cdf_p[(std::size_t)variable].push_back(
                    cumulative_probability
                );
            }
        }
    }

    return output;
}
