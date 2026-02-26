# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# Random Forest to GEE
# Author: Matt Macander, Timm Nawrocki
# Last Updated: 2026-02-25
# Usage: Must be executed in an Anaconda Python 3.12+ distribution.
# Description: "Random Forest to GEE" is a set of functions to convert a Random Forest model trained in scikit-learn or imbalanced-learn to GEE-compatible tree strings.
# ---------------------------------------------------------------------------

def rf_to_gee_strings(model, feature_names, model_type):
    """
    Converts a scikit-learn RandomForest model to a list of GEE-compatible strings.
    This optimized version avoids slow DataFrame manipulations and uses direct
    tree traversal for much better performance with large models.
    """

    # Import packages
    import numpy as np
    from sklearn.tree import _tree

    # Helper function to sanitize floats for GEE's strict parser
    def safe_format(x, precision=6):
        if np.isinf(x):
            # Replace infinity with a safely massive integer bound
            return "999999999.999999" if x > 0 else "-999999999.999999"
        elif np.isnan(x):
            return "0.000000"
        else:
            return f"{x:.{precision}f}"

    # Define string placeholder
    strings = []
    is_regressor = model_type == 'regressor'

    # Iterate through trees to compile tree string
    for tree in model.estimators_:
        tree_ = tree.tree_
        tree_string_parts = []

        # Determine root value
        if is_regressor:
            root_val = tree_.value[0][0][0]
        else:
            node_values = tree_.value[0][0]
            root_val = node_values[1] / np.sum(node_values) if np.sum(node_values) > 0 else 0

        # Add root string with formatted placeholder metrics expected by GEE
        tree_string_parts.append(f"1) root {tree_.n_node_samples[0]} 9999 9999 ({safe_format(root_val)})\n")

        # Process the rest of the tree using a DFS stack
        stack = []
        if tree_.feature[0] != _tree.TREE_UNDEFINED:
            feature_name = feature_names[tree_.feature[0]]
            threshold = tree_.threshold[0]

            # Push right child first (LIFO order ensures left child pops first)
            stack.append((tree_.children_right[0], 1, 3, ">", feature_name, threshold))
            stack.append((tree_.children_left[0], 1, 2, "<=", feature_name, threshold))

        while stack:
            node_id, depth, gee_id, sign, parent_feature, parent_threshold = stack.pop()

            # Determine value for the current node
            if is_regressor:
                val = tree_.value[node_id][0][0]
            else:
                nv = tree_.value[node_id][0]
                val = nv[1] / np.sum(nv) if np.sum(nv) > 0 else 0

            # Run variables through the safe formatter to catch inf/nan
            t_str = safe_format(parent_threshold)
            v_str = safe_format(val)
            imp_str = safe_format(tree_.impurity[node_id], precision=4)

            # Check if current node is a leaf
            is_leaf = tree_.feature[node_id] == _tree.TREE_UNDEFINED
            leaf_str = " *\n" if is_leaf else "\n"
            indent = "  " * depth

            # Construct and append the current node string
            tree_string_parts.append(
                f"{indent}{gee_id}) {parent_feature} {sign} {t_str} "
                f"{tree_.n_node_samples[node_id]} {imp_str} {v_str}{leaf_str}"
            )

            # If it's not a leaf, push children to the stack
            if not is_leaf:
                feature_name = feature_names[tree_.feature[node_id]]
                threshold = tree_.threshold[node_id]

                stack.append((tree_.children_right[node_id], depth + 1, gee_id * 2 + 1, ">", feature_name, threshold))
                stack.append((tree_.children_left[node_id], depth + 1, gee_id * 2, "<=", feature_name, threshold))

        # Combine list into a single string
        strings.append("".join(tree_string_parts))

    return strings
