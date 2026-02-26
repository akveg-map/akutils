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
            # Note: Outputs the probability of class index 1.
            node_values = tree_.value[0][0]
            root_val = node_values[1] / np.sum(node_values) if np.sum(node_values) > 0 else 0

        # Add root string with formatted placeholder metrics expected by GEE
        tree_string_parts.append(f"1) root {tree_.n_node_samples[0]} 9999 9999 ({root_val})\n")

        # Process the rest of the tree using a Depth-First Search (DFS) stack.
        # Stack elements contain:
        # (sklearn_node_id, depth, gee_node_id, sign, parent_feature_name, parent_threshold)
        stack = []
        if tree_.feature[0] != _tree.TREE_UNDEFINED:
            feature_name = feature_names[tree_.feature[0]]
            threshold = tree_.threshold[0]

            # Push right child first so left child is popped and processed first (LIFO order)
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

            # Check if current node is a leaf
            is_leaf = tree_.feature[node_id] == _tree.TREE_UNDEFINED
            leaf_str = " *\n" if is_leaf else "\n"

            # Formatting variables
            indent = "  " * depth

            # Construct and append the current node string utilizing properties from both parent and current child
            tree_string_parts.append(
                f"{indent}{gee_id}) {parent_feature} {sign} {parent_threshold:.6f} "
                f"{tree_.n_node_samples[node_id]} {tree_.impurity[node_id]:.4f} {val:.6f}{leaf_str}"
            )

            # If it's not a leaf, determine its splitting condition and append its children to the stack
            if not is_leaf:
                feature_name = feature_names[tree_.feature[node_id]]
                threshold = tree_.threshold[node_id]

                # Push right child first (LIFO stack ensures pre-order execution)
                stack.append((tree_.children_right[node_id], depth + 1, gee_id * 2 + 1, ">", feature_name, threshold))
                stack.append((tree_.children_left[node_id], depth + 1, gee_id * 2, "<=", feature_name, threshold))

        # Combine list of string parts into a single string representing the entire tree
        strings.append("".join(tree_string_parts))

    return strings
