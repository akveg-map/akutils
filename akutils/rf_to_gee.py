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

        # First pass: Get GEE node numbers using a level-order traversal
        node_to_cnt = {0: 1}
        queue = [(0, 1)]
        head = 0
        while head < len(queue):
            current_node, current_cnt = queue[head]
            head += 1
            if tree_.feature[current_node] != _tree.TREE_UNDEFINED:
                left_child = tree_.children_left[current_node]
                right_child = tree_.children_right[current_node]

                left_cnt = current_cnt * 2
                right_cnt = current_cnt * 2 + 1

                node_to_cnt[left_child] = left_cnt
                node_to_cnt[right_child] = right_cnt

                queue.append((left_child, left_cnt))
                queue.append((right_child, right_cnt))

        # Second pass: Build the string using a recursive depth-first traversal
        tree_string_parts = []

        # Add root line
        root_impurity = tree_.impurity[0]
        root_samples = tree_.n_node_samples[0]
        tree_string_parts.append(f"1) root {root_samples} 9999 9999 ({root_impurity:.4f})\n")

        def recurse(node_id, depth):
            """Recursive helper to build the string for each node."""
            if tree_.feature[node_id] == _tree.TREE_UNDEFINED:
                return

            # The splitting criteria belong to the parent
            feature_name = feature_names[tree_.feature[node_id]]
            threshold = tree_.threshold[node_id]
            indent = "  " * depth

            # Evaluate both children in a loop (Left, then Right)
            children = [
                (tree_.children_left[node_id], "<="),
                (tree_.children_right[node_id], ">")
            ]

            for child_id, sign in children:
                cnt = node_to_cnt.get(child_id, 0)

                # Properties below belong to the child
                n_samples = tree_.n_node_samples[child_id]
                impurity = tree_.impurity[child_id]
                is_leaf = tree_.feature[child_id] == _tree.TREE_UNDEFINED

                # Determine value
                if is_regressor:
                    value = tree_.value[child_id][0][0]
                else:
                    # Note: This specifically outputs the probability of class index 1.
                    # For standard classification (outputting the class label),
                    # you would use: value = np.argmax(tree_.value[child_id][0])
                    node_values = tree_.value[child_id][0]
                    value = node_values[1] / np.sum(node_values) if np.sum(node_values) > 0 else 0

                tail = " *\n" if is_leaf else "\n"

                tree_string_parts.append(
                    f"{indent}{cnt}) {feature_name} {sign} {threshold:.6f} {n_samples} {impurity:.4f} {value:.6f}{tail}"
                )

                # Recurse if it's not a leaf
                if not is_leaf:
                    recurse(child_id, depth + 1)

        recurse(0, 1)
        strings.append("".join(tree_string_parts))

    return strings
