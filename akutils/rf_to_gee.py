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
    strings = []
    for tree in model.estimators_:
        tree_ = tree.tree_
        is_regressor = model_type == 'regressor'

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
        tree_string_parts.append(f"1) root {root_samples} 9999 9999 ({root_impurity})\n")

        def recurse(node_id, depth):
            """Recursive helper to build the string for each node."""
            if tree_.feature[node_id] == _tree.TREE_UNDEFINED:
                return

            # --- Left Branch ---
            left_child_id = tree_.children_left[node_id]
            sign = "<="

            # Get node properties
            feature_name = feature_names[tree_.feature[node_id]]
            threshold = tree_.threshold[node_id]
            n_samples = tree_.n_node_samples[node_id]
            impurity = tree_.impurity[node_id]
            cnt = node_to_cnt.get(left_child_id, 0)
            indent = "  " * depth

            # Determine value and tail
            is_leaf = tree_.feature[left_child_id] == _tree.TREE_UNDEFINED
            if is_leaf:
                if is_regressor:
                    value = tree_.value[left_child_id][0][0]
                else:
                    value = tree_.value[left_child_id][0][1] / np.sum(tree_.value[left_child_id][0]) if np.sum(
                        tree_.value[left_child_id][0]) > 0 else 0
                tail = " *\n"
            else:  # Not a leaf, use parent's value
                if is_regressor:
                    value = tree_.value[node_id][0][0]
                else:
                    value = tree_.value[node_id][0][1] / np.sum(tree_.value[node_id][0]) if np.sum(
                        tree_.value[node_id][0]) > 0 else 0
                tail = "\n"

            tree_string_parts.append(
                f"{indent}{cnt}) {feature_name} {sign} {threshold:.6f} {n_samples} {impurity:.4f} {value:.6f}{tail}")
            if not is_leaf:
                recurse(left_child_id, depth + 1)

            # --- Right Branch ---
            right_child_id = tree_.children_right[node_id]
            sign = ">"
            cnt = node_to_cnt.get(right_child_id, 0)

            is_leaf = tree_.feature[right_child_id] == _tree.TREE_UNDEFINED
            if is_leaf:
                if is_regressor:
                    value = tree_.value[right_child_id][0][0]
                else:
                    value = tree_.value[right_child_id][0][1] / np.sum(tree_.value[right_child_id][0]) if np.sum(
                        tree_.value[right_child_id][0]) > 0 else 0
                tail = " *\n"
            else:  # Not a leaf, use parent's value
                if is_regressor:
                    value = tree_.value[node_id][0][0]
                else:
                    value = tree_.value[node_id][0][1] / np.sum(tree_.value[node_id][0]) if np.sum(
                        tree_.value[node_id][0]) > 0 else 0
                tail = "\n"

            tree_string_parts.append(
                f"{indent}{cnt}) {feature_name} {sign} {threshold:.6f} {n_samples} {impurity:.4f} {value:.6f}{tail}")
            if not is_leaf:
                recurse(right_child_id, depth + 1)

        recurse(0, 1)
        strings.append("".join(tree_string_parts))

    return strings
