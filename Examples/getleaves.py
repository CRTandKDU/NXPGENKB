from sklearn.tree import DecisionTreeClassifier

def get_leaf_conditions(model):
    tree = model.tree_
    feature_names = getattr(
        model,
        "feature_names_in_",
        [f"feature_{i}" for i in range(model.n_features_in_)]
    )

    paths = []

    def walk(node, conditions):
        left = tree.children_left[node]
        right = tree.children_right[node]

        if left == right:
            paths.append(conditions)
            return

        feature = feature_names[tree.feature[node]]
        threshold = tree.threshold[node]

        walk(
            left,
            conditions + [(feature, "<=", threshold)]
        )
        walk(
            right,
            conditions + [(feature, ">", threshold)]
        )

    walk(0, [])
    return paths


def get_leaf_paths(model: DecisionTreeClassifier):
    """
    Return all root-to-leaf paths of a fitted sklearn DecisionTreeClassifier.

    Each path is represented as a list of conditions, followed by the
    prediction information at the leaf.

    Parameters
    ----------
    model : DecisionTreeClassifier
        A fitted sklearn DecisionTreeClassifier.

    Returns
    -------
    list of dict
        One dictionary per leaf, containing:
        - "conditions": list of strings describing the path
        - "class": predicted class
        - "value": class counts at the leaf
    """
    tree = model.tree_
    feature_names = getattr(model, "feature_names_in_", None)

    if feature_names is None:
        feature_names = [
            f"feature_{i}" for i in range(model.n_features_in_)
        ]

    paths = []

    def recurse(node, conditions):
        left = tree.children_left[node]
        right = tree.children_right[node]

        # Leaf node
        if left == right:
            value = tree.value[node][0]
            predicted_class = model.classes_[value.argmax()]

            paths.append({
                "conditions": conditions.copy(),
                "class": predicted_class,
                "value": value.copy(),
            })
            return

        feature = tree.feature[node]
        threshold = tree.threshold[node]
        feature_name = feature_names[feature]

        # Left branch: feature <= threshold
        recurse(
            left,
            conditions + [
                f"{feature_name} <= {threshold:.6g}"
            ]
        )

        # Right branch: feature > threshold
        recurse(
            right,
            conditions + [
                f"{feature_name} > {threshold:.6g}"
            ]
        )

    recurse(0, [])

    return paths
