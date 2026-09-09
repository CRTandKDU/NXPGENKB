#  _______________________________________ 
# < Some utils for NXP40Y rule generation >
#  --------------------------------------- 
#         \   ^__^
#          \  (oo)\_______
#             (__)\       )\/\
#                 ||----w |
#                 ||     ||

from contextlib import redirect_stdout

#----------------------------------------------------------------------
# PRISM (J. Cendrowska, 1988)
#----------------------------------------------------------------------

def nxp_prism__bin_rules( terms_list, prism ):
    bin_rule_str = ""
    for term in terms_list:
        arr = [k for k, v in prism.int_to_values_map[ term[0] ].items() if v == str(term[1]) ]
        idx = int( arr[0] )
        inf = prism.bin_ranges[ term[0] ][idx]
        sup = prism.bin_ranges[ term[0] ][idx + 1]
        txt = '\n#+BEGIN_RULE\n'
        txt += f'!{term[0]} nxp@ s( {inf:.2f}) nxp2f f>\n'
        txt += f'!{term[0]} nxp@ s( {sup:.2f}) nxp2f f<\n'
        txt += 'THEN {}_{}\n'.format( str(term[0]), str(term[1]) )
        txt += '#+END_RULE\n'
        bin_rule_str += txt
    return bin_rule_str


def nxp_prism_rule( target_col, target_val, terms_list, prism ):
    rule_str = '#+BEGIN_RULE\n'
    for term in terms_list:
        rule_str += 'YES {}_{}\n'.format( str(term[0]), str(term[1]) )
    rule_str += 'THEN H{}_{}\n'.format( target_col, target_val )
    rule_str += '#+END_RULE\n'
    # bin rules
    bin_rule_str = nxp_prism__bin_rules( terms_list, prism )
    return rule_str + bin_rule_str

#----------------------------------------------------------------------
# DT to Rules conversion for sklearn 'DecisionTreeClassifier'
#----------------------------------------------------------------------
from sklearn.tree import DecisionTreeClassifier

def nxp__get_leaf_conditions(model):
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
            value           = tree.value[node][0]
            predicted_class = model.classes_[value.argmax()]

            paths.append({
                "conditions" : conditions.copy(),
                "class"      : predicted_class,
                "value"      : value.copy(),
            })

            # paths.append(conditions)
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


def nxp__paths_to_kb( paths ):
    for i, path in enumerate(paths, 1):
        print( f'#+BEGIN_RULE R_{i}' )
        for condition in path["conditions"]:
            sign, op, val = condition
            print( f'!{sign} nxp@ s( {val:.2f}) nxp2f {"f>" if op == ">" else "f<"}' )
        print( f'THEN HYPO_{path['class']}' )
        print( '#+END_RULE\n' )


def nxp__df_infos( df, heading ):
    if df is not None :
        print( f'* {heading}' )
        print( df.info() )

        
def nxp_dtclf_to_kb( model,
                     df_f = None,
                     df_t = None,
                     outfile=None ):
    paths = nxp__get_leaf_conditions( model )
    #
    if( None != outfile ):
        with open( outfile, 'w') as f:
            with redirect_stdout(f):
                nxp__df_infos( df_f, 'Dataset Features' )
                nxp__df_infos( df_t, 'Dataset Targets' )
                nxp__paths_to_kb( paths )
    else:
        nxp__df_infos( df_f, 'Dataset Features' )
        nxp__df_infos( df_t, 'Dataset Targets' )
        nxp__paths_to_kb( paths )
