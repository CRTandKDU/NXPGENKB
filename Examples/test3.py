#  _____________________________ 
# < DT Learning on UCI datasets >
#  ----------------------------- 
#         \   ^__^
#          \  (oo)\_______
#             (__)\       )\/\
#                 ||----w |
#                 ||     ||

import argparse
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo 

from sklearn.datasets import make_classification, make_blobs, fetch_openml
import sklearn.datasets as datasets
from sklearn import tree


import sys
sys.path.append('../')

# from prism_rules import PrismRules

import nxp_rules as nxpgenkb

#-------------------------------------------------------------------------------
# The command line and learning parameters
#-------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='NXPGENKB explorer')
parser.add_argument('--nbins', type=int, default=3, metavar='#bins',
                    help='number of bins for categorizing features (default 3)' )
parser.add_argument('--uci', type=int, default=60, metavar='UCI#',
                    help='UCI model number (default=60, liver disorder)' )
parser.add_argument('--output', type=str, default=None, metavar='KB',
                    help='output KB org file' )
parser.add_argument('--format', type=str, default='PRISM', metavar='format',
                    choices=['PRISM', 'NXP40Y'],
                    help='format for printing rules, one of "PRISM", "NXP40Y"' )
parser.add_argument('--stats', action='store_true',
                    help='display some stats')
parser.add_argument('--verbose', action='store_true',
                    help='verbosity')

args = parser.parse_args()

# [[https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic]]
# [[https://archive.ics.uci.edu/dataset/60/liver+disorders]]

#-------------------------------------------------------------------------------
# The dataset
#-------------------------------------------------------------------------------
uci_data = fetch_ucirepo( id=args.uci ) 
print( uci_data.data.features )
print( uci_data.data.targets )

df_f = pd.DataFrame( uci_data.data.features )
df_t = pd.DataFrame( uci_data.data.targets )


# df      = pd.DataFrame(data.data, columns=data.feature_names)
# df['Y'] = data['target']

print( df_f.info() )
print( df_t.info() )
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(df_f)


clf = tree.DecisionTreeClassifier()
clf = clf.fit( df_f, df_t )

r = tree.export_text(clf, feature_names=df_f.columns )
print(r)

n_nodes        = clf.tree_.node_count
children_left  = clf.tree_.children_left
children_right = clf.tree_.children_right
feature        = clf.tree_.feature
threshold      = clf.tree_.threshold

print( f'#nodes={n_nodes}\nLeft={children_left}\nRight={children_right}' )
print( f'Feature={[df_f.columns[i] for i in feature]}, Threshold={threshold}' )

# The tree structure can be traversed to compute various properties such
# as the depth of each node and whether or not it is a leaf.
node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
is_leaves  = np.zeros(shape=n_nodes, dtype=bool)
stack = [(0, -1)]  # seed is the root node id and its parent depth
while len(stack) > 0:
    node_id, parent_depth = stack.pop()
    node_depth[node_id] = parent_depth + 1

    # If we have a test node
    if (children_left[node_id] != children_right[node_id]):
        stack.append((children_left[node_id], parent_depth + 1))
        stack.append((children_right[node_id], parent_depth + 1))
    else:
        is_leaves[node_id] = True

print( node_depth )
print( is_leaves )
print( clf.tree_.n_outputs )
print( clf.tree_.n_classes )


paths = nxpgenkb.get_leaf_paths(clf)

for i, path in enumerate(paths, 1):
    print(f"Leaf {i}:")
    for condition in path["conditions"]:
        print(f"  {condition}")
    print(f"  -> class: {path['class']}")
    print()

# nxpgenkb.nxp_dtclf_to_kb( clf, outfile=f'../../IUP/uci{args.uci}.org' )
nxpgenkb.nxp_dtclf_to_kb( clf )

