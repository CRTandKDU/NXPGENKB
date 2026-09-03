    #  _______________________________ 
    # < Rule-Learning on UCI datasets >
    #  ------------------------------- 
    #         \   ^__^
    #          \  (oo)\_______
    #             (__)\       )\/\
    #                 ||----w |
    #                 ||     ||

import argparse
from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_blobs, fetch_openml
import sklearn.datasets as datasets

import sys
sys.path.append('../')

from prism_rules import PrismRules

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

df = pd.concat( [ pd.DataFrame( uci_data.data.features ),
                  pd.DataFrame( uci_data.data.targets ) ],
                axis=1 )

# df      = pd.DataFrame(data.data, columns=data.feature_names)
# df['Y'] = data['target']

print( f"Last is: {df.columns[-1]}" )
print( df.info() )

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(df)

#-------------------------------------------------------------------------------
# PRISM Rule Leaning (Cendrowska J.)
#-------------------------------------------------------------------------------
prism   = PrismRules( nbins=args.nbins,
                      verbose = 1 if args.verbose else 0 )
r       = prism.get_prism_rules(df, df.columns[-1],
                                display_stats = args.stats,
                                fmt           = args.format,
                                outfile       = args.output )

#-------------------------------------------------------------------------------
# Printouts
#-------------------------------------------------------------------------------
print( '\n------- Datatypes -------\n' )
for col_name in df.columns:
    print( '{:<32} {}'.format( col_name, prism.bin_ranges[ col_name ]
                               if ( pd.api.types.is_numeric_dtype(df[col_name]) and df[col_name].nunique() > 10 )
                               else df[col_name].unique() ) )
print()


for col_name in df.columns:
    print( prism.int_to_values_map[col_name] )
    
# for col_name in df.columns:
#     print()
#     print("*********************************************************************************")
#     print(f"Rules for {col_name}")
#     print("*********************************************************************************")
    
#     _ = prism.get_prism_rules(df, col_name, display_stats=False)
    
