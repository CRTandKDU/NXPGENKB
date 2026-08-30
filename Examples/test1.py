# PRISM on datasets from the UCI repo
from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_blobs, fetch_openml
import sklearn.datasets as datasets

import sys
sys.path.append('../')
from prism_rules import PrismRules

uci_data = fetch_ucirepo(id=60) 
print( uci_data.data.features )
print( uci_data.data.targets )

df = pd.concat( [ pd.DataFrame( uci_data.data.features ),
                  pd.DataFrame( uci_data.data.targets ) ],
                axis=1 )

# df      = pd.DataFrame(data.data, columns=data.feature_names)
# df['Y'] = data['target']


print( df.info() )

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(df)

prism   = PrismRules( nbins=3, verbose=1 )
r =  prism.get_prism_rules(df, 'drinks', display_stats=False,
                           fmt='NXP40Y', outfile=None )

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
    
