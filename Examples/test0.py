import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_blobs, fetch_openml
import sklearn.datasets as datasets

import sys
sys.path.append('../')
from prism_rules import PrismRules

data    = datasets.load_wine()
df      = pd.DataFrame(data.data, columns=data.feature_names)
df['Y'] = data['target']


print( df.info() )

prism   = PrismRules( nbins=4 )
r =  prism.get_prism_rules(df, 'Y', display_stats=False, fmt='NXP40Y' )

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
    
