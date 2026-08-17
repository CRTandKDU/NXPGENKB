# Some utils for NXP 40y rule generation from PRISM

def nxp_prism__bin_rules( terms_list, prism ):
    bin_rule_str = ""
    for term in terms_list:
        arr = [k for k, v in prism.int_to_values_map[ term[0] ].items() if v == str(term[1]) ]
        idx = int( arr[0] )
        inf = prism.bin_ranges[ term[0] ][idx]
        sup = prism.bin_ranges[ term[0] ][idx + 1]
        txt = '\n#+BEGIN_RULE\n'
        txt += f'!{term[0]} s( {inf:.2f}) f>\n'
        txt += f'!{term[0]} s( {sup:.2f}) f<\n'
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

