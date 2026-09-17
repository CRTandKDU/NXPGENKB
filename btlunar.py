# Backtracking algorithms for Lunar Landing
# Wednesday, September 16, 2026
import sys
import time
import numpy as np
import nlunar

RANGE_ACTIONS = [ 0, 5, 10, 15, 20, 25 ]
MAX_STEPS     = 50

def f_score( state ):
    return np.linalg.norm( state[:2] )


def get_min_score( state, range_actions=RANGE_ACTIONS ):
    scores = np.array( [ f_score( nlunar.f_state( state, action ) ) for action in range_actions ] )
    return scores.min(), np.argmin( scores )


def get_score_path( state, step, commands ):
    scores = np.array( [ f_score( nlunar.f_state( state, action ) ) for action in RANGE_ACTIONS ] )
    paths  = [ RANGE_ACTIONS[i] for i in np.argsort( scores ) ]
    return paths


def get_ord_path( state, step, commands ):
    return RANGE_ACTIONS


def get_path( state, step, commands, heuristics=get_ord_path ):
    if step > MAX_STEPS:
        return False, step, commands
    #
    res = nlunar.ok_state( state )
    if 0 == res:
        return False, step, commands
    if 1 == res:
        return True, step, commands
    for action in heuristics( state, step, commands ):
        _state = nlunar.f_state( state, action )
        foundp, _step, _commands = get_path( _state, step+1, commands + [ action ], heuristics=heuristics )
        if foundp :
            return True, _step, _commands
    return False, step, commands


        
if __name__ == "__main__":
    s0 = np.array( [65, 504, 200] if len(sys.argv) == 1 else sys.argv[1].split(','), dtype=np.int64 )
    #
    beg = time.perf_counter()
    foundp, step, commands = get_path( s0, 0, [], heuristics=get_score_path )
    end = time.perf_counter()
    print( s0, foundp, step, commands, sum( commands ), f'{end-beg:2.4f}' )
    
    
    
