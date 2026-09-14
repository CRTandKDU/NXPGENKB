import numpy as np

G = 5
W = np.array([[ 1, 0, 0],
              [-1, 1, 0],
              [ 0, 0, 1]], dtype=np.int64)
B = np.array([ -1, .5, -1], dtype=np.float64)
C = np.array([ G, -G*.5, 0], dtype=np.float64)

W_inv = np.array([[ 1, 0, 0],
                  [ 1, 1, 0],
                  [ 0, 0, 1]], dtype=np.int64)
B_inv = np.array([ 1, .5, 1], dtype=np.float64)
C_inv = np.array([ -G, -G*.5, 0], dtype=np.float64)

_adjust, _tolerance = 3, 3


def f_state( state, command ):
    "State transformation function."
    return np.matmul( W, state ) + np.int64( command * B  + C )

def f_inv_state( state, command ):
    "Inverse transformation"
    return np.matmul( W_inv, state ) + np.int64( command * B_inv + C_inv )

def ok_state( state ):
    # Invalid last command
    if( state[2] < 0 ):
        return 0
    # Negative altitude
    if( state[1] < 0 ):
        return 0
    # No more fuel
    if( 0 == state[2] and state[0] > _tolerance ):
        return 0
    # Landed
    if( state[1] <= _adjust and state[0] <= _tolerance ):
        return 1
    # In flight
    return 2

def play_seq( s0, commands ):
    s        = s0
    for c in commands :
        s = f_state(s, c)
        if( 2 != ok_state(s) ):
            break
    return s, ok_state(s)

def play_inv_seq( sfinal, commands ):
    s        = sfinal
    for c in commands :
        s = f_inv_state(s, c)
        if( 2 != ok_state(s) ):
            break
    return s, ok_state(s)
    

if __name__ == "__main__":
    # s, res = play_seq( np.array( [20, 10, 120], dtype=np.int64 ),
    #                    np.array( [ 25, 0, 0, 0, 0, 0], dtype=np.int64 ) )
    # print( f'Final state: {s} ({res})' )
    #
    s, res = play_inv_seq( np.array( [0, 0, 10], dtype=np.int64 ),
                           np.array( [25, 25, 25, 25, 0, 0, 0, 0, 10], dtype = np.int64 ) )
    print( f'Initial state: {s} ({res})' )
    s, res = play_seq( np.array( [65, 501, 120], dtype=np.int64 ),
                       np.flip( np.array( [25, 25, 25, 25, 0, 0, 0, 0, 10], dtype = np.int64 ) ) )
    print( f'Final state: {s} ({res})' )
                       
    
