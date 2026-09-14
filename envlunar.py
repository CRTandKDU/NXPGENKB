import numpy as np
import gymnasium as gym

import nlunar

class LLenv( gym.Env ):
    def __init__( self ):
        self._agent_location    = np.array( [ 65, 504, 150 ], dtype=np.int64 )
        self._target_location   = np.array( [ 0, 0, 10 ], dtype=np.int64 )
        # Define what the agent can observe
        # Dict space gives us structured, human-readable observations
        self.observation_space = gym.spaces.Dict(
            {
                "agent":  gym.spaces.Box( low   = np.array([-5000,-100,-1000]),
                                          high  = np.array([5000,5000,200]),
                                          dtype = np.int64 ),
                "target": gym.spaces.Box( low   = np.array([-500,0,0]),
                                          high  = np.array([500,5000,200]),
                                          dtype = np.int64 )
            }
        )
        # Define what actions are available (5 units increment up to 100)
        self.action_space = gym.spaces.Discrete( 20 )

    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent and target positions
        """
        return {"agent": self._agent_location, "target": self._target_location}
        
    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target
        """
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }
    
    def reset(self, seed = None, options = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        # Randomly place the agent anywhere on the grid
        self._agent_location = np.array( [ 65, 504, 150 ], dtype=np.int64 )

        # Randomly place target, ensuring it's different from agent position
        self._target_location   = np.array( [ 0, 0, 10 ], dtype=np.int64 )
        # self._target_location = self._agent_location
        # while np.array_equal(self._target_location, self._agent_location):
        #     self._target_location = self.np_random.integers(
        #         0, self.size, size=2, dtype=int
        #     )

        observation = self._get_obs()
        info = self._get_info()
        return observation, info
        

    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0 to 95 fuel units)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        self._agent_location = nlunar.f_state( self._agent_location, action )
        res                  = nlunar.ok_state( self._agent_location )
        terminated  = (1 == res)
        truncated   = True if 0 == res else False
        reward      = 1 if terminated else -0.01 if 0 == res else -1
        observation = self._get_obs()
        info        = self._get_info()

        return observation, reward, terminated, truncated, info

    
# Register the environment so we can create it with gym.make()
gym.register(
    id                = "LunarLandingHP-25",
    entry_point       = LLenv,
    max_episode_steps = 20,  # Prevent infinite episodes
)
