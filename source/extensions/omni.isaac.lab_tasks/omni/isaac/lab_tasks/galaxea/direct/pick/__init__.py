import os
import gymnasium as gym
from.bin_pick_env import YonduBinPickingEnv
from .pick_env_cfg import (
   YonduBinPickEnvCfg
)

##
# Register Gym environments.
##

gym.register(
    id="Isaac-R1-Bin-Pick-Env-v0",
    entry_point="omni.isaac.lab_tasks.galaxea.direct.pick:YonduBinPickingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": YonduBinPickEnvCfg,
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        # "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.FrankaCabinetPPORunnerCfg,
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)