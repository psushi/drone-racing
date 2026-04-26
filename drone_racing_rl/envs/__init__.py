"""Environment registrations for drone racing simulation and deployment."""

from gymnasium import register

# region SimEnvs

register(
    id="DroneRacing-v0",
    entry_point="drone_racing_rl.envs.drone_race:DroneRaceEnv",
    vector_entry_point="drone_racing_rl.envs.drone_race:VecDroneRaceEnv",
    max_episode_steps=1500,
    disable_env_checker=True,
)

# region RealEnvs

register(
    id="RealDroneRacing-v0",
    entry_point="drone_racing_rl.envs.real_race_env:RealDroneRaceEnv",
    disable_env_checker=True,
)
