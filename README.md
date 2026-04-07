[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Pzw1OSwB)
# ECE484 Drone Track
<p align="center">
  <img width="460" height="300" src="quadrotor-sketch.png">
</p>

## Introduction

In this project, you will develop safe and robust control systems for autonomous drone racing. There are two phases:

1. **Simulation**: Develop and test controllers in a physics-based simulation environment.
2. **Deployment**: Deploy and further optimize controllers on CrazyFlie hardware in the lab.

You are required to achieve safe autonomous flight in both phases. Teams that deploy successfully will be allowed to participate in a competition, in which lap times will be compared on various tracks.

The drone must navigate through a series of gates while avoiding obstacles. The track layout, gate positions, and obstacle positions are provided to your controller via observations.

**Join the course Slack for announcements, Q&A, and team coordination**: [ECE484 Drones Slack](https://join.slack.com/t/ece484drones/shared_invite/zt-3ubfnhimf-FQl2mhduGHjcZ~vIrxXbwQ)

## Setup

**Prerequisites**: Linux (Ubuntu recommended), Python 3.11+

1. Clone the repository:
   ```
   git clone <repo-url>
   cd ece484-fly
   ```
2. Install [Pixi](https://pixi.sh):
   ```
   curl -fsSL https://pixi.sh/install.sh | sh
   ```
3. Activate the simulation environment:
   ```
   pixi shell
   ```

## Project Structure

```
ece484-fly/
├── ece484_fly/
│   ├── control/          # Your controllers go here
│   │   └── controller.py # Base Controller class (must inherit from this)
│   ├── envs/             # Simulation environments (do not modify)
│   └── utils/            # Utility functions
├── scripts/
│   ├── sim.py            # Run simulation
│   ├── deploy.py         # Deploy to hardware
│   └── check_track.py    # Validate real track setup
└── config/
    └── level1.toml       # Track and environment configuration
```

## Writing a Controller

Create a new Python file in `ece484_fly/control/` that inherits from the `Controller` base class. You must implement the `compute_control` method. Only define **one** controller class per file.

```python
import numpy as np
from numpy.typing import NDArray
from ece484_fly.control.controller import Controller

class MyController(Controller):
    def __init__(self, obs, info, config):
        super().__init__(obs, info, config)
        # Initialize your controller here (pre-plan trajectories, load models, etc.)

    def compute_control(self, obs, info=None) -> NDArray[np.floating]:
        # Your control logic here
        # Return a state command or attitude command (see below)
        ...
```

### Observation Space

Your controller receives an `obs` dictionary at each time step with the following fields:

| Key | Shape | Description |
|-----|-------|-------------|
| `pos` | `(3,)` | Drone position `[x, y, z]` in meters |
| `quat` | `(4,)` | Drone orientation as quaternion `[qx, qy, qz, qw]` |
| `vel` | `(3,)` | Drone velocity `[vx, vy, vz]` in m/s |
| `ang_vel` | `(3,)` | Drone angular velocity `[wx, wy, wz]` in rad/s |
| `target_gate` | `int` | Index of the next gate to pass (`-1` if all gates passed) |
| `gates_pos` | `(n_gates, 3)` | Positions of all gates |
| `gates_quat` | `(n_gates, 4)` | Orientations of all gates as quaternions |
| `gates_visited` | `(n_gates,)` | Boolean flags for which gates have been passed |
| `obstacles_pos` | `(n_obstacles, 3)` | Positions of all obstacles |
| `obstacles_visited` | `(n_obstacles,)` | Boolean flags for obstacles detected |

> **Note**: Gate and obstacle positions report their nominal (config) positions until the drone is within `sensor_range` (default 0.7m), at which point the exact position is provided. In hardware, state estimation (position, velocity, orientation) is provided by an external motion capture system.

### Action Space

Your `compute_control` method must return one of:

- **State command** (13 values): `[x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate]` — desired position, velocity, acceleration, and yaw/angular rates in absolute coordinates.
- **Attitude command** (4 values): `[thrust, roll, pitch, yaw]` — direct attitude control.

The control mode is set in the config file (`control_mode = "state"` or `"attitude"`).

### Callback Methods (Optional)

- `step_callback(action, obs, reward, terminated, truncated, info)` — Called after each step. Use for learning, adaptation, or to signal episode termination.
- `episode_callback()` — Called after each episode. Use for logging, training, or resetting state.

## Configuration

The track configuration is defined in `config/level1.toml`. Key settings:

- **Gates**: Defined by position `[x, y, z]` and rotation `[roll, pitch, yaw]`. Gates are 0.72m wide (frame) with a 0.4m opening. Tall gates are at 1.195m, short gates at 0.695m.
- **Obstacles**: Cylinders, 0.03m diameter, ~1.52m tall.
- **Control mode**: `"state"` or `"attitude"` (set in `[env]` section).
- **Environment frequency**: 100 Hz (how often `compute_control` is called).
- **Simulation frequency**: 500 Hz (physics update rate).
- **Disturbances and randomizations**: Configured in `[env.disturbances]` and `[env.randomizations]` sections. These add noise to actions, dynamics, drone mass/inertia, and starting position to encourage robust controllers.

## Simulation

Activate the environment and run your controller:

```bash
pixi shell
python3 scripts/sim.py --config level1.toml --controller my_controller.py
```

The controller file is loaded from `ece484_fly/control/`. You can also set the controller in `level1.toml` under `[controller] file`.

## Hardware Deployment

### Environment Setup

Activate the deployment environment:
```bash
pixi shell -e deploy
```

### Motion Capture Configuration

The following changes must be made in `ros_ws`:

1. In `ros_ws/src/motion_capture_tracking/motion_capture_tracking/config/cfg.yaml`, set `type` to `vrpn`, `hostname` to `192.168.1.114`, and `port` to `3883`.
2. In `ros_ws/src/motion_capture_tracking/motion_capture_tracking/launch/node.launch`, set `motion_capture_type` and `motion_capture_hostname` accordingly.
3. In `ros_ws/src/motion_capture_tracking/motion_capture_tracking/src/motion_capture_tracking_node.cpp`, replace `vicon` with `vrpn` on Line 14.

Re-activate the deployment environment, then run `source install/setup.bash` in the ROS workspace if needed.

### USB Setup for CrazyRadio

Run the following once to configure USB permissions:

```bash
cat <<EOF | sudo tee /etc/udev/rules.d/99-bitcraze.rules > /dev/null
# Crazyradio (normal operation)
SUBSYSTEM=="usb", ATTRS{idVendor}=="1915", ATTRS{idProduct}=="7777", MODE="0664", GROUP="plugdev"
# Bootloader
SUBSYSTEM=="usb", ATTRS{idVendor}=="1915", ATTRS{idProduct}=="0101", MODE="0664", GROUP="plugdev"
# Crazyflie (over USB)
SUBSYSTEM=="usb", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="5740", MODE="0664", GROUP="plugdev"
EOF

sudo groupadd plugdev
sudo usermod -a -G plugdev $USER

sudo udevadm control --reload-rules
sudo udevadm trigger
```

### Running on Hardware

Open three separate terminals:

```bash
# Terminal 1: Motion capture tracking
ros2 launch motion_capture_tracking launch.py

# Terminal 2: State estimator
python3 -m drone_estimators.ros_nodes.ros2_node --drone_name <drone_name (eg. cf1)>

# Terminal 3: Deploy your controller
python3 scripts/deploy.py --config level1.toml --controller my_controller.py
```

## Must Have

Implement one or both of the following safety features in your stack. 

**Emergency Kill Switch**: Implement a background listener (e.g., keyboard or gamepad button) that terminates the flight when triggered. 

**Manual Takeover**: Implement a mode switch that allows an operator to take over control mid-flight via a gamepad or joystick. Pressing the button again should return to autonomous mode. The switch must be seamless — no abrupt jumps in commanded state.

## Resources

You are free to implement any control strategy. Here are some references to get started:

**Fundamentals**
- [Aerial Robotics — Vijay Kumar (Coursera)](https://www.coursera.org/learn/robotics-flight) — Video lectures on quadrotor dynamics, PID, and trajectory planning.
- [CrazyFlie Controller Documentation](https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/functional-areas/sensor-to-control/controllers/) — The cascaded PID architecture running on your hardware.

**Trajectory Planning**
- [Minimum Snap Trajectory Generation — Mellinger & Kumar (ICRA 2011)](https://ieeexplore.ieee.org/document/5980409) — Polynomial trajectory generation through waypoints.
- [Polynomial Trajectories in Dense Environments — Richter, Bry, Roy](https://arxiv.org/abs/1603.04622) — Minimum snap with corridor constraints for obstacle avoidance.

**Path Planning**
- [Planning Algorithms — LaValle](http://planning.cs.uiuc.edu/) — Free textbook. Chapters 5 (sampling-based) and 14 (differential constraints) cover A*, RRT, PRM.
- [Safe Flight Corridors — Liu et al.](https://arxiv.org/abs/1703.07640) — A* search to safe corridors to minimum snap optimization for quadrotors.

**Model Predictive Control**
- [Model Predictive Contouring Control for Drone Racing — Foehn et al.](https://arxiv.org/abs/2108.13205) — Time-optimal MPC for racing through gates.
- [acados Documentation](https://docs.acados.org/) — Fast embedded optimal control solver. See `tools/setup_acados.sh` in this repo.

**Reinforcement Learning**
- [Champion-Level Drone Racing Using Deep RL — Kaufmann et al. (Nature 2023)](https://arxiv.org/abs/2306.16772) — RL that beat human world champions at drone racing, with sim-to-real transfer.
- [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones) — Open-source Gymnasium environment for CrazyFlie-scale RL training.

> **Note on RL**: Training policies is non-trivial, requires significant compute, and can be difficult to transfer to real hardware. If you choose this approach, test sim-to-real transfer early.

## Hardware Details

- **Drone**: CrazyFlie 2.1B with 350mAh battery (`cf21B_500`)
- **Onboard sensors**: IMU only (no cameras)
- **State estimation**: Provided externally via motion capture system (position, orientation, velocity)
- **Communication**: CrazyRadio 2.0 USB dongle
