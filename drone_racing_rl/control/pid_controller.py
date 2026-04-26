import numpy as np
from numpy.typing import NDArray
from drone_racing_rl.control.controller import Controller


class MyController(Controller):
    def __init__(self, obs, info, config):
        super().__init__(obs, info, config)

        # PID controller (I might be useless)
        self.dt = 1.0 / 100.0

        # altitude
        self.kp_z = 0.9
        self.kd_z = 0.25
        self.ki_z = 0.01
        self.int_ez = 0.0

        # x
        self.kp_x = 0.35
        self.kd_x = 0.23
        self.ki_x = 0.01
        self.int_ex = 0.0

        # y
        self.kp_y = 0.35
        self.kd_y = 0.23
        self.ki_y = 0.01
        self.int_ey = 0.0

        # yaw
        self.ky = 0.7

        self.hover_thrust = 0.45
        self.target_yaw = 0.0

        # limits
        self.max_roll = 4.0
        self.max_pitch = 4.0
        self.max_yaw = 1.0
        self.max_thrust = 2.0

        # gate passing logic
        self.phase = 1                   
        self.prev_target_gate_idx = -1
        self.approach_dist = 0.2
        self.pass_dist = 0.5

        # for phase 1->2: switch only when it is closed to the target position 
        #                 and really aligned with the center of the gate
        self.phase_switch_forward = 1.5    # distance to the target position
        self.phase_switch_lateral = 0.25   # well centered
        self.phase_switch_vertical = 0.1   # difference in z direction

        # phase 3: hold after passing a gate
        self.hold_steps = 30
        self.hold_counter = 0
        self.hold_target = None

        self.kill_requested = False


    def quat_to_yaw(self, quat):
        qx, qy, qz, qw = quat
        yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        return yaw

    def wrap_angle(self, a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    def compute_control(self, obs, info=None) -> NDArray[np.floating]:

        if self.kill_requested:
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=float)

        pos = obs["pos"]
        vel = obs["vel"]
        target_gate_idx = obs["target_gate"]

        if target_gate_idx != -1 :
            gate_center = obs["gates_pos"][target_gate_idx].copy()
            gates_quat = obs["gates_quat"][target_gate_idx].copy()
            gate_yaw = self.quat_to_yaw(gates_quat)
        
        yaw_cur = self.quat_to_yaw(obs["quat"])

        # choose the target position
        # there are 3 phases
        # phase 1: targets a point in front of the gate center
        # phase 2: targets a point behind the gate center
        # phase 3: After the gate index updates, we ignore the new gate for some seconds and hold the original target
        if target_gate_idx == -1:   # all gates finished
            target_pos = pos.copy()
           
        elif self.phase == 3 and self.hold_counter > 0: # phase 3->1
            target_pos = self.hold_target.copy()
            self.hold_counter -= 1
            if self.hold_counter == 0:
                self.phase = 1
                self.hold_target = None
                self.target_yaw = gate_yaw
                self.int_ex = 0.0
                self.int_ey = 0.0
                self.int_ez = 0.0

        else:
            # the state of the gate
            forward = np.array([np.cos(gate_yaw), np.sin(gate_yaw), 0.0]) # gate forward axis in world xy
            lateral = np.array([-np.sin(gate_yaw), np.cos(gate_yaw), 0.0]) # gate lateral axis in world xy
            
            # if the gate index is changed, go to the phase 3
            if self.prev_target_gate_idx != -1 and target_gate_idx != self.prev_target_gate_idx: 
                self.phase = 3
                self.hold_counter = self.hold_steps
                self.prev_target_gate_idx = target_gate_idx
                target_pos = self.hold_target.copy()

            else:
                # the first time
                if target_gate_idx != self.prev_target_gate_idx:
                    self.phase = 1
                    self.prev_target_gate_idx = target_gate_idx
                    self.target_yaw = gate_yaw
                    self.int_ex = 0.0
                    self.int_ey = 0.0
                    self.int_ez = 0.0

                # the point in front of the gate center
                approach_point = gate_center - self.approach_dist * forward
                approach_point[2] = gate_center[2]

                # the point behind the gate center
                pass_point = gate_center + self.pass_dist * forward
                pass_point[2] = gate_center[2]

                # the error calculation
                rel = pos - gate_center
                gate_dis = np.linalg.norm((approach_point[0:2]-pos[0:2]))
                gate_lateral_err = np.dot(rel, lateral)
                gate_vertical_err = rel[2]

                # phase 1->2 : only pass if close and centered
                if (
                    self.phase == 1
                    and abs(gate_dis) < self.phase_switch_forward
                    and abs(gate_lateral_err) < self.phase_switch_lateral
                    and abs(gate_vertical_err) < self.phase_switch_vertical
                ):
                    self.phase = 2
                    self.hold_target = pass_point
                    self.int_ex = 0.0
                    self.int_ey = 0.0
                    self.int_ez = 0.0

                if self.phase == 1:
                    target_pos = approach_point.copy()

                else:
                    target_pos = pass_point.copy()

        # avoid obstacle logic:
        avoid_offset = np.zeros(3)
        for i, obs_pos in enumerate(obs["obstacles_pos"]):
            if not obs["obstacles_visited"][i]:
                continue

            diff = pos[:2] - obs_pos[:2]
            dist = np.linalg.norm(diff)

            influence_r = 0.3
            if dist < influence_r:
                direction = diff / dist # norm
                strength = max(1.0 * (influence_r - dist) / influence_r, 0.0)
                avoid_offset[:2] += strength * direction

        target_pos = target_pos + avoid_offset

        # calculate the target_yaw
        # in phase 1, it should face to the gate
        # in phase 2~3, just let it equal to the gate yaw
        # if there is no gate, keep the current yaw
        if target_gate_idx != -1 :
            if self.phase == 1 or self.phase == 3:
                dif = gate_center[:2] - pos[:2]
                new_target_yaw = np.arctan2(dif[1], dif[0])
                self.target_yaw = self.target_yaw + self.wrap_angle(new_target_yaw - self.target_yaw)
            else:
                self.target_yaw = gate_yaw
        else:
            self.target_yaw = yaw_cur

        # the global error for position and velocity
        pos_err = target_pos - pos
        vel_err = -vel

        # calculate the local error for position and velocity
        cy = np.cos(yaw_cur)
        sy = np.sin(yaw_cur)

        ex_body =  cy * pos_err[0] + sy * pos_err[1]
        ey_body = -sy * pos_err[0] + cy * pos_err[1]

        if abs(ex_body) < 0.2:
            self.int_ex += ex_body * self.dt
        if abs(ey_body) < 0.2:
            self.int_ey += ey_body * self.dt
        if abs(pos_err[2]) < 0.2:
            self.int_ez += pos_err[2] * self.dt

        evx_body =  cy * vel_err[0] + sy * vel_err[1]
        evy_body = -sy * vel_err[0] + cy * vel_err[1]

        # calculate the pitch, roll, yaw, thrust
        pitch = self.kp_x * ex_body + self.kd_x * evx_body + self.ki_x * self.int_ex
        pitch = np.clip(pitch, -self.max_pitch, self.max_pitch)

        roll  = -(self.kp_y * ey_body + self.kd_y * evy_body + self.ki_y * self.int_ey)
        roll  = np.clip(roll, -self.max_roll, self.max_roll)

        yaw_err = np.clip(self.wrap_angle(self.target_yaw - yaw_cur), -self.max_yaw, self.max_yaw)
        yaw = self.ky * (yaw_cur + yaw_err)
        
        thrust = self.hover_thrust + self.kp_z * pos_err[2] + self.kd_z * vel_err[2] + self.ki_z * self.int_ez
        thrust = np.clip(thrust, 0.1, self.max_thrust)

        # action
        action = np.array([roll, pitch, yaw , thrust], dtype=float)

        # output for debugging
        print(
            # "target index", target_gate_idx,
            # "phase", self.phase,
            # "cur_yaw", yaw_cur,
            # "target_yaw", self.target_yaw,
            "pos:", np.round(pos, 3),
            # "gate_center", gate_center
            # "target:", np.round(target_pos, 3),
            # "error:", np.round(pos_err, 3),
            # "vel:", np.round(vel, 3),
            # "action:", np.round(action, 3),
        )

        return action

    def step_callback(self, action, obs, reward, terminated, truncated, info) -> bool:

        if self.kill_requested:
            return True

        # if the drone is too near the obstacle, shutdown!
        for pos in obs["obstacles_pos"]:
            if np.linalg.norm(obs["pos"][:2] - pos[:2]) < 0.05:
                return True

        return terminated
        
