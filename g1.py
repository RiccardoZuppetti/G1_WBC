# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import io
from typing import List, Optional

import carb
import numpy as np
import omni
import omni.kit.commands
import torch
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.prims import define_prim, get_prim_at_path
from omni.isaac.core.utils.rotations import quat_to_rot_matrix
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.nucleus import get_assets_root_path
from pxr import Gf


class G1WBC:
    """The G1 WBC Policy"""

    def __init__(
        self,
        prim_path: str,
        name: str = "g1",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:

        self._stage = get_current_stage()
        self._prim_path = prim_path
        prim = get_prim_at_path(self._prim_path)
        assets_root_path = get_assets_root_path()
        if not prim.IsValid():
            prim = define_prim(self._prim_path, "Xform")
            if usd_path:
                prim.GetReferences().AddReference(usd_path)
            else:
                if assets_root_path is None:
                    carb.log_error("Could not find Isaac Sim assets folder")

                asset_path = "/home/ubuntu/IsaacLab/g1_new.usd" # urdf converted to usd (characterized by 27 DOFs as required by the control policy)

                prim.GetReferences().AddReference(asset_path)

        self.robot = Articulation(prim_path=self._prim_path, name=name, position=position, orientation=orientation)

        self._dof_control_modes: List[int] = list()

        # Policy
        file_content = omni.client.read_file("/home/ubuntu/Downloads/isaacgym/python/OpenHomie/HomieRL/legged_gym/logs/exported/policies/policy.pt")[2]
        file = io.BytesIO(memoryview(file_content).tobytes())

        self._policy = torch.jit.load(file)
        self._base_vel_lin_scale = 1
        self._base_vel_ang_scale = 1
        self._action_scale = 0.25

        # default joint_pos for the imported G1 in the following
        # Joints list: ['left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint', 'left_hip_roll_joint', 'right_hip_roll_joint', 'left_hip_yaw_joint', 'right_hip_yaw_joint', 'left_knee_joint', 'right_knee_joint', 'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint', 'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_ankle_roll_joint', 'right_ankle_roll_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 'left_elbow_joint', 'right_elbow_joint', 'left_wrist_roll_joint', 'right_wrist_roll_joint', 'left_wrist_pitch_joint', 'right_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_wrist_yaw_joint']

        self._default_joint_pos = [
            -0.1, -0.1, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.3, 0.3,
            0.0, 0.0, -0.2,
            -0.2, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0
        ]

        self._previous_action = np.zeros(12)
        self._policy_counter = 0

    def _compute_observation(self, command):

        lin_vel_I = self.robot.get_linear_velocity()
        ang_vel_I = self.robot.get_angular_velocity()
        pos_IB, q_IB = self.robot.get_world_pose()

        R_IB = quat_to_rot_matrix(q_IB)
        R_BI = R_IB.transpose()
        lin_vel_b = np.matmul(R_BI, lin_vel_I)
        ang_vel_b = np.matmul(R_BI, ang_vel_I)
        gravity_b = np.matmul(R_BI, np.array([0.0, 0.0, -1.0]))

        obs = np.zeros(76)
        # Base ang vel
        obs[0:3] = self._base_vel_ang_scale * ang_vel_b
        # Gravity
        obs[3:6] = gravity_b        
        # commands
        obs[6] = self._base_vel_lin_scale * command[0]
        obs[7] = self._base_vel_lin_scale * command[1]
        obs[8] = self._base_vel_ang_scale * command[2]
        obs[9] = self._base_vel_lin_scale * command[3]
        # Joint states
        current_joint_pos = self.robot.get_joint_positions()
        current_joint_vel = self.robot.get_joint_velocities()
        obs[10:37] = current_joint_pos - self._default_joint_pos
        obs[37:64] = current_joint_vel
        # Previous Action
        obs[64:76] = self._previous_action

        return obs

    def advance(self, dt, command):

        if self._policy_counter % 4 == 0:
            obs = self._compute_observation(command)
            with torch.no_grad():
                obs = torch.from_numpy(obs).view(1, -1).float()
                self.action = self._policy(obs).detach().view(-1).numpy()
            self._previous_action = self.action.copy()

        action = ArticulationAction(joint_positions=self._default_joint_pos + (self.action * self._action_scale))
        self.robot.apply_action(action)

        self._policy_counter += 1

    def initialize(self, physics_sim_view=None) -> None:

        self.robot.initialize(physics_sim_view=physics_sim_view)
        self.robot.get_articulation_controller().set_effort_modes("force")
        self.robot.get_articulation_controller().switch_control_mode("position")
        # initialize robot parameter, set joint properties based on the values from env param

        # DOFs names: ['left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint', 'left_hip_roll_joint', 'right_hip_roll_joint', 'left_hip_yaw_joint', 'right_hip_yaw_joint', 'left_knee_joint', 'right_knee_joint', 'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint', 'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_ankle_roll_joint', 'right_ankle_roll_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 'left_elbow_joint', 'right_elbow_joint', 'left_wrist_roll_joint', 'right_wrist_roll_joint', 'left_wrist_pitch_joint', 'right_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_wrist_yaw_joint']

        stiffness = np.array([
            100, 100, 300,
            100, 100, 100,
            100, 150, 150,
            200, 200, 40,
            40, 200, 200,
            40, 40, 200,
            200, 100, 100,
            20, 20, 20,
            20, 20, 20
        ])


        damping = np.array([
            2, 2, 5,
            2, 2, 2,
            2, 4, 4,
            4, 4, 2,
            2, 4, 4,
            2, 2, 4,
            4, 1, 1,
            0.5, 0.5, 0.5,
            0.5, 0.5, 0.5
        ])


        max_effort = np.array([
            300, 300, 300, 300, 300,
            300, 300, 300, 300, 300,
            300, 100, 100, 300, 300,
            100, 100, 300, 300, 300,
            300, 50, 50, 50, 50, 50, 50
        ])


        max_vel = np.zeros(27) + 100.0
        self.robot._articulation_view.set_gains(stiffness, damping)
        self.robot._articulation_view.set_max_efforts(max_effort)
        self.robot._articulation_view.set_max_joint_velocities(max_vel)

    def post_reset(self) -> None:
        """
        Post Reset robot articulation
        """
        self.robot.post_reset()