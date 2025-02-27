# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import carb
import numpy as np
import omni.appwindow  # Contains handle to keyboard
from omni.isaac.core import World
from omni.isaac.core.utils.prims import define_prim, get_prim_at_path
from omni.isaac.nucleus import get_assets_root_path
from g1 import G1WBC

from omni.isaac.core.utils.extensions import enable_extension

# enable ROS2 bridge extension
# enable_extension("omni.isaac.ros2_bridge")

simulation_app.update()

import time

# Note that this is not the system level rclpy, but one compiled for omniverse
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty

first_step = True
reset_needed = False

# initialize robot on first step, run robot advance
def on_physics_step(step_size) -> None:
    global first_step
    global reset_needed
    if first_step:
        g1.initialize()
        first_step = False
    elif reset_needed:
        my_world.reset(True)
        reset_needed = False
        first_step = True
    else:
        g1.advance(step_size, base_command)


# spawn world
my_world = World(stage_units_in_meters=1.0, physics_dt=1 / 500, rendering_dt=1 / 50)
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")

# spawn warehouse scene
prim = define_prim("/World/Ground", "Xform")
asset_path = assets_root_path + "/Isaac/Environments/Grid/default_environment.usd"
prim.GetReferences().AddReference(asset_path)

# spawn robot
g1 = G1WBC(
    prim_path="/World/G1",
    name="G1",
    position=np.array([0, 0, 0.75]),
)

my_world.add_physics_callback("physics_step", callback_fn=on_physics_step)
my_world.reset()

# robot command
base_command = np.zeros(4)

i = 0
while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_stopped():
        reset_needed = True
    if my_world.is_playing():
        while(True):
            # move forward
            base_command = np.array([1, 0, 0, 0.75]) # x_vel, y_vel, yaw, height


simulation_app.close()