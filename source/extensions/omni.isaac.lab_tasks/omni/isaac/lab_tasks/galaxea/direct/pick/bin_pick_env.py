# A bin picking environment  from __future__ import annotations

import torch
import copy
import random
from omni.isaac.core.prims import XFormPrimView
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, RigidObject, AssetBase
from omni.isaac.lab.envs import DirectRLEnv
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.utils.math import (
    sample_uniform,
    combine_frame_transforms,
    subtract_frame_transforms,
    quat_from_euler_xyz,
    quat_mul,
    skew_symmetric_matrix,
    matrix_from_quat,
)
from omni.isaac.lab.sensors.frame_transformer import FrameTransformer
from omni.isaac.lab.sensors import Camera
from omni.isaac.lab.controllers import (
    DifferentialIKController,
    DifferentialIKControllerCfg,
)

from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
import math

from .pick_env_cfg import YonduBinPickEnvCfg

class YonduBinPickingEnv(DirectRLEnv):
    # processed only at init time
    def __init__(
        self,
        cfg: YonduBinPickEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)

        # action type
        self.action_type = self.cfg.action_type

        # joint limits
        self.robot_joint_lower_limits = self._robot.data.soft_joint_pos_limits[
            0, :, 0
        ].to(device=self.device)
        self.robot_joint_upper_limits = self._robot.data.soft_joint_pos_limits[
            0, :, 1
        ].to(device=self.device)

        # track goal reset state
        self.reset_goal_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # default goal pose, i.e. the init target pose in the robot base_link frame
        self.goal_rot = torch.zeros(
            (self.num_envs, 4), dtype=torch.float, device=self.device
        )
        self.goal_rot[:, 0] = 1.0
        self.goal_pos = torch.zeros(
            (self.num_envs, 3), dtype=torch.float, device=self.device
        )
        self.goal_pos[:, :] = torch.tensor([0.4, 0.0, 1.2], device=self.device)

        # vis markers
        self.vis_markers = VisualizationMarkers(self.cfg.vis_markers_cfg)
        self.marker_indices = [
            i
            for i in range(self.vis_markers.num_prototypes)
            for _ in range(self.scene.num_envs)
        ]
        self.set_debug_vis(self.cfg.debug_vis)

        # end-effector offset w.r.t the *_arm_link6 frame
        self.ee_offset_pos = torch.tensor([0.0, 0.0, 0.15], device=self.device).repeat(
            self.num_envs, 1
        )
        self.ee_offset_quat = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=self.device
        ).repeat(self.num_envs, 1)
        #TODO:UNCOMMENT 
        # left/right arm/gripper joint ids
        self._setup_robot()
        # ik controller
        self._setup_ik_controller()

        self.succ = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.dt = self.cfg.sim.dt * self.cfg.decimation

        self.object_id = random.randint(0, 2)
        self.init_pos = torch.zeros(size=(self.num_envs, 3), device=self.device)

        print("Yondu Bin Picking is initialized. ActionType: ", self.action_type)
        
    def _setup_scene(self):
        self._object = [0]
        self._robot = Articulation(self.cfg.robot_cfg)
        self._drop_height = 0.91
        # add robot, object

        object0_cfg = copy.deepcopy(self.cfg.banana_cfg)
        object0_pos = (0.2, 1.2, self._drop_height) #(0.6, -0.6, self._drop_height)
        object0_cfg.init_state.pos = object0_pos
        object0_cfg.spawn.scale = (0.3, 0.3, 0.3)
        self._object[0] = RigidObject(object0_cfg)
        self._object.append(RigidObject(object0_cfg))
        
        #TODO: ADD ADDITIONAL OBJECTS AND DROP BINS IN RANDOMIZED LOCATION

        # add shelf which is a static object
        if self.cfg.shelf_cfg.spawn is not None:
            self.cfg.shelf_cfg.spawn.func(
                self.cfg.shelf_cfg.prim_path,
                self.cfg.shelf_cfg.spawn,
                translation=self.cfg.shelf_cfg.init_state.pos,
                orientation=self.cfg.shelf_cfg.init_state.rot,
            )
        # add cart which is a static object
        if self.cfg.cart_cfg.spawn is not None:
            self.cfg.cart_cfg.spawn.func(
                self.cfg.cart_cfg.prim_path,
                self.cfg.cart_cfg.spawn,
                translation=self.cfg.cart_cfg.init_state.pos,
                orientation=self.cfg.cart_cfg.init_state.rot,
            )

        # add camera
        if self.cfg.enable_camera:
            self._front_camera = Camera(self.cfg.front_camera_cfg)
            self._left_wrist_camera = Camera(self.cfg.left_wrist_camera_cfg)
            self._right_wrist_camera = Camera(self.cfg.right_wrist_camera_cfg)
            self.scene.sensors["front_camera"] = self._front_camera
            self.scene.sensors["left_wrist_camera"] = self._left_wrist_camera
            self.scene.sensors["right_wrist_camera"] = self._right_wrist_camera

        # frame transformer 
        self._left_ee_frame = FrameTransformer(self.cfg.left_ee_frame_cfg)
        self._right_ee_frame = FrameTransformer(self.cfg.right_ee_frame_cfg)

        # add to scene
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["object0"] = self._object[0]
        # self.scene.rigid_objects["object1"] = self._object[1]
        # self.scene.rigid_objects["object2"] = self._object[2]
        # self.scene.rigid_objects["object3"] = self._object[3]
        self.scene.sensors["left_ee_frame"] = self._left_ee_frame
        self.scene.sensors["right_ee_frame"] = self._right_ee_frame
        self.scene.extras["shelf"] = XFormPrimView(
            self.cfg.shelf_cfg.prim_path, reset_xform_properties=False
        )

        # add ground plane
        spawn_ground_plane(
            prim_path="/World/ground", cfg=GroundPlaneCfg(color=(1.0, 1.0, 1.0))
        )

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        print("Scene is set up.")

if __name__ == "main":
    import omni.isaac.core.utils.extensions as extensions
    # Load the Isaac Sim app and extensions
    extensions.load_extension("omni.isaac.kit")
    extensions.load_extension("omni.isaac.core")    
    cfg = YonduBinPickEnvCfg()
    env = YonduBinPickingEnv(cfg=cfg)
    env._setup_scene()
    