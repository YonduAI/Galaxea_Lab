from __future__ import annotations

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import (
    ArticulationCfg,
    RigidObjectCfg,
    AssetBaseCfg,
)
from omni.isaac.lab_assets import (
    SHELF_BIN_CFG,
    SHELF_CFG,
    CART_CFG,
    CART_BIN_CFG
)

from omni.isaac.lab.sensors import CameraCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import (
    FrameTransformerCfg,
    OffsetCfg,
)

from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip
from omni.isaac.lab.markers import VisualizationMarkersCfg
from omni.isaac.lab.envs import DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.terrains import TerrainImporterCfg


@configclass
class PickEnvCfg(DirectRLEnvCfg):
    episode_length_s = 4.0  #
    decimation = 2
    num_actions = 14  # (6 + 1) * 2, 6 for arm joints, 1 for gripper state, two arms
    num_observations = 44  # without point clouds or images
    num_states = 0
    debug_vis = False
    enable_camera = True

    # must be named as "sim", can not be renamed with other str
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200.0,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene, must be named as "scene", can not be renamed with other str
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=3.0)

    # robot
    robot_cfg: ArticulationCfg = GALAXEA_R1_HIGH_PD_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )
    # eef frame: this is actually defining a piece of the arm most likely the end effector link
    left_ee_marker_cfg = FRAME_MARKER_CFG.copy()
    left_ee_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    left_ee_marker_cfg.prim_path = "/Visuals/FrameTransformer/LeftEE"
    left_ee_frame_cfg: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        debug_vis=debug_vis,
        visualizer_cfg=left_ee_marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/left_arm_link6",
                name="left_ee",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.15),  # offset from the link6 to the gripper tip
                ),
            ),
        ],
    )

    right_ee_marker_cfg = FRAME_MARKER_CFG.copy()
    right_ee_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    right_ee_marker_cfg.prim_path = "/Visuals/FrameTransformer/RightEE"
    right_ee_frame_cfg: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        debug_vis=debug_vis,
        visualizer_cfg=right_ee_marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/right_arm_link6",
                name="right_ee",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.15),  # offset from the link6 to the gripper tip
                ),
            ),
        ],
    )

    front_camera_cfg: CameraCfg = GALAXEA_CAMERA_CFG.replace(
        prim_path="/World/envs/env_.*/Robot/torso_link4/front_camera",
        height=240,
        width=320,
        offset=CameraCfg.OffsetCfg(
            pos=(0.066, 0.0, 0.482),
            rot=(0.2706, -0.6533, 0.6533, -0.2706),
            convention="ros",
        ),
    )

    left_wrist_camera_cfg: CameraCfg = GALAXEA_CAMERA_CFG.replace(
        prim_path="/World/envs/env_.*/Robot/left_arm_link6/left_wrist_camera",
        height=240,
        width=320,
        offset=CameraCfg.OffsetCfg(
            pos=(0.068, 0.0, 0.057),
            rot=(-0.683, 0.183, 0.183, -0.683),
            convention="ros",
        ),
    )
    right_wrist_camera_cfg: CameraCfg = left_wrist_camera_cfg.replace(
        prim_path="/World/envs/env_.*/Robot/right_arm_link6/right_wrist_camera",
    )

    # visualization markers
    vis_markers_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Markers",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.1, 0.1, 0.1),
            ),
            "object_center": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.1, 0.1, 0.1),
            ),
        },
    )

    # action type
    action_type = "joint_position"  # ik_rel, ik_abs
    goal_threshold = 0.04  # distance threshold for goal completion
    minimal_height = 0.1  # minimal height for object to be lifted

    # reward scales
    reward_reaching_object_scale = 1.0
    reward_lifting_object_scale = 15.0
    reward_tracking_goal_coarse_scale = 16.0
    reward_tracking_goal_fine_scale = 5.0
    reward_action_penalty_scale = -1e-4
    reward_joint_vel_penalty_scale = -1e-4


@configclass
class YonduBinPickEnv(PickEnvCfg):
    shelf: AssetBaseCfg = SHELF_CFG.copy()

    cart: AssetBaseCfg = CART_CFG.copy()

    cart_bin: RigidObjectCfg = CART_BIN_CFG.copy()

    shelf_bin: RigidObjectCfg = SHELF_BIN_CFG.copy()

