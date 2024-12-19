import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab_assets import ISAACLAB_ASSETS_DATA_DIR
from omni.isaac.lab.assets import (
    RigidObjectCfg,
    AssetBaseCfg,
)
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg


SHELF_CFG = AssetBaseCfg(
    prim_path="/World/env1/Shelf",
    spawn=sim_utils.UsdFileCfg(
        usd_path="http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Environments/Hospital/Props/SM_MedShelf_01d.usd",
        scale=(1.0, 1.0, 1.0)
    ),
    init_state=AssetBaseCfg.InitialStateCfg(
        pos=(0.0, 1.5, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0)
    )
)

CART_CFG = AssetBaseCfg(
    prim_path="/World/env1/Cart",
    spawn=sim_utils.UsdFileCfg(
        usd_path="http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Environments/Hospital/Props/SM_Cart_01a.usd",
        scale=(0.9, 0.9, 0.9)
    ),
    init_state=AssetBaseCfg.InitialStateCfg(
        pos=(0.0, -1.5, 0.0),
        rot=(0.7071, 0.0, 0.0, 0.70712)
    )
)


SHELF_BIN_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/ShelfBin",
    debug_vis=True,
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/yondu/Galaxea_Lab/isaac-sim-assets-1-4.0.0/Assets/Isaac/4.0/Isaac/IsaacLab/Objects/SM_CratePlastic_A_02.usd",
        scale=(0.5, 0.5, 0.5),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.2, 1.2, 0.9),
        rot=(1.0, 0, 0, 0.),
    ),
)

CART_BIN_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/CartBin",
    debug_vis=True,
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/yondu/Galaxea_Lab/isaac-sim-assets-1-4.0.0/Assets/Isaac/4.0/Isaac/IsaacLab/Objects/SM_CratePlastic_A_03.usd",
        scale=(0.9, 0.9, 0.9),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0, -1.3, 1.21179),
        rot=(1.0, 0, 0, 1.0),
    ),
)