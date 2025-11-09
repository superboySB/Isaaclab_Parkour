from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from parkour_isaaclab.terrains.parkour_terrain_importer import ParkourTerrainImporter
from parkour_tasks.extreme_parkour_task.config.go2 import agents 
from isaaclab.sensors import RayCasterCameraCfg
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg
from isaaclab.envs import ViewerCfg
import os, torch 
from pathlib import Path
from parkour_isaaclab.actuators.parkour_actuator_cfg import ParkourDCMotorCfg

def quat_from_euler_xyz_tuple(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> tuple:
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    # compute quaternion
    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp
    convert = torch.stack([qw, qx, qy, qz], dim=-1) * torch.tensor([1.,1.,1.,-1])
    return tuple(convert.numpy().tolist())

@configclass
class ParkourDefaultSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    terrain = TerrainImporterCfg(
        class_type= ParkourTerrainImporter,
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=None,
        max_init_terrain_level=2,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    def __post_init__(self):
        repo_root = Path(__file__).resolve().parents[2]
        local_asset_root = repo_root / "assets" / "nucleus" / "Isaac" / "4.5" / "Isaac"

        local_go2_candidates = [
            local_asset_root / "IsaacLab" / "Robots" / "Unitree" / "Go2" / "go2.usd",
        ]
        for candidate in local_go2_candidates:
            if candidate.exists():
                self.robot.spawn.usd_path = candidate.as_posix()
                break
        else:
            print("[WARN] Local Go2 USD not found under assets/. Falling back to default asset path (requires network).")
        sky_tex = local_asset_root / "Materials" / "Textures" / "Skies" / "PolyHaven" / "kloofendal_43d_clear_puresky_4k.hdr"
        if sky_tex.exists():
            self.sky_light.spawn.texture_file = sky_tex.as_posix()

        marble_mdl = (
            local_asset_root
            / "IsaacLab"
            / "Materials"
            / "TilesMarbleSpiderWhiteBrickBondHoned"
            / "TilesMarbleSpiderWhiteBrickBondHoned.mdl"
        )
        if marble_mdl.exists():
            self.terrain.visual_material.mdl_path = marble_mdl.as_posix()

        arrow_usd = local_asset_root / "Props" / "UIElements" / "arrow_x.usd"
        if arrow_usd.exists():
            arrow_path = arrow_usd.as_posix()
            try:
                from isaaclab.markers.config import (
                    BLUE_ARROW_X_MARKER_CFG,
                    FRAME_MARKER_CFG,
                    GREEN_ARROW_X_MARKER_CFG,
                    RED_ARROW_X_MARKER_CFG,
                )

                for marker_cfg in (GREEN_ARROW_X_MARKER_CFG, BLUE_ARROW_X_MARKER_CFG, RED_ARROW_X_MARKER_CFG):
                    marker_cfg.markers["arrow"].usd_path = arrow_path
                if "frame" in FRAME_MARKER_CFG.markers:
                    frame_usd = local_asset_root / "Props" / "UIElements" / "frame_prim.usd"
                    if frame_usd.exists():
                        FRAME_MARKER_CFG.markers["frame"].usd_path = frame_usd.as_posix()
            except ImportError:
                pass

        self.robot.spawn.articulation_props.enabled_self_collisions = True
        self.robot.actuators['base_legs'] = ParkourDCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit={
                        '.*_hip_joint':35.0,
                        '.*_thigh_joint':40.0,
                        '.*_calf_joint':40.0,
                        },
            saturation_effort={
                        '.*_hip_joint':35.0,
                        '.*_thigh_joint':45.0,
                        '.*_calf_joint':45.0,
                        },
            velocity_limit={
                        '.*_hip_joint':52.4,
                        '.*_thigh_joint':30.1,
                        '.*_calf_joint':30.1,
                        },
            stiffness=40.0,
            damping=1.0,
            friction=0.0,
        )

## we are now using a raycaster based camera, not a pinhole camera. see tail issue https://github.com/isaac-sim/IsaacLab/issues/719
CAMERA_CFG = RayCasterCameraCfg( 
    prim_path= '{ENV_REGEX_NS}/Robot/base',
    data_types=["distance_to_camera"],
    offset=RayCasterCameraCfg.OffsetCfg(
        pos=(0.33, 0.0, 0.08), 
        rot=quat_from_euler_xyz_tuple(*tuple(torch.deg2rad(torch.tensor([180,70,-90])))), 
        convention="ros"
        ),
    depth_clipping_behavior = 'max',
    pattern_cfg = PinholeCameraPatternCfg(
        focal_length=11.041, 
        horizontal_aperture=20.955,
        vertical_aperture = 12.240,
        height=60,
        width=106,
    ),
    mesh_prim_paths=["/World/ground"],
    max_distance = 2.,
)

CAMERA_USD_CFG = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base/d435",
    spawn=sim_utils.UsdFileCfg(usd_path=os.path.join(agents.__path__[0],'d435.usd')),
    init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.33, 0.0, 0.08), 
            rot=quat_from_euler_xyz_tuple(*tuple(torch.deg2rad(torch.tensor([180,90,-90]))))
    )
)
VIEWER = ViewerCfg(
    eye=(-2.0, 2.6, 1.8),
    lookat=(0.0, 0.0, 0.8),
    origin_type="world",
    asset_name=None,
)


def apply_local_visualizers(env_cfg):
    """Override debug visualizer assets (arrows) to use local USDs if available."""
    repo_root = Path(__file__).resolve().parents[2]
    arrow_usd = (
        repo_root / "assets" / "nucleus" / "Isaac" / "4.5" / "Isaac" / "Props" / "UIElements" / "arrow_x.usd"
    )
    if not arrow_usd.exists():
        return
    arrow_path = arrow_usd.as_posix()

    def _patch(markers_cfg):
        if markers_cfg and hasattr(markers_cfg, "markers") and "arrow" in markers_cfg.markers:
            markers_cfg.markers["arrow"].usd_path = arrow_path

    commands_cfg = getattr(env_cfg, "commands", None)
    if commands_cfg:
        base_cmd = getattr(commands_cfg, "base_velocity", None)
        if base_cmd:
            _patch(getattr(base_cmd, "goal_vel_visualizer_cfg", None))
            _patch(getattr(base_cmd, "current_vel_visualizer_cfg", None))

    parkours_cfg = getattr(env_cfg, "parkours", None)
    if parkours_cfg:
        base_parkour = getattr(parkours_cfg, "base_parkour", None)
        if base_parkour:
            for attr in ("future_arrow_visualizer_cfg", "current_arrow_visualizer_cfg"):
                _patch(getattr(base_parkour, attr, None))
