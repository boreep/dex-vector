from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from typing import Union

import numpy as np
import yaml
import os

from dex_retargeting import yourdfpy as urdf
from dex_retargeting.kinematics_adaptor import MimicJointKinematicAdaptor,Mimic3JointKinematicAdaptor
from dex_retargeting.optimizer_utils import LPFilter
from dex_retargeting.robot_wrapper import RobotWrapper
from dex_retargeting.seq_retarget import SeqRetargeting
from dex_retargeting.yourdfpy import DUMMY_JOINT_NAMES


@dataclass
class RetargetingConfig:
    type: str
    urdf_path: str

    add_dummy_free_joint: bool = False  # 是否在机器人根部添加自由关节，使其可以自由移动
    target_link_human_indices: Optional[np.ndarray] = None  # 目标链接对应的人手关节索引
    wrist_link_name: Optional[str] = None  # 机器人手腕对应的关节
    target_link_names: Optional[List[str]] = None  # 位置映射的目标链接名称
    target_joint_names: Optional[List[str]] = None  # 需要优化的机器人关节名称
    target_origin_link_names: Optional[List[str]] = None  # 向量映射的起始链接
    target_task_link_names: Optional[List[str]] = None  # 向量映射的目标链接
    finger_tip_link_names: Optional[List[str]] = None  # DexPilot 方法中的手指末端链接
    scaling_factor: float = 1.0  # 机器人手相对于人手的缩放因子


    normal_delta: float = 4e-3  # 正常误差项
    huber_delta: float = 2e-2  # Huber 误差项（鲁棒性优化）
    project_dist: float = 0.03  # DexPilot 方法的投影距离参数
    escape_dist: float = 0.05  # DexPilot 方法的逃逸距离参数
    has_joint_limits: bool = True  # 是否考虑机器人关节的限制
    ignore_mimic_joint: bool = False  # 是否忽略 Mimic Joint 关节
    low_pass_alpha: float = 0  # 低通滤波参数


    _TYPE = ["vector", "position", "dexpilot"]
    _DEFAULT_URDF_DIR = "./"


    def __post_init__(self):
        #用于参数合法性检查
        # Retargeting type check
        self.type = self.type.lower()
        if self.type not in self._TYPE:
            raise ValueError(f"Retargeting type must be one of {self._TYPE}")

        # Vector retargeting requires: target_origin_link_names + target_task_link_names
        # Position retargeting requires: target_link_names
        if self.type == "vector":
            if self.target_origin_link_names is None or self.target_task_link_names is None:
                raise ValueError(f"Vector retargeting requires: target_origin_link_names + target_task_link_names")
            if len(self.target_task_link_names) != len(self.target_origin_link_names):
                raise ValueError(f"Vector retargeting origin and task links dim mismatch")
            if self.target_link_human_indices.shape != (2, len(self.target_origin_link_names)):
                raise ValueError(f"Vector retargeting link names and link indices dim mismatch")
            if self.target_link_human_indices is None:
                raise ValueError(f"Vector retargeting requires: target_link_human_indices")


        # URDF path check
        urdf_path = Path(self.urdf_path)
        if not urdf_path.is_absolute():
            urdf_path = self._DEFAULT_URDF_DIR / urdf_path
            urdf_path = urdf_path.absolute()
        if not urdf_path.exists():
            raise ValueError(f"URDF path {urdf_path} does not exist")
        self.urdf_path = str(urdf_path)

#本函数用于指定存储URDF文件的默认目录路径。
    @classmethod
    def set_default_urdf_dir(cls, urdf_dir: Union[str, Path]):
        path = Path(urdf_dir)
        if not path.exists():
            raise ValueError(f"URDF dir {urdf_dir} not exists.")
        cls._DEFAULT_URDF_DIR = urdf_dir

    @classmethod
    def load_from_file(cls, config_path: Union[str, Path], override: Optional[Dict] = None):
        path = Path(config_path)
        if not path.is_absolute():
            path = path.absolute()

        with path.open("r") as f:
            yaml_config = yaml.load(f, Loader=yaml.FullLoader)
            cfg = yaml_config["retargeting"]
            return cls.from_dict(cfg, override)

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any], override: Optional[Dict] = None):
        if "target_link_human_indices" in cfg:
            cfg["target_link_human_indices"] = np.array(cfg["target_link_human_indices"])
        if override is not None:
            for key, value in override.items():
                cfg[key] = value
        config = RetargetingConfig(**cfg)
        return config

    def build(self) -> SeqRetargeting:

        # 加载和处理 URDF 模型（包括添加虚拟关节）。
        # 使用 Pinocchio 模型支持运动学计算。
        # 根据配置选择并初始化合适的优化器。
        # 根据需求启用 Mimic Joint 支持。
        # 配置低通滤波器以平滑结果。
        # 构建一个完整的重定向对象。
        
        from dex_retargeting.optimizer import (
            VectorOptimizer,

        )
        import tempfile

        # Process the URDF with yourdfpy to better find file path
        robot_urdf = urdf.URDF.load(
            self.urdf_path, add_dummy_free_joints=self.add_dummy_free_joint, build_scene_graph=False
        )
        urdf_name = self.urdf_path.split(os.path.sep)[-1]
        temp_dir = tempfile.mkdtemp(prefix="dex_retargeting-")
        temp_path = f"{temp_dir}/{urdf_name}"
        robot_urdf.write_xml_file(temp_path)

        # 加载pinocchio模型
        robot = RobotWrapper(temp_path)

        # Add 6D dummy joint to target joint names so that it will also be optimized
        if self.add_dummy_free_joint and self.target_joint_names is not None:
            self.target_joint_names = DUMMY_JOINT_NAMES + self.target_joint_names
        joint_names = self.target_joint_names if self.target_joint_names is not None else robot.dof_joint_names

        if self.type == "vector":

            optimizer = VectorOptimizer(
                robot,
                joint_names,
                target_origin_link_names=self.target_origin_link_names,
                target_task_link_names=self.target_task_link_names,
                target_link_human_indices=self.target_link_human_indices,
                scaling=self.scaling_factor,
                norm_delta=self.normal_delta,
                huber_delta=self.huber_delta,
            )

        else:
            raise RuntimeError()

        if 0 <= self.low_pass_alpha <= 1:
            lp_filter = LPFilter(self.low_pass_alpha)
        else:
            lp_filter = None

        # 在这里支持了被动关节(某些关节间存在固定比例的机械耦合关系) 机械耦合关节
        has_mimic_joints, source_names, mimic_names, multipliers, offsets ,multipliers1, multipliers2, multipliers3 = parse_mimic_joint(robot_urdf)
        
        if has_mimic_joints and not self.ignore_mimic_joint:
            if len(multipliers3)==0:
                print(f"has_mimic3_joints: {len(multipliers3)}")  
                print("选择线性的MimicJointKinematicAdaptor")
                adaptor = MimicJointKinematicAdaptor(
                    robot,
                    target_joint_names=joint_names,
                    source_joint_names=source_names,
                    mimic_joint_names=mimic_names,
                    multipliers=multipliers,
                    offsets=offsets,
                )
                optimizer.set_kinematic_adaptor(adaptor)
                print(
                    "\033[34m",
                    "Mimic joint adaptor enabled. The mimic joint tags in the URDF will be considered during retargeting.\n"
                    "To disable mimic joint adaptor, consider setting ignore_mimic_joint=True in the configuration.",
                    "\033[39m",
                )
        
            else:
                print("选择三阶的MimicJointKinematicAdaptor")
                adaptor = Mimic3JointKinematicAdaptor(
                    robot,
                    target_joint_names=joint_names,
                    source_joint_names=source_names,
                    mimic_joint_names=mimic_names,
                    multipliers1=multipliers1,
                    multipliers2=multipliers2,
                    multipliers3=multipliers3,
                    offsets=offsets,
                )
                optimizer.set_kinematic_adaptor(adaptor)
                print(
                    "\033[34m",
                    "Mimic3 joint adaptor enabled. The mimic3 joint tags in the URDF will be considered during retargeting.\n"
                    "To disable mimic joint adaptor, consider setting ignore_mimic_joint=True in the configuration.",
                    "\033[39m",
                )
            
        retargeting = SeqRetargeting(
            optimizer,
            has_joint_limits=self.has_joint_limits,
            lp_filter=lp_filter,
        )
        return retargeting


def get_retargeting_config(config_path: Union[str, Path]) -> RetargetingConfig:
    config = RetargetingConfig.load_from_file(config_path)
    return config

def parse_mimic_joint(robot_urdf: urdf.URDF) -> Tuple[bool, List[str], List[str], List[float], List[float], List[float], List[float], List[float]]:
    mimic_joint_names = []
    source_joint_names = []
    multipliers = []
    multipliers1 = []
    multipliers2=[]
    multipliers3=[]
    offsets = []
    for name, joint in robot_urdf.joint_map.items():
        if joint.mimic is not None:
            mimic_joint_names.append(name)
            source_joint_names.append(joint.mimic.joint)
            multipliers.append(joint.mimic.multiplier)
            offsets.append(joint.mimic.offset)
        if hasattr(joint.mimic, 'multiplier3'):
            multipliers1.append(joint.mimic.multiplier1)
            multipliers2.append(joint.mimic.multiplier2)
            multipliers3.append(joint.mimic.multiplier3)

    return len(mimic_joint_names) > 0, source_joint_names, mimic_joint_names, multipliers, offsets, multipliers1, multipliers2, multipliers3

