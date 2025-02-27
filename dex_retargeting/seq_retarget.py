import time
from typing import Optional

import numpy as np
from pytransform3d import rotations

from dex_retargeting.constants import OPERATOR2MANO, HandType
from dex_retargeting.optimizer import Optimizer
from dex_retargeting.optimizer_utils import LPFilter


class SeqRetargeting:
    def __init__(
        self,
        optimizer: Optimizer,
        has_joint_limits=True,
        lp_filter: Optional[LPFilter] = None,
    ):
        self.optimizer = optimizer
        robot = self.optimizer.robot

        # Joint limit
        self.has_joint_limits = has_joint_limits
        joint_limits = np.ones_like(robot.joint_limits)
        joint_limits[:, 0] = -1e4  #设置了极大范围，相当于无约束
        joint_limits[:, 1] = 1e4
        if has_joint_limits:
            joint_limits[:] = robot.joint_limits[:]
            self.optimizer.set_joint_limit(joint_limits[self.optimizer.idx_pin2target])
        self.joint_limits = joint_limits[self.optimizer.idx_pin2target]

        # Temporal information
        #记录上一帧的关节角度，用于优化初始值。
        self.last_qpos = joint_limits.mean(1)[self.optimizer.idx_pin2target].astype(np.float32)
        self.accumulated_time = 0
        self.num_retargeting = 0

        # Filter
        self.filter = lp_filter

        # Warm started
        self.is_warm_started = False

#vector模式不会用到这个函数
    def warm_start(
        self,
        wrist_pos: np.ndarray,
        wrist_quat: np.ndarray,
        hand_type: HandType = HandType.right,
        is_mano_convention: bool = False,
    ):
        """
        直接初始化手腕的位姿，跳过优化计算，使得机器人手部快速进入合理的初始状态。

        Initialize the wrist joint pose using analytical computation instead of retargeting optimization.
        这个函数特别用于position重定向模式下的带浮动关节的6d自由手, i.e. has 6D free joint
        You are not expected to use this function for vector retargeting, e.g. when you are working on teleoperation

        Args:
            wrist_pos: position of the hand wrist, typically from human hand pose
            wrist_quat: quaternion of the hand wrist, the same convention as the operator frame definition if not is_mano_convention
            hand_type: hand type, used to determine the operator2mano matrix
            is_mano_convention: whether the wrist_quat is in mano convention
        """
        # This function can only be used when the first joints of robot are free joints

        if len(wrist_pos) != 3:
            raise ValueError(f"Wrist pos: {wrist_pos} is not a 3-dim vector.")
        if len(wrist_quat) != 4:
            raise ValueError(f"Wrist quat: {wrist_quat} is not a 4-dim vector.")

        operator2mano = OPERATOR2MANO[hand_type] if is_mano_convention else np.eye(3)
        robot = self.optimizer.robot
        target_wrist_pose = np.eye(4)
        target_wrist_pose[:3, :3] = rotations.matrix_from_quaternion(wrist_quat) @ operator2mano.T
        target_wrist_pose[:3, 3] = wrist_pos

        name_list = [
            "dummy_x_translation_joint",
            "dummy_y_translation_joint",
            "dummy_z_translation_joint",
            "dummy_x_rotation_joint",
            "dummy_y_rotation_joint",
            "dummy_z_rotation_joint",
        ]
        wrist_link_id = robot.get_joint_parent_child_frames(name_list[5])[1]

        # Set the dummy joints angles to zero
        old_qpos = robot.q0
        new_qpos = old_qpos.copy()
        for num, joint_name in enumerate(self.optimizer.target_joint_names):
            if joint_name in name_list:
                new_qpos[num] = 0

        robot.compute_forward_kinematics(new_qpos)
        root2wrist = robot.get_link_pose_inv(wrist_link_id)
        target_root_pose = target_wrist_pose @ root2wrist

        euler = rotations.euler_from_matrix(target_root_pose[:3, :3], 0, 1, 2, extrinsic=False)
        pose_vec = np.concatenate([target_root_pose[:3, 3], euler])

        # Find the dummy joints
        for num, joint_name in enumerate(self.optimizer.target_joint_names):
            if joint_name in name_list:
                index = name_list.index(joint_name)
                self.last_qpos[num] = pose_vec[index]

        self.is_warm_started = True

    def retarget(self, ref_value, fixed_qpos=np.array([])):
        """
        对参考值进行重定位，以计算机器人的关节位置。

        此函数的目的是根据给定的参考值和固定的关节位置，计算出机器人的关节位置。
        它通过优化过程实现这一点，并且可以应用额外的适配器和滤波器来调整最终的关节位置。

        参数:
        - ref_value: 参考值，用于计算目标关节位置。
        - fixed_qpos: 固定的关节位置数组，默认为空数组。这些位置不会被优化器改变。

        返回:
        - robot_qpos: 计算得到的机器人关节位置数组。
        """
        # 记录函数开始执行的时间
        tic = time.perf_counter()

        # 执行重定位计算，考虑参考值、固定关节位置和上一次的关节位置
        qpos = self.optimizer.retarget(
            ref_value=ref_value.astype(np.float32),
            fixed_qpos=fixed_qpos.astype(np.float32),
            last_qpos=np.clip(self.last_qpos, self.joint_limits[:, 0], self.joint_limits[:, 1]),
            #last_qpos 作为优化的初始值，但会被 clip() 限制在关节范围内。
        )

        # 累加时间差，用于跟踪总执行时间
        self.accumulated_time += time.perf_counter() - tic
        # 增加重定位次数计数
        self.num_retargeting += 1
        # 更新上一次的关节位置
        self.last_qpos = qpos

        # 初始化机器人的关节位置数组
        robot_qpos = np.zeros(self.optimizer.robot.dof)
        # 将固定的关节位置赋值给机器人关节位置数组中的相应索引
        robot_qpos[self.optimizer.idx_pin2fixed] = fixed_qpos
        # 将计算得到的关节位置赋值给机器人关节位置数组中的相应索引
        robot_qpos[self.optimizer.idx_pin2target] = qpos

        # 如果存在adaptor，应用adaptor调整关节位置
        if self.optimizer.adaptor is not None:
            robot_qpos = self.optimizer.adaptor.forward_qpos(robot_qpos)

        # 如果存在滤波器，应用滤波器处理关节位置
        if self.filter is not None:
            robot_qpos = self.filter.next(robot_qpos)
        
        # 返回最终的机器人关节位置
        return robot_qpos

    def set_qpos(self, robot_qpos: np.ndarray):
        """
        直接设置last_qpos，通常用于外部同步机器人状态

        本函数通过选择特定索引的关节位置值来更新优化器的目标关节位置.

        参数:
        - robot_qpos (np.ndarray): 机器人当前的关节位置数组.

        返回:
        无返回值，但会更新实例变量 last_qpos 来存储当前的目标关节位置.
        """
        # 通过优化器的索引将机器人关节位置映射到目标关节位置
        target_qpos = robot_qpos[self.optimizer.idx_pin2target]
        
        # 更新最后的目标关节位置
        self.last_qpos = target_qpos

    def get_qpos(self, fixed_qpos: Optional[np.ndarray] = None):
        """
        获取当前的机器人关节位置。即返回机器人的关节位置数组。

        该函数计算并返回机器人的关节位置。如果提供了fixed_qpos参数，
        则将机器人的部分关节位置固定为该参数指定的值。

        参数:
        - fixed_qpos (Optional[np.ndarray]): 可选参数，指定要固定的关节位置数组。
            如果提供，函数将确保这些关节位置被设置为指定的值。

        返回:
        - robot_qpos (np.ndarray): 当前机器人的关节位置数组。
        """
        # 初始化机器人关节位置数组，长度为机器人的自由度数量
        robot_qpos = np.zeros(self.optimizer.robot.dof)

        # 将上一次的关节位置映射到目标位置索引上
        robot_qpos[self.optimizer.idx_pin2target] = self.last_qpos

        # 如果提供了固定的关节位置，则将其映射到对应的固定位置索引上
        if fixed_qpos is not None:
            robot_qpos[self.optimizer.idx_pin2fixed] = fixed_qpos

        # 返回计算得到的机器人关节位置数组
        return robot_qpos

    def verbose(self):
        #记录重定向次数、累计时间、最后的优化距离。
        min_value = self.optimizer.opt.last_optimum_value()
        print(f"Retargeting {self.num_retargeting} times takes: {self.accumulated_time}s")
        print(f"Last distance: {min_value}")

    def reset(self):
        #清空 last_qpos 并重置计数器。
        self.last_qpos = self.joint_limits.mean(1).astype(np.float32)
        self.num_retargeting = 0
        self.accumulated_time = 0

    @property
    def joint_names(self):
        return self.optimizer.robot.dof_joint_names
