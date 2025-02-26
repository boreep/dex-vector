import enum
from pathlib import Path
from typing import Optional

import numpy as np


#用于VR
OPERATOR2MANO_RIGHT = np.array(
    [
        [0, 0, 1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)

OPERATOR2MANO_LEFT = np.array(
    [
        [0, 0, -1],
        [1, 0, 0],
        [0, 1, 0],
    ]
)
#原始
# OPERATOR2MANO_RIGHT = np.array(
#     [
#         [0, 0, -1],
#         [1, 0, 0],
#         [0, 1, 0],
#     ]
# )

# OPERATOR2MANO_LEFT = np.array(
#     [
#         [0, 0, 1],
#         [-1, 0, 0],
#         [0, -1, 0],
#     ]
# )


class RobotName(enum.Enum):
    inspire = enum.auto()
    roh=enum.auto()



class RetargetingType(enum.Enum):
    vector = enum.auto()  # For teleoperation, no finger closing prior



class HandType(enum.Enum):
    right = enum.auto()
    left = enum.auto()


ROBOT_NAME_MAP = {
    RobotName.inspire: "inspire_hand",
    RobotName.roh: "roh_hand",
}

ROBOT_NAMES = list(ROBOT_NAME_MAP.keys())


def get_default_config_path(
    robot_name: RobotName, retargeting_type: RetargetingType, hand_type: HandType
) -> Optional[Path]:
    config_path = Path(__file__).parent / "configs"

    robot_name_str = ROBOT_NAME_MAP[robot_name]
    hand_type_str = hand_type.name
    config_name = f"{robot_name_str}_{hand_type_str}.yml"
    return config_path / config_name


OPERATOR2MANO = {
    HandType.right: OPERATOR2MANO_RIGHT,
    HandType.left: OPERATOR2MANO_LEFT,
}
