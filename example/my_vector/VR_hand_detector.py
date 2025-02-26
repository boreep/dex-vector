# -*- coding: utf-8 -*-
import numpy as np
import zmq
from scipy.spatial.transform import Rotation

##最终JointPos的坐标左右手统一为：z轴正方向由手腕到中指，x轴正方向由手背指向手掌
#对左手：y轴正方向为从食指到小指横向，对右手：y轴正方向为从小指到食指横向
OPERATOR2MANO_RIGHT = np.array(
    [
        [0, 0, 1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)#沿着y轴旋转90度

OPERATOR2MANO_LEFT = np.array(
    [
        [0, 0, -1],
        [1, 0, 0],
        [0, 1, 0],
    ]
)#沿着y轴旋转-90度
rows=84
cols=7

class VRHandDetector:

    def __init__(self, address="tcp://172.17.129.195:12346"):

        self.address = address
        self.context=zmq.Context()
        self.socket=self.context.socket(zmq.SUB)
        self.socket.connect(self.address)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, '') 
        print(f"已准备接受消息来自： {self.address}")


    def recv_origin_array(self):

        message = self.socket.recv()
        # 将接收到的消息转换为 numpy 数组
        array_data = np.frombuffer(message, dtype=np.float32) # 假设 float32 类型
        float_matrix = array_data.reshape(rows, cols) 
        #打印出来接受到数据大小
        print(f"接收到数据维度：{float_matrix.shape}")

        return float_matrix   #维度为[84,7],7代表三维坐标和四元数

       
    @staticmethod
    def detect(self,left_hand_data,right_hand_data): #传入维度为两个[26,7]
        if left_hand_data is None or left_hand_data.size == 0:

            return None, None, None
        left_keypoint_3d=left_hand_data[:,:3]
        right_keypoint_3d=right_hand_data[:,:3]

        left_keypoint_3d = left_keypoint_3d - left_keypoint_3d[1:2, :]
        right_keypoint_3d = right_keypoint_3d - right_keypoint_3d[1:2, :]


        left_mediapipe_palm_rot=Rotation.from_quat(left_hand_data[0,3:7]).as_matrix()
        left_joint_pos =  left_keypoint_3d @ left_mediapipe_palm_rot @OPERATOR2MANO_LEFT#维度为[26:3]

        right_mediapipe_palm_rot=Rotation.from_quat(right_hand_data[0,3:7]).as_matrix()
        right_joint_pos =  right_keypoint_3d @ right_mediapipe_palm_rot @OPERATOR2MANO_RIGHT#维度为[26:3]


        left_keypoint_2d = left_joint_pos[:, 1:3]

        return left_joint_pos, right_joint_pos, left_keypoint_2d,

    @staticmethod
    def left_hand_detect(self,left_hand_data): #传入维度为[26,7]
        if left_hand_data is None :

            return None, None
        left_keypoint_3d=left_hand_data[:,:3]

        left_keypoint_3d = left_keypoint_3d - left_keypoint_3d[1:2, :]

        left_mediapipe_palm_rot=Rotation.from_quat(left_hand_data[0,3:7]).as_matrix()
        left_joint_pos =  left_keypoint_3d @ left_mediapipe_palm_rot @OPERATOR2MANO_LEFT#维度为[26:3]

        left_keypoint_2d = left_joint_pos[:, 1:3]

        return left_joint_pos,  left_keypoint_2d,



#用于求手掌平面的旋转矩阵，在此处不需要使用
    # @staticmethod
    # def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
    #     """
    #     Compute the 3D coordinate frame (orientation only) from detected 3d key points
    #     :param points: keypoint3 detected from MediaPipe detector. Order: [wrist, index, middle, pinky]
    #     :return: the coordinate frame of wrist in MANO convention
    #     """
    #     assert keypoint_3d_array.shape == (26, 3)
    #     points = keypoint_3d_array[[1, 7, 12], :]

    #     # Compute vector from palm to the first joint of middle finger
    #     x_vector = points[0] - points[2]

    #     # Normal fitting with SVD
    #     points = points - np.mean(points, axis=0, keepdims=True)
    #     u, s, v = np.linalg.svd(points)

    #     normal = v[2, :]

    #     # Gram–Schmidt Orthonormalize
    #     x = x_vector - np.sum(x_vector * normal) * normal
    #     x = x / np.linalg.norm(x)
    #     z = np.cross(x, normal)

    #     # We assume that the vector from pinky to index is similar the z axis in MANO convention
    #     if np.sum(z * (points[1] - points[2])) < 0:
    #         normal *= -1
    #         z *= -1
    #     frame = np.stack([x, normal, z], axis=1)
    #     return frame