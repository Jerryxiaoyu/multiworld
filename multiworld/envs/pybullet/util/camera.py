import numpy  as np
import math

import os
import json
#import torch
import transformations
import pybullet as p

def intrinsic_to_projection_matrix(intrinsics, height, width, near, far,
                                   upside_down=True):
    """Convert the camera intrinsics to the projection matrix.

    Takes a Hartley-Zisserman intrinsic matrix and returns a Bullet/OpenGL
    style projection matrix. We pad with zeros on right and bottom and a 1
    in the corner.

    Uses algorithm found at:
    https://strawlab.org/2011/11/05/augmented-reality-with-OpenGL#note-about-image-coordinates

    Args:
        intrinsics: The camera intrinsincs matrix.
        height: The image height.
        width: The image width.
        near: The distance to the near plane.
        far: The distance to the far plane.

    Returns:
        projection_matrix: The projection matrix.
    """
    projection_matrix = np.empty((4, 4), dtype=np.float32)

    f_x = intrinsics[0, 0]
    f_y = intrinsics[1, 1]
    x_0 = intrinsics[0, 2]
    y_0 = intrinsics[1, 2]
    s = intrinsics[0, 1]

    if upside_down:
        x_0 = width - x_0
        y_0 = height - y_0

    projection_matrix[0, 0] = 2 * f_x / width
    projection_matrix[0, 1] = -2 * s / width
    projection_matrix[0, 2] = (width - 2 * x_0) / width
    projection_matrix[0, 3] = 0

    projection_matrix[1, 0] = 0
    projection_matrix[1, 1] = 2 * f_y / height
    projection_matrix[1, 2] = (-height + 2 * y_0) / height
    projection_matrix[1, 3] = 0

    projection_matrix[2, 0] = 0
    projection_matrix[2, 1] = 0
    projection_matrix[2, 2] = (-far - near) / (far - near)
    projection_matrix[2, 3] = -2 * far * near / (far - near)

    projection_matrix[3, 0] = 0
    projection_matrix[3, 1] = 0
    projection_matrix[3, 2] = -1
    projection_matrix[3, 3] = 0

    projection_matrix = list(projection_matrix.transpose().flatten())

    return projection_matrix


def extrinsic_to_view_matrix(translation, rotation, distance):
    """Convert the camera extrinsics to the view matrix.

    The function takes HZ-style rotation matrix R and translation matrix t
    and converts them to a Bullet/OpenGL style view matrix. the derivation
    is pretty simple if you consider x_camera = R * x_world + t.

    Args:
        distance: The distance from the camera to the focus.

    Returns:
        view_matrix: The view matrix.
    """
    # The camera position in the world frame.
    camera_position = rotation.T.dot(-translation)

    # The focus in the world frame.
    focus = rotation.T.dot(np.array([0, 0, distance]) - translation)

    # The up vector is the Y-axis of the camera in the world frame.
    up_vector = rotation.T.dot(np.array([0, 1, 0]))

    # Compute the view matrix.
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=camera_position,
        cameraTargetPosition=focus,
        cameraUpVector=up_vector)

    return view_matrix

class pybullet_camera(object):
    def __init__(self, pybullet_server, camera_params, save_path=None):
        self._p = pybullet_server
        self.camTarget = camera_params['target_pos']
        self.dist =  camera_params['distance']
        self.yaw =  camera_params['yaw']
        self.pitch =  camera_params['pitch']
        self.roll =  camera_params['roll']
        self.fov =  camera_params['fov']
        self.nearPlane =  camera_params['near']
        self.farPlane =  camera_params['far']
        self.width = camera_params['image_width']
        self.height = camera_params['image_height']


        self._init_camera()

        self.save_path = save_path

        #print('view mat:', self.viewMatrix)

        # self.save_camera_info()

    def _init_camera(self):
        assert self.roll == 0
        self.aspect = float(self.width) / float(self.height)

        self.view_matrix_list = self._p.computeViewMatrixFromYawPitchRoll(
                                cameraTargetPosition=self.camTarget,
                                distance=self.dist,
                                yaw=self.yaw,
                                pitch=self.pitch,
                                roll=self.roll,
                                upAxisIndex=2)

        self.proj_matrix_list = self._p.computeProjectionMatrixFOV(
                                fov=self.fov, aspect=self.aspect,
                                nearVal=self.nearPlane, farVal=self.farPlane)

        self._camera_info()
        self.horizon, self.vertical = self._cal_hor_ver()
        self.projMatrix = np.array(self.proj_matrix_list).reshape(4, 4).transpose()
        self.viewMatrix = np.array(self.view_matrix_list).reshape(4, 4).transpose()

    def _camera_info(self):
        assert self.dist > 0

        upAxis = 2
        eyePos = np.zeros(3)

        yawRad = self.yaw * 0.01745329251994329547  # rad per anger
        pitchRad = self.pitch * 0.01745329251994329547
        rollRad = 0.0

        if upAxis == 1:
            forwardAxis = 2
            camUpVector = np.array([0, 1, 0])
            eyeRot = transformations.euler_matrix(rollRad, -pitchRad, yawRad, 'syxz')[0:3,0:3]
        elif upAxis == 2:
            forwardAxis = 1
            camUpVector = np.array([0, 0, 1])
            eyeRot = transformations.euler_matrix(rollRad, pitchRad, yawRad, 'syxz')[0:3,0:3]
        else:
            raise NotImplementedError

        eyePos[forwardAxis] = -self.dist
        eyePos = eyeRot.dot(eyePos)
        self.camUpVector = eyeRot.dot(camUpVector)
        self.camPos = eyePos + self.camTarget

        #print("camPos", self.camPos)
        #print("camTargetPos", self.camTarget)
        #print('camUpVector', self.camUpVector)

        z_axis = self.camPos - self.camTarget
        z_axis = z_axis/np.linalg.norm(z_axis)

        y_axis = self.camUpVector/np.linalg.norm(self.camUpVector)
        x_axis = np.cross(y_axis, z_axis)

        self.cam_Matrix = np.eye(4)
        self.cam_Matrix[0:3, 0] = x_axis
        self.cam_Matrix[0:3, 1] = y_axis
        self.cam_Matrix[0:3, 2] = z_axis
        self.cam_Matrix[0:3,3] = self.camPos

        self.camRot = self.cam_Matrix[0:3, 0:3]

        #print('camMat:', self.cam_Matrix)

    def _cal_hor_ver(self):
        # rayFrom = self.camPos
        rayForward = np.array(self.camTarget) - np.array(self.camPos)
        rayForward = rayForward / np.linalg.norm(rayForward)
        rayForward *= self.farPlane

        vertical = self.camUpVector
        horizon = np.cross(rayForward, vertical)
        horizon = horizon / np.linalg.norm(horizon)
        vertical = np.cross(horizon, rayForward)
        vertical = vertical / np.linalg.norm(vertical)
        tanfov = math.tan(0.5 * self.fov / 180.0 * math.pi)

        horizon *= 2. * self.farPlane * tanfov
        vertical *= 2. * self.farPlane * tanfov
        aspect = float(self.width) / float(self.height)
        horizon *= aspect

        return horizon, vertical

    def save_camera_info(self):
        params = {'fov': self.fov,
                  'project_mat': self.proj_matrix.tolist(),
                  'width': self.width,
                  'height': self.height,
                  'nearPlane': self.nearPlane,
                  'farPlane': self.farPlane,
                  'horizon': self.horizon.tolist(),
                  'vertical': self.vertical.tolist()
                  }
        if self.save_path is None:
            self.save_path = 'camera_info.json'
        else:
            self.save_path = os.path.join(self.save_path, 'camera_info.json')
        with open(self.save_path, 'w') as f:
            json.dump(params, f)

    def getRayFromTo(self, pixelX, pixelY):
        """
        X: width
        Y: height
        """
        rayForward = [(self.camTarget[0] - self.camPos[0]), (self.camTarget[1] - self.camPos[1]),
                      (self.camTarget[2] - self.camPos[2])]
        lenFwd = math.sqrt(
            rayForward[0] * rayForward[0] + rayForward[1] * rayForward[1] + rayForward[2] * rayForward[2])
        invLen = self.farPlane * 1. / lenFwd
        rayForward = [invLen * rayForward[0], invLen * rayForward[1], invLen * rayForward[2]]
        rayFrom = self.camPos
        oneOverWidth = float(1) / float(self.width)
        oneOverHeight = float(1) / float(self.height)

        dHor = [self.horizon[0] * oneOverWidth, self.horizon[1] * oneOverWidth, self.horizon[2] * oneOverWidth]
        dVer = [self.vertical[0] * oneOverHeight, self.vertical[1] * oneOverHeight, self.vertical[2] * oneOverHeight]
        rayToCenter = [rayFrom[0] + rayForward[0], rayFrom[1] + rayForward[1], rayFrom[2] + rayForward[2]]
        ortho = [- 0.5 * self.horizon[0] + 0.5 * self.vertical[0] + float(pixelX) * dHor[0] - float(pixelY) * dVer[0],
                 - 0.5 * self.horizon[1] + 0.5 * self.vertical[1] + float(pixelX) * dHor[1] - float(pixelY) * dVer[1],
                 - 0.5 * self.horizon[2] + 0.5 * self.vertical[2] + float(pixelX) * dHor[2] - float(pixelY) * dVer[2]]

        rayTo = [rayFrom[0] + rayForward[0] + ortho[0],
                 rayFrom[1] + rayForward[1] + ortho[1],
                 rayFrom[2] + rayForward[2] + ortho[2]]

        lenOrtho = math.sqrt(ortho[0] * ortho[0] + ortho[1] * ortho[1] + ortho[2] * ortho[2])
        alpha = math.atan(lenOrtho / self.farPlane)
        # print(alpha)

        return rayFrom, rayTo, alpha

    def get_depth_image_from_buffer(self, detph_buffer):
        """
        calculate the real detph value from raw depthbuffers ( read from pybullet API)
        ref to : https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#
        :param detph_buffer: array shape (h, w, 1),or (h, w) or (b, h,w,1),  map [0,1] to real depth
        :return:
        """
        depth_img = self.farPlane * self.nearPlane / (self.farPlane - (self.farPlane - self.nearPlane) * detph_buffer)
        return depth_img

    def get_depth_image_from_buffer2(self, detph_buffer):
        """
        calculate the real detph value from raw depthbuffers ( read from pybullet API)
        ref to : https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#
        :param detph_buffer: array shape (h, w, 1),or (h, w) or (b, h,w,1),  map [0,1] to real depth
        :return:
        """

        z_b = detph_buffer
        z_n = 2.0 * z_b - 1.0
        z_e = (2.0 * self.nearPlane * self.farPlane /
               (self.farPlane + self.nearPlane - z_n * (self.farPlane - self.nearPlane)))
        depth = z_e

        return depth


    def cal_pixel_3d_in_world(self, w, h, depthbuffer_value):
        """
        given a pair of pixels in an image, return the 3D coordinate in the world
        :param h:
        :param w:
        :param depthbuffer_value:
        :return:
        """
        depthImg = depthbuffer_value

        rayFrom, rayTo, alpha = self.getRayFromTo(w, h)
        rf = np.array(rayFrom)
        rt = np.array(rayTo)
        vec = rt - rf
        l = np.sqrt(np.dot(vec, vec))

        depth = self.farPlane * self.nearPlane / (self.farPlane - (self.farPlane - self.nearPlane) * depthImg)
        depth_r = depth / math.cos(alpha)
        newTo = (depth_r / l) * vec + rf

        return newTo

    def cal_3D_in_world_from_PixelAndDepth(self, w, h, depth):
        """
        given a pair of pixels in an image, return the 3D coordinate in the world
        :param h:
        :param w:
        :param depth:
        :return:np.array (3, )
        """

        rayFrom, rayTo, alpha = self.getRayFromTo(w, h)
        rf = np.array(rayFrom)
        rt = np.array(rayTo)
        vec = rt - rf
        l = np.sqrt(np.dot(vec, vec))

        depth_r = depth / math.cos(alpha)
        newTo = (depth_r / l) * vec + rf

        return newTo

    def cal_3D_in_world_from_batchPixelAndDepth(self, u_vec, v_vec, depth_vec):
        """
        given a pair of pixels in an image, return the 3D coordinate in the world
        :param u_vec: W pix coor [n, ]
        :param v_vec: H pix coor [n, ]
        :param depth:
        :return: [3, n]
        """

        rayForward = [(self.camTarget[0] - self.camPos[0]), (self.camTarget[1] - self.camPos[1]),
                      (self.camTarget[2] - self.camPos[2])]
        lenFwd = math.sqrt(
            rayForward[0] * rayForward[0] + rayForward[1] * rayForward[1] + rayForward[2] * rayForward[2])
        invLen = self.farPlane * 1. / lenFwd
        rayForward = [invLen * rayForward[0], invLen * rayForward[1], invLen * rayForward[2]]
        rayFrom = self.camPos
        oneOverWidth = float(1) / float(self.width)
        oneOverHeight = float(1) / float(self.height)

        dHor = [self.horizon[0] * oneOverWidth, self.horizon[1] * oneOverWidth, self.horizon[2] * oneOverWidth]
        dVer = [self.vertical[0] * oneOverHeight, self.vertical[1] * oneOverHeight, self.vertical[2] * oneOverHeight]
        rayToCenter = [rayFrom[0] + rayForward[0], rayFrom[1] + rayForward[1], rayFrom[2] + rayForward[2]]

        dHor = torch.tensor(dHor).float()
        dVer = torch.tensor(dVer).float()

        horizon = torch.tensor(self.horizon).float()
        vertical = torch.tensor(self.vertical).float()

        rayForward = torch.tensor(rayForward).float().view(-1, 1)
        rayFrom = torch.tensor(rayFrom).float().view(-1, 1)

        ortho_base = (- 0.5 * horizon + 0.5 * vertical).view(-1, 1)
        ortho = ortho_base + torch.matmul(dHor.view(-1, 1), u_vec.float().view(1, -1)) \
                - torch.matmul(dVer.view(-1, 1), v_vec.float().view(1, -1))

        rayTo = rayFrom + rayForward + ortho
        vec = rayTo - rayFrom
        l = torch.sqrt(torch.pow(vec, 2).sum(dim=0))

        lenOrtho = torch.sqrt(torch.pow(ortho, 2).sum(dim=0))

        alpha = torch.atan(lenOrtho / self.farPlane)
        depth_r = depth_vec / torch.cos(alpha)

        newTo = vec.mul((depth_r / l).repeat(3, 1)) + rayFrom

        return newTo

    def update_viewMatrix_from_params(self, camera_params):
        raise NotImplementedError
        # TODO test
        self.horizon, self.vertical = self._cal_hor_ver()
        self.projMatrix = np.array(self.proj_matrix_list).reshape(4, 4).transpose()
        self.viewMatrix = np.array(self.view_matrix_list).reshape(4, 4).transpose()

    def update_viewMatrix_from_pose(self, camera_pose):
        """
        update camera pose from a homo-transformation matrix SE(3)
        :param camera_pose: np.array(4,4), camera eye position and orn in term of SE(3)
        z axis points back obejct(z轴背向物体)
        y axis positive denotes image positive

        :return:
        """

        self.camPos = camera_pose[0:3, 3]
        self.camRot = camera_pose[0:3, 0:3]
        self.cam_Matrix = camera_pose

        self.camUpVector = self.camRot[0:3, 1]

        focus_in_camera1 = np.eye(4)
        focus_in_camera1[0:3, 3] = [0, 0, -self.dist]
        focus_in_world = camera_pose.dot(focus_in_camera1)
        self.camTarget = focus_in_world[0:3, 3]


        self.view_matrix_list= self._p.computeViewMatrix(self.camPos,
                                           self.camTarget,
                                           self.camUpVector)

        self.horizon, self.vertical = self._cal_hor_ver()
        self.projMatrix = np.array(self.proj_matrix_list).reshape(4, 4).transpose()
        self.viewMatrix = np.array(self.view_matrix_list).reshape(4, 4).transpose()

    def cal_pixel_coordinates_from_3D(self, pos_in_world):
        pos_w = np.ones(4).reshape((-1, 1))
        pos_w[0:3, 0] = pos_in_world

        p_e = self.viewMatrix.dot(pos_w)
        # world_to_camera = np.eye(4)
        # world_to_camera[0:3, 0:3]= self.camera_Matrix[0:3,0:3].transpose()
        # world_to_camera[0:3, 3] = -self.camera_Matrix[0:3,3]
        #
        # p_e = world_to_camera.dot(pos_w)

        p_c = self.projMatrix.dot(p_e)

        w = int((p_c[0, 0] / p_c[3, 0] + 1) * self.width / 2.0)
        h = self.height - int((p_c[1, 0] / p_c[3, 0] + 1) * self.height / 2.0)

        return (w, h)

    def cal_pixel_coordinates_from_batch3D(self, pos_vec_in_world):
        """
        :param pos_vec_in_world: (n, 3)
        :return:
        """
        assert pos_vec_in_world.shape[1] == 3

        pos_homo = pos_vec_in_world.transpose()
        pos_w = np.concatenate((pos_homo, np.ones(pos_homo.shape[1]).reshape(1, -1)), axis=0)

        p_e = self.viewMatrix.dot(pos_w)
        p_c = self.projMatrix.dot(p_e)

        u_vec = np.round((p_c[0, :] / p_c[3, :] + 1) * self.width / 2.0).astype(np.int)
        v_vec = np.round(self.height - ((p_c[1, :] / p_c[3, :] + 1) * self.height / 2.0) ).astype(np.int)

        # calculate the depth value = distance between point and nearPlane + near Plane
        z_n = self.camRot[0:3, 2]
        focus_p = self.camTarget
        Q = pos_vec_in_world.transpose()
        d_vec = np.abs((z_n.dot(Q) - z_n.dot(focus_p)) /
                       np.sqrt(np.power(z_n, 2).sum())) + self.nearPlane

        return u_vec, v_vec, d_vec

    @staticmethod
    def calcuate_camera_matrix_from_OPENCV_pose(camera_in_world_wrt_opencv):
        raise  NotImplementedError
        # TODO test
        assert np.array(camera_in_world_wrt_opencv).shape == (4, 4)
        focus_length = 0.1  # it's a fake value that has nothing with the result

        cam_in_world = camera_in_world_wrt_opencv

        cameraUpVector = -cam_in_world[0:3, 1]
        cameraEyePosition = cam_in_world[0:3, 3]

        focus_in_camera1 = np.eye(4)
        focus_in_camera1[0:3, 3] = [0, 0, focus_length]
        focus_in_world = cam_in_world.dot(focus_in_camera1)
        cameraTargetPosition = focus_in_world[0:3, 3]
        view_matrix = p.computeViewMatrix(cameraEyePosition,
                                           cameraTargetPosition,
                                           cameraUpVector)

        return view_matrix

    @staticmethod
    def calcuate_camera_matrix_from_OPENGL_pose(camera_in_world_wrt_opengl):
        """

        :param camera_in_world_wrt_opengl: np.array(4,4), camera eye position and orn in term of SE(3)
        z axis points back obejct(z轴背向物体)
        y axis positive denotes image positive

        :return:
        """
        assert np.array(camera_in_world_wrt_opengl).shape == (4,4)
        focus_length = 0.1 # it's a fake value that has nothing with the result

        cam_in_world = camera_in_world_wrt_opengl

        cameraUpVector = cam_in_world[0:3, 1]
        cameraEyePosition = cam_in_world[0:3, 3]

        focus_in_camera1 = np.eye(4)
        focus_in_camera1[0:3, 3] = [0, 0, -focus_length]
        focus_in_world = cam_in_world.dot(focus_in_camera1)
        cameraTargetPosition = focus_in_world[0:3, 3]
        view_matrix1 = p.computeViewMatrix(cameraEyePosition,
                                           cameraTargetPosition,
                                           cameraUpVector)

        return view_matrix1
