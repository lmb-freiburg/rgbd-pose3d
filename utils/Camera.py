import numpy as np


class Camera(object):
    def __init__(self, K):
        self.K = K
        self.K_inv = np.linalg.inv(K)

    def project(self, xyz_coords):
        """ Projects a (x, y, z) tuple of world coords into the image frame. """
        xyz_coords = np.reshape(xyz_coords, [-1, 3])
        uv_coords = np.matmul(xyz_coords, np.transpose(self.K, [1, 0]))
        return self._from_hom(uv_coords)

    def backproject(self, uv_coords, z_coords):
        """ Projects a (x, y, z) tuple of world coords into the world frame. """
        uv_coords = np.reshape(uv_coords, [-1, 2])
        z_coords = np.reshape(z_coords, [-1, 1])

        uv_coords_h = self._to_hom(uv_coords)
        z_coords = np.reshape(z_coords, [-1, 1])
        xyz_coords = z_coords * np.matmul(uv_coords_h, np.transpose(self.K_inv, [1, 0]))
        return xyz_coords

    @staticmethod
    def get_meshgrid_vector(shape_hw, mask=None):
        """ Given an imageshape it outputs all coordinates as [N, dim] matrix. """
        if mask is None:
            H, W = np.meshgrid(range(0, shape_hw[0]), range(0, shape_hw[1]), indexing='ij')
            h_vec = np.reshape(H, [-1])
            w_vec = np.reshape(W, [-1])
            coords = np.stack([w_vec, h_vec], 1)
        else:
            H, W = np.meshgrid(range(0, shape_hw[0]), range(0, shape_hw[1]), indexing='ij')
            h_vec = np.reshape(H[mask], [-1])
            w_vec = np.reshape(W[mask], [-1])
            coords = np.stack([w_vec, h_vec], 1)
        return coords

    @staticmethod
    def _to_hom(coords):
        """ Turns the [N, D] coord matrix into homogeneous coordinates [N, D+1]. """
        coords_h = np.concatenate([coords, np.ones((coords.shape[0], 1))], 1)
        return coords_h

    @staticmethod
    def _from_hom(coords_h):
        """ Turns the homogeneous coordinates [N, D+1] into [N, D]. """
        coords = coords_h[:, :-1] / (coords_h[:, -1:] + 1e-10)
        return coords