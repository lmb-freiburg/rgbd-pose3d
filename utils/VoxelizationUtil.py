import numpy as np
from collections import namedtuple

from utils.Camera import Camera

VoxelTrafoParamsNew = namedtuple('VoxelTrafoParamsNew', ['voxel_root', 'voxel_scale'])


class VoxelizationUtil(object):
    def __init__(self):
        assert 0, "Abstract class"

    @classmethod
    def voxelize_person(cls, cam, depth_warped, mask,
                 coord2d, coord_vis,
                 root_z_value, grid_size, grid_size_m=None,
                 f=1.0, root_id=1, coord2d_root=None):
        """ Creates a voxelgrid from given input. """
        if grid_size_m is None:
            # cube around the neck, camera coord system
            grid_size_m = np.array([[-1.1, -0.4, -1.1],
                                    [1.1, 1.8, 1.1]])

        if coord2d_root is None:
            coord2d_root = coord2d[root_id, :]

        # randomly sample the cube a bit different in size
        grid_size_m *= f

        grid_size = np.reshape(grid_size, [1, 3]).astype('int32')
        root_xyz = cam.backproject(coord2d_root, root_z_value)  # neck world coordinates

        # get a voxel located at the root_xyz
        voxel_grid, trafo_params = cls.voxelize(cam, depth_warped, mask, root_xyz, grid_size, grid_size_m, f)

        # 5. Calculate pseudo 2D coordinates at neck depth in voxel coords
        coord_pseudo2d = cam.backproject(coord2d, root_z_value)  # project all points onto the neck depth
        coord_pseudo2d_vox = cls.trafo_xyz_coords_to_vox_new(coord_pseudo2d, trafo_params)

        return voxel_grid, coord_pseudo2d_vox[:, :2], trafo_params

    @classmethod
    def voxelize(cls, cam, depth_warped, mask, voxel_root, grid_size, grid_size_m, f=1.0):
        """ Creates a voxelgrid from given input. """
        grid_size = np.reshape(grid_size, [1, 3]).astype('int32')

        # Copy inputs
        depth_warped = np.copy(depth_warped)
        mask = np.copy(mask)
        voxel_root = voxel_root.copy()

        # 1. Vectorize depth and project into world
        uv_vec = cam.get_meshgrid_vector(depth_warped.shape, mask)
        z_vec = np.reshape(depth_warped[mask], [-1]) / 1000.0
        pcl_xyz = cam.backproject(uv_vec, z_vec)
        # print('pcl_xyz', pcl_xyz.shape)

        # 2. Discard unnecessary parts of the pointcloud
        pcl_xyz_rel = pcl_xyz - voxel_root
        cond_x = np.logical_and(pcl_xyz_rel[:, 0] < grid_size_m[1, 0], pcl_xyz_rel[:, 0] > grid_size_m[0, 0])
        cond_y = np.logical_and(pcl_xyz_rel[:, 1] < grid_size_m[1, 1], pcl_xyz_rel[:, 1] > grid_size_m[0, 1])
        cond_z = np.logical_and(pcl_xyz_rel[:, 2] < grid_size_m[1, 2], pcl_xyz_rel[:, 2] > grid_size_m[0, 2])
        cond = np.logical_and(cond_x, np.logical_and(cond_y, cond_z))
        pcl_xyz_rel = pcl_xyz_rel[cond, :]
        # print('pcl_xyz_rel', pcl_xyz_rel.shape)

        # 3. Scale down to voxel size and quantize
        pcl_xyz_01 = (pcl_xyz_rel - grid_size_m[0, :]) / (grid_size_m[1, :] - grid_size_m[0, :])
        pcl_xyz_vox = pcl_xyz_01 * grid_size

        # 4. Set values in the grid
        voxel_grid = np.zeros((grid_size[0, :]))
        pcl_xyz_vox = pcl_xyz_vox.astype('int32')
        voxel_grid[pcl_xyz_vox[:, 0],
                   pcl_xyz_vox[:, 1],
                   pcl_xyz_vox[:, 2]] = 1.0

        # 7. Trafo params
        voxel_root += grid_size_m[0, :]
        voxel_scale = (grid_size_m[1, :] - grid_size_m[0, :]) / grid_size
        trafo_params = VoxelTrafoParamsNew(voxel_root=voxel_root, voxel_scale=voxel_scale)

        return voxel_grid, trafo_params

    @classmethod
    def trafo_xyz_coords_to_vox_new(cls, coord_xyz, trafo_param):
        """ Transforms xyz coordinates defined by trafo_param into voxelized coordinates. """
        assert len(coord_xyz.shape) == 2, "coord_xyz has a bad shape."
        assert coord_xyz.shape[1] == 3, "coord_xyz has a bad last dimension."

        coord_xyz_vox = (coord_xyz - trafo_param.voxel_root) / trafo_param.voxel_scale
        return coord_xyz_vox

    @classmethod
    def trafo_vox_coords_to_xyz_new(cls, coord_xyz_vox, trafo_param):
        """ Transforms from voxelized coords into real xyz. """
        assert len(coord_xyz_vox.shape) == 2, "coord_xyz_vox has a bad shape."
        assert coord_xyz_vox.shape[1] == 3, "coord_xyz_vox has a bad last dimension."

        coord_xyz = coord_xyz_vox * trafo_param.voxel_scale + trafo_param.voxel_root
        return coord_xyz

