# CPU fallback implementation for hard_voxelize
import torch
import numpy as np


def hard_voxelize(points, voxels, coors, num_points_per_voxel, 
                  voxel_size, coors_range, max_points, max_voxels, 
                  ndim_minus_1, deterministic):
    """CPU implementation of hard voxelization.
    
    Args:
        points: [N, ndim] float tensor with point coordinates
        voxels: [max_voxels, max_points, ndim] output tensor
        coors: [max_voxels, 3] output tensor for coordinates
        num_points_per_voxel: [max_voxels] output tensor
        voxel_size: [3] list of voxel dimensions
        coors_range: [6] list of coordinate range [x_min, y_min, z_min, x_max, y_max, z_max]
        max_points: max points per voxel
        max_voxels: max number of voxels
        ndim_minus_1: not used, for compatibility
        deterministic: whether to use deterministic version
        
    Returns:
        voxel_num: number of voxels created
    """
    # Convert to numpy for easier processing
    points_np = points.cpu().numpy() if points.is_cuda else points.numpy()
    voxel_size_np = np.array(voxel_size)
    coors_range_np = np.array(coors_range)
    
    # Calculate voxel coordinates
    # coors_range format: [x_min, y_min, z_min, x_max, y_max, z_max]
    point_xyz = points_np[:, :3]
    
    # Normalize to grid indices
    grid_indices = ((point_xyz - coors_range_np[:3]) / voxel_size_np).astype(np.int32)
    
    # Filter out points outside range
    mask = (grid_indices >= 0).all(axis=1)
    grid_shape = ((coors_range_np[3:] - coors_range_np[:3]) / voxel_size_np).astype(np.int32)
    mask &= (grid_indices < grid_shape).all(axis=1)
    
    grid_indices = grid_indices[mask]
    points_filtered = points_np[mask]
    
    # Convert grid indices to linear indices for hashing
    linear_indices = (grid_indices[:, 0] * (grid_shape[1] * grid_shape[2]) +
                      grid_indices[:, 1] * grid_shape[2] +
                      grid_indices[:, 2])
    
    # Find unique voxels
    unique_linear_idx, inverse_indices = np.unique(linear_indices, return_inverse=True)
    
    voxel_num = min(len(unique_linear_idx), max_voxels)
    
    # Fill voxels
    for voxel_id in range(voxel_num):
        mask = (inverse_indices == voxel_id)
        pts_in_voxel = points_filtered[mask]
        
        num_pts = min(len(pts_in_voxel), max_points)
        voxels[voxel_id, :num_pts] = torch.from_numpy(pts_in_voxel[:num_pts])
        num_points_per_voxel[voxel_id] = num_pts
        
        # Convert linear index back to 3D coordinates
        linear_idx = unique_linear_idx[voxel_id]
        coor_x = linear_idx // (grid_shape[1] * grid_shape[2])
        linear_idx %= (grid_shape[1] * grid_shape[2])
        coor_y = linear_idx // grid_shape[2]
        coor_z = linear_idx % grid_shape[2]
        
        coors[voxel_id, 0] = coor_x
        coors[voxel_id, 1] = coor_y
        coors[voxel_id, 2] = coor_z
    
    return voxel_num
