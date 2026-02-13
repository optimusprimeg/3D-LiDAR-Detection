import open3d as o3d
import numpy as np
import os

input_dir = "/workspaces/3D-LiDAR-Detection/a9_dataset_r02_s01/point_clouds/s110_lidar_ouster_south"
output_dir = "/workspaces/3D-LiDAR-Detection/a9_dataset_r02_s01/velodyne/south"

os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_dir):
    if file.endswith(".pcd"):
        pcd = o3d.io.read_point_cloud(os.path.join(input_dir, file))

        points = np.asarray(pcd.points)

        # intensity often missing, add dummy if needed
        intensity = np.zeros((points.shape[0], 1))
        points = np.hstack((points, intensity))

        out_path = os.path.join(output_dir, file.replace(".pcd", ".bin"))
        points.astype(np.float32).tofile(out_path)

        print("Converted:", file)
