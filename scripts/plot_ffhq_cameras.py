import json
import torch
import matplotlib.pyplot as plt


def plot_cameras(
    cameras_pos: torch.Tensor, 
    camera_axis: torch.Tensor, 
    output_path: str, 
    axis_limits: dict,
    reduction_axes: float,
    multiplier_axis: float = 1) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(
        cameras_pos[:, 0], 
        cameras_pos[:, 1], 
        cameras_pos[:, 2], 
        c='black', 
        marker='o', 
        label='Cameras centers'
    )
    # Plot point clouds
    for cur_camera_pos, cur_camera_axis in zip(cameras_pos, camera_axis):
        # Plot the x vector
        ax.quiver(*cur_camera_pos, *cur_camera_axis[0] / reduction_axes, color='r', linewidth=2)
        # Plot the y vector
        ax.quiver(*cur_camera_pos, *cur_camera_axis[1] / reduction_axes, color='g', linewidth=2)
        # Plot the z vector
        ax.quiver(*cur_camera_pos, *cur_camera_axis[2] / reduction_axes, color='b', linewidth=2)
        
    # Origin for the vectors
    origin = [0, 0, 0]

    # Define the components of the unit vectors
    x_vector = [1* multiplier_axis , 0, 0] 
    y_vector = [0, 1 * multiplier_axis, 0]
    z_vector = [0, 0, 1 * multiplier_axis]

    # Plot the x vector
    ax.quiver(*origin, *x_vector, color='r', linewidth=2)
    # Plot the y vector
    ax.quiver(*origin, *y_vector, color='g', linewidth=2)
    # Plot the z vector
    ax.quiver(*origin, *z_vector, color='b', linewidth=2)
    
    # Fix axis
    ax.set_xlim(axis_limits["x"][0], axis_limits["x"][1])
    ax.set_ylim(axis_limits["y"][0], axis_limits["y"][1])
    ax.set_zlim(axis_limits["z"][0], axis_limits["z"][1])
    # Add labels
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    # Add a legend
    ax.legend(loc='upper left', bbox_to_anchor=(0, 0.15))
    # Adjust camera angles
    ax.view_init(elev=45, azim=17, roll=102)
    # Save image
    plt.savefig(output_path, dpi=300)  # High-resolution image
    
    
def main(confs: dict) -> None:
    
    with open(confs["file_path"], "r") as f:
        dataset_cameras = json.load(f)

    cameras_pos = []
    camera_axis = []
    for idx in range(25):
        cur_cameras = torch.tensor([float(x) for x in dataset_cameras["labels"][idx][1]], dtype=torch.float32)[:16].reshape(4, 4)
        cur_cameras = torch.linalg.inv(cur_cameras)
        camera_center = - cur_cameras[:3, :3].T @ cur_cameras[:3, -1]
        print(camera_center)
        cameras_pos.append(camera_center)
        camera_axis.append(cur_cameras[:3, :3])

    axis_limits = {
        "x": [-2, 2],
        "y": [-3, 3],
        "z": [-2, 3]
    }

    plot_cameras(
        cameras_pos=torch.stack(cameras_pos), 
        camera_axis=torch.stack(camera_axis), 
        output_path="ffhq_cameras.png", 
        axis_limits=axis_limits,
        reduction_axes=1,
        multiplier_axis=2
    )


if __name__ == "__main__":
    confs = {
        "file_path": "/data1/users/alvaro/cgs-gan_dataset/FFHQC/dataset.json"
    }
    main(confs=confs)