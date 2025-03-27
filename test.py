# import torch
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# class RewardFunction:
#     def __init__(self, min_dist=0.04, max_dist=0.06):
#         self.min_dist = min_dist
#         self.max_dist = max_dist
#
#     def _reward_feet_away(self, foot_dist):
#         fd = self.min_dist
#         max_df = self.max_dist
#         d_min = torch.clamp(foot_dist - fd, -0.04, 0.)
#         d_max = torch.clamp(foot_dist - max_df, 0, 0.04)
#         return (torch.exp(-torch.abs(d_min) * 50) + torch.exp(-torch.abs(d_max) * 50)) / 2 - 0.57
#
#
# # Create a range of foot distances
# foot_distances = torch.linspace(0.0, 0.2, 100)
# reward_fn = RewardFunction()
# rewards = reward_fn._reward_feet_away(foot_distances)
#
# # Plot the reward curve
# plt.figure(figsize=(8, 5))
# plt.plot(foot_distances.numpy(), rewards.numpy(), label='Reward')
# plt.xlabel('Foot Distance')
# plt.ylabel('Reward')
# plt.title('Reward vs. Foot Distance')
# plt.legend()
# plt.grid()
# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
#
# # Define reward function parameters
# tracking_sigma = 2.0  # Example value for self.cfg.rewards.tracking_sigma
# v_cmd_xy = 0.5  # Fixed commanded velocity (m/s)
#
# # Generate actual velocity values
# v_xy = np.linspace(0, 1.5, 100)
#
# # Compute velocity error based on the minimum condition in the reward function
# vel_err = np.minimum(v_xy, v_cmd_xy + 0.1) - v_cmd_xy
#
# # Compute reward values
# reward = np.exp(-vel_err**2 * tracking_sigma)
#
# # Plot the reward function
# plt.figure(figsize=(6, 4))
# plt.plot(v_xy, reward, label=r'$e^{-\sigma (\text{vel\_err})^2}$', color='b')
# plt.axvline(x=v_cmd_xy, linestyle="--", color="red", label="Commanded velocity")
# plt.axvline(x=v_cmd_xy + 0.1, linestyle="--", color="green", label="Upper threshold")
# plt.show()

# import torch
# import math
# import matplotlib.pyplot as plt
#
# # Define parameters
# parkour_vel_tolerance = 0.3  # Example value
# tracking_sigma = 5  # Example value
#
# # Define lin_vel_error range
# lin_vel_error = torch.linspace(0, 1.5, 100)  # Adjust range as needed
#
# # Compute reward function
# rew = torch.where(
#     lin_vel_error < parkour_vel_tolerance,
#     torch.exp(-lin_vel_error * 0.3),
#     torch.exp(-(lin_vel_error - parkour_vel_tolerance) * tracking_sigma) - 1 + math.exp(-parkour_vel_tolerance * 0.3)
# )
#
# # Convert to numpy for plotting
# lin_vel_error_np = lin_vel_error.numpy()
# rew_np = rew.numpy()
#
# # Plot the function
# plt.figure(figsize=(8, 5))
# plt.plot(lin_vel_error_np, rew_np, label="Reward Function", color='b')
# plt.axvline(x=parkour_vel_tolerance, color='r', linestyle='--', label="Tolerance Threshold")
# plt.xlabel("Linear Velocity Error")
# plt.ylabel("Reward")
# plt.title("Reward Function vs. Linear Velocity Error")
# plt.legend()
# plt.grid(True)
# plt.show()


import numpy as np
import noise
import matplotlib.pyplot as plt

from legged_gym.utils.terrain.utils import convert_heightfield_to_trimesh
import open3d as o3d


def generate_fractal_noise_2d(shape, scale=100, octaves=5, persistence=0.5, lacunarity=2.0, seed=None):
    """
    Generate 2D fractal noise using Perlin noise.

    Parameters:
        shape (tuple): (height, width) of the output noise grid.
        scale (float): Frequency scale of the noise.
        octaves (int): Number of noise layers.
        persistence (float): Amplitude reduction per octave (0-1).
        lacunarity (float): Frequency multiplier per octave (>1).
        seed (int, optional): Random seed for noise generation.

    Returns:
        np.ndarray: 2D fractal noise array.
    """
    height, width = shape
    noise_array = np.zeros((height, width))

    if seed is not None:
        np.random.seed(seed)

    for y in range(height):
        for x in range(width):
            noise_value = 0
            frequency = 1.0 / scale
            amplitude = 1.0
            for _ in range(octaves):
                noise_value += amplitude * noise.pnoise2(x * frequency, y * frequency, repeatx=width, repeaty=height)
                frequency *= lacunarity
                amplitude *= persistence

            noise_array[y, x] = noise_value

    # Normalize noise to range [0, 1]
    noise_array = (noise_array - noise_array.min()) / (noise_array.max() - noise_array.min())
    return noise_array


# Example usage
noise_2d = generate_fractal_noise_2d((256, 256), scale=50, octaves=6, persistence=0.5, lacunarity=2.0, seed=42)

ver, tri = convert_heightfield_to_trimesh(noise_2d, 0.05, 0.5, 1.0)
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(ver)
mesh.triangles = o3d.utility.Vector3iVector(tri)
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])

# Visualize the fractal noise
plt.imshow(noise_2d, cmap="gray")
plt.colorbar()
plt.title("2D Fractal Noise")
plt.show()
