from isaacgym import gymapi, gymtorch

# Initialize Gym
gym = gymapi.acquire_gym()

# Set simulator and physics parameters
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim_params.physx.use_gpu = True  # Set to False if using CPU
sim_params.use_gpu_pipeline = True

# Create simulator
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
if sim is None:
    raise Exception("Failed to create sim")

# Create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

# Create ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

# Set up environment
num_envs = 1
spacing = 2.0
envs = []

# Load URDF asset
asset_root = "/home/harry/projects/parkour_genesis/legged_gym/robots/g1"  # <-- Change this to your URDF directory
urdf_file = "g1_15dof.urdf"  # <-- Match RL framework
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True  # ← Match RL framework
asset_options.collapse_fixed_joints = True  # ← This is the key difference!
asset_options.disable_gravity = False
asset_options.armature = 0.01
asset_options.default_dof_drive_mode = 3

robot_asset = gym.load_asset(sim, asset_root, urdf_file, asset_options)

# Print asset info to compare with RL framework
num_dof = gym.get_asset_dof_count(robot_asset)
num_bodies = gym.get_asset_rigid_body_count(robot_asset)
print(f"Asset loaded: {num_dof} DOFs, {num_bodies} bodies")

# Print DOF names
print("DOF names:")
for i in range(num_dof):
    dof_name = gym.get_asset_dof_name(robot_asset, i)
    print(f"  {i}: {dof_name}")

# Print body names  
print("Body names:")
for i in range(num_bodies):
    body_name = gym.get_asset_rigid_body_name(robot_asset, i)
    print(f"  {i}: {body_name}")

# Create environments and actors
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

for i in range(num_envs):
    env = gym.create_env(sim, env_lower, env_upper, num_envs)
    envs.append(env)

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, 0.1)  # Start slightly above ground

    actor_handle = gym.create_actor(env, robot_asset, pose, "robot", i, 1)
    print(f"Created actor with handle: {actor_handle}")
    
    # Set all joints to zero position BEFORE prepare_sim
    print("Setting all joints to zero position...")
    dof_states = gym.get_actor_dof_states(env, actor_handle, gymapi.STATE_ALL)
    dof_states['pos'][:] = 0.0  # Set all positions to zero
    dof_states['vel'][:] = 0.0  # Set all velocities to zero
    gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)

gym.prepare_sim(sim)

# Alternative: Set joint positions to zero AFTER prepare_sim using tensor API
print("Setting joints to zero using tensor API...")


# Get the DOF state tensor
gym.refresh_dof_state_tensor(sim)
dof_state_tensor = gym.acquire_dof_state_tensor(sim)
dof_state = gymtorch.wrap_tensor(dof_state_tensor)

# Set all positions and velocities to zero
dof_state[:, 0] = 0.0  # positions
dof_state[:, 1] = 0.0  # velocities

# Apply the changes to simulation
gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_state))

# Simulate
while not gym.query_viewer_has_closed(viewer):
    # 3. Step graphics and render viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # 4. Sync to real-time
    gym.sync_frame_time(sim)

# Cleanup
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
