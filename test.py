import trimesh

# Create large ground plane
ground = trimesh.creation.box(extents=(1000, 1000, 1), transform=trimesh.transformations.translation_matrix([0, 0, -0.5]))

# Create a "cutout" volume (a cube where the stair will go)
hole = trimesh.creation.box(extents=(2, 2, 2), transform=trimesh.transformations.translation_matrix([0, 0, -1]))

# Subtract the hole from the ground
ground_with_hole = trimesh.boolean.difference([ground, hole], engine="scad")  # engine can be "blender", "scad", or "igl"

# Add downward stair mesh
stair = trimesh.creation.box(extents=(2, 2, 2), transform=trimesh.transformations.translation_matrix([0, 0, -2]))

# Combine ground with stairs
final_env = trimesh.util.concatenate([ground_with_hole, stair])
