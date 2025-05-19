import numpy as np
import open3d as o3d

# Define vertices and faces (triangles)
verts = np.array([
    [-5.e+02, -5.e+02, -1.e-02],
    [-5.e+02, -5.e+02,  0.e+00],
    [-5.e+02,  5.e+02, -1.e-02],
    [-5.e+02,  5.e+02,  0.e+00],
    [ 5.e+02, -5.e+02, -1.e-02],
    [ 5.e+02, -5.e+02,  0.e+00],
    [ 5.e+02,  5.e+02, -1.e-02],
    [ 5.e+02,  5.e+02,  0.e+00]
], dtype=np.float64)

faces = np.array([
    [1, 3, 0],
    [4, 1, 0],
    [0, 3, 2],
    [2, 4, 0],
    [1, 7, 3],
    [5, 1, 4],
    [5, 7, 1],
    [3, 7, 2],
    [6, 4, 2],
    [2, 7, 6],
    [6, 5, 4],
    [7, 5, 6]
], dtype=np.int32)

# Create Open3D TriangleMesh
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(verts)
mesh.triangles = o3d.utility.Vector3iVector(faces)

# Compute normals for better visualization
mesh.compute_vertex_normals()

# Optional: Paint the mesh in a color (e.g., light gray)
mesh.paint_uniform_color([0.7, 0.7, 0.7])

# Visualize
o3d.visualization.draw_geometries([mesh], window_name="Triangle Mesh")