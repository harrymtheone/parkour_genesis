### Convert URDF to USD

Since IsaacSim only support importing USD file to simulation, you should convert your URDF file to USD format.

To do this, run 

````

./isaaclab.sh -p scripts/tools/convert_urdf.py \
path/to/your.urdf \
path/to/store.usd \
--merge-joints   \
--joint-stiffness 0.0   \
--joint-damping 0.0   \
--joint-target-type none

````

The output is not just a single USD file, so you'd better put them under a directory.



### Test snippet

````
self.sim.reset()
print("[INFO]: Setup complete...")
while self.simulation_app.is_running():
    self.sim.render()
````
