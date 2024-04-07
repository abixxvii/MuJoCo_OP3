import mujoco
import numpy as np
import matplotlib.pyplot as plt

# Define the Mujoco XML with your specified content
boxball = """
<mujoco>
	<option gravity="0 0 -9.8"/>
	<asset>
		<!-- Texture for the background -->
		<texture type="skybox" builtin="gradient" rgb1="0 0 0" rgb2="0 0 0" width="512" height="512"/>
	</asset>
	<worldbody>
		<!-- Ground -->
		<body name="ground">
			<geom type="plane" size="10 10 0.1" rgba="0.8 0.8 0.8 1"/>
		</body>

		<!-- Box -->
		<body name="box" pos="0 0 0">
			<geom name="box_geom" type="box" size="0.5 0.5 0.5" rgba="1 0 0 1" solref="0.02 1" friction="1.1 0.005 0.0001"/>
		</body>

		<!-- Ball -->
		<body name="ball" pos="0 0 10">
			<joint type="free"/>
			<geom name="ball_geom" type="sphere" size="0.25" rgba="0 1 0 1" mass="10" solref="0.02 1" friction="1 1 0.0001"/>
		</body>
	</worldbody>
</mujoco>
"""

# Load the model
model = mujoco.MjModel.from_xml_string(boxball)
data = mujoco.MjData(model)

# Simulate the collision and record the data
n_frames = 100
times = []
velocities = []

for i in range(n_frames):
    mujoco.mj_step(model, data)
    times.append(data.time)
    
    # Extract velocity of the ball
    vel = np.zeros(6)
    ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, ball_id, vel, 0)  # 1 for local, 0 for world frame
    velocities.append(vel)
    
    print(f"Time: {data.time}, Velocity: {vel}")

# Plotting the velocities over time
velocities = np.array(velocities)
plt.plot(times, velocities[:, -1])  # Plot the last component (angular velocity around z-axis)
plt.xlabel('Time (s)')
plt.ylabel('Linear Velocity (m/s)')
plt.title('Linear Velocity Over Time')
plt.grid(True)
plt.show()
