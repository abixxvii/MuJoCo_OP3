import mujoco
import mujoco_viewer
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

# Create the viewer
#viewer = mujoco_viewer.MjViewer()

#model = mujoco.MjModel.from_xml_path('boxball.xml')
#ata = mujoco.MjData(model)

# create the viewer object
#viewer = mujoco_viewer.MujocoViewer(model, data)

# simulate and render
#for _ in range(10000):
#    if viewer.is_alive:
#        mujoco.mj_step(model, data)
#        viewer.render()
#    else:
#        break

# close
#viewer.close()

# Function to get object linear velocity
def get_object_linear_velocity(model, data, obj_id):
    lin_vel = np.zeros(6)  # Changed to a 6-element array to accommodate the result
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_GEOM, obj_id, lin_vel, 0)
    print(f"Linear velocity: {lin_vel}")
    return lin_vel[:3]  # Return only the first 3 elements (linear velocity)


# Simulate the collision and record the data
n_frames = 100
times = []
box_vel = []
ball_vel = []

for i in range(n_frames):
    mujoco.mj_step(model, data)
    times.append(data.time)
    box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "box_geom")
    ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
    print(f"Box ID: {box_id}, Ball ID: {ball_id}")
    box_vel.append(get_object_linear_velocity(model, data, box_id))  # Box velocity
    ball_vel.append(get_object_linear_velocity(model, data, ball_id))  # Ball velocity
    print(f"Box velocity at timestep {i}: {box_vel[-1]}")
    print(f"Ball velocity at timestep {i}: {ball_vel[-1]}")

# Calculate coefficient of restitution
# Assuming the ball is released from rest and only bounces off the ground
initial_velocity = [0, 0, -9.8]  # Initial velocity of the ball
final_velocity = ball_vel[-1]  # Final velocity of the ball after bouncing

print("Initial velocity:", initial_velocity)
print("Final velocity:", final_velocity)

# Calculate the coefficient of restitution (COR)
cor = final_velocity / initial_velocity
print(f"Coefficient of Restitution (COR): {cor}")

# Plotting the velocities over time
box_vel = np.array(box_vel)
ball_vel = np.array(ball_vel)

plt.plot(times, box_vel[:, 2], label='Box Z-velocity')
plt.plot(times, ball_vel[:, 2], label='Ball Z-velocity')
plt.xlabel('Time (s)')
plt.ylabel('Z-velocity')
plt.title('Box and Ball Z-velocity Over Time')
plt.legend()
plt.grid(True)
plt.show()
