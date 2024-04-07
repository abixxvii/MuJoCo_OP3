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

# Function to get object linear velocity
def get_object_linear_velocity(model, data, obj_id):
    lin_vel = np.zeros(6)  # Changed to a 6-element array to accommodate the result
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_GEOM, obj_id, lin_vel, 0)
    return lin_vel[:3]  # Return only the first 3 elements (linear velocity)

def calculate_coefficient_of_restitution(final_sphere_vel, final_box_vel):
    """Calculate the coefficient of restitution."""
    try:
        # Print velocities for debugging
        print("Final Sphere Velocity:", final_sphere_vel)
        print("Final Box Velocity:", final_box_vel)

        # Define the collision direction (for example, along the z-axis)
        collision_direction = np.array([0, 0, 1])

        # Calculate relative velocity along the collision direction
        relative_velocity_before = np.dot(final_box_vel[:3] - final_sphere_vel[:3], collision_direction)

        # Assuming the initial velocity of the sphere is zero
        relative_velocity_after = final_sphere_vel[2]  # Only consider z-component

        # Handle division by zero
        if relative_velocity_before == 0:
            cor = np.nan
        else:
            cor = relative_velocity_after / relative_velocity_before

        return cor
    except Exception as e:
        print("Error calculating coefficient of restitution:", e)
        return np.nan


def verify_restitution(v_initial, cor):
    """Verify if the simulation coincides with expected restitution."""
    try:
        # Calculate expected final velocity
        v_final_expected = v_initial * cor
        
        # Simulate the model to get the final velocity from the environment
        final_sphere_vel, _ = simulate_model(xml_file)
        
        # Extract the z-component of the final velocity of the ball
        v_final_simulated = final_sphere_vel[2]
        
        # Compare the expected and simulated final velocities
        if np.isclose(v_final_expected, v_final_simulated, rtol=1e-3):
            print("Simulation coincides with expected restitution.")
        else:
            print("Simulation does not coincide with expected restitution.")
            print("Expected final velocity:", v_final_expected)
            print("Simulated final velocity:", v_final_simulated)
    except Exception as e:
        print("Error verifying restitution:", e)

def calculate_initial_velocity(height):
    """Calculate the initial velocity of the ball based on the release height."""
    # Using the formula for initial velocity when an object is dropped from a certain height:
    initial_velocity = np.sqrt(2 * 9.8 * height)  # Assuming acceleration due to gravity is 9.8 m/s^2
    return initial_velocity

# Calculate initial velocity based on release height
release_height = 10  # Example value, replace with actual release height
v_initial = calculate_initial_velocity(release_height)

# Get coefficient of restitution from simulation
cor = calculate_coefficient_of_restitution(final_sphere_vel, final_box_vel)

# Verify restitution
verify_restitution(v_initial, cor)


# Simulate the collision and record the data
n_frames = 100
times = []
ball_vel = []

for i in range(n_frames):
    mujoco.mj_step(model, data)
    times.append(data.time)
    ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
    ball_vel.append(get_object_linear_velocity(model, data, ball_id))  # Ball velocity

# Extract final velocity
final_velocity = ball_vel[-1]

# Define release height of the ball
release_height = 10  # meters

# Calculate expected impact velocity
expected_final_velocity = np.sqrt(2 * 9.8 * release_height)

# Calculate coefficient of restitution (COR)
cor = final_velocity[2] / expected_final_velocity
print(f"Expected impact velocity: {expected_final_velocity}")
print(f"Final velocity from simulation: {final_velocity[2]}")
print(f"Coefficient of Restitution (COR): {cor}")

# Plotting the ball velocity over time
ball_vel = np.array(ball_vel)
plt.plot(times, ball_vel[:, 2], label='Ball Z-velocity')
plt.xlabel('Time (s)')
plt.ylabel('Z-velocity')
plt.title('Ball Z-velocity Over Time')
plt.legend()
plt.grid(True)
plt.show()
