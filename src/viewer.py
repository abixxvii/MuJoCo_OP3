import mujoco
import mujoco_viewer
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

# Function to modify XML file solimp and solref values (optional)
def modify_xml(xml_file):
    """Modifies the solref values for box and ball geometries in the specified XML file."""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    for geom in root.findall(".//geom"):
        if geom.attrib.get("name") == "box_geom":
            geom.attrib["solref"] = "0.02 1"  # Change solref value for box
        elif geom.attrib.get("name") == "ball_geom":
            geom.attrib["solref"] = "0.02 2"  # Change solref value for ball

    tree.write(xml_file)

# Function to modify the height of the ball in the XML file
def modify_ball_height(xml_file, new_height):
    """Modifies the height of the ball in the specified XML file."""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Find the <body> tag representing the ball
    ball_body = root.find(".//body[@name='ball']")

    # Update the height (z-coordinate) in the pos attribute
    ball_pos = ball_body.attrib.get("pos")
    if ball_pos:
        ball_pos_values = ball_pos.split()
        if len(ball_pos_values) == 3:
            ball_pos_values[2] = str(new_height)  # Update the z-coordinate to the new height
            ball_body.set("pos", " ".join(ball_pos_values))

    tree.write(xml_file)

# Modify the XML file before loading the Mujoco model
modify_xml(r'C:\Users\abish\Desktop\MuJoCo files\Boxball\boxball.xml')

# Modify the height of the ball
modify_ball_height(r'C:\Users\abish\Desktop\MuJoCo files\Boxball\boxball.xml', 15)  # Change the height as desired

# Load the Mujoco model
model = mujoco.MjModel.from_xml_path(r'C:\Users\abish\Desktop\MuJoCo files\Boxball\boxball.xml')
data = mujoco.MjData(model)

# Retrieve object IDs
sphere_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "box_geom")

print(f"Sphere ID: {sphere_id}")
print(f"Box ID: {box_id}")
print(f"Total Bodies: {model.nbody}")

# Function to retrieve linear velocity of an object
def get_object_linear_velocity(model, data, obj_id):
    """Retrieves the linear velocity of an object."""
    lin_vel = np.zeros(6)  # Create array to store linear velocity
    obj_type = mujoco.mjtObj.mjOBJ_GEOM  # Use geometry object type constant
    mujoco.mj_objectVelocity(model, data, obj_type, obj_id, lin_vel, 0)  # Use 0 for global coordinates
    return lin_vel

# Simulate and render
viewer = mujoco_viewer.MujocoViewer(model, data)

# Define time steps at which to print velocities
time_steps_to_print = [100, 500, 1000]

# Store velocities at specified time steps
velocities_at_time_steps = {}

for step in range(1, 1001):
    if viewer.is_alive:
        mujoco.mj_step(model, data)  # Update data object before retrieving velocity
        viewer.render()

        if step in time_steps_to_print:
            # Get velocities at the current time step
            sphere_velocity = get_object_linear_velocity(model, data, sphere_id)
            box_velocity = get_object_linear_velocity(model, data, box_id)
            velocities_at_time_steps[step] = {'sphere': sphere_velocity, 'box': box_velocity}

    else:
        break

viewer.close()

# Print velocities at specified time steps
for step, velocities in velocities_at_time_steps.items():
    print(f"Time Step: {step}")
    print("Sphere Velocity:", velocities['sphere'])
    print("Box Velocity:", velocities['box'])
    print()

# Initialize lists to store velocities over time
ball_velocities = []
box_velocities = []

# Initialize coefficient of restitution
cor = 1.0

# Initialize final_sphere_vel outside the loop
final_sphere_vel = np.zeros(6)

# Initialize collision flag
collision_occurred = False

# Get initial velocities
initial_sphere_vel = get_object_linear_velocity(model, data, sphere_id)
initial_box_vel = get_object_linear_velocity(model, data, box_id)

# Run the simulation loop
for _ in range(1000):
    if viewer.is_alive:
        mujoco.mj_step(model, data)  # Update data object before retrieving velocity
        viewer.render()

        # Get initial velocities
        initial_sphere_vel = get_object_linear_velocity(model, data, sphere_id)
        initial_box_vel = get_object_linear_velocity(model, data, box_id)

        # Capture velocities at each time step
        ball_velocities.append(initial_sphere_vel)
        box_velocities.append(initial_box_vel)
        
        # Calculate relative velocity along the collision direction
        collision_direction = np.array([0, 0, 1])
        relative_velocity_before = np.dot(initial_box_vel[:3] - initial_sphere_vel[:3], collision_direction)

        # Update coefficient of restitution based on collision
        if relative_velocity_before != 0 and not collision_occurred:
            final_sphere_vel = get_object_linear_velocity(model, data, sphere_id)
            final_box_vel = get_object_linear_velocity(model, data, box_id)
            relative_velocity_after = np.dot(final_box_vel[:3] - final_sphere_vel[:3], collision_direction)
            cor = relative_velocity_after / relative_velocity_before
            collision_occurred = True

        # Reset collision flag
        if collision_occurred and relative_velocity_before == 0:
            collision_occurred = False

    else:
        break

viewer.close()

# Plot velocities over time
time_steps = range(len(ball_velocities))
ball_magnitudes = [np.linalg.norm(vel[:3]) for vel in ball_velocities]
box_magnitudes = [np.linalg.norm(vel[:3]) for vel in box_velocities]

plt.plot(time_steps, ball_magnitudes, label='Ball Velocity Magnitude', color='blue', linestyle='-', marker='o')
plt.plot(time_steps, box_magnitudes, label='Box Velocity Magnitude', color='red', linestyle='--', marker='x')
plt.xlabel('Time Step')
plt.ylabel('Velocity Magnitude (m/s)')
plt.title('Object Velocity Magnitudes Over Time')
plt.legend()
plt.show()

# Print the coefficient of restitution
print("Coefficient of Restitution:", cor)

# Calculate the expected final velocity of the ball based on the coefficient of restitution
expected_final_sphere_vel = initial_sphere_vel * cor

# Print or log the expected and simulated final velocities
print("Expected Final Sphere Velocity (m/s):")
print(expected_final_sphere_vel)
print("Simulated Final Sphere Velocity (m/s):")
print(final_sphere_vel)

# Compare the expected and simulated final velocities
velocity_difference = np.linalg.norm(expected_final_sphere_vel - final_sphere_vel)
print("Difference in Velocities (m/s):", velocity_difference)

# Release height of the ball
release_height = 15  # Input the release height in meters

# Calculate the initial velocity using kinematic equations
initial_velocity = np.sqrt(2 * 9.81 * release_height)

# Calculate the expected final velocity using the coefficient of restitution
expected_final_velocity = initial_velocity * cor

# Print the calculated initial and expected final velocities
print("Calculated Initial Velocity (m/s):", initial_velocity)
print("Expected Final Velocity (m/s):", expected_final_velocity)
