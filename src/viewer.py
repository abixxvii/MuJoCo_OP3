import mujoco
import mujoco_viewer
import numpy as np

# Function to modify XML file solimp and solref values (optional)
def modify_xml(xml_file):
    """Modifies solimp and solref values in the specified XML file."""
    with open(xml_file, 'r') as f:
        xml_content = f.read()

    # Modify solimp and solref values as needed
    modified_xml_content = xml_content.replace('solref="0.02 1"', 'solref="0.02 0.5"')  # Example modification

    # Write the modified XML content back to the file
    with open(xml_file, 'w') as f:
        f.write(modified_xml_content)

# Load the Mujoco model
model = mujoco.MjModel.from_xml_path('boxball.xml')
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

# Get initial velocities
initial_sphere_vel = get_object_linear_velocity(model, data, sphere_id)
initial_box_vel = get_object_linear_velocity(model, data, box_id)

# Simulate and render
viewer = mujoco_viewer.MujocoViewer(model, data)
for _ in range(1000):
    if viewer.is_alive:
        mujoco.mj_step(model, data)  # Update data object before retrieving velocity
        viewer.render()

        # Capture final velocities in the last step
        if _ == 999:
            final_sphere_vel = get_object_linear_velocity(model, data, sphere_id)
            final_box_vel = get_object_linear_velocity(model, data, box_id)
            
    else:
        break

viewer.close()

# Print the velocities
print("Sphere Initial Velocity (m/s):")
print(initial_sphere_vel)
print("Sphere Final Velocity (m/s):")
print(final_sphere_vel)
print("Box Initial Velocity (m/s):")
print(initial_box_vel)
print("Box Final Velocity (m/s):")
print(final_box_vel)

# Define the collision direction (for example, along the z-axis)
collision_direction = np.array([0, 0, 1])

# Calculate relative velocity along the collision direction
relative_velocity_before = np.dot(initial_box_vel[:3] - initial_sphere_vel[:3], collision_direction)
relative_velocity_after = np.dot(final_box_vel[:3] - final_sphere_vel[:3], collision_direction)

# Calculate coefficient of restitution
if relative_velocity_before != 0:
    cor = relative_velocity_after / relative_velocity_before
else:
    cor = np.nan

print("Coefficient of Restitution:", cor)
