import mujoco
import mujoco_viewer
import numpy as np

# Function to modify XML file solref and solimp values
def modify_xml(xml_file, solref_value, solimp_value):
    """Modifies solref and solimp values in the specified XML file."""
    with open(xml_file, 'r') as f:
        xml_content = f.read()

    # Modify solref and solimp values
    modified_xml_content = xml_content.replace('solref="0.02 0.85"', f'solref="0.02 {solref_value}"')
    modified_xml_content = modified_xml_content.replace('solimp="0.9 0.95 0.001"', f'solimp="{solimp_value}"')

    # Write the modified XML content back to the file
    with open(xml_file, 'w') as f:
        f.write(modified_xml_content)

# Optionally modify XML file
modify_xml('boxball.xml', 0.85, '0.8 0.9 0.005')  # Change solref to 10 and solimp to '0.8 0.9 0.005'

# Load the MuJoCo model
model = mujoco.MjModel.from_xml_path('boxball.xml')
data = mujoco.MjData(model)

# Retrieve object IDs
sphere_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "box_geom")

# Function to retrieve linear velocity of an object
def get_object_linear_velocity(model, data, obj_id):
    """Retrieves the linear velocity of an object."""
    lin_vel = np.zeros(6)  # Create array to store linear velocity
    obj_type = mujoco.mjtObj.mjOBJ_GEOM  # Use geometry object type constant
    mujoco.mj_objectVelocity(model, data, obj_type, obj_id, lin_vel, 0)  # Use 0 for global coordinates
    return lin_vel

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

# Print object IDs
print("Sphere ID:", sphere_id)
print("Box ID:", box_id)

# Print the velocities
print("Sphere Final Velocity (m/s):")
print(final_sphere_vel)
print("Box Final Velocity (m/s):")
print(final_box_vel)

# Define the collision direction (for example, along the z-axis)
collision_direction = np.array([0, 0, 1])

# Calculate relative velocity along the collision direction
relative_velocity_before = np.dot(final_box_vel[:3] - final_sphere_vel[:3], collision_direction)

# Assuming the initial velocity of the sphere is zero
relative_velocity_after = final_sphere_vel[2]  # Only consider z-component

# Calculate coefficient of restitution
if relative_velocity_before != 0:
    cor = relative_velocity_after / relative_velocity_before
else:
    cor = np.nan

print("Coefficient of Restitution:", cor)
