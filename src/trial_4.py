import mujoco
import mujoco_viewer
import numpy as np

def modify_xml(xml_file, solref_value, solimp_value):
    """Modifies solref and solimp values in the specified XML file."""
    try:
        with open(xml_file, 'r') as f:
            xml_content = f.read()

        # Modify solref and solimp values
        modified_xml_content = xml_content.replace('solref="0.02 0.85"', f'solref="0.02 {solref_value}"')
        modified_xml_content = modified_xml_content.replace('solimp="0.9 0.95 0.001"', f'solimp="{solimp_value}"')

        # Write the modified XML content back to the file
        with open(xml_file, 'w') as f:
            f.write(modified_xml_content)

        print("XML file successfully modified.")
    except Exception as e:
        print("Error modifying XML file:", e)

def calculate_initial_velocity(release_height):
    """Calculate the initial velocity of the ball based on release height."""
    try:
        # Constants
        g = 9.8  # Acceleration due to gravity

        # Calculate initial velocity using energy conservation principle
        v_initial = np.sqrt(2 * g * release_height)
        
        return v_initial
    except Exception as e:
        print("Error calculating initial velocity:", e)
        return None

def simulate_model(xml_file):
    """Simulates the MuJoCo model and captures final velocities."""
    try:
        # Load the MuJoCo model
        model = mujoco.MjModel.from_xml_path(xml_file)
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
        final_sphere_vel = None
        final_box_vel = None
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

        # Close the viewer
        viewer.close()

        # Return the final velocities
        return final_sphere_vel, final_box_vel
    except Exception as e:
        print("Error simulating model:", e)
        return None, None

def calculate_coefficient_of_restitution(final_sphere_vel, final_box_vel):
    """Calculate the coefficient of restitution."""
    try:
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

def verify_restitution(xml_file, v_initial, cor):
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

# Define release height (replace with actual height)
release_height = 10  # Example value, replace with actual release height

# Calculate initial velocity based on release height
v_initial = calculate_initial_velocity(release_height)

# Define range of solref values to test
solref_values = np.linspace(0.5, 1.5, 5)

# Lists to store results
cor_values = []

# Iterate over each solref value
for solref in solref_values:
    try:
        # Modify XML file with current solref value
        modify_xml('boxball.xml', solref, '0.5 1.5 0.005')
        
        # Simulate the modified model
        final_sphere_vel, final_box_vel = simulate_model('boxball.xml')
        
        # Calculate coefficient of restitution
        cor = calculate_coefficient_of_restitution(final_sphere_vel, final_box_vel)
        cor_values.append(cor)
        
        # Verify restitution
        verify_restitution('boxball.xml', v_initial, cor)
        
    except Exception as e:
        print("Error in iteration with solref value", solref, ":", e)

# Print results
print("Solref Values:", solref_values)
print("Coefficient of Restitution Values:", cor_values)
