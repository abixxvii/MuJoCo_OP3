import mujoco
import mujoco_viewer
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

def modify_xml(xml_file, solref_box, solref_ball, release_height):
    """Modifies the solref values for box and ball geometries and the release height in the specified XML file."""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    for geom in root.findall(".//geom"):
        if geom.attrib.get("name") == "box_geom":
            geom.attrib["solref"] = solref_box  # Change solref value for box
        elif geom.attrib.get("name") == "ball_geom":
            geom.attrib["solref"] = solref_ball  # Change solref value for ball

    for body in root.findall(".//body"):
        if body.attrib.get("name") == "ball":
            body.attrib["pos"] = f"0 0 {release_height}"  # Change release height for the ball

    tree.write(xml_file)

# Define the range of solref values and release heights to iterate over
solref_values = [("0.02 1", "0.02 10"), ("0.03 2", "0.03 20"), ("0.04 3", "0.04 30")]  # Example solref values for box and ball
release_heights = [5, 10, 15]  # Example release heights in meters

for solref_box, solref_ball in solref_values:
    for release_height in release_heights:
        # Modify the XML file with the current solref values and release height
        modify_xml(r'C:\Users\abish\Desktop\MuJoCo files\Boxball\boxball.xml', solref_box, solref_ball, release_height)

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

        # Get initial velocities
        initial_sphere_vel = get_object_linear_velocity(model, data, sphere_id)
        initial_box_vel = get_object_linear_velocity(model, data, box_id)

        # Simulate and render
        viewer = mujoco_viewer.MujocoViewer(model, data)

        # Store velocities over time
        ball_velocities = []
        box_velocities = []

        for _ in range(1000):
            if viewer.is_alive:
                mujoco.mj_step(model, data)  # Update data object before retrieving velocity
                viewer.render()

                # Capture final velocities in the last step
                if _ == 999:
                    final_sphere_vel = get_object_linear_velocity(model, data, sphere_id)
                    final_box_vel = get_object_linear_velocity(model, data, box_id)
                
                # Capture velocities at each time step
                ball_velocities.append(get_object_linear_velocity(model, data, sphere_id))
                box_velocities.append(get_object_linear_velocity(model, data, box_id))
            else:
                break

        viewer.close()

        # Plot velocities over time
        time_steps = range(len(ball_velocities))
        plt.plot(time_steps, ball_velocities, label='Ball Velocity')
        plt.plot(time_steps, box_velocities, label='Box Velocity')
        plt.xlabel('Time Step')
        plt.ylabel('Velocity (m/s)')
        plt.title('Object Velocities Over Time')
        plt.legend()
        plt.show()

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
            # Handle division by zero case
            if relative_velocity_after == 0:
                cor = 1.0  # If both velocities are zero, assume perfect elasticity (COR = 1)
            else:
                cor = np.inf  # If only the initial velocity is zero, assume perfectly inelastic collision (COR = infinity)

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
        print("Release Height (m):", release_height)

        # Calculate the initial velocity using kinematic equations
        initial_velocity = np.sqrt(2 * 9.81 * release_height)

        # Calculate the expected final velocity using the coefficient of restitution
        expected_final_velocity = initial_velocity * cor

        # Print the calculated initial and expected final velocities
        print("Calculated Initial Velocity (m/s):", initial_velocity)
        print("Expected Final Velocity (m/s):", expected_final_velocity)
