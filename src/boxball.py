import mujoco
import mujoco_viewer

def create_box_ball_model(box_solref=(0.02, 0), ball_solref=(0.02, 1)):
    # Create the Mujoco model
    model = mujoco.MjModel.from_xml_path('scene.xml')

    # Set mj_options struct parameters
    # model.opt.gravity = [0, 0, ]  # Set gravity

    # Set up other mj_options parameters as needed

    # Create the Mujoco viewer
    data = mujoco.MjData(model)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    return model, viewer

if __name__ == '__main__':
    # Create the model and viewer
    model, viewer = create_box_ball_model()

    # Simulate and render
    for _ in range(10000):
        if viewer.is_alive:
            mujoco.mj_step(model, viewer.data)
            viewer.render()
        else:
            break

    # Close the viewer
    viewer.close()
