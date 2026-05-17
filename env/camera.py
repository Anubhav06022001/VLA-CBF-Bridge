import mujoco as mj
import numpy as np

def render_rgb(model, data, scene, context, cam, width=224, height=224):

    # Create viewport
    viewport = mj.MjrRect(0, 0, width, height)

    # Update scene for this camera
    mj.mjv_updateScene(
        model,
        data,
        mj.MjvOption(),
        None,
        cam,
        mj.mjtCatBit.mjCAT_ALL.value,
        scene
    )

    # Render
    mj.mjr_render(viewport, scene, context)

    # Read pixels
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    depth = np.zeros((height, width), dtype=np.float32)

    mj.mjr_readPixels(rgb, depth, viewport, context)

    # OpenGL renders upside-down
    rgb = np.flipud(rgb).copy()

    return rgb
