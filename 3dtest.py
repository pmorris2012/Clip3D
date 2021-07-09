
import numpy as np
import sys

from vispy import app, visuals, scene

# build visuals
Plot3D = scene.visuals.create_visual_node(visuals.LinePlotVisual)

# build canvas
canvas = scene.SceneCanvas(keys='interactive', title='plot3d', show=True)

# Add a ViewBox to let the user zoom/rotate
view = canvas.central_widget.add_view()
view.camera = 'turntable'
view.camera.fov = 45
view.camera.distance = 6

# prepare data
N = 60
x = np.sin(np.linspace(-2, 2, N)*np.pi)
y = np.cos(np.linspace(-2, 2, N)*np.pi)
z = np.linspace(-2, 2, N)

for i in np.linspace(-1, 1, 100):
    # plot
    pos = np.c_[x-i, y+i, z]
    Plot3D(pos, width=10.0, color='white',
        edge_color='w', face_color=(0.2, 0.2, 1, 0.8),
        parent=view.scene)


if __name__ == '__main__':
    if sys.flags.interactive != 1:
        app.run()