
import numpy as np
import sys

from vispy import app, visuals, scene

# build visuals
Plot3D = scene.visuals.create_visual_node(visuals.line.line.LineVisual)

# build canvas
canvas = scene.SceneCanvas(keys='interactive', title='plot3d', show=True)

# Add a ViewBox to let the user zoom/rotate
view = canvas.central_widget.add_view()
view.camera = 'turntable'
view.camera.fov = 45
view.camera.distance = 6

# prepare data
x, y, z, segments = [], [], [], []
for start, i in enumerate(np.linspace(-5, 5, 1000)):
    N = 6000
    x.append(np.sin(np.linspace(-5-i, 5+1, N)*np.pi))
    y.append(np.cos(np.linspace(-5+i, 5-i, N)*np.pi))
    z.append(np.linspace(-5-i, 5-i, N))
    start_idx = 1000 * start
    idxs = np.arange(start_idx, start_idx+N-1)
    idxs = np.stack([idxs, idxs+1], axis=-1)
    segments.append(idxs)
x, y, z = np.concatenate(x), np.concatenate(y), np.concatenate(z)
segments = np.concatenate(segments, axis=0)

# plot
pos = np.c_[x, y, z]
Plot3D(pos, width=10.0,
    color=(1.0, 0.0, 0.0, 1.0), method='gl',
    connect=segments,
    parent=view.scene)


if __name__ == '__main__':
    if sys.flags.interactive != 1:
        app.run()