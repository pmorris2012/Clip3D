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

def lorenz(x, y, z, s=10, r=28, b=2.667):
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot


dt = 0.01
num_steps = 30
grid_dim = 40

# Need one more for the initial values
xs = np.empty((num_steps + 1, grid_dim**3))
ys = np.empty((num_steps + 1, grid_dim**3))
zs = np.empty((num_steps + 1, grid_dim**3))

start = np.linspace(-20, 20, grid_dim)
start = np.meshgrid(start, start, start)

colors = np.linspace(0, 1, grid_dim)
colors = np.meshgrid(colors, colors, colors)
colors = np.stack([*[c.reshape(-1) for c in colors], np.ones(grid_dim**3)], axis=-1)
colors = np.repeat(colors, num_steps + 1, axis=0)

# Set initial values
xs[0], ys[0], zs[0] = start[0].reshape(-1), start[1].reshape(-1), start[2].reshape(-1)
connect = np.full(num_steps+1, fill_value=True, dtype=np.bool8)
connect[-1] = False
connect = np.tile(connect, grid_dim**3)

# Step through "time", calculating the partial derivatives at the current point
# and using them to estimate the next point
for i in range(num_steps):
    x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
    xs[i + 1] = xs[i] + (x_dot * dt)
    ys[i + 1] = ys[i] + (y_dot * dt)
    zs[i + 1] = zs[i] + (z_dot * dt)

# plot
pos = np.c_[xs.T.reshape(-1), ys.T.reshape(-1), zs.T.reshape(-1)]
Plot3D(pos, width=10.0,
    color=colors,# method='gl',
    connect=connect,
    parent=view.scene)


if __name__ == '__main__':
    if sys.flags.interactive != 1:
        app.run()