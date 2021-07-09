import clip_inspect
import torch
import numpy as np
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm

model = clip_inspect.load_model("ViT-B-32.pt")
device = clip_inspect.get_device()

def from_spherical(angle_grid, r):
    return torch.stack([
        r * torch.cos(angle_grid[...,0]) * torch.sin(angle_grid[...,1]),
        r * torch.sin(angle_grid[...,0]) * torch.sin(angle_grid[...,1]),
        r * torch.cos(angle_grid[..., 1])
    ], dim=-1)

def to_spherical(xyz):
    return torch.stack([
        torch.atan2(xyz[...,1], xyz[...,0]),
        torch.atan2(torch.linalg.norm(xyz[...,:2], dim=-1), xyz[...,2]),
        torch.linalg.norm(xyz, dim=-1)
    ], dim=-1)

def to_spherical_4d(wxyz):
    return torch.stack([
        torch.atan2(torch.linalg.norm(wxyz[...,1:4], dim=-1), wxyz[...,0]),
        torch.atan2(torch.linalg.norm(wxyz[...,2:4], dim=-1), wxyz[...,1]),
        torch.atan2(wxyz[...,3], wxyz[...,2] + torch.linalg.norm(wxyz[...,2:4], dim=-1)),
        torch.linalg.norm(wxyz, dim=-1)
    ], dim=-1)

def stack_coords(model, layer, visual=False, n=300, in_dims=[0, 1, 2], steps=20, sizes=(-np.pi, np.pi), device=None):
    grid = clip_inspect.get_grid(n=n, dims=2, sizes=sizes)
    block = clip_inspect.CLIPResblockFF(model, layer-1, visual=visual)
    if device is not None:
        grid = grid.to(device)
        block = block.to(device)
    r = 1
    proj = from_spherical(grid, r)
    coords = [proj.cpu()]
    proj = block.components(proj, dims=in_dims)
    for i in tqdm(range(steps)):
        out_dict = block(
            proj,
            in_project=False,
            out_project=False,
            return_keys=["res"]
        )
        proj = out_dict["res"]
        out_coords = block.components(out_dict["res"], dims=in_dims + [3], reverse=True).cpu()
        out_coords = to_spherical_4d(out_coords)[...,:-1]
        coords.append(out_coords)
    return torch.stack(coords)

steps = 1000
n = 500
layer = 9
visual = False
draw_step = 5
draw_limit = 2

#load_path = None
load_path = "saves/CLIP_text_layer_9_n_500_steps_1000.npy"

if load_path is not None:
    _coords = np.load(load_path)
else:
    _coords = stack_coords(
        model, 
        layer, 
        visual=visual, 
        n=n, 
        in_dims=[0, 1, 2],
        steps=steps,
        sizes=(-np.pi, np.pi),
        device=device
    ).numpy()

    name = "visual" if visual else "text"
    title = F"saves/CLIP_{name}_layer_{layer}_n_{n}_steps_{steps}.npy"
    np.save(title, _coords)

draw_idx = 0
coords = _coords[[draw_idx,draw_idx+draw_step]]

steps = coords.shape[0]

connect = np.full(coords.shape[:-1], fill_value=True, dtype=np.bool8)
connect[-1,...] = False
should_draw = np.abs(coords[1:,...] - coords[:-1,...]).sum(axis=-1) <= draw_limit
connect[:-1,...] = should_draw

#color = np.array([plt.cm.viridis(i/(steps-1)) for i in range(steps)])
#color[:,-1] = np.linspace(0.5, 1, steps)
#color = np.tile(color, (n**2, 1))

color = np.array([plt.cm.viridis(i/((n**2)-1)) for i in range(n**2)])
color = np.repeat(color, steps, axis=0)
color[:,-1] = np.tile(np.linspace(0.01, 1, steps), n**2)

from vispy import app, visuals, scene

# build visuals
#Plot3D = scene.visuals.create_visual_node(visuals.line.line.LineVisual)

# build canvas
name = "Visual" if visual else "Text"
title = F"CLIP {name} Layer {layer}"
canvas = scene.SceneCanvas(keys='interactive', title=title, show=True)

# Add a ViewBox to let the user zoom/rotate
view = canvas.central_widget.add_view()
view.camera = 'fly'
view.camera.fov = 45
view.camera.distance = 6

pos = coords.transpose(1,2,0,3).reshape(-1,3)
connect = connect.transpose(1, 2, 0).reshape(-1)
plot3d = scene.Line(pos, width=30.0,
    color=color,# method='gl',
    connect=connect,
    parent=view.scene)

def update(ev):
    global draw_idx, pos, _coords, connect, plot3d
    draw_idx += 1
    if (draw_idx + draw_step) >= _coords.shape[0]:
        draw_idx = 0

    coords = _coords[[draw_idx,draw_idx+draw_step]]

    connect = np.full(coords.shape[:-1], fill_value=True, dtype=np.bool8)
    connect[-1,...] = False
    should_draw = np.abs(coords[1:,...] - coords[:-1,...]).sum(axis=-1) <= draw_limit
    connect[:-1,...] = should_draw

    pos = coords.transpose(1,2,0,3).reshape(-1,3)
    connect = connect.transpose(1, 2, 0).reshape(-1)
    plot3d.set_data(pos=pos, connect=connect)

timer = app.Timer(interval=1.0/60.0, connect=update, start=True)


if __name__ == '__main__':
    if sys.flags.interactive != 1:
        app.run()