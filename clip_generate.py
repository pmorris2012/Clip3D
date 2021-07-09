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

for layer in [5]:
    steps = 1000
    n = 500
    #layer = 4
    visual = False
    draw_step = 5
    draw_limit = 2

    load_path = None
    #load_path = "saves/CLIP_text_layer_7_n_500_steps_1000_test.npy"

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
        title = F"saves/CLIP_{name}_layer_{layer}_n_{n}_steps_{steps}_test.npy"
        np.save(title, _coords)