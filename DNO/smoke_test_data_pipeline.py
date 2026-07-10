import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.DNO import DNO
from utils25 import flood_data


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("===== DNO DATA-PIPELINE SMOKE TEST =====")
print("Device:", device)

# Synthetic dataset dimensions matching the DNO configuration.
size_x = 64
size_y = 64
total_timesteps = 25
num_channels = 5
t_in = 1
t_out = 24

# Shape: [X, Y, T, channels]
pde = torch.zeros(
    size_x,
    size_y,
    total_timesteps,
    num_channels,
    dtype=torch.float32,
)

# H, U and V
pde[..., 0] = 0.10
pde[..., 1] = 0.01
pde[..., 2] = -0.01

# Rainfall
pde[..., 3] = 2.0

# Simple synthetic DEM
dem = torch.linspace(0.0, 100.0, size_x).reshape(size_x, 1, 1)
pde[..., 4] = dem.expand(size_x, size_y, total_timesteps)

with tempfile.TemporaryDirectory() as temporary_directory:
    data_directory = Path(temporary_directory)
    data_path = data_directory / "case_000.pt"

    torch.save(pde, data_path)

    print("Saved synthetic tensor:", tuple(pde.shape))
    print("Temporary PT file:", data_path.name)

    dataset = flood_data(
        path_root=data_directory,
        strategy="oneshot",
        T_in=t_in,
        T_out=t_out,
        train=True,
    )

    print("Dataset length:", len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    x, y, mask = next(iter(loader))

    print("Loader x shape:", tuple(x.shape))
    print("Loader y shape:", tuple(y.shape))
    print("Loader mask shape:", tuple(mask.shape))

    expected_x_shape = (1, 64, 64, 24, 1, 5)
    expected_y_shape = (1, 64, 64, 24, 3)

    assert tuple(x.shape) == expected_x_shape
    assert tuple(y.shape) == expected_y_shape
    assert tuple(mask.shape) == expected_y_shape
    assert torch.isfinite(x).all()
    assert torch.isfinite(y).all()

    model = DNO(
        num_channels=5,
        width=10,
        initial_step=1,
        pad=2,
        factor=1,
    ).to(device)

    model.eval()
    x = x.to(device)

    with torch.inference_mode():
        prediction = model(x)

    print("Prediction shape:", tuple(prediction.shape))
    print("Prediction finite:", bool(torch.isfinite(prediction).all().item()))
    print("Model padding after forward:", model.padding)

    assert tuple(prediction.shape) == expected_y_shape
    assert torch.isfinite(prediction).all()
    assert model.padding == 2

print("DNO data-pipeline smoke test completed successfully.")
